#!/usr/bin/env python3
"""Layer HashK-PLE mode into qwen4_exp_nvfp4.py (on top of the packed-NVFP4 edits).
Env SGLANG_QWEN4_PLE_HASHK=<artifact.pt> supersedes the packed mode: k=2 hashed
sub-tables + per-head ridge projection replace the 28.8 GB packed table."""
import io
import py_compile

p = "/home/serverdestroyers/flashnext/qwen4_exp_nvfp4.py"
s = io.open(p, encoding="utf-8").read()

HELPERS = '''
# ==== HashK-PLE: k-sub-table hash-compressed n-gram embedding ====
_HASHK_STATE = None


def _hashk_path():
    import os

    return os.environ.get("SGLANG_QWEN4_PLE_HASHK", "")


def _hashk_lsr(x, k):
    return (x >> k) & ((1 << (64 - k)) - 1)


def _hashk_splitmix(x):
    x = x + (-7046029254386353131)
    x = (x ^ _hashk_lsr(x, 30)) * (-4658895280553007687)
    x = (x ^ _hashk_lsr(x, 27)) * (-7723592293110705685)
    return x ^ _hashk_lsr(x, 31)


def _hashk_load(device):
    global _HASHK_STATE
    if _HASHK_STATE is not None:
        return _HASHK_STATE
    art = torch.load(_hashk_path(), map_location="cpu", weights_only=False)
    st = {
        "A": art["A"].to(device),
        "B": art["B"].to(device),
        "W": art["W"].to(device).to(torch.bfloat16),
        "offs": torch.tensor(art["offs"], device=device, dtype=torch.int64),
        "subsz": torch.tensor(art["sub_sizes"], device=device, dtype=torch.int64),
        "suboff": torch.tensor(art["sub_offs"][:-1], device=device, dtype=torch.int64),
        "heads": torch.arange(art["heads"], device=device, dtype=torch.int64),
        "salts": art["salts"],
        "hsalt": int(art["hsalt"]),
        "mulc": int(art["mulc"]),
        "half": int(art["half"]),
    }
    _HASHK_STATE = st
    logger.info(
        "HashK PLE loaded: R=%s sub-rows=%d (~%.1f GB) from %s",
        art.get("R"), st["A"].shape[0],
        (st["A"].numel() + st["B"].numel()) / 1e9, _hashk_path(),
    )
    return st


def _hashk_gather(emb, ids: torch.Tensor) -> torch.Tensor:
    st = _HASHK_STATE
    local = ids.to(torch.int64) - st["offs"]  # [T, 16] per-column head offsets
    hterm = st["heads"] * st["hsalt"]
    base = (local + 1) * st["mulc"]
    sA = st["suboff"] + torch.remainder(
        _hashk_splitmix(base + st["salts"][0] + hterm), st["subsz"]
    )
    sB = st["suboff"] + torch.remainder(
        _hashk_splitmix(base + st["salts"][1] + hterm), st["subsz"]
    )
    hat = torch.cat(
        [st["A"][sA].to(torch.bfloat16), st["B"][sB].to(torch.bfloat16)], dim=-1
    )  # [T, 16, 160]
    return torch.einsum("thd,hde->the", hat, st["W"])


# ==== end HashK-PLE ====


def _nvfp4_ple_enabled() -> bool:'''
old = "def _nvfp4_ple_enabled() -> bool:"
assert s.count(old) == 1
s = s.replace(old, HELPERS)

# __init__: hashk takes precedence over packed
old2 = """        if _nvfp4_ple_enabled():
            if getattr(config, "ple_offload_embedding", False):
                logger.info(
                    "NVFP4 PLE: disabling ple_offload_embedding (UMA box, packed storage)"
                )
                config.ple_offload_embedding = False
            _nvfp4_convert_embedding(self.ngram_embedding)"""
new2 = """        if _hashk_path():
            if getattr(config, "ple_offload_embedding", False):
                config.ple_offload_embedding = False
            w = self.ngram_embedding.weight
            device = w.device
            del self.ngram_embedding._parameters["weight"]
            torch.cuda.empty_cache()
            _hashk_load(device)
            self.ngram_embedding.hashk_mode = True
        elif _nvfp4_ple_enabled():
            if getattr(config, "ple_offload_embedding", False):
                logger.info(
                    "NVFP4 PLE: disabling ple_offload_embedding (UMA box, packed storage)"
                )
                config.ple_offload_embedding = False
            _nvfp4_convert_embedding(self.ngram_embedding)"""
assert s.count(old2) == 1
s = s.replace(old2, new2)

# loader: skip PLE shards entirely in hashk mode
old3 = """            emb = ple_mod.ngram_embedding
            if getattr(emb, "nvfp4_packed", False):"""
new3 = """            emb = ple_mod.ngram_embedding
            if getattr(emb, "hashk_mode", False):
                loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
                return True
            if getattr(emb, "nvfp4_packed", False):"""
assert s.count(old3) == 1
s = s.replace(old3, new3)

# gather branch
old4 = """        if getattr(self.ngram_embedding, "nvfp4_packed", False):
            embeddings = _nvfp4_gather(self.ngram_embedding, lookup_ids)"""
new4 = """        if getattr(self.ngram_embedding, "hashk_mode", False):
            embeddings = _hashk_gather(self.ngram_embedding, lookup_ids)
        elif getattr(self.ngram_embedding, "nvfp4_packed", False):
            embeddings = _nvfp4_gather(self.ngram_embedding, lookup_ids)"""
assert s.count(old4) == 1
s = s.replace(old4, new4)

io.open(p, "w", encoding="utf-8").write(s)
py_compile.compile(p, doraise=True)
print("HashK mode layered into qwen4_exp_nvfp4.py")
