#!/usr/bin/env python3
"""Produce qwen4_exp_nvfp4.py: qwen4_exp.py + packed-NVFP4 PLE support.
Anchored string edits; every anchor asserted unique so drift fails loudly."""
import io, sys

SRC = "/home/serverdestroyers/flashnext/qwen4_exp.py"
DST = "/home/serverdestroyers/flashnext/qwen4_exp_nvfp4.py"

s = io.open(SRC, encoding="utf-8").read()

def sub1(old, new):
    global s
    assert s.count(old) == 1, f"anchor not unique ({s.count(old)}x): {old[:80]!r}"
    s = s.replace(old, new)

# ---- Edit A: module-level helpers, inserted before _get_ple_forward_mode ----
HELPERS = '''
# ==== NVFP4-packed PLE support (local patch; env SGLANG_QWEN4_PLE_NVFP4=1) ====
_NVFP4_LUT = None
_NVFP4_MIDS = None


def _nvfp4_ple_enabled() -> bool:
    import os

    return os.environ.get("SGLANG_QWEN4_PLE_NVFP4", "0") == "1"


def _nvfp4_get_lut(device):
    global _NVFP4_LUT
    if _NVFP4_LUT is None or _NVFP4_LUT.device != device:
        mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        _NVFP4_LUT = torch.tensor(
            mags + [-m for m in mags], dtype=torch.bfloat16, device=device
        )
    return _NVFP4_LUT


def _nvfp4_get_mids(device):
    global _NVFP4_MIDS
    if _NVFP4_MIDS is None or _NVFP4_MIDS.device != device:
        _NVFP4_MIDS = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            dtype=torch.float32,
            device=device,
        )
    return _NVFP4_MIDS


def _nvfp4_quantize_rows(w, chunk_rows: int = 262144):
    """rows [N, D] (fp8/bf16, any device) -> (uint8 [N, D//2], bf16 [N, D//16]) on cuda."""
    n, d = w.shape
    assert d % 16 == 0
    dev = torch.device("cuda")
    packed_out = torch.empty(n, d // 2, dtype=torch.uint8, device=dev)
    gscale_out = torch.empty(n, d // 16, dtype=torch.float8_e4m3fn, device=dev)
    mids = _nvfp4_get_mids(dev)
    for st in range(0, n, chunk_rows):
        en = min(st + chunk_rows, n)
        x = w[st:en].to(device=dev).to(torch.float32).view(en - st, d // 16, 16)
        amax = x.abs().amax(dim=-1)
        scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
        q = x / scale.unsqueeze(-1)
        idx = torch.bucketize(q.abs(), mids).to(torch.uint8)
        code = torch.where(q < 0, idx + 8, idx).view(en - st, d)
        packed_out[st:en] = code[:, 0::2] | (code[:, 1::2] << 4)
        gscale_out[st:en] = scale.to(torch.float8_e4m3fn)
        del x, amax, scale, q, idx, code
    return packed_out, gscale_out


def _nvfp4_convert_embedding(emb) -> None:
    """Swap a VocabParallelEmbedding weight for packed-NVFP4 buffers (tp1 only)."""
    w = emb.weight
    n, d = w.shape
    device = w.device
    del emb._parameters["weight"]
    emb.register_buffer(
        "weight_packed",
        torch.zeros(n, d // 2, dtype=torch.uint8, device=device),
        persistent=False,
    )
    emb.register_buffer(
        "weight_gscale",
        torch.zeros(n, d // 16, dtype=torch.float8_e4m3fn, device=device),
        persistent=False,
    )
    emb.nvfp4_packed = True
    torch.cuda.empty_cache()


def _nvfp4_gather(emb, ids: torch.Tensor) -> torch.Tensor:
    d = emb.weight_packed.shape[1] * 2
    p = emb.weight_packed[ids]
    lut = _nvfp4_get_lut(p.device)
    lo = lut[(p & 0xF).to(torch.long)]
    hi = lut[(p >> 4).to(torch.long)]
    codes = torch.stack((lo, hi), dim=-1).reshape(*ids.shape, d)
    gs = emb.weight_gscale[ids].to(torch.bfloat16)
    out = codes.view(*ids.shape, d // 16, 16) * gs.unsqueeze(-1)
    return out.reshape(*ids.shape, d)


# ==== end NVFP4-packed PLE support ====


def _get_ple_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:'''
sub1("def _get_ple_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:", HELPERS)

# ---- Edit B: convert to packed storage at the end of NGramEmbedding.__init__ ----
sub1(
    '''        self.ngram_embedding.register_buffer(
            "weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=True
        )''',
    '''        self.ngram_embedding.register_buffer(
            "weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=True
        )
        if _nvfp4_ple_enabled():
            if getattr(config, "ple_offload_embedding", False):
                logger.info(
                    "NVFP4 PLE: disabling ple_offload_embedding (UMA box, packed storage)"
                )
                config.ple_offload_embedding = False
            _nvfp4_convert_embedding(self.ngram_embedding)
            logger.info(
                "PLE embedding using packed NVFP4 storage: rows=%d dim=%d",
                self.ngram_embedding.weight_packed.shape[0],
                self.ngram_embedding.weight_packed.shape[1] * 2,
            )''',
)

# ---- Edit C: gather branch in _embed_ngram_ids ----
sub1(
    '''        embeddings = self.ngram_embedding(lookup_ids)
        embeddings = embeddings * self.ngram_embedding.weight_scale''',
    '''        if getattr(self.ngram_embedding, "nvfp4_packed", False):
            embeddings = _nvfp4_gather(self.ngram_embedding, lookup_ids)
        else:
            embeddings = self.ngram_embedding(lookup_ids)
        embeddings = embeddings * self.ngram_embedding.weight_scale''',
)

# ---- Edit D: loader branch (quantize each fp8 shard on the fly) ----
sub1(
    '''            emb = ple_mod.ngram_embedding
            if (
                loaded_weight.dtype == torch.float8_e4m3fn
                and emb.weight.dtype != torch.float8_e4m3fn
            ):''',
    '''            emb = ple_mod.ngram_embedding
            if getattr(emb, "nvfp4_packed", False):
                shard_size = (
                    emb.org_vocab_size + ple_num_sync_shards - 1
                ) // ple_num_sync_shards
                row_start = shard_idx * shard_size
                row_end = row_start + loaded_weight.shape[0]
                tp_start = emb.shard_indices.org_vocab_start_index
                tp_end = emb.shard_indices.org_vocab_end_index
                ov_start = max(row_start, tp_start)
                ov_end = min(row_end, tp_end)
                if ov_start < ov_end:
                    local_start = ov_start - tp_start
                    src_start = ov_start - row_start
                    n_rows = ov_end - ov_start
                    pk, gs = _nvfp4_quantize_rows(
                        loaded_weight[src_start : src_start + n_rows]
                    )
                    emb.weight_packed[local_start : local_start + n_rows].copy_(pk)
                    emb.weight_gscale[local_start : local_start + n_rows].copy_(gs)
                    del pk, gs
                loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
                torch.cuda.empty_cache()
                return True
            if (
                loaded_weight.dtype == torch.float8_e4m3fn
                and emb.weight.dtype != torch.float8_e4m3fn
            ):''',
)

io.open(DST, "w", encoding="utf-8").write(s)
import py_compile

py_compile.compile(DST, doraise=True)
print("patched + compiles OK:", DST)
