#!/usr/bin/env python3
"""Build a hash-compressed PLE ("HashK") from Qwen3.8-Flash-Next's FP8 n-gram table.

Per head h (16 heads, prime-sized vocabs):
  - k=2 sub-tables, dims split 0:80 / 80:160, independent splitmix64 hashes
  - compression ratio R: sub-table size S_h = ceil(V_h / R)
  - slot value = mean of all original rows hashing to it (unbiased, trainless)
  - per-head 160x160 ridge-fitted projection W_h mapping reconstructions back
    toward true rows (the "linear transformation" stage)
Saves fp8 tables + bf16 W to an artifact for the runtime patch.
"""
import json, os, shutil, sys, time
import torch
import sympy
from safetensors import safe_open

R = int(os.environ.get("HASHK_R", "4"))
OUT = os.environ.get("HASHK_OUT", "/out/ple_hashk_R%d.pt" % R)
MODEL_ID = os.environ.get("HASHK_MODEL_ID", "RadixArk/Qwen3.8-Flash-Next-NVFP4")
SNAPSHOT = os.environ.get("HASHK_SNAPSHOT", "")  # explicit path; skips all lookup
HF_CACHE = os.environ.get("HF_HOME", "/root/.cache/huggingface")
SNAP_GLOB = os.path.join(
    HF_CACHE, "hub", "models--" + MODEL_ID.replace("/", "--"), "snapshots"
)
DEV = "cuda"
CKPT_GB = 135

NGRAM_BASE = 20_000_000
HEADS = 16
DIM = 160
HALF = 80
SPLIT_PARTS = 128  # split_ngram_parts

GAMMA = -7046029254386353131  # 0x9E3779B97F4A7C15 as signed int64
M1 = -4658895280553007687     # 0xBF58476D1CE4E5B9
M2 = -7723592293110705685     # 0x94D049BB133111EB
SALTS = (1234567891011121314, -8765432109876543211)  # sub-table salts
HSALT = 998244353


def _lsr(x, k):
    return (x >> k) & ((1 << (64 - k)) - 1)


def splitmix(x):
    x = x + GAMMA
    x = (x ^ _lsr(x, 30)) * M1
    x = (x ^ _lsr(x, 27)) * M2
    return x ^ _lsr(x, 31)


def hash_slot(local_idx, head, sub, size):
    x = (local_idx + 1) * 2862933555777941757 + SALTS[sub] + head * HSALT
    return torch.remainder(splitmix(x), size)


def head_sizes():
    sizes, offs, total = [], [], 0
    prime = NGRAM_BASE - 1
    for h in range(HEADS):
        prime = int(sympy.nextprime(prime))
        sizes.append(prime)
        offs.append(total)
        total += prime
    return sizes, offs, total


def _has_index(path):
    return bool(path) and os.path.isfile(
        os.path.join(path, "model.safetensors.index.json")
    )


def resolve_snapshot():
    """Return a checkpoint dir holding model.safetensors.index.json.

    Order: explicit HASHK_SNAPSHOT, then the mounted HF cache, then download.
    Set HASHK_NO_DOWNLOAD=1 to fail with instructions instead of fetching.
    """
    if SNAPSHOT:
        if not _has_index(SNAPSHOT):
            sys.exit(
                f"HASHK_SNAPSHOT={SNAPSHOT} has no model.safetensors.index.json"
            )
        return SNAPSHOT

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        snapshot_download = None

    if snapshot_download is not None:
        try:
            cached = snapshot_download(MODEL_ID, local_files_only=True)
            if _has_index(cached):
                return cached
        except Exception:
            pass  # not cached (or incomplete) -> fall through

    # Manually-populated caches may not match the hub layout exactly.
    if os.path.isdir(SNAP_GLOB):
        for name in sorted(os.listdir(SNAP_GLOB)):
            cand = os.path.join(SNAP_GLOB, name)
            if _has_index(cand):
                return cand

    if snapshot_download is None or os.environ.get("HASHK_NO_DOWNLOAD"):
        sys.exit(
            f"{MODEL_ID} is not in the mounted HF cache ({HF_CACHE}).\n"
            f"Fetch it first (~{CKPT_GB} GB — the same checkpoint launch.sh serves):\n"
            f"  huggingface-cli download {MODEL_ID}\n"
            "or point this script at an existing copy with "
            "HASHK_SNAPSHOT=/path/to/snapshot."
        )

    free_gb = shutil.disk_usage(HF_CACHE if os.path.isdir(HF_CACHE) else "/").free / 1e9
    if free_gb < CKPT_GB * 1.05:
        sys.exit(
            f"{MODEL_ID} needs ~{CKPT_GB} GB but the cache volume has only "
            f"{free_gb:.0f} GB free. Free space or mount a bigger volume at "
            f"{HF_CACHE}."
        )

    print(
        f"[hashk] {MODEL_ID} not cached — downloading ~{CKPT_GB} GB into {HF_CACHE} "
        "(resumable; launch.sh serves this same checkpoint)",
        flush=True,
    )
    got = snapshot_download(MODEL_ID, max_workers=8)
    if not _has_index(got):
        sys.exit(f"download finished but {got} has no model.safetensors.index.json")
    return got


def main():
    snap = resolve_snapshot()
    print(f"[hashk] checkpoint: {snap}", flush=True)
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    shard_names = {}
    for name, f in wmap.items():
        if ".ngram_embedding.shard_" in name and name.endswith(".weight"):
            n = int(name.rsplit("shard_", 1)[1].split(".")[0])
            shard_names[n] = (name, os.path.join(snap, f))
    assert len(shard_names) == SPLIT_PARTS, len(shard_names)

    sizes, offs, total = head_sizes()
    print(f"vocab total={total}  R={R}", flush=True)
    ss = (total + SPLIT_PARTS - 1) // SPLIT_PARTS  # rows per shard

    sub_sizes = [(v + R - 1) // R for v in sizes]
    sub_offs = [0]
    for s in sub_sizes:
        sub_offs.append(sub_offs[-1] + s)
    csize = sub_offs[-1]
    print(f"compressed rows per sub-table={csize} (~{csize*HALF*2/1e9:.1f} GB fp8 total)", flush=True)

    sumA = torch.zeros(csize, HALF, dtype=torch.float32, device=DEV)
    sumB = torch.zeros(csize, HALF, dtype=torch.float32, device=DEV)
    cntA = torch.zeros(csize, dtype=torch.float32, device=DEV)
    cntB = torch.zeros(csize, dtype=torch.float32, device=DEV)

    SAMPLE_EVERY = 640  # ~500k global sample rows
    samp_rows, samp_gidx = [], []

    t0 = time.time()
    sizes_t = torch.tensor(sizes, device=DEV)
    offs_t = torch.tensor(offs + [total], device=DEV)
    subsz_t = torch.tensor(sub_sizes, device=DEV)
    suboff_t = torch.tensor(sub_offs[:-1], device=DEV)

    for n in range(SPLIT_PARTS):
        name, path = shard_names[n]
        with safe_open(path, framework="pt", device="cpu") as f:
            w = f.get_tensor(name)
        g0 = n * ss
        rows = w.shape[0]
        w32 = w.to(DEV).to(torch.float32)
        gidx = torch.arange(g0, g0 + rows, device=DEV, dtype=torch.int64)
        valid = gidx < total
        gidx = gidx[valid]
        w32 = w32[valid]
        head = torch.bucketize(gidx, offs_t[1:], right=False)
        local = gidx - offs_t[head]
        sz = subsz_t[head]
        soff = suboff_t[head]
        sA = soff + hash_slot(local, head, 0, sz)
        sB = soff + hash_slot(local, head, 1, sz)
        sumA.index_add_(0, sA, w32[:, :HALF])
        sumB.index_add_(0, sB, w32[:, HALF:])
        cntA.index_add_(0, sA, torch.ones_like(sA, dtype=torch.float32))
        cntB.index_add_(0, sB, torch.ones_like(sB, dtype=torch.float32))
        pick = (gidx % SAMPLE_EVERY) == 0
        if pick.any():
            samp_rows.append(w32[pick].cpu())
            samp_gidx.append(gidx[pick].cpu())
        if n % 16 == 0:
            print(f"shard {n}/{SPLIT_PARTS} {time.time()-t0:.0f}s", flush=True)
        del w, w32
    meanA = sumA / cntA.clamp_min(1).unsqueeze(1)
    meanB = sumB / cntB.clamp_min(1).unsqueeze(1)
    del sumA, sumB
    torch.cuda.empty_cache()

    samp = torch.cat(samp_rows).to(DEV)
    sgi = torch.cat(samp_gidx).to(DEV)
    print(f"sample rows: {samp.shape[0]}", flush=True)

    Ws = torch.zeros(HEADS, DIM, DIM, dtype=torch.float32, device=DEV)
    lam = 1e-3
    for h in range(HEADS):
        m = (sgi >= offs[h]) & (sgi < offs[h] + sizes[h])
        gi = sgi[m]
        y = samp[m]
        local = gi - offs[h]
        sA = sub_offs[h] + hash_slot(local, torch.tensor(h, device=DEV), 0, torch.tensor(sub_sizes[h], device=DEV))
        sB = sub_offs[h] + hash_slot(local, torch.tensor(h, device=DEV), 1, torch.tensor(sub_sizes[h], device=DEV))
        hat = torch.cat([meanA[sA], meanB[sB]], dim=1)
        cos0 = torch.nn.functional.cosine_similarity(hat, y, dim=1).mean().item()
        X, Y = hat, y
        A_ = X.T @ X + lam * torch.eye(DIM, device=DEV)
        W = torch.linalg.solve(A_, X.T @ Y)
        cos1 = torch.nn.functional.cosine_similarity(X @ W, Y, dim=1).mean().item()
        Ws[h] = W
        print(f"head {h}: n={y.shape[0]} cos_raw={cos0:.4f} cos_proj={cos1:.4f}", flush=True)

    art = {
        "R": R, "heads": HEADS, "dim": DIM, "half": HALF,
        "sizes": sizes, "offs": offs, "sub_sizes": sub_sizes, "sub_offs": sub_offs,
        "salts": SALTS, "hsalt": HSALT, "mulc": 2862933555777941757,
        "A": meanA.to(torch.float8_e4m3fn).cpu(),
        "B": meanB.to(torch.float8_e4m3fn).cpu(),
        "W": Ws.to(torch.bfloat16).cpu(),
    }
    torch.save(art, OUT)
    print(f"saved {OUT} ({os.path.getsize(OUT)/1e9:.1f} GB) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
