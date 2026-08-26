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
import json, os, sys, time
import torch
import sympy
from safetensors import safe_open

R = int(os.environ.get("HASHK_R", "4"))
OUT = os.environ.get("HASHK_OUT", "/out/ple_hashk_R%d.pt" % R)
SNAP_GLOB = "/root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots"
DEV = "cuda"

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


def main():
    snap = os.path.join(SNAP_GLOB, os.listdir(SNAP_GLOB)[0])
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
