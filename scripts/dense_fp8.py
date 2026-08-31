#!/usr/bin/env python3
"""Convert Flash-Next's largest BF16 dense GEMMs to FP8 (E4M3).

Why FP8 and not 4-bit: sglang merges in_proj_qkv+in_proj_z into one
in_proj_qkvz parameter and loads each as a shard sized by logical width.
NVFP4 packs 2 values per byte, so those shard sizes never match. FP8 is
1 byte/element, so the merge works untouched.

input_scale comes from live calibration (calib_amax_merged.json); sglang's
modelopt FP8 path requires it and the checkpoint ships none for dense modules.
Module names in the emitted config use sglang's INTERNAL prefixes
(model.layers.N....), which calibration revealed differ from the checkpoint's
HF names (model.language_model.layers.N....).
"""
import json, os, struct, sys, shutil, collections
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0
SRC, OUT, CALIB = sys.argv[1], sys.argv[2], sys.argv[3]

FP8_SUFFIXES = (
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
)


def hf_to_internal(name: str) -> str:
    """checkpoint name -> sglang runtime prefix (verified by calibration)."""
    return name.replace("model.language_model.", "model.")


def is_fp8_target(n):
    return (not n.startswith(("model.visual", "mtp."))
            and any(n.endswith(s) for s in FP8_SUFFIXES))


amax = json.load(open(CALIB))


def amax_for(module_internal):
    """Merged params share one activation scale; qkv and z both feed qkvz."""
    if module_internal in amax:
        return amax[module_internal]
    merged = module_internal.replace(".in_proj_qkv", ".in_proj_qkvz") \
                            .replace(".in_proj_z", ".in_proj_qkvz")
    return amax.get(merged)


idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wmap = idx["weight_map"]
affected = sorted({sh for n, sh in wmap.items() if is_fp8_target(n)})
print(f"{len(affected)} of {len(set(wmap.values()))} shards affected")

os.makedirs(OUT, exist_ok=True)
for sh in sorted(set(wmap.values())):
    if sh not in affected:
        dst = os.path.join(OUT, sh)
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(os.path.join(SRC, sh)), dst)

new_map = {n: sh for n, sh in wmap.items() if sh not in affected}
converted, saved, skipped = [], 0, []
for i, sh in enumerate(affected, 1):
    tensors = {}
    with safe_open(f"{SRC}/{sh}", framework="pt") as f:
        for n in f.keys():
            t = f.get_tensor(n)
            base = n[: -len(".weight")] if n.endswith(".weight") else None
            internal = hf_to_internal(base) if base else None
            a = amax_for(internal) if internal else None
            if is_fp8_target(n) and t.dtype in (torch.bfloat16, torch.float16):
                if a is None:
                    skipped.append(internal)
                    tensors[n] = t
                    continue
                w = t.float()
                w_amax = w.abs().max().clamp(min=1e-8)
                w_scale = (w_amax / FP8_MAX)
                q = (w / w_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                tensors[n] = q
                tensors[base + ".weight_scale"] = w_scale.to(torch.float32).reshape(())
                tensors[base + ".input_scale"] = torch.tensor(
                    a / FP8_MAX, dtype=torch.float32)
                saved += t.numel() * 2 - q.numel()
                converted.append(internal)
            else:
                tensors[n] = t
    save_file(tensors, f"{OUT}/{sh}", metadata={"format": "pt"})
    for n in tensors:
        new_map[n] = sh
    print(f"  shard {i}/{len(affected)} {sh}", flush=True)

idx["weight_map"] = new_map
json.dump(idx, open(f"{OUT}/model.safetensors.index.json", "w"))

# ---- config: FP8 for converted dense, NVFP4 for every packed expert ----
hdrs = {}
for sh in sorted(set(new_map.values())):
    with open(os.path.realpath(os.path.join(OUT, sh)), "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        hdrs.update(json.loads(f.read(hl)))

ql = {}
for n, m in hdrs.items():
    if n == "__metadata__" or not n.endswith(".weight_scale_2"):
        continue
    mod = n[: -len(".weight_scale_2")]
    w = hdrs.get(mod + ".weight")
    if w and w["dtype"] == "U8":
        ql[hf_to_internal(mod)] = {"quant_algo": "NVFP4", "group_size": 16}
for mod in converted:
    ql[mod] = {"quant_algo": "FP8"}

q = json.load(open(f"{SRC}/hf_quant_config.json"))
z = q.get("quantization", q)
z["quant_algo"] = "MIXED_PRECISION"
z["quantized_layers"] = ql
z.pop("exclude_modules", None)
json.dump(q, open(f"{OUT}/hf_quant_config.json", "w"))

for f in os.listdir(SRC):
    if f.endswith(".safetensors") or f in ("model.safetensors.index.json",
                                           "hf_quant_config.json"):
        continue
    p = os.path.join(SRC, f)
    if os.path.isfile(p):
        shutil.copy2(p, OUT)

c = collections.Counter(v["quant_algo"] for v in ql.values())
print(f"FP8-converted {len(converted)} modules, saved {saved/1e9:.2f} GB")
if skipped:
    print(f"skipped (no calibration): {len(skipped)}")
print(f"declared {len(ql)} modules: {dict(c)}")
print("FP8_CONVERT_DONE")
