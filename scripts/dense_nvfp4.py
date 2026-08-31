#!/usr/bin/env python3
"""Convert Flash-Next's BF16 dense GEMMs to W4A16_NVFP4 using modelopt's own
quantizer, so the encoding is bit-identical to how the experts were made.

Why: only 10/512 experts fire per token, so the already-NVFP4 experts are just
13% of each token's memory read. The BF16 dense path (linear_attn, self_attn,
lm_head) is ~87%. Cutting it 16->4.5 bpw is the lever for decode throughput.

W4A16 (BF16 activations) is used deliberately: it needs no activation
calibration, unlike W4A4 which requires a static input_scale.

Modes:
  validate  - round-trip NVIDIA's own tensors through modelopt; require exact bytes
  convert   - write a new checkpoint with dense modules quantized
"""
import json, os, struct, sys, shutil, collections
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from modelopt.torch.quantization.qtensor import NVFP4QTensor

BLOCK = 16
# Dense GEMMs worth converting, largest-first. Deliberately excluded:
#   embed_tokens (lookup, not a GEMM), *_hyper_connection (small mixing
#   matrices, sensitivity unknown), mlp.gate / shared_expert_gate (router:
#   tiny and decisive for expert choice), norms, mtp (draft head).
# NOTE: fused/sharded modules (linear_attn.in_proj_qkv, self_attn q/k/v) are
# EXCLUDED. sglang splits those by logical sizes at load time, which is
# incompatible with 2-per-byte packing -> assert param_data.shape ==
# loaded_weight.shape. Only standalone projections are converted.
# sglang MERGES in_proj_qkv + in_proj_z into one in_proj_qkvz parameter
# (qwen3_5.py:448) and fuses self_attn q/k/v, loading each as a shard sized by
# LOGICAL width -- incompatible with 2-per-byte packing. Only genuinely
# standalone RowParallel/head projections are safe to convert.
TARGET_SUFFIXES = (
    "linear_attn.out_proj.weight",
    "self_attn.o_proj.weight",
    "lm_head.weight",
)


def is_target(name: str) -> bool:
    if name.startswith("model.visual") or name.startswith("mtp."):
        return False
    return any(name.endswith(s) for s in TARGET_SUFFIXES)


def unpack_parts(qt, scale, gscale):
    """Normalize modelopt's return into (packed_u8, e4m3_scale, f32_global)."""
    packed = qt._quantized_data if hasattr(qt, "_quantized_data") else qt
    return packed, scale, gscale


def validate(D):
    idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
    name = next(n for n in idx
                if n.endswith(".gate_proj.weight") and "mtp" not in n)
    base = name[: -len(".weight")]
    with safe_open(f"{D}/{idx[name]}", framework="pt") as f:
        w_q = f.get_tensor(name)
        s1 = f.get_tensor(base + ".weight_scale")
        s2 = f.get_tensor(base + ".weight_scale_2")
    if w_q.dim() == 3:
        w_q, s1 = w_q[0], s1[0]
    # NVFP4QTensor carries the LOGICAL shape; stored data is packed 2-per-byte
    logical = torch.Size((w_q.shape[0], w_q.shape[1] * 2))
    ref = NVFP4QTensor(logical, torch.bfloat16, w_q).dequantize(
        scale=s1, double_scale=s2, block_sizes={-1: BLOCK})
    out = NVFP4QTensor.quantize(ref, BLOCK, weights_scaling_factor=s1,
                                weights_scaling_factor_2=s2)
    qt = out[0] if isinstance(out, (tuple, list)) else out
    packed = qt._quantized_data if hasattr(qt, "_quantized_data") else qt
    match = torch.equal(packed.view(torch.uint8).cpu(), w_q.view(torch.uint8).cpu())
    print(f"{name[:60]:60} bytes_match={match}")
    print("QUANTIZER_VALIDATED" if match else "QUANTIZER_MISMATCH")
    return match


def convert(D, OUT):
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(open(f"{D}/model.safetensors.index.json"))
    wmap = idx["weight_map"]
    shards = sorted(set(wmap.values()))
    # Only a handful of shards hold the dense GEMMs; symlink the rest so the
    # conversion costs minutes and a few GB instead of rewriting 135 GB.
    affected = sorted({sh for n, sh in wmap.items() if is_target(n)})
    print(f"{len(affected)} of {len(shards)} shards need rewriting: {affected}")
    for sh in shards:
        if sh not in affected:
            dst = os.path.join(OUT, sh)
            if not os.path.exists(dst):
                os.symlink(os.path.realpath(os.path.join(D, sh)), dst)
    new_map, converted, saved = {}, [], 0
    for n, sh in wmap.items():
        if sh not in affected:
            new_map[n] = sh
    for i, sh in enumerate(affected, 1):
        tensors = {}
        with safe_open(f"{D}/{sh}", framework="pt") as f:
            for n in f.keys():
                t = f.get_tensor(n)
                if is_target(n) and t.dtype in (torch.bfloat16, torch.float16):
                    before = t.numel() * t.element_size()
                    out = NVFP4QTensor.quantize(t.cuda().float(), BLOCK)
                    qt = out[0] if isinstance(out, (tuple, list)) else out
                    sc = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
                    gs = out[2] if isinstance(out, (tuple, list)) and len(out) > 2 else None
                    packed = (qt._quantized_data if hasattr(qt, "_quantized_data") else qt).cpu()
                    b = n[: -len(".weight")]
                    tensors[n] = packed
                    if sc is not None:
                        tensors[b + ".weight_scale"] = sc.cpu()
                    if gs is not None:
                        tensors[b + ".weight_scale_2"] = gs.cpu().to(torch.float32)
                    saved += before - packed.numel()
                    converted.append(b)
                else:
                    tensors[n] = t
        save_file(tensors, f"{OUT}/{sh}", metadata={"format": "pt"})
        for n in tensors:
            new_map[n] = sh
        print(f"  shard {i}/{len(affected)} {sh} ({len(tensors)} tensors)", flush=True)
    idx["weight_map"] = new_map
    idx.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(os.path.realpath(f"{OUT}/{s}")) for s in shards)
    json.dump(idx, open(f"{OUT}/model.safetensors.index.json", "w"))

    # config: MIXED_PRECISION, dense -> W4A16_NVFP4, experts stay NVFP4
    q = json.load(open(f"{D}/hf_quant_config.json"))
    z = q.get("quantization", q)
    ql = {}
    for m in converted:
        ql[m] = {"quant_algo": "W4A16_NVFP4", "group_size": BLOCK}
    z["quantized_layers"] = ql
    z["quant_algo"] = "MIXED_PRECISION"
    json.dump(q, open(f"{OUT}/hf_quant_config.json", "w"), indent=1)

    for f in os.listdir(D):
        if f.endswith(".safetensors") or f in ("model.safetensors.index.json",
                                               "hf_quant_config.json"):
            continue
        p = os.path.join(D, f)
        if os.path.isfile(p):
            shutil.copy2(p, OUT)
    print(f"converted {len(converted)} dense modules, saved {saved/1e9:.2f} GB")
    print("CONVERT_DONE")


if __name__ == "__main__":
    mode, D = sys.argv[1], sys.argv[2]
    if mode == "validate":
        sys.exit(0 if validate(D) else 1)
    convert(D, sys.argv[3])
