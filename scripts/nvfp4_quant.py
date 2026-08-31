#!/usr/bin/env python3
"""NVFP4 quantizer + a correctness gate.

NVFP4 = E2M1 values (4-bit, 2 per byte) with an E4M3 scale per 16-element block
along the input dim, plus one FP32 global scale. Before converting anything, we
validate the implementation by round-tripping NVIDIA's own expert tensors:
dequantize an official tensor, re-quantize with this code, and require the
packed bytes and block scales to come back identical. If they don't, the
implementation is wrong and we stop.
"""
import sys, torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP8_MAX = 448.0
E2M1_MAX = 6.0
BLOCK = 16


def _to_e2m1_code(x: torch.Tensor) -> torch.Tensor:
    """Round |x| to nearest E2M1 level, return 4-bit codes (sign<<3 | idx)."""
    sign = (x < 0).to(torch.uint8)
    a = x.abs()
    lut = E2M1.to(a.device, a.dtype)
    # nearest level by absolute distance (ties -> lower index, matches RN)
    idx = (a.unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.uint8)
    return (sign << 3) | idx


def _from_e2m1_code(code: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    lut = E2M1.to(code.device, dtype)
    val = lut[(code & 0x7).long()]
    return torch.where((code & 0x8) > 0, -val, val)


def quantize_nvfp4(w: torch.Tensor):
    """BF16/FP32 [out, in] -> (packed_u8 [out, in/2], scale_e4m3 [out, in/16],
    global_scale_f32 scalar). Mirrors modelopt's two-level scheme."""
    w = w.to(torch.float32)
    out_f, in_f = w.shape
    assert in_f % BLOCK == 0, f"input dim {in_f} not divisible by {BLOCK}"
    amax = w.abs().max()
    s2 = (amax / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)          # global fp32
    blocks = w.view(out_f, in_f // BLOCK, BLOCK)
    b_amax = blocks.abs().amax(dim=-1)                            # [out, nblk]
    b_scale = (b_amax / E2M1_MAX / s2).clamp(min=1e-12)
    b_scale_e4m3 = b_scale.to(torch.float8_e4m3fn)                # store as fp8
    eff = b_scale_e4m3.to(torch.float32) * s2                     # actual scale
    q = blocks / eff.unsqueeze(-1).clamp(min=1e-12)
    q = q.clamp(-E2M1_MAX, E2M1_MAX)
    codes = _to_e2m1_code(q).view(out_f, in_f)
    lo, hi = codes[:, 0::2], codes[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()
    return packed, b_scale_e4m3, s2.to(torch.float32)


def dequantize_nvfp4(packed, scale_e4m3, s2):
    out_f, half = packed.shape
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).view(out_f, half * 2)
    vals = _from_e2m1_code(codes)
    eff = (scale_e4m3.to(torch.float32) * s2).unsqueeze(-1)
    return (vals.view(out_f, -1, BLOCK) * eff).view(out_f, half * 2)


if __name__ == "__main__":
    # ---- correctness gate against NVIDIA's own tensors ----
    from safetensors import safe_open
    import json, glob, os
    D = sys.argv[1]
    idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
    targets = [n for n in idx if n.endswith(".gate_proj.weight")
               and "layers.0." in n and "mtp" not in n][:1]
    if not targets:
        targets = [n for n in idx if n.endswith("up_proj.weight") and "mtp" not in n][:1]
    ok = True
    for name in targets:
        base = name[:-len(".weight")]
        sh = idx[name]
        with safe_open(f"{D}/{sh}", framework="pt") as f:
            w_q = f.get_tensor(name)
            s1 = f.get_tensor(base + ".weight_scale")
            s2 = f.get_tensor(base + ".weight_scale_2")
        # official -> float -> our quantizer -> compare
        if w_q.dim() == 3:      # [experts, out, in/2] : test first expert
            w_q, s1 = w_q[0], s1[0]
        w_ref = dequantize_nvfp4(w_q, s1, s2)
        p2, sc2, g2 = quantize_nvfp4(w_ref)
        same_bytes = torch.equal(p2, w_q)
        same_scale = torch.equal(sc2.view(torch.uint8), s1.view(torch.uint8))
        err = (dequantize_nvfp4(p2, sc2, g2) - w_ref).abs().max().item()
        print(f"{name[:58]:58} bytes_match={same_bytes} scales_match={same_scale} "
              f"roundtrip_maxerr={err:.3e}")
        ok &= same_bytes and same_scale
    print("QUANTIZER_VALIDATED" if ok else "QUANTIZER_MISMATCH")
