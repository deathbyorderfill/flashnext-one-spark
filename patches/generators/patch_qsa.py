#!/usr/bin/env python3
"""Patch qwen_sparse_attn_backend.py: replace the FA varlen decode fallback
with a pure-torch SDPA implementation (exact for q_len==1)."""
import io
import py_compile

p = "/home/serverdestroyers/flashnext/qwen_sparse_attn_backend.py"
s = io.open(p, encoding="utf-8").read()

SHIM = '''def _torch_sdpa_varlen_func(
    q=None, k=None, v=None, cu_seqlens_q=None, cu_seqlens_k=None,
    max_seqlen_q=None, max_seqlen_k=None, softmax_scale=None, causal=True, **kw
):
    """Pure-torch replacement for the FA varlen decode fallback (q_len==1 only).

    The image's FA4-cute sm120 varlen kernels fail MLIR tracing (TMA-O rank bug,
    then PackGQA layout bug); with a single query token per sequence this SDPA
    path is exact and cheap, and every op is CUDA-graph capturable.
    """
    import torch.nn.functional as _F

    assert max_seqlen_q == 1, "torch varlen shim supports decode (q_len==1) only"
    B, H, D = q.shape
    Hkv = k.shape[1]
    topk = int(max_seqlen_k)
    lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.long)
    ar = torch.arange(topk, device=q.device)
    idx = (cu_seqlens_k[:-1].to(torch.long).unsqueeze(1) + ar.unsqueeze(0)).clamp_max(
        k.shape[0] - 1
    )
    mask = ar.unsqueeze(0) < lens.clamp_min(1).unsqueeze(1)
    kh = k[idx].permute(0, 2, 1, 3)
    vh = v[idx].permute(0, 2, 1, 3)
    if Hkv != H:
        rep = H // Hkv
        kh = kh.repeat_interleave(rep, dim=1)
        vh = vh.repeat_interleave(rep, dim=1)
    out = _F.scaled_dot_product_attention(
        q.unsqueeze(2), kh, vh, attn_mask=mask.view(B, 1, 1, topk),
        scale=softmax_scale,
    )
    out = out.squeeze(2)
    out = torch.where(lens.view(B, 1, 1) > 0, out, torch.zeros_like(out))
    return out


@lru_cache(maxsize=1)
def _resolve_flash_attn_varlen_func():'''

old = "@lru_cache(maxsize=1)\ndef _resolve_flash_attn_varlen_func():"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, SHIM)

old2 = """    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        pass"""
new2 = """    return _torch_sdpa_varlen_func
    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        pass"""
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, new2)

io.open(p, "w", encoding="utf-8").write(s)
py_compile.compile(p, doraise=True)
print("qsa backend patched + compiles")
