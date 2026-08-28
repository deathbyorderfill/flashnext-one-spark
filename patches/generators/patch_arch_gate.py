#!/usr/bin/env python3
"""Rewrite _resolve_trtllm_sparse_decode from unconditional-None to the
per-arch gate recommended in the sgl-project/sglang#36556 thread:
SM100 family and SM120 take trtllm-gen decode; SM121 (where those kernels
silently corrupt) takes the patched varlen fallback. On SM121 behavior is
identical to the previous unconditional None."""
import io, py_compile, sys

p = sys.argv[1] if len(sys.argv) > 1 else "/home/serverdestroyers/flashnext/qwen_sparse_attn_backend.py"

OLD = '''@lru_cache(maxsize=1)
def _resolve_trtllm_sparse_decode():
    """trtllm-gen paged decode for the post-gather sparse attention.

    On Blackwell the FA4 cute varlen fallback runs a prefill-shaped kernel
    at decode row counts; the trtllm-gen decode kernel over a page-aligned
    scratch measures ~35% faster for the gather+attention pair.
    """
    from sglang.srt.utils import is_sm100_supported

    if not is_sm100_supported():
        return None
    # sm121/GB10: flashinfer trtllm-gen decode kernels are SM100-only and
    # silently corrupt output here -- force the (patched) varlen fallback.
    return None
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    except ImportError:
        return None
    return trtllm_batch_decode_with_kv_cache'''

NEW = '''@lru_cache(maxsize=1)
def _resolve_trtllm_sparse_decode():
    """trtllm-gen paged decode for the post-gather sparse attention.

    Per-arch gate (sgl-project/sglang#36556 discussion): the trtllm-gen
    sparse decode kernels are validated correct on SM100 and SM120, but on
    SM121/GB10 they run without error and silently corrupt output (first
    token right, then NaN logits). SM121 takes the varlen fallback instead,
    which the direct-gather rewrite below makes safe.
    """
    import torch

    major, minor = torch.cuda.get_device_capability()
    if not (major == 10 or (major, minor) == (12, 0)):
        return None
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    except ImportError:
        return None
    return trtllm_batch_decode_with_kv_cache'''

s = io.open(p, encoding="utf-8").read()
assert s.count(OLD) == 1, f"anchor x{s.count(OLD)} in {p}"
io.open(p, "w", encoding="utf-8", newline="\n").write(s.replace(OLD, NEW))
py_compile.compile(p, doraise=True)
print("arch gate applied + compile OK:", p)
