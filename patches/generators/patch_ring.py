#!/usr/bin/env python3
"""QSA pending-ring widening: decouple the ring window from the compression
ratio so speculative_num_draft_tokens can exceed 4. Env SGLANG_QSA_RING_WINDOW
(multiple of compress_ratio, >= ratio) sets the per-request ring stride;
unset/invalid -> ratio (identical to stock). Groups still compress by ratio.
Edits: qsa_metadata.py (2 builders), qsa_kv_pool.py (alloc), and the guard in
the already-mounted qwen_sparse_attn_backend.py."""
import io, py_compile, sys

D = "/home/serverdestroyers/flashnext/"

# helper injected into metadata + pool
HELPER = '''

def _qsa_ring_stride(compress_ratio: int) -> int:
    """Per-request pending-ring stride. SGLANG_QSA_RING_WINDOW widens it beyond
    compress_ratio (must be a multiple of ratio) so a speculative verify window
    up to (stride - ratio) drafts fits without pos%stride collisions. Default =
    compress_ratio (stock behaviour)."""
    import os
    try:
        rw = int(os.environ.get("SGLANG_QSA_RING_WINDOW", "0"))
    except ValueError:
        rw = 0
    if rw >= compress_ratio and rw % compress_ratio == 0:
        return rw
    return compress_ratio
'''

# ---- metadata.py ----
mp = D + "qsa_metadata.py"
s = io.open(mp, encoding="utf-8").read()
# insert helper before the first builder
anchor = "def build_pending_ring_slots("
assert s.count(anchor) == 1
s = s.replace(anchor, HELPER.lstrip("\n") + "\n\n" + anchor, 1)
# pending builder: main formula + non-pending scratch
s = s.replace(
    "    slots = requests * compress_ratio + positions % compress_ratio\n"
    "    if is_extend:",
    "    stride = _qsa_ring_stride(compress_ratio)\n"
    "    slots = requests * stride + positions % stride\n"
    "    if is_extend:",
)
s = s.replace(
    "        slots = torch.where(pending, slots, positions % compress_ratio)",
    "        slots = torch.where(pending, slots, positions % compress_ratio)  # scratch [0,ratio)",
)
# group builder
s = s.replace(
    "    return requests[:, None] * compress_ratio + positions % compress_ratio",
    "    stride = _qsa_ring_stride(compress_ratio)\n"
    "    return requests[:, None] * stride + positions % stride",
)
io.open(mp, "w", encoding="utf-8").write(s)
py_compile.compile(mp, doraise=True)

# ---- qsa_kv_pool.py ----
pp = D + "qsa_kv_pool.py"
s = io.open(pp, encoding="utf-8").read()
# add module helper near top (after imports) — reuse same function name
assert "\nimport torch" in s
s = s.replace("\nimport torch", "\nimport torch" + HELPER, 1)
s = s.replace(
    "        ring_slots = self.qsa_num_request_slots * self.qsa_compress_ratio",
    "        self.qsa_ring_window = _qsa_ring_stride(self.qsa_compress_ratio)\n"
    "        ring_slots = self.qsa_num_request_slots * self.qsa_ring_window",
)
io.open(pp, "w", encoding="utf-8").write(s)
py_compile.compile(pp, doraise=True)

# ---- guard in the already-mounted backend ----
bp = D + "qwen_sparse_attn_backend.py"
s = io.open(bp, encoding="utf-8").read()
old_guard = '''        draft_tokens = int(getattr(spec_info, "draft_token_num", 0) or 0)
        if draft_tokens > self.compress_ratio:'''
new_guard = '''        draft_tokens = int(getattr(spec_info, "draft_token_num", 0) or 0)
        import os as _os
        try:
            _rw = int(_os.environ.get("SGLANG_QSA_RING_WINDOW", "0"))
        except ValueError:
            _rw = 0
        _stride = _rw if (_rw >= self.compress_ratio and _rw % self.compress_ratio == 0) else self.compress_ratio
        # ring holds a pending tail (< ratio) + the verify window; a window of
        # (stride - ratio) drafts never collides mod stride.
        if draft_tokens > _stride - self.compress_ratio:'''
assert s.count(old_guard) == 1, f"guard anchor x{s.count(old_guard)}"
s = s.replace(old_guard, new_guard)
# the error message references compress_ratio; leave it, it's still informative
io.open(bp, "w", encoding="utf-8").write(s)
py_compile.compile(bp, doraise=True)

print("all three files patched + compile OK")
