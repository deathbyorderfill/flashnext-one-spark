#!/usr/bin/env python3
"""Append activation-amax calibration hooks to qwen4_exp_mmap.py.

FP8 in sglang's modelopt path requires a static per-tensor input_scale
(= amax/448); the checkpoint ships those only for experts, not for the dense
modules we want to convert. This collects them by hooking the live model:
set SGLANG_CALIB_OUT=/path.json, send prompts, and the amax per module prefix
is dumped on exit. Names come from named_modules() on the real model, so they
are exactly the prefixes sglang resolves quantization against.
"""
PATCH = '''

# ---------------------------------------------------------------- calibration
import os as _calib_os
if _calib_os.environ.get("SGLANG_CALIB_OUT"):
    import atexit as _atexit, json as _json, torch as _torch, signal as _signal

    _CALIB_AMAX = {}
    _CALIB_NAMES = {}
    _CALIB_TARGET = (
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
        "linear_attn.in_proj_qkvz", "linear_attn.out_proj",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.qkv_proj", "self_attn.o_proj", "lm_head",
    )

    def _calib_dump(*_a):
        out = _calib_os.environ["SGLANG_CALIB_OUT"]
        try:
            with open(out, "w") as f:
                _json.dump(_CALIB_AMAX, f)
            print(f"[calib] wrote {len(_CALIB_AMAX)} module amax -> {out}", flush=True)
        except Exception as e:
            print(f"[calib] dump failed: {e}", flush=True)

    _atexit.register(_calib_dump)
    try:
        _signal.signal(_signal.SIGTERM, lambda *a: (_calib_dump(), _signal.default_int_handler))
    except Exception:
        pass

    def _calib_pre_hook(mod, args):
        name = _CALIB_NAMES.get(id(mod))
        if name is None or not args:
            return
        x = args[0]
        if _torch.is_tensor(x) and x.numel():
            try:
                v = x.detach().abs().amax().float().item()
            except Exception:
                return
            if v > _CALIB_AMAX.get(name, 0.0):
                _CALIB_AMAX[name] = v

    _CALIB_INSTALLED = {"done": False}

    def _calib_install(root):
        if _CALIB_INSTALLED["done"]:
            return
        _CALIB_INSTALLED["done"] = True
        n = 0
        for name, mod in root.named_modules():
            if any(name.endswith(t) for t in _CALIB_TARGET):
                _CALIB_NAMES[id(mod)] = name
                mod.register_forward_pre_hook(_calib_pre_hook)
                n += 1
        print(f"[calib] hooked {n} modules", flush=True)

    _orig_fwd = Qwen4ExpForConditionalGeneration.forward

    def _calib_forward(self, *a, **kw):
        _calib_install(self)
        return _orig_fwd(self, *a, **kw)

    Qwen4ExpForConditionalGeneration.forward = _calib_forward
'''

import sys
p = sys.argv[1]
s = open(p).read()
if "SGLANG_CALIB_OUT" in s:
    print("already patched")
else:
    open(p, "w").write(s + PATCH)
    print("calibration hooks appended")
