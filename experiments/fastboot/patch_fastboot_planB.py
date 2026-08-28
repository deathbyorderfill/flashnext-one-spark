#!/usr/bin/env python3
"""Anchored edit: wire fastboot dump/restore into DefaultModelLoader.
load_weights_and_postprocess.

Restore mode skips ONLY the shard load (model.load_weights); the
process_weights_after_loading loop still runs — it sets non-tensor module
state (padded dims, split sizes, kernel runners) that forward kernels need —
and the dumped tensors are applied AFTER it, overwriting whatever processing
computed from the constructed garbage values. Dump mode captures tensors after
the same loop. Gated on SGLANG_FASTBOOT_DIR + Qwen4Exp* classes."""
import io, py_compile, sys

p = sys.argv[1] if len(sys.argv) > 1 else "/home/serverdestroyers/flashnext/loader.py"
s = io.open(p, encoding="utf-8").read()

OLD_HEAD = '''    @staticmethod
    def load_weights_and_postprocess(model, weights, target_device):
        # Used in tests to verify memory savings when using online quantization.
        if is_cuda_alike():'''
NEW_HEAD = '''    @staticmethod
    def load_weights_and_postprocess(model, weights, target_device):
        import os as _os

        _fb_dir = _os.environ.get("SGLANG_FASTBOOT_DIR", "")
        _fb_key = type(model).__name__
        _fb_on = bool(_fb_dir) and _fb_key.startswith("Qwen4Exp")
        _fb_restore = _fb_on and _os.path.isfile(
            _os.path.join(_fb_dir, _fb_key, "MANIFEST.json")
        )
        # Used in tests to verify memory savings when using online quantization.
        if is_cuda_alike():'''
assert s.count(OLD_HEAD) == 1, f"head anchor x{s.count(OLD_HEAD)}"
s = s.replace(OLD_HEAD, NEW_HEAD)

OLD_LW1 = '''            ):
                model.load_weights(weights)
            if target_device.type == "cuda":'''
NEW_LW1 = '''            ):
                if not _fb_restore:
                    model.load_weights(weights)
            if target_device.type == "cuda":'''
assert s.count(OLD_LW1) == 1, f"lw1 anchor x{s.count(OLD_LW1)}"
s = s.replace(OLD_LW1, NEW_LW1)

OLD_LW2 = '''        else:
            model.load_weights(weights)

        # Used in tests to verify memory savings when using online quantization.'''
NEW_LW2 = '''        else:
            if not _fb_restore:
                model.load_weights(weights)

        # Used in tests to verify memory savings when using online quantization.'''
assert s.count(OLD_LW2) == 1, f"lw2 anchor x{s.count(OLD_LW2)}"
s = s.replace(OLD_LW2, NEW_LW2)

OLD_TAIL = '''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)


class LayeredModelLoader(DefaultModelLoader):'''
NEW_TAIL = '''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        if _fb_on:
            from sglang.srt.model_loader.fastboot_qwen4 import (
                dump_after,
                try_restore,
            )

            if _fb_restore:
                try_restore(model, _fb_dir, _fb_key, target_device)
            else:
                dump_after(model, _fb_dir, _fb_key)


class LayeredModelLoader(DefaultModelLoader):'''
assert s.count(OLD_TAIL) == 1, f"tail anchor x{s.count(OLD_TAIL)}"
s = s.replace(OLD_TAIL, NEW_TAIL)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
py_compile.compile(p, doraise=True)
print("fastboot (post-processing restore) wired into loader + compile OK:", p)
