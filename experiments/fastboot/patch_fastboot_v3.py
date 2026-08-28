#!/usr/bin/env python3
"""Anchored edit v3: fastboot with FULL skip on restore (no shard load, no
process_weights_after_loading) — testing whether the alias-aware v2 format
alone makes the skip safe. Dump still happens after the processing loop."""
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
        if _fb_on:
            from sglang.srt.model_loader.fastboot_qwen4 import (
                dump_after,
                try_restore,
            )

            if try_restore(model, _fb_dir, _fb_key, target_device):
                return
        # Used in tests to verify memory savings when using online quantization.
        if is_cuda_alike():'''
assert s.count(OLD_HEAD) == 1, f"head anchor x{s.count(OLD_HEAD)}"
s = s.replace(OLD_HEAD, NEW_HEAD)

OLD_TAIL = '''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)


class LayeredModelLoader(DefaultModelLoader):'''
NEW_TAIL = '''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        if _fb_on:
            dump_after(model, _fb_dir, _fb_key)


class LayeredModelLoader(DefaultModelLoader):'''
assert s.count(OLD_TAIL) == 1, f"tail anchor x{s.count(OLD_TAIL)}"
s = s.replace(OLD_TAIL, NEW_TAIL)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
py_compile.compile(p, doraise=True)
print("fastboot v3 (full skip on restore) wired + compile OK:", p)
