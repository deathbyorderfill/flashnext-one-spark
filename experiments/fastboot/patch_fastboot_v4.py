#!/usr/bin/env python3
"""Anchored edit v4: whole-model pickle fastboot in DefaultModelLoader.
load_model. On a miss: normal load+postprocess, then torch.save(model) —
capturing tensors, storage aliasing, AND all python-side module state in one
artifact. On a hit: discard the freshly constructed skeleton and torch.load
the pickled module. Gated on SGLANG_FASTBOOT_DIR + Qwen4Exp* classes."""
import io, py_compile, sys

p = sys.argv[1] if len(sys.argv) > 1 else "/home/serverdestroyers/flashnext/loader.py"
s = io.open(p, encoding="utf-8").read()

OLD = '''            self.load_weights_and_postprocess(
                model, self._get_all_weights(model_config, model), target_device
            )

        self.counter_after_loading_weights = time.perf_counter()
        return model.eval()'''
NEW = '''            import gc as _gc
            import logging as _logging
            import os as _fbos
            import time as _fbtime

            _fb_dir = _fbos.environ.get("SGLANG_FASTBOOT_DIR", "")
            _fb_key = type(model).__name__
            _fb_pkl = (
                _fbos.path.join(_fb_dir, _fb_key + ".model.pt")
                if _fb_dir and _fb_key.startswith("Qwen4Exp")
                else ""
            )
            _fb_log = _logging.getLogger(__name__)
            if _fb_pkl and _fbos.path.isfile(_fb_pkl):
                _t0 = _fbtime.time()
                del model
                _gc.collect()
                torch.cuda.empty_cache()
                with open(_fb_pkl, "rb") as _f:
                    model = torch.load(_f, weights_only=False)
                    try:
                        _fbos.posix_fadvise(
                            _f.fileno(), 0, 0, _fbos.POSIX_FADV_DONTNEED
                        )
                    except (AttributeError, OSError):
                        pass
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                _fb_log.info(
                    "[fastboot] unpickled %s in %.0fs (load+postprocess skipped)",
                    _fb_key, _fbtime.time() - _t0,
                )
            else:
                self.load_weights_and_postprocess(
                    model, self._get_all_weights(model_config, model), target_device
                )
                if _fb_pkl:
                    _t0 = _fbtime.time()
                    try:
                        torch.save(model, _fb_pkl)
                        _fb_log.info(
                            "[fastboot] pickled %s in %.0fs",
                            _fb_key, _fbtime.time() - _t0,
                        )
                    except Exception as _e:
                        _fb_log.warning(
                            "[fastboot] pickle failed (%s: %s); serving normally",
                            type(_e).__name__, str(_e)[:200],
                        )
                        try:
                            _fbos.remove(_fb_pkl)
                        except OSError:
                            pass

        self.counter_after_loading_weights = time.perf_counter()
        return model.eval()'''
assert s.count(OLD) == 1, f"anchor x{s.count(OLD)}"
s = s.replace(OLD, NEW)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
py_compile.compile(p, doraise=True)
print("fastboot v4 (whole-model pickle) wired + compile OK:", p)
