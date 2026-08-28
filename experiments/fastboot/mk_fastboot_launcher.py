#!/usr/bin/env python3
"""Create /tmp/launch_fastboot.sh: canonical launcher + loader/fastboot mounts."""
import io

SRC = "/home/serverdestroyers/flashnext/launch_flashnext.sh"
DST = "/tmp/launch_fastboot.sh"

s = io.open(SRC, encoding="utf-8").read()
anchor = "  -e SGLANG_QWEN4_PLE_HASHK=/patches/ple_hashk_R4.pt \\\n"
assert s.count(anchor) == 1, f"anchor x{s.count(anchor)}"
add = (
    "  -v /home/serverdestroyers/flashnext/loader.py:"
    "/sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py:ro \\\n"
    "  -v /home/serverdestroyers/flashnext/fastboot_qwen4.py:"
    "/sgl-workspace/sglang/python/sglang/srt/model_loader/fastboot_qwen4.py:ro \\\n"
    "  -e SGLANG_FASTBOOT_DIR=/patches/fastboot \\\n"
)
io.open(DST, "w", encoding="utf-8", newline="\n").write(s.replace(anchor, add + anchor))
print("test launcher written:", DST)
