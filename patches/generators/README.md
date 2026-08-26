# Patch generators

These scripts document every modification in `patches/` as anchored string
edits against the pristine files from `lmsysorg/sglang:qwen38flashnext`.
They are the authoritative diff record: each edit asserts its anchor is
unique, so drift against a different image version fails loudly instead of
silently mis-patching.

Order of application (paths inside the scripts point at a working dir;
adjust or read them as documentation):

1. `make_nvfp4_patch.py`  - qwen4_exp.py: load-time NVFP4 PLE packing (packed mode)
2. `patch_hashk.py`       - qwen4_exp.py: HashK compressed-table mode (layered on 1)
3. `patch_qsa.py`         - backend: torch-SDPA varlen shim (kept for warmup paths)
4. `patch_qsa2.py`        - backend: direct gather+SDPA decode fallback, replacing
                            the broken `_compact_kv` pack path
   (the trtllm-decode disable and the fp8->bf16 `tl.dot` casts in
   sparse_attn.py / flash_fwd.py's TMA-O varlen guard were applied as
   one-line edits; see the shipped patched files and README's bug table)

The shipped files in `patches/` are the final result; you do not need to run
these generators to use the repo.
