# Fastboot experiment — PARKED (negative result, mechanism fully mapped)

Goal: cut the ~9-minute boot to ~3 by serializing the model's post-processed
memory once and restoring it on later boots, skipping the per-tensor NVFP4
prep. **Outcome after 8 instrumented boots: not achievable by tensor
serialization alone.** The findings below are the requirements document for
anyone attempting the real fix.

## What was proven

1. **I/O was never the cost.** Restoring all 97 GB of final tensors takes
   ~60–70 s. The boot budget lives in weight *processing*, and the expensive
   part is `process_weights_after_loading` (~6 of the 9 minutes), not the
   shard reads.
2. **`state_dict()` is not the model's memory.** Non-persistent buffers
   (`gemma_weight`, `cos_sin_cache`, …) appear in neither `state_dict()` nor
   `vars(module)` — only in `module._buffers`. Missing them produces illegal
   memory accesses at first forward.
3. **Tensors alias heavily.** 276 of the main model's 2,198 tensors are views
   into shared storage (GDN fused qkvzba splits and friends). Restoring them
   as independent tensors breaks stride arithmetic → IMA in
   `fused_qkvzba_split_reshape_cat_contiguous`. The dump must be
   storage-level: save each unique storage once, record every tensor as
   (storage, dtype, shape, stride, offset), rebuild views on restore
   (`fastboot_qwen4.py` implements this; it also shrank the dump
   89.4 → 78.8 GB).
4. **Two UMA-specific landmines** (GB10 unified memory):
   - the page cache of the dump files competes with GPU allocations in the
     same pool — `posix_fadvise(DONTNEED)` after each file read, or KV-pool
     sizing starves;
   - interleaving alloc-new/free-old across ~2k tensors fragments the caching
     allocator by ~17 GB — free everything first, then load into a clean
     allocator.
5. **The decisive blocker:** `process_weights_after_loading` sets load-bearing
   *non-tensor* module state (padded dims, split sizes, kernel runner
   objects). Skip it and forward IMAs even with byte-perfect tensors
   (verified with CUDA_LAUNCH_BLOCKING). Keep it and the boot is correct but
   no faster (604 s measured, probes byte-healthy) because the loop itself is
   the cost.
6. **Whole-model pickling is dead on arrival:** `torch.save(model)` hits
   `cannot pickle ProcessGroup` (main model) and `cannot pickle Stream`
   (draft model) — unpicklable handles nest inside modules and runners even
   at tp=1, with no clean serialization boundary.

## What a real fix looks like

Serialize the *effects* of processing, not just tensors: a processed-
checkpoint export format (per-module tensor payload + an allowlist of the
plain-python attrs processing sets, with runner objects rebuilt cheaply from
those attrs at load). That is upstream-shaped work in the quant methods
themselves. The DeepSeek Spark image's 106-GB-in-150-s restore shows the
ceiling once processing effects are cached.

## Files

- `fastboot_qwen4.py` — alias-aware storage-level dump/restore module
  (bind-mount into `sglang/srt/model_loader/`).
- `patch_fastboot_planB.py` — loader edit: skip shard load, keep processing,
  restore after (the correct-but-not-faster configuration).
- `patch_fastboot_v3.py` — loader edit: full skip (fails: blocker #5).
- `patch_fastboot_v4.py` — whole-model pickle attempt (fails: blocker #6),
  with graceful fallback to normal serving.
- `mk_fastboot_launcher.py` — builds the test launcher with the extra mounts.

None of this is wired into `launch.sh`; it is reference material.
