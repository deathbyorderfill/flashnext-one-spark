# Qwen3.8-Flash-Next 180B on ONE DGX Spark — working layout (2026-08-26)

## What runs
- Container `flashnext` → sglang server on 0.0.0.0:30000 (OpenAI-compatible)
- Model: RadixArk/Qwen3.8-Flash-Next-NVFP4 (135 GB checkpoint in the shared HF cache)
- Image: lmsysorg/sglang:qwen38flashnext (ONLY sglang with qwen4_exp; nightlies lack it)
- Full 262,144-token context (KV pool 262,272), fp8 KV cache
- Measured: prefill 2,047 tok/s (222,527-token needle test, exact recall), decode ~10-16 tok/s
- Coding bench: 10/12 first pass (classic 6/6; 2 fails were 8k-truncation rambles, not errors)

## Why it fits (~97-100 GB weights on a ~115 GB-usable UMA box)
- 60.4 GB FP4 MoE experts (stock RadixArk quant)
- 51.2 GB FP8 PLE n-gram table → **packed to NVFP4 AT LOAD TIME by our patch**
  (320,001,536 rows x 160 dims -> uint8 codes [rows,80] + fp8 group scales [rows,10] = 28.8 GB)
- 8 GB BF16 misc (MTP head never loads; vision tower skipped via --language-only)

## The 4 patched files (bind-mounted over the image; sources + generators in this dir)
1. qwen4_exp_nvfp4.py  -> srt/models/qwen4_exp.py
   Env SGLANG_QWEN4_PLE_NVFP4=1: packed PLE buffers at init (layer 2 -> 51G fp8 frees
   before the 60G MoE allocates), per-shard GPU quantize on load (empty_cache per shard
   is REQUIRED or ~5G of allocator cache masquerades as weights), LUT-dequant gather.
   Also force-disables ple_offload_embedding (pointless on UMA).
   Generator: make_nvfp4_patch.py (applies anchored edits to a pristine qwen4_exp.py).
2. flash_fwd.py -> flash_attn/cute/flash_fwd.py
   One line: use_tma_O = arch>=sm_90 AND mCuSeqlensQ is None (varlen TMA-O epilogue is
   rank-broken; without this boot dies with MLIR "weakly congruent").
3. qwen_sparse_attn_backend.py -> srt/layers/attention/
   (a) trtllm-gen decode resolver returns None: those kernels are SM100-only and
       SILENTLY emit garbage on SM121 ("first token right, then !!!!" = NaN->token 0).
   (b) decode fallback replaced with direct torch gather + masked SDPA: the image's
       _compact_kv triton kernel does NOT compact (valid rows keep original column
       offsets, cu_seqlens promises contiguity -> uninitialized NaN holes).
   (c) gathered K/V cast .to(q.dtype) (fp8 KV vs bf16 SDPA).
   Generators: patch_qsa.py + patch_qsa2.py.
4. sparse_attn.py -> srt/layers/attention/qsa/
   Long-prefill sparse triton kernel: tl.dot on raw fp8 K ("Unsupported rhs dtype
   fp8e4nv" — only compiles at long-context shapes) -> cast keys to q dtype and
   compute the P*V dot in bf16.

## Launch
./launch_flashnext.sh   (this dir) — canonical command. Boot ~15 min (11 min load incl
per-boot PLE requantize). NO restart policy: docker start flashnext after any exit.

## Tuning ledger (KV pool vs settings)
- bf16 KV, mem 0.96, mamba 12: pool 19,776 tokens
- fp8 KV,  mem 0.97, mamba 6:  pool 220,928 (safe config, ~2.5G host slack)
- fp8 KV,  mem 0.98, mamba 5:  pool 262,272 (CURRENT; ~1G host slack — do NOT run
  desktop/Chrome; hourly weather crons are the OOM risk. Fallback = the 0.97 line.)

## Server defaults
--default-chat-template-kwargs '{"enable_thinking": false}' (omp can't send kwargs);
per-request chat_template_kwargs {"enable_thinking": true} re-enables thinking.
omp provider: flashnext/flash-next (~/.omp/agent/models.yml).

## UPDATE (2026-08-26 late): HashK-PLE + NEXTN speculative decoding
- PLE replaced by hash-compressed tables (`ple_hashk_R4.pt`, 12.8 GB, built by
  build_hashk_ple.py): k=2 dim-split sub-tables, R=4, per-head ridge projection.
  Reconstruction cosine ~0.50 (= 1/sqrt(R) mean-pool limit) yet EMPIRICALLY BETTER:
  coding 12/12 (full table: 10/12 w/ rambling truncations), needle exact, boot 6.5m.
  Env: SGLANG_QWEN4_PLE_HASHK=/patches/ple_hashk_R4.pt (supersedes packed mode).
- Freed ~16 GB -> MTP head loads -> NEXTN spec decode (steps 3, draft tokens 4):
  code decode 26-33 tok/s (~2x). REQUIRES --mamba-scheduler-strategy extra_buffer
  --mamba-track-interval 64 (DeltaNet state checkpoints for draft rollback;
  without them, rejected drafts corrupt state -> mid-stream "!!!!" NaN).
- mem-fraction 0.95 (pool ~700-900k tokens >> 262k need; keeps ~3G host slack).
- Old full-table packed mode still available: swap env to SGLANG_QWEN4_PLE_NVFP4=1
  and remove spec-decode flags (28.8 GB table, no MTP room).
