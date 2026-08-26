# Qwen3.8-Flash-Next 180B on ONE DGX Spark

Run **Qwen3.8-Flash-Next** — a 180B-parameter MoE whose reference deployment
requires **two GB300s** — on a **single NVIDIA DGX Spark** (GB10, 128 GB unified
memory), at the **full 262,144-token context**, with speculative decoding.

```
context     262,144 tokens          decode      ~27 tok/s (NEXTN spec decode)
prefill     2,000+ tok/s cold       warm cache  up to 143,465 tok/s (67x)
weights     ~97 GB resident         checkpoint  135 GB (RadixArk NVFP4)
```

Validated on this exact build: exact needle recall from a **222,527-token**
haystack, and **12/12** on an executed-code benchmark.

## Why this shouldn't fit — and how it does

The [RadixArk NVFP4 checkpoint](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
is 135 GB against ~115 GB usable on a Spark. The blocker is the model's
**PLE n-gram embedding table**: 320,001,536 rows x 160 dims of FP8 = **51.2 GB**
that no quantization toolchain touches, because an embedding is a *gather*, not
a GEMM — and the reference runtime dequantizes it to BF16 (102 GB!) at load.

This repo makes it fit twice over:

**1. Load-time NVFP4 packing** (`PLE_MODE=packed`, 28.8 GB)
Each of the 128 FP8 shards is quantized to packed NVFP4 (4-bit E2M1 codes +
FP8 group-16 scales) *on the GPU during weight load* — no offline conversion,
no checkpoint rewrite — and the gather runs through a LUT dequant. Total
weights ≈ 100 GB. Fits, barely.

**2. HashK compression** (`PLE_MODE=hashk`, default, **12.8 GB**)
The table is already hash-addressed (no dictionary), so it can be re-hashed
into a 4x smaller store, trainlessly:

- **polynomial re-hash** of each head's local index into a fixed-size sub-table
- **k=2 sub-tables** splitting the 160 dims into two 80-dim halves with
  *independent* hashes, so collisions decorrelate across halves
- slot value = **mean of all original rows** hashing to it (unbiased)
- a per-head **160x160 ridge-fitted projection** mapping reconstructions back
  toward the true rows
- the model's own PLELayer conv + grouped-norm **gating** filters the retrieved
  values into the residual stream

Reconstruction cosine is ~0.50 — the exact 1/sqrt(R) mean-pooling limit — yet
the model **degrades gracefully and even improves on some axes**: the
compressed build scored 12/12 on our executed-code benchmark where the full
table scored 10/12 (it eliminated two runaway-verbosity failures), with
identical needle recall. The freed ~16 GB funds the 8 GB MTP draft head, which
buys **NEXTN speculative decoding** and roughly **2x decode speed**.

## Results

Measured on one GB10 DGX Spark, this repo's default config. Full methodology in
[`docs/RESULTS.md`](docs/RESULTS.md); raw data in
[`docs/bench_results_suite.json`](docs/bench_results_suite.json).

**Context depth (cold, concurrency 1)**

| Depth | Tokens | TTFT | Prefill | Decode |
|---|---|---|---|---|
| 8k | 8,088 | 5.57 s | 1,452 tok/s | 26.1 tok/s |
| 32k | 32,409 | 14.04 s | 2,309 tok/s | 17.9 tok/s |
| 128k | 129,503 | 60.5 s | 2,140 tok/s | 26.7 tok/s |

**Prefix cache (radix + DeltaNet state checkpoints)**

| Depth | Cold | Warm | Ratio |
|---|---|---|---|
| 8k | 2,314 tok/s | 10,152 tok/s | 4.4x |
| 32k | 2,306 tok/s | 64,399 tok/s | 27.9x |
| 128k | 2,145 tok/s | 143,465 tok/s | **66.9x** |

**Reasoning effort** (this build honors per-request `reasoning_effort`)

| Variant | Out tok | Time to answer | Decode | Correct |
|---|---|---|---|---|
| default (thinking on) | 3,541 | 149.4 s | 23.8 tok/s | yes |
| medium | 1,210 | 43.6 s | 27.9 tok/s | yes |
| thinking off | 981 | 34.4 s | 28.7 tok/s | yes |

## Quickstart

Requirements: one DGX Spark (GB10/SM121), Docker + NVIDIA Container Toolkit,
~150 GB free disk, no desktop session hogging unified memory.

```bash
git clone <this repo> && cd flashnext-one-spark

# 1. Build the HashK artifact (~6 min GPU; streams the checkpoint's PLE shards).
#    Downloads the 135 GB checkpoint into $HF_CACHE first if absent.
docker run --rm --gpus all \
  -v ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
  -v $PWD:/out --entrypoint python3 \
  lmsysorg/sglang:qwen38flashnext /out/tools/build_hashk_ple.py
mv ple_hashk_R4.pt .   # artifact lands in repo root (12.8 GB, gitignored)

# 2. Launch (~9 min boot; ~20 min on the very first run).
./launch.sh

# 3. Talk to it (OpenAI-compatible).
curl localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "flash-next",
  "messages": [{"role": "user", "content": "Hello from a 180B model on one Spark."}]}'
```

Knobs: `THINKING=off|low|medium|xhigh`, `MEM_FRACTION`, `CTX`, `PORT`,
`PLE_MODE=packed` for the lossless-leaning mode (drops spec decode). See
`launch.sh` header.

## The four upstream bugs this repo fixes

None of this worked out of the box. The cookbook image was validated on GB300
(SM100); the SM121 + long-context + fallback paths had never been run, and we
hit four genuine bugs — each shipped here as a bind-mounted patched file
(`patches/`), with the anchored-edit generators that document the exact diffs
(`patches/generators/`):

| # | File | Bug | Symptom |
|---|---|---|---|
| 1 | `flash_fwd.py` | TMA-O enabled for varlen; the ragged epilogue is rank-broken (the correct guard survives in a comment one line up) | MLIR "weakly congruent" crash at boot |
| 2 | `qwen_sparse_attn_backend.py` | flashinfer **trtllm-gen decode kernels are SM100-only and silently emit garbage on SM121** | first token correct, then `!!!!` forever (NaN -> token 0) |
| 3 | `qwen_sparse_attn_backend.py` | the `_compact_kv` Triton kernel **does not compact**: valid rows keep their original column offsets while `cu_seqlens` promises contiguity, so interleaved `-1` top-k indices leave uninitialized NaN holes | same `!!!!` signature via the fallback path |
| 4 | `sparse_attn.py` | long-prefill sparse kernel feeds fp8-loaded K straight into `tl.dot` | "Unsupported rhs dtype fp8e4nv" — compiles fine on short prompts, kills the server on the first ~100k+ request |

Fixes 2+3 replace the decode fallback with a direct
`req_to_token`-gather + masked SDPA — exact for single-token queries, hole-
tolerant, CUDA-graph-safe. If you serve Flash-Next on a Spark by any other
means, you will meet these bugs; upstream issues/PRs welcome.

## One more landmine: speculative decoding needs state checkpoints

NEXTN rejects draft tokens, and rejected tokens must **rewind the DeltaNet
recurrent state**. That requires `--mamba-scheduler-strategy extra_buffer
--mamba-track-interval 64` (already in `launch.sh`). Without them the server
looks fine on prose and corrupts mid-stream on high-acceptance (code)
workloads. The full landmine ledger is in
[`docs/LANDMINES.md`](docs/LANDMINES.md).

## Repo layout

```
launch.sh                  one-command server launch (both PLE modes)
patches/                   4 patched files, bind-mounted over the image
patches/generators/        anchored-edit scripts documenting each diff
tools/build_hashk_ple.py   builds the 12.8 GB HashK artifact from the checkpoint
tools/bench_results_suite.py  reproduces every table above
docs/                      results, raw data, landmine ledger
```

## Ideas for improvement

- **Product quantization of the PLE** — codebook PQ at ~8 B/row would be
  ~2.6 GB with far higher fidelity than hash-mean (the R=4 mean-pool sits at
  its cos≈0.5 information limit). Same runtime shape: gather + decode.
- **Learned HashK** — alternating least squares over (A, B, W) instead of
  plain means; or distill the table against model outputs.
- **Fix `_compact_kv` upstream** in Triton (true compaction) and add an SM121
  guard to the trtllm decode resolver.
- **Quality evals** — perplexity and rare-phrase completion, where an n-gram
  table should matter most; our 12-task + needle coverage is necessary, not
  sufficient.
- **2-Spark TP2** with the uncompressed table, as an A/B quality reference.

## Credits

- [Qwen](https://huggingface.co/Qwen) — Qwen3.8-Flash-Next
- [RadixArk](https://huggingface.co/RadixArk) — the NVFP4 checkpoint this runs from
- [sglang](https://github.com/sgl-project/sglang) and the `qwen38flashnext`
  cookbook image — the runtime all patches apply to
- MiaAI-Lab's one-Spark DeepSeek recipes for proving the genre

## License

MIT for everything original here (`launch.sh`, `tools/`, generators, docs).
Files under `patches/` are **modified copies** of sglang (Apache-2.0) and
flash-attention (BSD-3) sources and remain under their original licenses —
see [`NOTICE.md`](NOTICE.md).
