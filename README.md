# Qwen3.8-Flash-Next 180B on ONE DGX Spark

Qwen3.8-Flash-Next officially needs **two DGX Sparks**. This repo runs it on **one
desk-side DGX Spark** — full 262,144-token context, speculative decoding,
27-46 tok/s — by compressing the one part of the model no quantizer can touch:
its **51 GB n-gram embedding table**.

The unlock isn't a better quantizer. The table is a *gather*, not a matmul, so
every 4-bit toolchain skips it and the reference runtime inflates it to 102 GB
of BF16 at load. Treat it as what it is — a hash-addressed memory with a
learned gate behind it — and it compresses 4× with **no training, in six
minutes**, and the model doesn't just survive: it stopped failing two of our
benchmark tasks.

```
context     262,144 tokens          decode      27-46 tok/s (NEXTN spec)
prefill     2,000+ tok/s cold       warm cache  up to 143,465 tok/s (67x)
weights     ~97 GB resident         checkpoint  135 GB (RadixArk NVFP4)
```

Validated on this exact build: exact needle recall from a **222,527-token**
haystack, and **12/12** on an executed-code benchmark.

## The problem: 135 GB into 115

The [RadixArk NVFP4 checkpoint](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
is already 4-bit everywhere 4-bit can reach, and it is *still* 135 GB against
~115 GB usable on a Spark — because 51.2 GB of it is the PLE table
(320,001,536 rows × 160 dims of FP8) that quantizers structurally cannot
handle. Two independent solutions here, both as bind-mount patches over the
stock cookbook image — pick either:

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
[`docs/bench_results_suite_v2.json`](docs/bench_results_suite_v2.json) (v1 baseline alongside).

**Context depth (cold, concurrency 1)**

| Depth | Tokens | TTFT | Prefill | Decode |
|---|---|---|---|---|
| 8k | 8,121 | 5.05 s | 1,608 tok/s | 31.6 tok/s |
| 32k | 32,541 | 15.93 s | 2,043 tok/s | 27.2 tok/s |
| 128k | 129,370 | 55.46 s | 2,333 tok/s | **46.4 tok/s** |

**Prefix cache (radix + DeltaNet state checkpoints)**

| Depth | Cold | Warm | Ratio |
|---|---|---|---|
| 8k | 2,578 tok/s | 8,275 tok/s | 3.2x |
| 32k | 2,587 tok/s | 133,240 tok/s | 51.5x |
| 128k | 2,482 tok/s | 139,194 tok/s | **56.1x** |

**v1 → v2: what one profiling pass bought** (grouped-bmm QSA gather, 8k prefill
chunks, thinking-stripped prefix cache — same suite before/after)

| Metric | v1 | v2 | Change |
|---|---|---|---|
| Decode @ 8k | 26.1 | 31.6 tok/s | +21% |
| Decode @ 32k | 17.9 | 27.2 tok/s | +52% |
| Decode @ 128k | 26.7 | **46.4 tok/s** | **+74%** |
| Aggregate, 4 streams | 31.3 | 96.3 tok/s | 3.1x |
| Aggregate, 8 streams | 71.5 | 157.1 tok/s | 2.2x |
| Concurrency scaling | 4.0x | 6.5x | — |
| Warm prefill @ 32k | 64,399 | 133,240 tok/s | 2.1x |
| Decode, effort variants | 24–29 | ~34–35 tok/s | +25–45% |

**Reasoning effort** (this build honors per-request `reasoning_effort`)

| Variant | Out tok | Time to answer | Decode | Correct |
|---|---|---|---|---|
| default (thinking on) | 1,220 | 35.3 s | 34.8 tok/s | yes |
| medium | 1,146 | 33.1 s | 34.9 tok/s | yes |
| thinking off | 1,684 | 49.4 s | 34.2 tok/s | yes |

## Quickstart

Requirements: one DGX Spark (GB10/SM121), Docker + NVIDIA Container Toolkit,
~150 GB free disk, no desktop session hogging unified memory.

```bash
git clone https://github.com/Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark.git
cd Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark

# 1. Build the HashK artifact (~6 min GPU; streams the checkpoint's PLE shards).
#    Downloads the 135 GB checkpoint into $HF_CACHE first if absent.
#    The artifact lands in the repo root as ple_hashk_R4.pt (12.8 GB, gitignored).
docker run --rm --gpus all \
  -v ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
  -v $PWD:/out --entrypoint python3 \
  lmsysorg/sglang:qwen38flashnext /out/tools/build_hashk_ple.py

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

None of this worked out of the box. The reference deployment splits the model
across two Sparks with tensor parallelism, where each device carries half the
weights and different kernel paths run. This repo's single-device,
long-context, fallback-heavy configuration had never been exercised, and we
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
tools/watchdog.sh          cron watchdog: revives a dead server and detects the
                           silent wedge (accept-len 1.00 while /health is green)
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
- **NGRAM/prompt-lookup speculation** — tested (relax the ngram guard in
  `_prepare_ple_batch`, then `--speculative-algorithm NGRAM`): runs
  corruption-free but loses to NEXTN here (11.8 vs 18.5 tok/s free-form;
  35.1 vs ~30 on verbatim reproduction), because the QSA compressed index
  ring caps `speculative_num_draft_tokens` at the compress ratio (4).
  Reworking that ring to hold multiple pending groups would unlock 16-token
  repetition drafts (~97 tok/s territory, per
  [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
  on llama.cpp) — the single highest-leverage decode item left.
- **Fast boot via preprocessed weights** — warm boots are ~9 min, and profiling
  shows it is per-tensor processing, not I/O: skipping the 51 GB of PLE shard
  reads entirely (filtered snapshot) saved only 8 s of a 480 s load. An
  instanttensor-style dump (serialize the post-processed tensors once,
  mmap-restore on boot) should reach ~3 min; the DeepSeek Spark image loads
  106 GB in 150 s this way.
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
