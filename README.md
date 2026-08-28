# Qwen3.8-Flash-Next 180B on ONE DGX Spark

Qwen3.8-Flash-Next officially needs **two DGX Sparks**. This repo runs it on **one
desk-side DGX Spark** — full 262,144-token context, speculative decoding,
~36 tok/s on code and ~57 tok/s aggregate across 4 concurrent sessions — by
compressing the one part of the model no quantizer can touch: its **51 GB
n-gram embedding table**.

The unlock isn't a better quantizer. The table is a *gather*, not a matmul, so
every 4-bit toolchain skips it and the reference runtime inflates it to 102 GB
of BF16 at load. Treat it as what it is — a hash-addressed memory with a
learned gate behind it — and it compresses 4× with **no training, in six
minutes**, and the model doesn't just survive: it stopped failing two of our
benchmark tasks, and scores **86/100 quality on an independent 88-scenario
agentic tool-calling benchmark**.

```
context     262,144 tokens          decode      ~36 tok/s code, ~21 free-form
concurrency 4 simultaneous sessions aggregate   ~57 tok/s across 4 streams
prefill     2,000+ tok/s cold       warm cache  up to 139,000 tok/s
weights     ~97 GB resident         checkpoint  135 GB (RadixArk NVFP4)
```

Validated on this exact build: exact needle recall from a **222,527-token**
haystack, **12/12** on an executed-code benchmark, and **151/176 (86/100
quality)** on [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench)
with 100% in 8 of 16 categories including structured output (12/12) and
restraint/refusal (6/6).

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
identical needle recall. Ablations confirm the scheme sits at its trainless
ceiling: XOR-bitwise hashing scores identically (any well-mixed hash saturates
the 1/sqrt(R) limit), signed accumulation is slightly *worse* (the rows share a
common component that mean-pooling preserves and sign-cancellation destroys),
and norm-weighted pooling loses to the plain mean (the table's row norms are
tight; there is no junk to down-weight). Only R=2, product quantization, or
training beats 0.50. The freed ~16 GB funds the 8 GB MTP draft head, which
buys **NEXTN speculative decoding** and roughly **2x decode speed**.

## Results

Measured on one GB10 DGX Spark, this repo's default config. Full methodology in
[`docs/RESULTS.md`](docs/RESULTS.md); raw data in
[`docs/bench_results_suite_v2.json`](docs/bench_results_suite_v2.json) (v1
baseline alongside) and [`docs/bench_cc3_results.json`](docs/bench_cc3_results.json).

**Context depth (cold, concurrency 1)**

| Depth | Tokens | TTFT | Prefill | Decode |
|---|---|---|---|---|
| 8k | 8,121 | 5.05 s | 1,608 tok/s | 31.6 tok/s |
| 32k | 32,541 | 15.93 s | 2,043 tok/s | 27.2 tok/s |
| 128k | 129,370 | 55.46 s | 2,333 tok/s | 46.4 tok/s |

> ⚠️ These depth rows were measured before we added output-validity checks to
> the suite (see the corruption section below). The 46.4 tok/s @128k figure —
> *faster* than 8k, which real attention never is — matches the degenerate-
> output speed signature and should be treated as unverified until the suite
> re-run lands. Single-stream decode on verified-clean output: **36 tok/s
> code / 21 free-form / 41 verbatim** (current config, [`tools/speed_probe.py`](tools/speed_probe.py)).

**Concurrency — corrected.** Earlier revisions of this README claimed 96/157
tok/s aggregate at 4/8 streams and "6.5× scaling". **Those numbers were wrong,
twice over**: the suite summed per-stream decode rates over *non-overlapping*
time windows (streams run in pairs when the mamba state cache caps true
concurrency), and it never validated output text. The honest, wall-clock
window-aggregate numbers ([`tools/bench_cc3.py`](tools/bench_cc3.py)):

| Config | True concurrency | 4-stream aggregate | Single-stream code |
|---|---|---|---|
| v2 (`fp32 ssm, cache 12`) | 2 (mamba-capped) | 39.0 tok/s | 33.6 |
| **v3 (`bf16 ssm, cache 24`) — default** | **4** | **57.2 tok/s (+47%)** | **36.0** |

The mechanics: each request needs 5 mamba state slots; `--max-mamba-cache-size`
is a slot *count*, so 12 slots = 2 concurrent requests regardless of
`--max-running-requests`. Halving the state dtype to bf16 and doubling the
count to 24 is **memory-neutral** and lifts true concurrency to 4 — all
verified corruption-free with unchanged accept length. Both flags are in
`launch.sh`.

**v1 → v2 → v3 decode (single stream, verified where noted)**

| Metric | v1 | v2 | v3 (validated) |
|---|---|---|---|
| Code | 26–33 | 34.2 | **36.0** |
| Free-form | ~18 | 19.9 | 21.8 |
| Verbatim | — | ~25 | 41.4 |
| 4-stream aggregate | — | 39.0* | **57.2** |

\* v2 aggregate re-measured with the corrected metric; the originally published
3.1×/6.5× scaling figures are retracted.

**Reasoning effort** (this build honors per-request `reasoning_effort`)

| Variant | Out tok | Time to answer | Decode | Correct |
|---|---|---|---|---|
| default (thinking on) | 1,220 | 35.3 s | 34.8 tok/s | yes |
| medium | 1,146 | 33.1 s | 34.9 tok/s | yes |
| thinking off | 1,684 | 49.4 s | 34.2 tok/s | yes |

**Agentic tool calling** ([tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench),
88 scenarios + hard mode, seed 42): **151/176 points, quality 86/100**.
100% in parameter precision, restraint/refusal, localization, structured
reasoning, instruction following, toolset scale, creative composition, and
structured output; hard mode 82%. Weakest: autonomous planning (50%) and
multi-turn state under correction pressure — model-level traits, not serving
artifacts. The responsiveness score (35/100) reflects the bench's sub-second
cloud-API curve against local ~4.6 s thinking-medium turns; quality is the
number that describes the model.

## Quickstart

Requirements: one DGX Spark (GB10/SM121), Docker + NVIDIA Container Toolkit,
~150 GB free disk, no desktop session hogging unified memory.

```bash
git clone https://github.com/deathbyorderfill/flashnext-one-spark.git
cd flashnext-one-spark

# 1. Build the HashK artifact (~6 min GPU; streams the checkpoint's PLE shards).
#    On a cold cache this first downloads the 135 GB checkpoint into $HF_CACHE,
#    so the initial run is dominated by the download, not the 6 min build.
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

Step 1 knobs: `HASHK_SNAPSHOT=/path/to/snapshot` uses a checkpoint you already
have instead of the HF cache; `HASHK_NO_DOWNLOAD=1` fails with instructions
rather than fetching; `HASHK_R` sets the compression ratio (default 4).
If you prefer to download separately first:
`huggingface-cli download RadixArk/Qwen3.8-Flash-Next-NVFP4`.

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
| 1 | `qwen_sparse_attn_backend.py` | flashinfer **trtllm-gen decode kernels silently emit garbage on SM121** (validated correct on SM100 and SM120) | first token correct, then `!!!!` forever (NaN -> token 0) |
| 2 | `qwen_sparse_attn_backend.py` | the `_compact_kv` Triton kernel **does not compact**: valid rows keep their original column offsets while `cu_seqlens` promises contiguity, so interleaved `-1` top-k indices leave uninitialized NaN holes | same `!!!!` signature via the fallback path |
| 3 | `flash_fwd.py` | TMA-O enabled for varlen; the ragged epilogue is rank-broken (the correct guard survives in a comment one line up) | MLIR "weakly congruent" crash at boot |
| 4 | `sparse_attn.py` | long-prefill sparse kernel feeds fp8-loaded K straight into `tl.dot` | "Unsupported rhs dtype fp8e4nv" — compiles fine on short prompts, kills the server on the first ~100k+ request |

Fix 1 now ships as a **per-arch gate** (SM100 family and SM120 take trtllm-gen;
SM121 takes the fallback), matching the evidence in the upstream discussion
around [sglang PR #36556](https://github.com/sgl-project/sglang/pull/36556) and
[issue #36558](https://github.com/sgl-project/sglang/issues/36558): SM120 users
report byte-identical A/B output on trtllm-gen up to 229k tokens, while SM121
corrupts on that path with real weights. A blanket "family 12" gate would ship
silent corruption to every DGX Spark; a blanket SM100-only gate regresses
SM120. Fixes 1+2 together replace the decode fallback with a direct
`req_to_token`-gather + masked SDPA — exact for single-token queries,
hole-tolerant, CUDA-graph-safe.

Bug 3's provenance is settled: the stock PyPI wheel
`flash_attn_4-4.0.0b19-py3-none-any.whl` ships with the varlen guard commented
out (`flash_attn/cute/flash_fwd.py:658-659`, md5-verified byte-identical to the
image's copy) — **upstream flash-attn-4 is affected as released**, not just
this image.

## ⚠️ Open investigation: activity-dependent deep-context corruption

Under some conditions, requests deeper than ~3–5k prompt tokens (where the QSA
sparse path engages) begin returning pure token-0 output (`!!!!`) — **prefill-
side** (the first token is already garbage), **single-stream** (no concurrency
needed at failure time), while shallow requests stay clean and `/health` stays
green. Degenerate output accepts every draft, so corrupt runs are ~2× *faster* —
which is how it inflates unvalidated benchmarks.

What we have established with controlled runs:

- A fresh boot serves deep requests clean; corruption probability **rises with
  cumulative traffic** on a boot, reaching 100% deterministic in the worst
  observed state.
- `POST /flush_cache` does **not** reliably clear it. **Only a restart does.**
- No retraction events required; a repro recipe (deep requests + concurrent
  bursts, [`tools/causal_chain.py`](tools/causal_chain.py) /
  [`tools/bisect_probe.py`](tools/bisect_probe.py)) has poisoned a fresh boot in
  one pass — and, on other nights, failed to poison in six passes. The trigger
  has an **environmental component** (suspected: UMA memory pressure from
  co-running load; the box idles at ~1 GB free) that is not yet pinned down.
- A config bisection (radix off / strip-thinking off / spec off / chunk 2048)
  produced only clean runs on a night when the **stock positive control was
  also clean** — so no flag has been validly exonerated or convicted.

Operational mitigations shipped here: [`tools/watchdog.sh`](tools/watchdog.sh)
(wedge detection) and [`tools/poison_sentinel.sh`](tools/poison_sentinel.sh)
(scheduled deep-probe canary that auto-restarts the server on a corrupt
verdict). If you serve long contexts, run the sentinel; the failure mode is
silent.

Sentinel status so far: the first scheduled probes ran during peak co-load —
one at **0 GB free host memory with 5 GB in swap** — and came back **clean**,
which weighs *against* the simple memory-pressure hypothesis. Current honest
position: reproduced twice on one evening under conditions not yet isolated,
never since, canary standing guard.

## Dead ends, documented

Negative results that cost boots so you don't have to repeat them:

- **GDN kernel backend swaps** (`--linear-attn-backend cutedsl|flashinfer`):
  assert `extra_buffer is not supported... use no_buffer` — i.e. they are
  incompatible with the mamba scheduler strategy that NEXTN speculative
  decoding *requires* for draft-rejection rollback. A kernel swap costs the 2×
  spec-decode win. `--mamba-backend flashinfer` boots and runs clean but is
  within noise of triton.
- **Tree drafting** (`--speculative-eagle-topk > 1`): hard
  `NotImplementedError` in `QwenSparseMultiStepDraftBackend` — and it is not a
  guard flip: sibling branches share a position, so they would collide in the
  same QSA pending-ring slot. Needs a branch dimension in the ring plus
  tree-mask verify.
- **Wider draft chains** (`SGLANG_QSA_RING_WINDOW`, shipped here, off by
  default): corruption-free but a workload knob, not a speedup — the stock MTP
  head is tuned for ~4 tokens. Draft-6 wins only on repetitive/structured
  output (+34% verbatim), and costs free-form. Full table in
  [`docs/RESULTS.md`](docs/RESULTS.md).
- **NGRAM/prompt-lookup speculation**: runs corruption-free (relax the ngram
  guard in `_prepare_ple_batch`) but loses to NEXTN at draft-4; only worth
  revisiting through the ring-window work as hybrid ngram-hit/MTP-miss.
- `--enable-torch-compile` (won't trace), `--num-continuous-decode-steps`
  (regresses with NEXTN), `--cuda-graph-backend-prefill breakable` (hard
  crash-loop during capture).

## Landmines

- **Speculative decoding needs state checkpoints.** NEXTN rejects draft
  tokens, and rejected tokens must rewind the DeltaNet recurrent state:
  `--mamba-scheduler-strategy extra_buffer --mamba-track-interval 64` (in
  `launch.sh`). Without them the server looks fine on prose and corrupts
  mid-stream on high-acceptance (code) workloads.
- **Concurrency is mamba-capped, not flag-capped.** See the corrected
  concurrency section; `--max-running-requests` is not what limits you.
- **Reasoning-parser whitespace artifact** (known issue): with the qwen3
  reasoning parser, `content` can arrive prefixed with `\n\n` after the think
  block is extracted. Harmless for chat UIs; breaks exact-match consumers
  (it cost us a benchmark scenario whose grader demanded an exact numeric
  string). Trim leading whitespace client-side until fixed.
- The full ledger is in [`docs/LANDMINES.md`](docs/LANDMINES.md).

## Repo layout

```
launch.sh                  one-command server launch (both PLE modes)
patches/                   patched files, bind-mounted over the image
patches/generators/        anchored-edit scripts documenting each diff
tools/build_hashk_ple.py   builds the 12.8 GB HashK artifact from the checkpoint
tools/bench_results_suite.py  the original suite (aggregate metric superseded)
tools/bench_cc3.py         corrected concurrency bench: wall-clock window
                           aggregate + per-request output-validity checks
tools/speed_probe.py       per-config speed probe (1-stream + 4-stream + deep)
tools/cc_probe.py          concurrency corruption probe (1/2/4/8 streams)
tools/causal_chain.py      deep-context corruption repro (boot -> storm -> deep)
tools/bisect_probe.py      one-pass poisoning recipe with verdict
tools/poison_sentinel.sh   cron canary: deep probe + auto-restart on corruption
tools/watchdog.sh          cron watchdog: revives a dead server and detects the
                           silent wedge (accept-len 1.00 while /health is green)
docs/                      results, raw data, landmine ledger
```

## Ideas for improvement

- **Product quantization of the PLE** — codebook PQ at ~8 B/row would be
  ~2.6 GB with far higher fidelity than hash-mean (the R=4 mean-pool sits at
  its cos≈0.5 information limit — confirmed from three independent ablation
  angles). Same runtime shape: gather + decode.
- **FP8 draft head** — the MTP head currently drafts with NVFP4 weights; a
  load-time dequant of just the `mtp.*` tensors to FP8 (~+3.5 GB, fundable by
  trimming the 2.5×-context KV pool) converts head precision directly into
  accept length, the current decode bottleneck.
- **Root-cause the deep-context corruption** — the repro tooling is here; the
  next step is per-layer NaN instrumentation on a poisoned boot.
- **Fast boot via preprocessed weights** — attempted and **parked with a
  mapped-out negative result** ([`experiments/fastboot/`](experiments/fastboot/)):
  restoring all 97 GB of final tensors takes only ~70 s, but
  `process_weights_after_loading` is both the bulk of the 9-minute boot *and*
  the creator of load-bearing non-tensor state (padded dims, kernel runners)
  that forward kernels require — skip it and the model IMAs even with
  byte-perfect tensors; whole-model pickling dies on unpicklable
  ProcessGroup/Stream handles. A real fix means serializing processing
  *effects* (processed-checkpoint export in the quant methods) — upstream-
  shaped work. The experiment also surfaced that 276 of 2,198 tensors are
  views into shared storage, and two UMA landmines (page-cache vs GPU-pool
  competition; allocator fragmentation) documented in the experiment README.
- **Quality evals** — perplexity and rare-phrase completion, where an n-gram
  table should matter most; our benchmark coverage is necessary, not
  sufficient.
- **2-Spark TP2** with the uncompressed table, as an A/B quality reference.

## Credits

- [Qwen](https://huggingface.co/Qwen) — Qwen3.8-Flash-Next
- [RadixArk](https://huggingface.co/RadixArk) — the NVFP4 checkpoint this runs from
- [sglang](https://github.com/sgl-project/sglang) and the `qwen38flashnext`
  cookbook image — the runtime all patches apply to
- [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench)
  — the agentic tool-calling benchmark
- MiaAI-Lab's one-Spark DeepSeek recipes for proving the genre

## License

MIT for everything original here (`launch.sh`, `tools/`, generators, docs).
Files under `patches/` are **modified copies** of sglang (Apache-2.0) and
flash-attention (BSD-3) sources and remain under their original licenses —
see [`NOTICE.md`](NOTICE.md).
