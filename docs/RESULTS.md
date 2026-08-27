# Qwen3.8-Flash-Next 180B on one DGX Spark — Results (v2)
SGLang cookbook + HashK-compressed PLE (12.8 GB) + NEXTN speculative decoding
+ v2 perf pass (grouped-bmm QSA gather, 8k prefill chunks, thinking-stripped
prefix cache). 262,144-token context. Cold = unique random prompt per run.
Raw data: bench_results_suite_v2.json (v1 baseline: bench_results_suite_v1.json).

## Context depth, concurrency 1, cold prefill
| Depth | Tokens  | TTFT    | Prefill      | Decode      |
|-------|---------|---------|--------------|-------------|
| 8k    | 8,121   | 5.05 s  | 1,608 tok/s  | 31.6 tok/s  |
| 32k   | 32,541  | 15.93 s | 2,043 tok/s  | 27.2 tok/s  |
| 128k  | 129,370 | 55.46 s | 2,333 tok/s  | **46.4 tok/s** |

## Concurrency at 32k (unique prompts per stream)
| Streams | Per-stream decode | Aggregate decode | Scaling |
|---------|-------------------|------------------|---------|
| 1       | 24.3 tok/s        | 24.3 tok/s       | 1.0x    |
| 4       | 24.1 tok/s        | 96.3 tok/s       | 4.0x    |
| 8       | 19.6 tok/s        | 157.1 tok/s      | **6.5x** |

## Prefix cache (radix cache + mamba extra_buffer tracking)
| Depth | Cold         | Warm           | Ratio |
|-------|--------------|----------------|-------|
| 8k    | 2,578 tok/s  | 8,275 tok/s    | 3.2x  |
| 32k   | 2,587 tok/s  | 133,240 tok/s  | 51.5x |
| 128k  | 2,482 tok/s  | 139,194 tok/s  | **56.1x** |

## Reasoning effort (per-request chat_template_kwargs; honored by this build)
| Variant               | Out tok | Time to answer | TTFT   | Decode      | Correct |
|-----------------------|---------|----------------|--------|-------------|---------|
| default (thinking on) | 1,220   | 35.3 s         | 0.28 s | 34.8 tok/s  | yes     |
| medium                | 1,146   | 33.1 s         | 0.20 s | 34.9 tok/s  | yes     |
| thinking off          | 1,684   | 49.4 s         | 0.20 s | 34.2 tok/s  | yes     |

## v1 -> v2
| Metric | v1 | v2 | Change |
|---|---|---|---|
| Decode @ 8k / 32k / 128k | 26.1 / 17.9 / 26.7 | 31.6 / 27.2 / 46.4 tok/s | +21% / +52% / +74% |
| Aggregate @ 4 / 8 streams | 31.3 / 71.5 | 96.3 / 157.1 tok/s | 3.1x / 2.2x |
| Warm prefill @ 32k | 64,399 | 133,240 tok/s | 2.1x |

Notes: decode is speculative (NEXTN k=3) so speed is content-dependent.
Effort "time to answer" varies with generated length (thinking-off produced a
longer answer in v2); the decode column is the comparable metric.
Test question: two-train meeting time (answer 10:36:40 -> 10:37); all correct.
