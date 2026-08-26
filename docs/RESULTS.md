# Qwen3.8-Flash-Next 180B on one DGX Spark — Results
SGLang cookbook + HashK-compressed PLE (12.8 GB) + NEXTN speculative decoding.
262,144-token context. Cold = unique random prompt per run (no prefix reuse).

## Context depth, concurrency 1, cold prefill
| Depth | Tokens  | TTFT    | Prefill      | Decode      |
|-------|---------|---------|--------------|-------------|
| 8k    | 8,088   | 5.57 s  | 1,452 tok/s  | 26.1 tok/s  |
| 32k   | 32,409  | 14.04 s | 2,309 tok/s  | 17.9 tok/s  |
| 128k  | 129,503 | 60.5 s  | 2,140 tok/s  | 26.7 tok/s  |

## Concurrency at 32k (unique prompts per stream)
| Streams | Per-stream decode | Aggregate decode | Scaling |
|---------|-------------------|------------------|---------|
| 1       | 17.8 tok/s        | 17.8 tok/s       | 1.0x    |
| 4       | 7.8 tok/s         | 31.3 tok/s       | 1.8x    |
| 8       | 8.9 tok/s         | 71.5 tok/s       | 4.0x    |

## Prefix cache (radix cache + mamba extra_buffer tracking)
| Depth | Cold         | Warm           | Ratio |
|-------|--------------|----------------|-------|
| 8k    | 2,314 tok/s  | 10,152 tok/s   | 4.4x  |
| 32k   | 2,306 tok/s  | 64,399 tok/s   | 27.9x |
| 128k  | 2,145 tok/s  | 143,465 tok/s  | **66.9x** |

## Reasoning effort (per-request chat_template_kwargs; honored by this build)
| Variant               | Out tok | Time to answer | TTFT   | Decode      | Correct |
|-----------------------|---------|----------------|--------|-------------|---------|
| default (thinking on) | 3,541   | 149.4 s        | 0.62 s | 23.8 tok/s  | yes     |
| medium                | 1,210   | 43.6 s         | 0.31 s | 27.9 tok/s  | yes     |
| thinking off          | 981     | 34.4 s         | 0.22 s | 28.7 tok/s  | yes     |

Notes: decode is spec-decode (NEXTN k=3) so speed is content-dependent; the 32k
decode dip and the 4-stream per-stream dip reflect decode overlapping other
streams prefill in the measurement window. usage does not expose a reasoning
token split on this build; out-token deltas show the effort knob working.
Test question: two-train meeting time (answer 10:36:40 -> 10:37); all correct.
