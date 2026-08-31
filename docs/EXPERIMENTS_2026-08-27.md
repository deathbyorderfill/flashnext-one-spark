# Experiment log — 2026-08-27 (UNTRACKED, do not commit until conclusions)

Campaign: (0) KV-pool -> mamba-cache rebalance, (1) Tier-1 spec-decode papers,
(2) Tier-2 MoE self-speculation papers. Baseline first.

Harness: `bench_cc.py` (on the Spark, `~/flashnext/`), writes to
`~/flashnext/exp_kv_mamba/<label>.json`. Deliberately separate from
`bench_results_suite.py` so the existing baseline JSON is never overwritten.
Counts `!` per response as a corruption canary.

---

## Run A — baseline, stock config (mamba 12)

Config: `--max-running-requests 8 --max-mamba-cache-size 12
--mem-fraction-static 0.95`, NEXTN draft-4, fully reverted stock server
(no profiler, no local patches — verified `git status` clean and zero
profiler refs on the box).

Reported by `/get_server_info`: `max_total_num_tokens=670528`,
`max_running_requests=8` (requested), `max_mamba_cache_size=12`.

| streams | per-stream decode | agg_sum | agg_true | wall | **corrupt tokens** |
|---|---|---|---|---|---|
| 1 | 11.21 | 11.2 | 11.2 | 9.0 s | **0** |
| 2 | 9.95 | 19.9 | 14.7 | 13.7 s | **256** |
| 4 | 10.63 | 42.5 | 44.8 | 26.5 s | **1024** |
| 8 | 8.27 | 66.1 | 169.6 | 51.8 s | **1280** |

### FINDING 1 (blocking): concurrency corrupts output, reproducibly

`max_tokens` was 256. The corruption counts are exact multiples of 256:

- 1 stream  -> 0 corrupt   (clean)
- 2 streams -> 256  = **1 of 2 streams 100% corrupt**
- 4 streams -> 1024 = **4 of 4 streams 100% corrupt**
- 8 streams -> 1280 = **5 of 8 streams 100% corrupt**

Whole responses are nothing but `!`. This is the documented NaN -> token-0
signature, but it is **not** either known cause: the `trtllm-gen SM100` and
`_compact_kv` patches are both mounted, and mamba checkpoint flags
(`extra_buffer`, `track-interval 64`) are both present. Single-stream is
consistently clean; the trigger is concurrency.

This is on the **fully reverted stock configuration** — no profiler, nothing
of ours in the process. It is a pre-existing production bug, not a regression
from this session's work.

### FINDING 2: corrupt streams *inflate* throughput, so every multi-stream
### number ever measured on this box is suspect

Scheduler log during the corrupt runs:

```
#running-req: 2, mamba num: 8, mamba usage: 0.67,
accept len: 4.00, accept rate: 1.00, gen throughput: 88.23 token/s
```

`accept rate: 1.00` at the draft-4 ceiling. Degenerate `!` repetition is
trivially predicted by the MTP draft head, so a corrupted stream accepts every
draft and runs ~3x faster than real generation. `agg_true=169.6` at 8 streams
is measuring corruption speed, not throughput.

Consequence: the KV/mamba experiment **cannot proceed** on this axis. Raising
`max-mamba-cache-size` to buy concurrency would be optimizing the exact axis
that produces corrupt output, and the resulting throughput numbers would be
inflated by the corruption itself. Fix the bug first, then re-baseline.

This also casts doubt on the v2 aggregate figures (4-stream 96.3, 8-stream
157.1 tok/s) — those were measured with the same concurrency that corrupts
here, and nothing in that suite checked output validity.

### FINDING 3: `max_running_requests=8` is not real; it is 2

`mamba num: 8, mamba usage: 0.67` with `#running-req: 2` -> **4 mamba slots per
request**, 12 total, so 2 concurrent is the hard cap. With 8 streams offered
the scheduler showed `#queue-req: 6`. Reaching 8 real concurrent requests needs
~32-36 mamba slots, not 12.

### OPEN: single-stream decode 11.2 tok/s vs 31.6 documented @ 8k

3x below `docs/RESULTS.md`. Either a measurement-method difference (this
harness counts streamed SSE content deltas and divides by wall-minus-TTFT;
the existing suite may count differently) or a real regression. Must be
reconciled before any A/B is trustworthy — an 11 tok/s baseline would make
every subsequent comparison meaningless.

---

## Run B — A/B: `--decode-attention-backend flashinfer` (bypasses QSA decode)

Motivated by the upstream thread, which used this override for the sm_120 A/B.
Experiment launcher `exp_launch_ab.sh` = `launch_flashnext.sh` + that one flag
(verified: 1-line diff). Confirmed active in server_args:
`decode_attention_backend='flashinfer'`. `max_total_num_tokens=643648`.

Corrupt tokens (max_tokens=256, so 256 = one fully corrupt stream):

| streams | A: QSA fallback | B: flashinfer | B2: repeat |
|---|---|---|---|
| 1 | 0 | 0 | 0 |
| 2 | **256** (1/2 corrupt) | **0** | **0** |
| 4 | **1024** (4/4 corrupt) | **0** | **0** |
| 8 | **1280** (5/8 corrupt) | **512** (2/8) | **512** (2/8) |

### !! FINDING 4 RETRACTED — the A-side does not reproduce !!

A repeat of the stock config after restoring it (`A2_stock_repeat`) came back
**0 corrupt at every stream count**, as did three further repeats (A3-A5).
Stock is therefore 1 corrupt run in 5, not deterministic.

| run | config | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| A  | stock QSA | 0 | 256 | 1024 | 1280 |
| A2 | stock QSA | 0 | 0 | 0 | 0 |
| A3 | stock QSA | 0 | 0 | 0 | 0 |
| A4 | stock QSA | 0 | 0 | 0 | 0 |
| A5 | stock QSA | 0 | 0 | 0 | 0 |
| B  | flashinfer | 0 | 0 | 0 | 512 |
| B2 | flashinfer | 0 | 0 | 0 | 512 |

The A/B is **inconclusive**. B's clean 2- and 4-stream results are consistent
with simply being clean runs rather than a backend effect. If anything the
naive rates point the other way: flashinfer corrupted in 2/2 runs at 8 streams,
stock in 1/5. n is far too small either way.

The conclusion below was drawn from n=1 on the A side and is withdrawn. Nothing
here attributes the corruption to the QSA fallback.

**Most likely confound for run A:** it ran on the boot where a single smoke-test
request had already shown `#running-req: 2`, i.e. an external client (omp?) was
hitting the server. Real concurrency during run A was probably higher than the
offered stream count, driving deeper preemption. Those logs are gone (two
container restarts since), so this cannot be verified retroactively — but every
future run must record `#running-req` / `#queue-req` and discard trials with
unaccounted load.

### (WITHDRAWN) FINDING 4: there are TWO independent bugs, not one

**Bug A — QSA fallback decode corrupts at batch >= 2.** Swapping the decode
attention backend eliminates corruption entirely at 2 and 4 streams. This is
the local `forward_decode` patch in `qwen_sparse_attn_backend.py`, i.e. our own
fallback, on the path sm_121 is forced onto because the trtllm resolver is
gated to `None`.

**Bug B — something else corrupts at deep queue depth.** At 8 streams the
corruption persists with dense decode, at *exactly* 2 of 8 streams across two
independent runs. Attention backend is not the variable. With
`max_running_requests=2`, offering 8 means 6 queued -> retraction/preemption
(`retraction_policy='length'`) and mamba state-slot recycling. Prime suspect is
DeltaNet recurrent state across preemption, which matches the previously
observed "leaked state slots / mamba num exceeds running requests" wedge.

Note 4 streams also queues (2 running, 2 queued) and is clean, so bug B needs
deeper queue pressure than a single round of preemption.

### Hypothesis for Bug A (from code reading, NOT yet confirmed)

`patches/qwen_sparse_attn_backend.py`, `forward_decode`:

```python
req_pool_idx = (metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices)      # scheduler ordering
valid = (positions >= 0) & (positions < sequence_lens.view(-1, 1))
slots = req_to_token[req_pool_idx.view(-1, 1).long(), positions.clamp_min(0)]
```

`topk_indices` rows follow the QSA metadata row layout; the fallback branch
takes `req_pool_indices` in scheduler order. Those agree trivially at batch 1.
At batch >= 2, if they diverge, row i is paired with the wrong request's pool
index and the gather reads another request's KV. Same failure class as the
original `_compact_kv` bug: two index spaces that agree only in the
single-request case. `sequence_lens.view(-1, 1)` has the same exposure — a
misaligned length lets positions past a shorter request's end pass the validity
mask and read uninitialized KV.

Cheapest probe: assert `row_req_pool_indices is not None` on the decode path and
log whenever batch > 1 takes the fallback branch.

### Throughput note: the baseline numbers were inflated by corruption

Real per-stream decode at 4 streams is ~5.6-6.3 tok/s (clean), versus 10.63 in
run A where all four streams were corrupt. Degenerate `!` output accepts every
draft (accept rate 1.00) and runs ~2x faster than real generation. Any
historical throughput figure measured at concurrency on this box without an
output-validity check is suspect in the same direction.

---

## Run S — isolated box, stock config, 4 trials x 4 streams

Box isolated first: all 7 cron entries paused (backup
`~/flashnext/crontab.backup.txt`), `vllm_gateway` (Caddy, fronts the model on
the tailnet) stopped, zero connections to :30000, GPU holding only sglang.
Left running deliberately: `tailscale` (box access) and PIDs 4365/2954
(`tv_stream.py`, `swing_pivot_trader.py` — 2d12h continuous live paper-trading
track record, 42/36 MB RSS, no GPU).

| trial | corrupt streams | peak #running-req | per-stream decode | agg_true |
|---|---|---|---|---|
| 1 | 0 | 2 | 8.11 | 40.5 |
| 2 | 0 | 2 | 6.67 | 32.9 |
| 3 | 0 | 2 | 7.19 | 31.6 |
| 4 | 0 | 2 | 6.67 | 31.5 |

**0/4 trials corrupt.** `peak_running_req = 2` in every trial confirms no
traffic beyond what the harness generated.

### FINDING 5: the confound is real, but 4 trials cannot establish absence

`P(0 corrupt in 4 | true rate 20%) = 0.8^4 = 0.41`. This result is fully
consistent with a 20% event rate and rules nothing out. It does argue against a
*high* rate (`P(0/4 | 50%) = 0.06`), which matters because run A had 4/4 streams
corrupt at this exact stream count — if the isolated box behaved like run A, 4
trials would have caught it.

Cron-timing correlation (suggestive, NOT established):

| run | wall time (EDT) | cron window | corrupt? |
|---|---|---|---|
| A | ~19:34 | after `*/30` fire at 19:30 | YES |
| B, B2 | ~19:47-19:58 | spans `55 6-23` fire at 19:55 | YES |
| A2-A5 | ~20:00-20:12 | overlaps `*/30` fire at 20:00 | NO |
| S 1-4 | ~20:20 | all crons paused | NO |

A2-A5 weaken the correlation: they also overlapped a cron window and stayed
clean. And a grep of `Weather/scripts/*.py` + `converged/*.sh` found **no**
reference to `:30000` or `spark-vllm`, so it is not even confirmed that the
weather jobs call the LLM. The mechanism might be host memory/CPU pressure
rather than LLM requests, or the association may be coincidence.

Honest state: corruption is real and was observed 3 times, always at
concurrency >= 2, never at 1 stream in any run. Under verified isolation it has
not recurred in 4 trials. That is not enough to call it fixed, explained, or
attributed.

## Next actions (paused for user)

1. Reconcile the 11.2 vs 31.6 tok/s discrepancy (measurement vs regression).
2. Root-cause the concurrency corruption. First probes, cheapest first:
   - does it reproduce with `--speculative-algorithm none`? (isolates spec
     decode / DeltaNet draft-rollback from attention)
   - does it reproduce with `--max-running-requests 1`?
   - does it reproduce with `--disable-cuda-graph`? (isolates graph replay
     across batch-size changes as concurrency ramps)
   - per-layer NaN probe at the first corrupt token, as used for the original
     `_compact_kv` bug
3. Only then resume the KV-pool -> mamba-cache sweep.

Tier 1 / Tier 2 assessment not started. Note in advance: SpecBlock, Bastion and
Draft-Less-Retrieve-More all require tree drafting (topk>1), which is a hard
`NotImplementedError` in `QwenSparseMultiStepDraftBackend`, and all four Tier-1
items are gated behind the QSA ring cap. They are implementation projects, not
flag flips.
