#!/usr/bin/env python3
"""Results-page benchmark suite for Flash-Next on the Spark (sglang, HashK+NEXTN).
Sections: depth sweep (cold), concurrency @32k, prefix cache cold/warm,
reasoning effort (default/medium/off). Writes bench_results_suite.json."""
import json, time, random, threading, urllib.request

BASE = "http://127.0.0.1:30000/v1/chat/completions"
WORDS = ("alpha bravo copper delta ember frost gable harbor iris juniper kelp "
         "lumen marrow nectar onyx pallet quartz rivet sable timber").split()
OUT = {"sections": {}}


def filler(n_words, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def stream_req(prompt, max_tok=256, kwargs=None, timeout=3600):
    payload = {"model": "m", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tok, "stream": True,
               "stream_options": {"include_usage": True}}
    if kwargs is not None:
        payload["chat_template_kwargs"] = kwargs
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; tlast = None; usage = None; text = []; rtok_seen = False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    tlast = now
                    if d.get("content"):
                        text.append(d["content"])
    wall = time.time() - t0
    u = usage or {}
    pt = u.get("prompt_tokens", 0)
    ct = u.get("completion_tokens", 0)
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    decode_t = (tlast - (t0 + ttft)) if (ttft and tlast and tlast > t0 + ttft) else None
    return {"prompt_tokens": pt, "completion_tokens": ct, "reasoning_tokens": rt,
            "ttft_s": round(ttft, 2) if ttft else None, "wall_s": round(wall, 2),
            "prefill_toks": round(pt / ttft, 0) if ttft and pt else None,
            "decode_toks": round((ct - 1) / decode_t, 2) if decode_t and ct > 1 else None,
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "text": "".join(text)}


def sec_depth():
    print("=== depth sweep (cold) ===", flush=True)
    rows = []
    for name, nw in (("8k", 6000), ("32k", 24000), ("128k", 96000)):
        p = filler(nw, seed=hash(name) % 10**6) + "\n\nSummarize the vocabulary above in one sentence."
        r = stream_req(p, 256, kwargs={"enable_thinking": False})
        row = {"depth": name, **{k: r[k] for k in ("prompt_tokens", "ttft_s", "prefill_toks", "decode_toks", "cached")}}
        rows.append(row)
        print(row, flush=True)
    OUT["sections"]["depth"] = rows


def sec_concurrency():
    print("=== concurrency @32k ===", flush=True)
    rows = []
    for streams in (1, 4, 8):
        prompts = [filler(24000, seed=7000 + streams * 100 + i)
                   + "\n\nSummarize the vocabulary above in one sentence." for i in range(streams)]
        results = [None] * streams
        def work(i):
            results[i] = stream_req(prompts[i], 256, kwargs={"enable_thinking": False})
        t0 = time.time()
        th = [threading.Thread(target=work, args=(i,)) for i in range(streams)]
        [t.start() for t in th]
        [t.join() for t in th]
        wall = time.time() - t0
        per = [r["decode_toks"] for r in results if r and r["decode_toks"]]
        total_ct = sum(r["completion_tokens"] for r in results if r)
        # aggregate decode: total completion tokens / (wall - first ttft) is messy
        # under staggered prefill; report total tokens / wall past last TTFT window
        agg = round(sum(per), 1) if per else None
        row = {"streams": streams, "per_stream_decode": round(sum(per) / len(per), 2) if per else None,
               "aggregate_decode": agg, "wall_s": round(wall, 1),
               "scaling": None}
        rows.append(row)
        print(row, flush=True)
    base = rows[0]["aggregate_decode"]
    for row in rows:
        if base and row["aggregate_decode"]:
            row["scaling"] = round(row["aggregate_decode"] / base, 1)
    OUT["sections"]["concurrency"] = rows


def sec_prefix():
    print("=== prefix cache cold/warm ===", flush=True)
    rows = []
    for name, nw in (("8k", 6000), ("32k", 24000), ("128k", 96000)):
        p = filler(nw, seed=31000 + nw) + "\n\nSummarize the vocabulary above in one sentence."
        cold = stream_req(p, 32, kwargs={"enable_thinking": False})
        warm = stream_req(p, 32, kwargs={"enable_thinking": False})
        row = {"depth": name, "cold_prefill": cold["prefill_toks"],
               "warm_prefill": warm["prefill_toks"], "warm_cached": warm["cached"],
               "ratio": round(warm["prefill_toks"] / cold["prefill_toks"], 1)
               if cold["prefill_toks"] and warm["prefill_toks"] else None}
        rows.append(row)
        print(row, flush=True)
    OUT["sections"]["prefix"] = rows


def sec_effort():
    print("=== reasoning effort ===", flush=True)
    Q = ("A train leaves city A at 9:00 travelling 80 km/h. Another leaves city B "
         "(240 km away) at 9:30 travelling toward A at 100 km/h. At what time do "
         "they meet? Show your answer as HH:MM.")
    variants = [
        ("default (thinking on)", {"enable_thinking": True}),
        ("medium", {"enable_thinking": True, "reasoning_effort": "medium"}),
        ("thinking off", {"enable_thinking": False}),
    ]
    rows = []
    for name, kw in variants:
        r = stream_req(Q, 4096, kwargs=kw)
        correct = "10:3" in r["text"] or "10.5" in r["text"]
        row = {"variant": name, "out_tok": r["completion_tokens"],
               "reasoning_tok": r["reasoning_tokens"],
               "time_to_answer_s": r["wall_s"], "ttft_s": r["ttft_s"],
               "decode_toks": r["decode_toks"], "correct": correct,
               "answer_tail": r["text"][-80:]}
        rows.append(row)
        print(row, flush=True)
    OUT["sections"]["effort"] = rows


if __name__ == "__main__":
    t0 = time.time()
    sec_depth()
    sec_concurrency()
    sec_prefix()
    sec_effort()
    OUT["total_s"] = round(time.time() - t0, 1)
    with open("/home/serverdestroyers/flashnext/bench_results_suite.json", "w") as f:
        json.dump(OUT, f, indent=2)
    print(f"done in {OUT['total_s']}s", flush=True)
