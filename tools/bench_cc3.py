#!/usr/bin/env python3
"""Corrected concurrency benchmark (v3 metric).

Same conditions as the v2 suite's concurrency section (32k-token filler
prompts, thinking off, 256 new tokens) but with honest metrics:
  - window_aggregate: total completion tokens / (last token time - first
    token time) -- true concurrent decode throughput over the decode window.
  - e2e_aggregate: total completion tokens / wall (includes prefill+queue).
  - v2_style_sum: sum of per-stream decode rates (the OLD, inflated metric,
    reported only to quantify the inflation).
  - per-request output-validity check (degenerate '!'/char-run/empty).
"""
import json, random, re, threading, time, urllib.request

BASE = "http://127.0.0.1:30000/v1/chat/completions"
WORDS = ("ledger harbor velvet quartz meadow lantern copper thicket marble "
         "ember willow canyon prism fathom orchard breeze cinder galley "
         "hollow ridge summit tundra vessel walnut zephyr basalt cobalt "
         "drift ellipse fjord").split()


def filler(n_words, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def corrupt(txt):
    if not txt.strip():
        return "empty"
    if "!!!!" in txt:
        return "bangs"
    m = re.search(r"(.)\1{39,}", txt)
    if m:
        return "charrun:" + repr(m.group(1))
    return ""


def stream_req(prompt, max_tok=256, timeout=3600):
    payload = {"model": "m", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tok, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    tlast = None
    usage = None
    text = []
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
    u = usage or {}
    ct = u.get("completion_tokens", 0)
    dec_rate = ct / (tlast - (t0 + ttft)) if (ttft and tlast and tlast > t0 + ttft) else None
    return {"ct": ct, "t_first": (t0 + ttft) if ttft else None, "t_last": tlast,
            "wall": time.time() - t0, "dec_rate": dec_rate,
            "bad": corrupt("".join(text))}


def level(streams):
    prompts = [filler(24000, seed=7000 + streams * 100 + i)
               + "\n\nSummarize the vocabulary above in one sentence."
               for i in range(streams)]
    results = [None] * streams
    def work(i):
        try:
            results[i] = stream_req(prompts[i])
        except Exception as e:
            results[i] = {"ct": 0, "t_first": None, "t_last": None,
                          "wall": 0, "dec_rate": None, "bad": "error:" + str(e)[:60]}
    t0 = time.time()
    th = [threading.Thread(target=work, args=(i,)) for i in range(streams)]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.time() - t0
    ok = [r for r in results if r]
    total = sum(r["ct"] for r in ok)
    firsts = [r["t_first"] for r in ok if r["t_first"]]
    lasts = [r["t_last"] for r in ok if r["t_last"]]
    window = (max(lasts) - min(firsts)) if firsts and lasts else None
    rates = [r["dec_rate"] for r in ok if r["dec_rate"]]
    bad = sum(1 for r in ok if r["bad"])
    row = {
        "streams": streams,
        "total_completion_tokens": total,
        "window_aggregate_toks": round(total / window, 1) if window else None,
        "e2e_aggregate_toks": round(total / wall, 1),
        "v2_style_sum_toks": round(sum(rates), 1) if rates else None,
        "mean_per_stream_decode": round(sum(rates) / len(rates), 1) if rates else None,
        "corrupt": bad,
        "wall_s": round(wall, 1),
    }
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    rows = [level(n) for n in (1, 2, 4, 8)]
    json.dump(rows, open("/home/serverdestroyers/flashnext/bench_cc3_results.json", "w"), indent=1)
    print("saved bench_cc3_results.json", flush=True)
