#!/usr/bin/env python3
"""2-stream decode benchmark for Flash-Next (goal: 40-50 tok/s aggregate).

Fixes over v1: warms the server first (51 GB PLE table is NVMe-mmap'd, so a
cold server faults pages on early requests), and counts ALL generated tokens
via usage.completion_tokens — production runs thinking-on, and reasoning
tokens arrive as reasoning_content deltas, invisible to a content-only counter.
"""
import json, threading, time, urllib.request, random, sys

PORT = sys.argv[1] if len(sys.argv) > 1 else "30000"
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"
random.seed(11)
WORDS = "system design latency cache token stream buffer kernel memory".split()


def filler(n):
    return " ".join(random.choice(WORDS) for _ in range(n))


def one(prompt, max_tok, out, idx):
    req = urllib.request.Request(
        BASE,
        data=json.dumps({"messages": [{"role": "user", "content": prompt}],
                         "temperature": 0, "max_tokens": max_tok,
                         "stream": True,
                         "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = None; n_chunks = 0; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    n_chunks += 1
                    if tf is None:
                        tf = time.time()
    te = time.time()
    n = (usage or {}).get("completion_tokens") or n_chunks
    out[idx] = {"tok": n, "ttft": (tf - t0) if tf else None,
                "decode": (n - 1) / (te - tf) if tf and n > 1 else 0}


def run(nstream, depth_words, max_tok=400, label="", quiet=False):
    prompt = filler(depth_words) + "\n\nWrite a thorough technical explanation of write-ahead logging."
    out = [None] * nstream
    ths = [threading.Thread(target=one, args=(prompt, max_tok, out, i))
           for i in range(nstream)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    ok = [o for o in out if o]
    per = [o["decode"] for o in ok]
    if quiet:
        return
    print(f"{label:26} streams={nstream} per-stream={sum(per)/len(per):6.2f} "
          f"aggregate={sum(per):6.2f} tok/s | tokens={sum(o['tok'] for o in ok):5d} "
          f"wall={wall:5.1f}s ttft={min(o['ttft'] for o in ok if o['ttft']):.2f}s",
          flush=True)
    return sum(per)


print(f"=== warming (PLE page cache + CUDA graphs), port {PORT} ===", flush=True)
for i in range(3):
    run(2, 200, max_tok=150, quiet=True)
print("=== measurements ===", flush=True)
run(1, 200, label="1 stream / short")
run(2, 200, label="2 streams / short")
run(2, 4000, label="2 streams / 4k ctx")
run(2, 16000, label="2 streams / 16k ctx")
print("BENCH2_DONE")
