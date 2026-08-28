#!/usr/bin/env python3
"""Independent concurrency-corruption probe for the flashnext server.

For each requested concurrency level: fire N distinct requests at once
(threads), then one solo "post-check" request to see whether the batch left
the engine wedged. Corruption = '!!!!', long single-char runs, or empty text.
"""
import json, re, sys, threading, time, urllib.request

URL = "http://127.0.0.1:30000/v1/chat/completions"
PROMPTS = [
    "Explain how a hash table handles collisions.",
    "Write a Python function that reverses a linked list.",
    "Describe the water cycle in plain language.",
    "What causes inflation? Give a concise answer.",
    "Write a short story opening set on a container ship.",
    "Explain TCP slow start to a junior engineer.",
    "Summarize how photosynthesis works.",
    "Write a SQL query counting orders per customer per month.",
]


def corrupt(txt):
    if not txt.strip():
        return "empty"
    if "!!!!" in txt:
        return "bangs"
    m = re.search(r"(.)\1{39,}", txt)
    if m:
        return "charrun:" + repr(m.group(1))
    return ""


def one(idx, out, max_tok=300):
    p = {
        "model": "m",
        "messages": [{"role": "user", "content": PROMPTS[idx % len(PROMPTS)]}],
        "temperature": 0,
        "max_tokens": max_tok,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(p).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        wall = time.time() - t0
        ct = r["usage"]["completion_tokens"]
        msg = r["choices"][0]["message"]
        txt = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
        out[idx] = (ct, wall, corrupt(txt), txt[:70].replace("\n", " "))
    except Exception as e:
        out[idx] = (0, time.time() - t0, "error:" + str(e)[:60], "")


def level(n):
    out = {}
    ts = [threading.Thread(target=one, args=(i, out)) for i in range(n)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.time() - t0
    bad = sum(1 for v in out.values() if v[2])
    agg = sum(v[0] for v in out.values()) / wall
    print(f"== {n} streams: wall {wall:.0f}s  aggregate {agg:.1f} tok/s  corrupt {bad}/{n}")
    for i in sorted(out):
        ct, w, c, head = out[i]
        print(f"   s{i}: {ct} tok {ct/max(w,0.01):.1f} tok/s  {'CORRUPT[' + c + ']' if c else 'ok':<22} {head}")
    post = {}
    one(0, post, max_tok=120)
    ct, w, c, head = post[0]
    print(f"   post-check solo: {ct} tok {ct/max(w,0.01):.1f} tok/s  "
          f"{'WEDGED[' + c + ']' if c else 'clean'}  {head}")
    return bad


if __name__ == "__main__":
    for n in [int(x) for x in sys.argv[1:]] or [1, 2, 4]:
        level(n)
