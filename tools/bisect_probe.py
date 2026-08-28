#!/usr/bin/env python3
"""Poisoning-recipe probe for one config variant.

Recipe (mirrors the confirmed repro): deep x2, storm, deep x2, storm,
deep x6. Verdict POISONED if any deep request corrupts, else CLEAN.
Seeds offset per variant so radix never crosses variants.
"""
import json, random, sys, threading, time, urllib.request

BASE = "http://127.0.0.1:30000/v1/chat/completions"
WORDS = ("ledger harbor velvet quartz meadow lantern copper thicket marble "
         "ember willow canyon prism fathom orchard breeze cinder galley "
         "hollow ridge summit tundra vessel walnut zephyr basalt cobalt "
         "drift ellipse fjord").split()
OFF = int(sys.argv[1]) if len(sys.argv) > 1 else 0
bad_total = 0


def deep(seed, label):
    global bad_total
    rng = random.Random(seed)
    prompt = " ".join(rng.choice(WORDS) for _ in range(5500)) \
        + "\n\nSummarize the vocabulary above in one sentence."
    p = {"model": "m", "messages": [{"role": "user", "content": prompt}],
         "temperature": 0, "max_tokens": 100,
         "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE, data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        txt = r["choices"][0]["message"].get("content") or ""
        bad = "!!!!" in txt or not txt.strip()
    except Exception as e:
        print(f"  [{label}] ERROR {str(e)[:60]}", flush=True)
        return
    bad_total += bad
    print(f"  [{label}] {'CORRUPT' if bad else 'ok'}", flush=True)


def storm(tag):
    def work(i):
        p = {"model": "m", "messages": [{"role": "user",
             "content": f"Write a detailed essay about {WORDS[i]} (variant {OFF}/{tag})."}],
             "temperature": 0, "max_tokens": 700}
        req = urllib.request.Request(BASE, data=json.dumps(p).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=900).read()
        except Exception as e:
            print(f"  storm{tag} s{i} error {str(e)[:50]}", flush=True)
    th = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    print(f"  [storm{tag}] done {time.time()-t0:.0f}s", flush=True)


deep(OFF + 1, "pre-A"); deep(OFF + 2, "pre-B")
storm(1)
deep(OFF + 3, "mid-A"); deep(OFF + 4, "mid-B")
storm(2)
for i in range(6):
    deep(OFF + 10 + i, f"post-{i}")
print(f"VERDICT: {'POISONED' if bad_total else 'CLEAN'} ({bad_total} corrupt)", flush=True)
