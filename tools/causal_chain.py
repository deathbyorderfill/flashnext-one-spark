#!/usr/bin/env python3
"""Causal-chain test for the state-poisoning corruption hypothesis.

Phase 1 (fresh boot): deep request first -> expect clean.
Phase 2: concurrency storm (8 streams, forces queueing + retraction).
Phase 3: fresh deep request -> corrupt if the storm poisons shared state.
Phase 4: idle 5 min, fresh deep request -> does it self-recover?
Every deep prompt uses a unique seed (no radix reuse between phases).
"""
import json, random, threading, time, urllib.request

BASE = "http://127.0.0.1:30000/v1/chat/completions"
WORDS = ("ledger harbor velvet quartz meadow lantern copper thicket marble "
         "ember willow canyon prism fathom orchard breeze cinder galley "
         "hollow ridge summit tundra vessel walnut zephyr basalt cobalt "
         "drift ellipse fjord").split()


def deep(seed, label, nw=5500):
    rng = random.Random(seed)
    prompt = " ".join(rng.choice(WORDS) for _ in range(nw)) \
        + "\n\nSummarize the vocabulary above in one sentence."
    p = {"model": "m", "messages": [{"role": "user", "content": prompt}],
         "temperature": 0, "max_tokens": 160,
         "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE, data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    wall = time.time() - t0
    u = r["usage"]
    txt = r["choices"][0]["message"].get("content") or ""
    bad = "!!!!" in txt or not txt.strip()
    print(f"[{label}] pt={u['prompt_tokens']} "
          f"{'CORRUPT' if bad else 'ok'} rate {u['completion_tokens']/wall:.1f} "
          f"{txt[:55]!r}", flush=True)
    return bad


def storm():
    prompts = ["Write a detailed essay about topic %d: %s." % (i, w)
               for i, w in enumerate(WORDS[:8])]
    def work(i):
        p = {"model": "m", "messages": [{"role": "user", "content": prompts[i]}],
             "temperature": 0, "max_tokens": 700}
        req = urllib.request.Request(BASE, data=json.dumps(p).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=900).read()
        except Exception as e:
            print("  storm stream", i, "error:", str(e)[:60], flush=True)
    th = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    print(f"[storm] 8 streams x 700 tok done in {time.time()-t0:.0f}s", flush=True)


print("=== phase 1: deep on fresh boot ===", flush=True)
p1 = deep(9001, "fresh-boot deep A")
deep(9002, "fresh-boot deep B")
print("=== phase 2: concurrency storm ===", flush=True)
storm()
print("=== phase 3: deep right after storm ===", flush=True)
p3a = deep(9003, "post-storm deep A")
p3b = deep(9004, "post-storm deep B")
if p3a or p3b:
    print("=== phase 4: idle 300s then deep ===", flush=True)
    time.sleep(300)
    deep(9005, "post-idle deep")
else:
    print("=== storm did not poison; trying second storm + immediate deep ===",
          flush=True)
    storm()
    deep(9006, "post-storm2 deep")
print("DONE", flush=True)
