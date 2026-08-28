#!/usr/bin/env python3
"""Per-variant speed benchmark: single-stream decode (freeform/code/repro),
4-stream concurrent aggregate (the user's real workload shape: 1 main + 3
subagents), deep-context corruption check, accept length. One JSON line out."""
import json, random, re, subprocess, sys, threading, time, urllib.request

BASE = "http://127.0.0.1:30000/v1/chat/completions"
NAME = sys.argv[1] if len(sys.argv) > 1 else "unnamed"
WORDS = ("ledger harbor velvet quartz meadow lantern copper thicket marble "
         "ember willow canyon prism fathom orchard breeze cinder galley "
         "hollow ridge summit tundra vessel walnut zephyr basalt cobalt "
         "drift ellipse fjord").split()


def req(prompt, max_tok, think=False):
    p = {"model": "m", "messages": [{"role": "user", "content": prompt}],
         "temperature": 0, "max_tokens": max_tok,
         "chat_template_kwargs": {"enable_thinking": think}}
    r = urllib.request.Request(BASE, data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(r, timeout=900).read())
    wall = time.time() - t0
    m = out["choices"][0]["message"]
    txt = (m.get("content") or "") + (m.get("reasoning_content") or "")
    ct = out["usage"]["completion_tokens"]
    bad = "!!!!" in txt or bool(re.search(r"(.)\1{39,}", txt))
    return ct, wall, bad


res = {"variant": NAME}
try:
    req("Say hi.", 20)  # warmup
    ct, w, b = req("Write a 250-word essay on lighthouses.", 300)
    res["freeform_toks"] = round(ct / w, 1); bad = b
    ct, w, b = req("Write a complete Python LRU cache class with O(1) get/put.", 500)
    res["code_toks"] = round(ct / w, 1); bad |= b
    ct, w, b = req("Count from 1 to 60, comma separated, no spaces.", 220)
    res["repro_toks"] = round(ct / w, 1); bad |= b

    outs = [None] * 4
    def work(i):
        try:
            outs[i] = req("Write a detailed 300-word essay about %s (v-%s)."
                          % (WORDS[i], NAME), 400)
        except Exception:
            outs[i] = (0, 1, True)
    th = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.time() - t0
    total = sum(o[0] for o in outs if o)
    bad |= any(o[2] for o in outs if o)
    res["agg4_toks"] = round(total / wall, 1)

    for i in range(2):
        rng = random.Random(hash(NAME) % 100000 + i)
        deep = " ".join(rng.choice(WORDS) for _ in range(5500)) \
            + "\n\nSummarize the vocabulary above in one sentence."
        ct, w, b = req(deep, 100)
        bad |= b
    res["corrupt"] = bool(bad)

    logs = subprocess.run(["docker", "logs", "--since", "6m", "flashnext"],
                          capture_output=True, text=True).stderr + \
           subprocess.run(["docker", "logs", "--since", "6m", "flashnext"],
                          capture_output=True, text=True).stdout
    acc = re.findall(r"accept len: ([0-9.]+)", logs)[-15:]
    res["accept_len"] = round(sum(map(float, acc)) / len(acc), 2) if acc else None
except Exception as e:
    res["error"] = str(e)[:120]
print("RESULT " + json.dumps(res), flush=True)
