#!/usr/bin/env bash
# Flash-Next watchdog: revives a dead server AND detects the wedged state.
# Rate-limited so a crash-looping boot cannot spiral (max 4 actions/day).
#
# Wedge mode (found 2026-08-27): after aborted client requests the engine can
# degrade silently -- zombie requests grind at ~5 tok/s, speculative decoding
# rejects every draft (accept len: 1.00), state slots leak -- while /health
# still returns 200. Signature: every recent "Decode batch" log line shows
# accept len: 1.00 with running requests. Healthy traffic shows ~3+.
LOG=~/flashnext/watchdog.log
limit_ok() {
  recent=$(grep -c "$(date -I)" "$LOG" 2>/dev/null || echo 0)
  [ "$recent" -lt 4 ]
}

st=$(docker ps --filter name=flashnext --format "{{.Status}}")

if ! curl -fsS -m 5 http://127.0.0.1:30000/health >/dev/null 2>&1; then
  [ -n "$st" ] && exit 0   # up but unhealthy = likely mid-boot; leave it alone
  limit_ok || exit 0
  docker start flashnext >/dev/null 2>&1 && \
    echo "$(date -Is) watchdog: restarted flashnext (down)" >> "$LOG"
  exit 0
fi

# Healthy endpoint -- check for the silent wedge.
lines=$(docker logs flashnext --since 3m 2>&1 | grep -a "Decode batch" | tail -8)
[ -z "$lines" ] && exit 0
total=$(echo "$lines" | wc -l)
wedged=$(echo "$lines" | grep -c "accept len: 1.00")
running=$(echo "$lines" | grep -cE "#running-req: [1-9]")
if [ "$total" -ge 6 ] && [ "$wedged" -eq "$total" ] && [ "$running" -eq "$total" ]; then
  limit_ok || exit 0
  echo "$(date -Is) watchdog: WEDGE detected (accept len 1.00 x$total) - restarting" >> "$LOG"
  docker restart flashnext >/dev/null 2>&1
fi
