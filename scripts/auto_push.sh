#!/usr/bin/env bash
# Periodically resamples whatever raw trade data has landed and pushes it (plus
# any code changes) to GitHub, so progress is never lost even though the full
# trade-history backfill takes many hours.
set -u
cd "$(dirname "$0")/.."
INTERVAL="${1:-300}"   # seconds between push cycles

while true; do
  python3 scripts/resample_to_bars.py >> data/resample.log 2>&1

  git add -A
  if ! git diff --cached --quiet; then
    N_TRADES=$(python3 -c "import json;print(json.load(open('data/raw_trades/checkpoint.json'))['total_trades'])" 2>/dev/null || echo "?")
    git commit -q -m "data: auto-sync (total trades so far: ${N_TRADES})"
    git push -q origin main 2>> data/push.log || git push -q origin master 2>> data/push.log
    echo "[$(date -u +%FT%TZ)] pushed commit, total_trades=${N_TRADES}" >> data/push.log
  fi
  sleep "$INTERVAL"
done
