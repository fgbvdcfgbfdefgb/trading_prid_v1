#!/usr/bin/env python3
"""
Resumable downloader for Kraken BTC/USD trade data (tick-level = true "sec-sec" source).

Kraken's OHLC REST endpoint only retains the most recent ~720 candles, but
/Trades supports pagination all the way back to the pair's inception (2013)
via the `since` cursor, and also acts as a genuine live feed once caught up
to "now". This single script is used for two different jobs (see scripts/
run_recent.sh and run_backfill.sh):

  - "recent" job:   since = now - LOOKBACK_DAYS, catches up fast, then keeps
                     polling live -> feeds the real-time simulator/dashboard.
  - "backfill" job: since = 0 (or wherever it left off), slow multi-hour crawl
                     of full history -> feeds long-horizon model pretraining.

Usage:
  python3 download_trades.py --checkpoint data/raw_trades/checkpoint_recent.json \
      --chunk-prefix trades_recent_ --since-days-ago 3 --poll-when-live 10

  python3 download_trades.py --checkpoint data/raw_trades/checkpoint_backfill.json \
      --chunk-prefix trades_hist_ --since 0
"""
import argparse
import csv
import gzip
import json
import os
import signal
import sys
import time
import urllib.request
import urllib.error

PAIR = "XBTUSD"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_trades")
CHUNK_ROWS = 2_000_000
ROTATE_SECONDS = 600   # also rotate chunks on a time basis (not just row count), so
                       # the resampler always has a recently-closed, readable file
                       # instead of waiting on a 2M-row threshold that may take days
SLEEP_BETWEEN_CALLS = 2.5
MAX_RETRIES = 8

os.makedirs(RAW_DIR, exist_ok=True)


def load_checkpoint(path, default_since_ns):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"since_ns": default_since_ns, "total_trades": 0, "chunk_idx": 0, "rows_in_chunk": 0}


def save_checkpoint(path, ckpt):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, path)


def chunk_path(prefix, idx):
    return os.path.join(RAW_DIR, f"{prefix}{idx:05d}.csv.gz")


def open_chunk_writer(prefix, idx, append):
    mode = "at" if append else "wt"
    f = gzip.open(chunk_path(prefix, idx), mode, newline="")
    w = csv.writer(f)
    if not append:
        w.writerow(["price", "volume", "time", "side", "order_type", "misc", "trade_id"])
    return f, w


def fetch(since_ns, retries=0):
    url = f"https://api.kraken.com/0/public/Trades?pair={PAIR}&since={since_ns}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "trading_prid_v1-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        return data["result"]
    except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as e:
        if retries >= MAX_RETRIES:
            raise
        backoff = min(60, 2 ** retries)
        print(f"[warn] fetch failed ({e}); retrying in {backoff}s", flush=True)
        time.sleep(backoff)
        return fetch(since_ns, retries + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--chunk-prefix", required=True)
    ap.add_argument("--since", default=None, help="raw since cursor (ns) to start from if no checkpoint")
    ap.add_argument("--since-days-ago", type=float, default=None, help="start `N` days before now if no checkpoint")
    ap.add_argument("--poll-when-live", type=float, default=0,
                     help="if >0, once caught up to now, keep polling live every N seconds forever")
    ap.add_argument("--stop-when-live", action="store_true",
                     help="exit cleanly once caught up to now (ignored if --poll-when-live set)")
    args = ap.parse_args()

    if args.since is not None:
        default_since = args.since
    elif args.since_days_ago is not None:
        default_since = str(int((time.time() - args.since_days_ago * 86400) * 1e9))
    else:
        default_since = "0"

    ckpt = load_checkpoint(args.checkpoint, default_since)
    since_ns = ckpt["since_ns"]
    chunk_idx = ckpt["chunk_idx"]
    rows_in_chunk = ckpt["rows_in_chunk"]
    total = ckpt["total_trades"]
    prefix = args.chunk_prefix

    # Safety: NEVER append to an existing gzip chunk across process restarts.
    # Appending a fresh gzip member to a file whose previous member may not
    # have been flushed cleanly (e.g. the process was SIGKILL'd) produces a
    # corrupted archive. Instead, if this run's target chunk file already
    # exists, start a brand-new chunk index -- cheap (small files) and safe.
    if os.path.exists(chunk_path(prefix, chunk_idx)):
        chunk_idx += 1
    rows_in_chunk = 0
    f, w = open_chunk_writer(prefix, chunk_idx, append=False)
    last_rotate = time.time()
    t_start = time.time()
    last_report = t_start
    empty_polls = 0

    def _graceful_exit(signum, frame):
        print(f"[signal] caught {signum}, flushing and closing cleanly...", flush=True)
        try:
            f.flush()
            f.close()
        finally:
            save_checkpoint(args.checkpoint, {"since_ns": since_ns, "total_trades": total,
                                               "chunk_idx": chunk_idx, "rows_in_chunk": rows_in_chunk})
        sys.exit(0)

    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)

    print(f"[start] job={prefix} resuming from since={since_ns} total_so_far={total} (new chunk {chunk_idx})", flush=True)

    try:
        while True:
            result = fetch(since_ns)
            pair_key = [k for k in result.keys() if k != "last"][0]
            trades = result[pair_key]
            last = result["last"]

            if not trades:
                empty_polls += 1
                since_ns = last
                save_checkpoint(args.checkpoint, {"since_ns": since_ns, "total_trades": total,
                                                   "chunk_idx": chunk_idx, "rows_in_chunk": rows_in_chunk})
                if args.poll_when_live > 0:
                    print(f"[live] caught up to now, total_trades={total:,}. Polling every {args.poll_when_live}s", flush=True)
                    time.sleep(args.poll_when_live)
                    continue
                elif args.stop_when_live:
                    print(f"[done] caught up to now, total_trades={total:,}. Exiting.", flush=True)
                    break
                else:
                    time.sleep(min(30, 2 * empty_polls))
                    continue
            empty_polls = 0

            for t in trades:
                row = t if len(t) >= 7 else (t + [""] * (7 - len(t)))
                w.writerow(row[:7])
                rows_in_chunk += 1
                total += 1

            since_ns = last

            if rows_in_chunk >= CHUNK_ROWS or (time.time() - last_rotate) >= ROTATE_SECONDS:
                f.close()
                chunk_idx += 1
                rows_in_chunk = 0
                f, w = open_chunk_writer(prefix, chunk_idx, append=False)
                last_rotate = time.time()

            save_checkpoint(args.checkpoint, {"since_ns": since_ns, "total_trades": total,
                                               "chunk_idx": chunk_idx, "rows_in_chunk": rows_in_chunk})

            now = time.time()
            if now - last_report > 15:
                last_ts = float(trades[-1][2])
                rate = total / (now - t_start) if now > t_start else 0
                print(f"[progress] job={prefix} total_trades={total:,} reached_time="
                      f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(last_ts))} UTC "
                      f"rate={rate:.1f}/s chunk={chunk_idx}", flush=True)
                last_report = now

            time.sleep(SLEEP_BETWEEN_CALLS)
    finally:
        f.close()


if __name__ == "__main__":
    main()
