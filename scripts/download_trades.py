#!/usr/bin/env python3
"""
Resumable full-history downloader for Kraken BTC/USD trade data (tick-level).
This is the real source of "second-by-second" data: Kraken's OHLC REST endpoint
only retains the most recent ~720 candles, but /Trades supports pagination all
the way back to the pair's inception (2013 for XBT/USD) via the `since` cursor.

Design:
  - Writes raw trades to rotating gzip CSV chunks under data/raw_trades/
  - Keeps a checkpoint.json with the last `since` cursor + totals so the job
    can be killed and restarted (`python3 download_trades.py`) without redoing work.
  - Paces requests to be a polite citizen of Kraken's public API.
  - When it catches up to "now", it switches to slow live polling so new trades
    keep streaming in (useful for the live simulator later).
"""
import csv
import gzip
import json
import os
import sys
import time
import urllib.request
import urllib.error

PAIR = "XBTUSD"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_trades")
CKPT_PATH = os.path.join(BASE_DIR, "data", "raw_trades", "checkpoint.json")
CHUNK_ROWS = 2_000_000           # rows per gz chunk file
SLEEP_BETWEEN_CALLS = 1.2        # seconds, polite pacing for public endpoint
MAX_RETRIES = 8

os.makedirs(RAW_DIR, exist_ok=True)


def load_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {"since_ns": "0", "total_trades": 0, "chunk_idx": 0, "rows_in_chunk": 0}


def save_checkpoint(ckpt):
    tmp = CKPT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, CKPT_PATH)


def chunk_path(idx):
    return os.path.join(RAW_DIR, f"trades_{idx:05d}.csv.gz")


def open_chunk_writer(idx, append):
    mode = "at" if append else "wt"
    f = gzip.open(chunk_path(idx), mode, newline="")
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
    ckpt = load_checkpoint()
    since_ns = ckpt["since_ns"]
    chunk_idx = ckpt["chunk_idx"]
    rows_in_chunk = ckpt["rows_in_chunk"]
    total = ckpt["total_trades"]

    f, w = open_chunk_writer(chunk_idx, append=os.path.exists(chunk_path(chunk_idx)))
    t_start = time.time()
    last_report = t_start
    empty_polls = 0

    print(f"[start] resuming from since={since_ns} total_so_far={total}", flush=True)

    try:
        while True:
            result = fetch(since_ns)
            pair_key = [k for k in result.keys() if k != "last"][0]
            trades = result[pair_key]
            last = result["last"]

            if not trades:
                empty_polls += 1
                sleep_s = min(30, 2 * empty_polls)
                time.sleep(sleep_s)
                since_ns = last
                continue
            empty_polls = 0

            for t in trades:
                price, volume, ts, side, otype, misc, trade_id = (t + [""])[:7] if len(t) < 7 else t
                w.writerow([price, volume, ts, side, otype, misc, trade_id])
                rows_in_chunk += 1
                total += 1

            since_ns = last

            if rows_in_chunk >= CHUNK_ROWS:
                f.close()
                chunk_idx += 1
                rows_in_chunk = 0
                f, w = open_chunk_writer(chunk_idx, append=False)

            ckpt = {"since_ns": since_ns, "total_trades": total,
                    "chunk_idx": chunk_idx, "rows_in_chunk": rows_in_chunk}
            save_checkpoint(ckpt)

            now = time.time()
            if now - last_report > 15:
                last_ts = float(trades[-1][2])
                rate = total / (now - t_start) if now > t_start else 0
                print(f"[progress] total_trades={total:,} reached_time={time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(last_ts))} UTC rate={rate:.1f}/s chunk={chunk_idx}", flush=True)
                last_report = now

            time.sleep(SLEEP_BETWEEN_CALLS)
    finally:
        f.close()


if __name__ == "__main__":
    main()
