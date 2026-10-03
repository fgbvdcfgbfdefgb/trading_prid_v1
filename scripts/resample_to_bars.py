#!/usr/bin/env python3
"""
Resamples raw tick/trade chunks (data/raw_trades/trades_*.csv.gz) into regular
OHLCV bars.

IMPORTANT memory note: this sandbox has ~2GB RAM. `df.resample(rule)` creates
ONE BUCKET PER TIME UNIT ACROSS THE FULL SPAN of the dataframe, even where
there's no data -- e.g. resampling a frame that spans 2016-to-today at
1-minute resolution materializes ~5M rows *regardless* of how few actual
trades are in it, because of the multi-year gap between the "hist" backfill
(still crawling through 2016 as of this writing) and the "recent" window
(today). That blew the sandbox's memory budget. Fix: resample in bounded
TIME CHUNKS (per ISO week for 1s bars, per year for 1m bars) and concatenate
the small resulting bar-level results -- memory now stays flat no matter how
far the full history eventually spans.

  - 1-MINUTE bars: full combined history (hist + recent), chunked by year.
  - 1-SECOND bars ("sec-sec" data, what the live simulator wants): only the
    contiguous `trades_recent_*` window, chunked by week.
"""
import glob
import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_trades")
OUT_1S = os.path.join(BASE_DIR, "data", "ohlc_1s")
OUT_1M = os.path.join(BASE_DIR, "data", "ohlc_1m")
STATE_PATH = os.path.join(BASE_DIR, "data", "resample_state.json")

os.makedirs(OUT_1S, exist_ok=True)
os.makedirs(OUT_1M, exist_ok=True)


def load_trades(pattern):
    files = sorted(glob.glob(os.path.join(RAW_DIR, pattern)))
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, compression="gzip")
            frames.append(df)
        except Exception as e:
            print(f"[warn] skipping unreadable {fp}: {e}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["price"] = df["price"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def ohlcv_chunk(df, rule):
    s = df.set_index("time")
    bars = s["price"].resample(rule).ohlc()
    vol = s["volume"].resample(rule).sum().rename("volume")
    ntr = s["price"].resample(rule).count().rename("n_trades")
    out = bars.join(vol).join(ntr)
    out["close"] = out["close"].ffill()
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out["close"])
    out["low"] = out["low"].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    out["n_trades"] = out["n_trades"].fillna(0).astype(int)
    return out.reset_index().rename(columns={"time": "timestamp"})


def resample_chunked(df, rule, chunk_freq):
    """Resamples df at `rule`, processing one `chunk_freq` period at a time
    so memory never scales with the dataframe's full time span -- only with
    how much data lands in any single chunk."""
    if df is None or df.empty:
        return None
    period = df["time"].dt.to_period(chunk_freq)
    results = []
    for _, grp in df.groupby(period, observed=True):
        if grp.empty:
            continue
        results.append(ohlcv_chunk(grp, rule))
    if not results:
        return None
    return pd.concat(results, ignore_index=True).drop_duplicates(subset="timestamp").sort_values("timestamp")


def main():
    hist = load_trades("trades_hist_*.csv.gz")
    recent = load_trades("trades_recent_*.csv.gz")

    # ---- 1-minute bars: full combined history, chunked by YEAR to bound memory ----
    bars_1m_parts = []
    for label, d in (("hist", hist), ("recent", recent)):
        if d is None or d.empty:
            continue
        print(f"[info] resampling {label} -> 1m bars ({len(d):,} trades, "
              f"{d['time'].min()} -> {d['time'].max()})")
        b = resample_chunked(d, "1min", "Y")
        if b is not None:
            bars_1m_parts.append(b)

    if bars_1m_parts:
        bars_1m = pd.concat(bars_1m_parts, ignore_index=True).drop_duplicates(subset="timestamp").sort_values("timestamp")
        bars_1m.to_parquet(os.path.join(OUT_1M, "btcusd_1m.parquet"), index=False)
        print(f"[info] wrote {len(bars_1m):,} 1-minute bars total")
    else:
        print("[info] no trades yet for 1-minute bars")

    # ---- 1-second bars: ONLY the contiguous recent window, chunked by WEEK ----
    if recent is not None and not recent.empty:
        print(f"[info] resampling recent -> 1s bars ({len(recent):,} trades)")
        bars_1s = resample_chunked(recent, "1s", "W")
        if bars_1s is not None:
            bars_1s["day"] = bars_1s["timestamp"].dt.strftime("%Y-%m-%d")
            for day, g in bars_1s.groupby("day"):
                g.drop(columns=["day"]).to_parquet(os.path.join(OUT_1S, f"btcusd_1s_{day}.parquet"), index=False)
            print(f"[info] wrote {len(bars_1s):,} 1-second bars across {bars_1s['day'].nunique()} day-files")
    else:
        print("[info] no recent-window trades yet for 1-second bars")

    with open(STATE_PATH, "w") as f:
        json.dump({
            "n_hist_trades": 0 if hist is None else len(hist),
            "n_recent_trades": 0 if recent is None else len(recent),
        }, f)


if __name__ == "__main__":
    main()
