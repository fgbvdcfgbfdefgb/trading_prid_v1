#!/usr/bin/env python3
"""
Resamples raw tick/trade chunks (data/raw_trades/trades_*.csv.gz) into regular
OHLCV bars at 1-second and 1-minute resolution, written as parquet under
data/ohlc_1s/ and data/ohlc_1m/ (one parquet file per UTC day for 1s bars,
one file total for 1m bars since it's much smaller).

Safe to re-run repeatedly (e.g. from a cron-like loop) - it tracks which raw
chunk files it has already folded in via a small state file, and re-derives
the bars for the (small) tail portion each time so late-arriving trades in the
currently-open chunk are reflected.
"""
import glob
import gzip
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


def load_all_trades():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "trades_*.csv.gz")))
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
    df = df.sort_values("time")
    return df


def ohlcv(df, rule):
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
    out = out.reset_index().rename(columns={"time": "timestamp"})
    return out


def main():
    df = load_all_trades()
    if df is None or df.empty:
        print("[info] no raw trades yet, nothing to resample")
        return

    print(f"[info] loaded {len(df):,} raw trades spanning {df['time'].min()} -> {df['time'].max()}")

    # 1-minute bars: whole history, single parquet (compact)
    bars_1m = ohlcv(df, "1min")
    bars_1m.to_parquet(os.path.join(OUT_1M, "btcusd_1m.parquet"), index=False)
    print(f"[info] wrote {len(bars_1m):,} 1-minute bars")

    # 1-second bars: this is the "sec-sec" data used by the live simulator.
    # Partition by UTC day to keep files manageable.
    bars_1s = ohlcv(df, "1s")
    bars_1s["day"] = bars_1s["timestamp"].dt.strftime("%Y-%m-%d")
    for day, g in bars_1s.groupby("day"):
        g.drop(columns=["day"]).to_parquet(os.path.join(OUT_1S, f"btcusd_1s_{day}.parquet"), index=False)
    print(f"[info] wrote {len(bars_1s):,} 1-second bars across {bars_1s['day'].nunique()} day-files")

    with open(STATE_PATH, "w") as f:
        json.dump({"n_trades": len(df), "last_trade_time": str(df["time"].max())}, f)


if __name__ == "__main__":
    main()
