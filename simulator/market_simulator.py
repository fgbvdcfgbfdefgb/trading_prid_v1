"""
Live market simulator.

Replays historical BTC/USD 1-second bars (built by scripts/resample_to_bars.py
from real Kraken trade data) one bar per wall-clock second, so the rest of the
stack (market analyser, predictor, trainer, dashboard) experiences it exactly
like a real live feed -- same cadence, same API, same "I don't know the future"
constraint.

If no real 1-second data has landed yet (the historical backfill can take
hours), it falls back to a synthetic geometric-Brownian-motion-with-jumps
generator seeded near the last known real BTC price, clearly flagged as
synthetic, so you can develop/test the full pipeline immediately instead of
waiting on the network job.
"""
import glob
import os
import random
import time
from dataclasses import dataclass

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OHLC_1S_DIR = os.path.join(BASE_DIR, "data", "ohlc_1s")


@dataclass
class Tick:
    t: float            # unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    n_trades: int
    synthetic: bool = False


class SyntheticFeed:
    """Fallback generator: random walk w/ mild jumps, used only until real
    resampled 1-second data is available."""

    def __init__(self, start_price=84500.0, seed=42):
        self.price = start_price
        self.rng = random.Random(seed)
        self.t = time.time()

    def next_tick(self) -> Tick:
        mu, sigma = 0.0, 0.0006
        ret = self.rng.gauss(mu, sigma)
        if self.rng.random() < 0.002:
            ret += self.rng.gauss(0, 0.01)  # occasional jump
        new_price = max(1.0, self.price * (1 + ret))
        o, c = self.price, new_price
        h, l = max(o, c) * (1 + abs(self.rng.gauss(0, 0.0002))), min(o, c) * (1 - abs(self.rng.gauss(0, 0.0002)))
        vol = abs(self.rng.gauss(0.05, 0.03))
        self.price = new_price
        self.t += 1
        return Tick(self.t, o, h, l, c, vol, self.rng.randint(0, 20), synthetic=True)


class HistoricalReplayFeed:
    """Replays real 1-second OHLCV bars loaded from data/ohlc_1s/*.parquet."""

    def __init__(self, loop=True):
        files = sorted(glob.glob(os.path.join(OHLC_1S_DIR, "*.parquet")))
        if not files:
            raise FileNotFoundError("no 1-second parquet files yet")
        frames = [pd.read_parquet(f) for f in files]
        self.df = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        self.i = 0
        self.loop = loop

    def __len__(self):
        return len(self.df)

    def next_tick(self) -> Tick:
        if self.i >= len(self.df):
            if self.loop:
                self.i = 0
            else:
                raise StopIteration
        row = self.df.iloc[self.i]
        self.i += 1
        return Tick(
            t=row["timestamp"].timestamp(),
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=float(row["volume"]), n_trades=int(row["n_trades"]),
            synthetic=False,
        )


class MarketSimulator:
    """Unified live-like feed: real replay when available, else synthetic.

    speed: wall-clock seconds between ticks. speed=1.0 => true real-time
    (1 tick/sec). speed=0 => as fast as possible (for offline training).
    """

    def __init__(self, speed: float = 1.0):
        self.speed = speed
        try:
            self.feed = HistoricalReplayFeed()
            self.mode = "historical_replay"
        except FileNotFoundError:
            self.feed = SyntheticFeed()
            self.mode = "synthetic_fallback"

    def stream(self):
        while True:
            tick = self.feed.next_tick()
            yield tick
            if self.speed > 0:
                time.sleep(self.speed)


if __name__ == "__main__":
    sim = MarketSimulator(speed=0.2)
    print(f"[simulator] mode={sim.mode}")
    for i, tick in enumerate(sim.stream()):
        print(tick)
        if i >= 10:
            break
