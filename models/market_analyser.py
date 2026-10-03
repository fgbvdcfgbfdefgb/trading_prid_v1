"""
Market Analyser -- runs on CPU.

Consumes a rolling raw window of recent 1-second OHLCV ticks and produces a
compact "market state" embedding: engineered microstructure features (returns,
realized volatility, volume imbalance, momentum at multiple horizons) refined
through a small feed-forward encoder. This is deliberately cheap (CPU-only)
because it has to run once *every second* in the live loop; the heavier
sequence model (the price predictor) is what runs on GPU.

Trained with a lightweight self-supervised objective: predict the next-second
return from the embedding (a proxy task that forces the embedding to capture
useful short-horizon structure). The price predictor downstream then learns
the harder 30-minute-ahead task on top of these embeddings.
"""
import collections
import math

import torch
import torch.nn as nn

RAW_FEATURES = 7     # log_ret, hl_range, body, volume, n_trades, vol_z, ret_z
WINDOW = 60           # seconds of raw history fed in per step
EMBED_DIM = 16


def extract_raw_features(window: list) -> list:
    """window: list[Tick]-like dicts with close/high/low/open/volume/n_trades.
    Returns a flat list of RAW_FEATURES per step (length WINDOW)."""
    feats = []
    closes = [w["close"] for w in window]
    vols = [w["volume"] for w in window]
    mean_v = sum(vols) / len(vols) if vols else 1.0
    std_v = (sum((v - mean_v) ** 2 for v in vols) / len(vols)) ** 0.5 if vols else 1.0
    std_v = std_v or 1.0
    rets = [0.0] + [math.log(max(closes[i], 1e-9) / max(closes[i - 1], 1e-9)) for i in range(1, len(closes))]
    mean_r = sum(rets) / len(rets) if rets else 0.0
    std_r = (sum((r - mean_r) ** 2 for r in rets) / len(rets)) ** 0.5 if rets else 1e-6
    std_r = std_r or 1e-6
    for i, w in enumerate(window):
        log_ret = rets[i]
        hl_range = (w["high"] - w["low"]) / max(w["close"], 1e-9)
        body = (w["close"] - w["open"]) / max(w["close"], 1e-9)
        volume = w["volume"]
        n_trades = w["n_trades"]
        vol_z = (volume - mean_v) / std_v
        ret_z = (log_ret - mean_r) / std_r
        feats.append([log_ret, hl_range, body, volume, n_trades, vol_z, ret_z])
    return feats


class MarketAnalyser(nn.Module):
    """Small CPU encoder: WINDOW x RAW_FEATURES -> EMBED_DIM state vector."""

    def __init__(self, window=WINDOW, raw_features=RAW_FEATURES, embed_dim=EMBED_DIM):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(
            nn.Conv1d(raw_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, embed_dim)
        # auxiliary self-supervised head: predict next-second return from embedding
        self.aux_head = nn.Linear(embed_dim, 1)
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        # x: (batch, window, raw_features) -> (batch, raw_features, window)
        x = x.transpose(1, 2)
        z = self.net(x).squeeze(-1)
        state = self.head(z)
        return state

    def predict_next_return(self, state):
        return self.aux_head(state)


class RollingWindow:
    """Maintains the last `window` raw ticks for feature extraction."""

    def __init__(self, window=WINDOW):
        self.window = window
        self.buf = collections.deque(maxlen=window)

    def push(self, tick_dict):
        self.buf.append(tick_dict)

    def ready(self):
        return len(self.buf) == self.window

    def features_tensor(self):
        feats = extract_raw_features(list(self.buf))
        return torch.tensor([feats], dtype=torch.float32)  # (1, window, raw_features)
