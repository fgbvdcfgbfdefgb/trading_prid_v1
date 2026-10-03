"""
Price Predictor -- runs on GPU, distributed-training-ready (torch.distributed
DistributedDataParallel). Consumes a sequence of MarketAnalyser embeddings
(one per second, up to a 30-minute = 1800-step lookback window) and predicts
the BTC/USD price 30 minutes (1800 seconds) ahead.

Architecture: a small Transformer encoder over the embedding sequence + a
regression head predicting the log-return to t+1800s (log-return, not raw
price, keeps the target scale well-behaved across price regimes).

Stability measures (since this feeds a live trading signal):
  - predicts log-return rather than raw price (bounded, stationary-ish)
  - output passed through tanh-scaled clamp to bound max predicted move
  - EMA smoothing of the prediction stream is applied *outside* this module,
    in training/online_trainer.py, so raw model jitter doesn't reach the user
  - gradient clipping + small learning rate + cosine decay in the trainer
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PricePredictor(nn.Module):
    def __init__(self, embed_dim=16, d_model=64, nhead=4, num_layers=3,
                 max_seq_len=1800, max_move=0.05):
        """max_move: hard cap on |predicted log-return| over the 30-min horizon
        (5% default) -- a crude but effective stability guardrail so a bad
        forward pass can't emit an absurd signal."""
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                            dim_feedforward=d_model * 4,
                                            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.max_move = max_move

    def forward(self, embed_seq, src_key_padding_mask=None):
        # embed_seq: (batch, seq_len, embed_dim)
        x = self.input_proj(embed_seq)
        x = self.pos_enc(x)
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        pooled = h[:, -1, :]  # last-token (causal summary) representation
        raw_logret = self.head(pooled).squeeze(-1)
        bounded_logret = self.max_move * torch.tanh(raw_logret)
        return bounded_logret  # predicted log-return from now to t+1800s


def build_model_and_optimizer(device, lr=1e-4, distributed=False, local_rank=0):
    model = PricePredictor().to(device)
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=200, T_mult=2)
    return model, optim, sched
