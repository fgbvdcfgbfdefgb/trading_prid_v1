"""
Online trainer -- the live loop that ties everything together:

  simulator (1 tick/sec) -> market_analyser (CPU) -> embedding
                                  -> price_predictor (GPU if available)
                                  -> prediction for t+1800s, EMA-smoothed
                                  -> logged every second (prediction_log.parquet)
                                  -> when ground truth for an old prediction
                                     arrives (1800s later), compute reward/loss
                                     and take a light online gradient step
                                  -> every 1800 ticks (= 1 "epoch" = 30 sim-min)
                                     save a checkpoint

Design notes on "stable predictions for trading":
  - predictor outputs a bounded log-return (see models/price_predictor.py)
  - the *displayed/used* prediction is an EMA of the raw model output, with a
    configurable alpha (lower alpha = smoother, laggier)
  - a max-step-change clamp further prevents single-tick spikes
  - these are engineering mitigations, not a guarantee -- no model can promise
    stability/profitability on live markets; treat this as a research signal.

This module is import-able (used by dashboard/server.py to run the loop in a
background thread) and runnable standalone: `python3 training/online_trainer.py`
"""
import collections
import math
import os
import threading
import time

import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.market_analyser import MarketAnalyser, RollingWindow, WINDOW
from models.price_predictor import PricePredictor
from simulator.market_simulator import MarketSimulator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "training", "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

HORIZON_SECONDS = 1800     # predict 30 min ahead
EPOCH_SECONDS = 1800       # 1 epoch == 30 sim-minutes
SEQ_LEN = 300              # seconds of embedding context fed to the predictor
                           # (kept < HORIZON for compute reasons; see README)
EMA_ALPHA = 0.05           # smoothing factor for displayed prediction (stability)
MAX_STEP_CHANGE = 0.0015   # max allowed change in smoothed logret per tick
REPLAY_CAPACITY = 20000
BATCH_SIZE = 32
MIN_REPLAY_FOR_TRAINING = 64


class OnlineTrainer:
    def __init__(self, speed=1.0, device=None, distributed=False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim = MarketSimulator(speed=speed)
        self.analyser = MarketAnalyser()  # always CPU by design (see module docstring)
        self.predictor = PricePredictor().to(self.device)
        if distributed and torch.cuda.device_count() > 1:
            self.predictor = torch.nn.DataParallel(self.predictor)  # simple multi-GPU path;
            # for true multi-node DDP use training/train_price_predictor_ddp.py instead.

        self.opt_pred = torch.optim.AdamW(self.predictor.parameters(), lr=1e-4, weight_decay=1e-5)
        self.opt_an = torch.optim.AdamW(self.analyser.parameters(), lr=3e-4)

        self.window = RollingWindow(window=WINDOW)
        self.embed_seq = collections.deque(maxlen=SEQ_LEN)
        self.pending = collections.deque()   # (t, price_t, embed_seq_snapshot)
        self.replay = collections.deque(maxlen=REPLAY_CAPACITY)

        self.tick_count = 0
        self.epoch = 0
        self.smoothed_logret = 0.0
        self.loss_ema = None
        self.reward_ema = None

        self.lock = threading.Lock()
        self.state = {
            "t": None, "market_price": None, "raw_pred_price": None,
            "smoothed_pred_price": None, "raw_logret": 0.0, "smoothed_logret": 0.0,
            "epoch": 0, "epoch_progress": 0.0, "loss_ema": None, "reward_ema": None,
            "lr": self.opt_pred.param_groups[0]["lr"], "buffer_size": 0,
            "mode": self.sim.mode, "device": str(self.device),
        }
        self._stop = False
        self._price_history = collections.deque(maxlen=5000)
        self._pred_history = collections.deque(maxlen=5000)

        self._log_buffer = []
        self._log_path = os.path.join(LOG_DIR, "prediction_log.parquet")

    # ---- external controls (used by the dashboard sliders) ----
    def set_lr(self, lr: float):
        for g in self.opt_pred.param_groups:
            g["lr"] = lr

    def set_ema_alpha(self, alpha: float):
        global EMA_ALPHA
        EMA_ALPHA = max(0.001, min(1.0, alpha))

    def get_state(self):
        with self.lock:
            return dict(self.state)

    def get_history(self):
        with self.lock:
            return list(self._price_history), list(self._pred_history)

    def stop(self):
        self._stop = True

    # ---- core loop ----
    def _checkpoint(self):
        path = os.path.join(CKPT_DIR, f"epoch_{self.epoch:05d}.pt")
        torch.save({
            "epoch": self.epoch,
            "tick_count": self.tick_count,
            "predictor_state": (self.predictor.module if hasattr(self.predictor, "module") else self.predictor).state_dict(),
            "analyser_state": self.analyser.state_dict(),
            "opt_pred": self.opt_pred.state_dict(),
            "opt_an": self.opt_an.state_dict(),
            "loss_ema": self.loss_ema,
            "reward_ema": self.reward_ema,
        }, path)
        return path

    def _flush_log(self, force=False):
        if not self._log_buffer or (len(self._log_buffer) < 300 and not force):
            return
        import pandas as pd
        df = pd.DataFrame(self._log_buffer)
        if os.path.exists(self._log_path):
            old = pd.read_parquet(self._log_path)
            df = pd.concat([old, df], ignore_index=True)
        df.to_parquet(self._log_path, index=False)
        self._log_buffer = []

    def run(self):
        for tick in self.sim.stream():
            if self._stop:
                break
            self.tick_count += 1
            raw = {"open": tick.open, "high": tick.high, "low": tick.low,
                   "close": tick.close, "volume": tick.volume, "n_trades": tick.n_trades}
            self.window.push(raw)

            raw_pred_price = None
            smoothed_pred_price = None
            raw_logret = 0.0

            if self.window.ready():
                feats = self.window.features_tensor()  # (1, WINDOW, RAW_FEATURES)
                with torch.no_grad():
                    state_embed = self.analyser(feats).squeeze(0)  # (EMBED_DIM,)
                self.embed_seq.append(state_embed)

                if len(self.embed_seq) >= 2:
                    seq = torch.stack(list(self.embed_seq)).unsqueeze(0)  # (1, L, D)
                    with torch.no_grad():
                        pred_logret = self.predictor(seq.to(self.device)).item()
                    raw_logret = pred_logret

                    # --- stability: EMA smoothing + max-step clamp ---
                    delta = pred_logret - self.smoothed_logret
                    delta = max(-MAX_STEP_CHANGE, min(MAX_STEP_CHANGE, delta))
                    self.smoothed_logret = (1 - EMA_ALPHA) * self.smoothed_logret + EMA_ALPHA * (self.smoothed_logret + delta)

                    raw_pred_price = tick.close * math.exp(raw_logret)
                    smoothed_pred_price = tick.close * math.exp(self.smoothed_logret)

                    self.pending.append((tick.t, tick.close, torch.stack(list(self.embed_seq)).clone()))

            # resolve matured predictions -> replay buffer
            while self.pending and (tick.t - self.pending[0][0]) >= HORIZON_SECONDS:
                t0, price0, seq0 = self.pending.popleft()
                actual_logret = math.log(max(tick.close, 1e-9) / max(price0, 1e-9))
                self.replay.append((seq0, actual_logret))

            # light online gradient step (every tick, if we have resolved data)
            if len(self.replay) >= MIN_REPLAY_FOR_TRAINING:
                self._train_step()

            # periodic self-supervised refresh of the analyser (cheap, CPU)
            if self.window.ready() and self.tick_count % 10 == 0:
                self._train_analyser_aux()

            # epoch boundary: every EPOCH_SECONDS simulated ticks
            epoch_progress = (self.tick_count % EPOCH_SECONDS) / EPOCH_SECONDS
            if self.tick_count > 0 and self.tick_count % EPOCH_SECONDS == 0:
                self.epoch += 1
                ckpt_path = self._checkpoint()
                print(f"[epoch {self.epoch}] checkpoint saved -> {ckpt_path}")

            # log every single second's prediction (for reward/loss auditing)
            self._log_buffer.append({
                "t": tick.t, "market_price": tick.close,
                "raw_pred_price": raw_pred_price, "smoothed_pred_price": smoothed_pred_price,
                "raw_logret": raw_logret, "smoothed_logret": self.smoothed_logret,
                "epoch": self.epoch, "tick": self.tick_count,
            })
            self._flush_log()

            with self.lock:
                self.state.update({
                    "t": tick.t, "market_price": tick.close,
                    "raw_pred_price": raw_pred_price, "smoothed_pred_price": smoothed_pred_price,
                    "raw_logret": raw_logret, "smoothed_logret": self.smoothed_logret,
                    "epoch": self.epoch, "epoch_progress": epoch_progress,
                    "loss_ema": self.loss_ema, "reward_ema": self.reward_ema,
                    "lr": self.opt_pred.param_groups[0]["lr"], "buffer_size": len(self.replay),
                    "mode": self.sim.mode, "device": str(self.device),
                })
                self._price_history.append((tick.t, tick.close))
                if smoothed_pred_price is not None:
                    self._pred_history.append((tick.t, smoothed_pred_price))

    def _train_step(self):
        import random
        batch = random.sample(list(self.replay), min(BATCH_SIZE, len(self.replay)))
        seqs = [b[0] for b in batch]
        targets = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=self.device)
        max_len = max(s.shape[0] for s in seqs)
        padded = torch.zeros(len(seqs), max_len, seqs[0].shape[1])
        mask = torch.ones(len(seqs), max_len, dtype=torch.bool)
        for i, s in enumerate(seqs):
            padded[i, : s.shape[0]] = s
            mask[i, : s.shape[0]] = False
        padded = padded.to(self.device)
        mask = mask.to(self.device)

        self.predictor.train()
        pred = self.predictor(padded, src_key_padding_mask=mask)
        loss = torch.nn.functional.mse_loss(pred, targets)
        reward = -loss.item()

        self.opt_pred.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        self.opt_pred.step()
        self.predictor.eval()

        l = loss.item()
        self.loss_ema = l if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * l
        self.reward_ema = reward if self.reward_ema is None else 0.98 * self.reward_ema + 0.02 * reward

    def _train_analyser_aux(self):
        feats = self.window.features_tensor()
        state = self.analyser(feats)
        pred_next_ret = self.analyser.predict_next_return(state).squeeze()
        target = torch.tensor(feats[0, -1, 0].item() if feats.shape[1] else 0.0)
        loss = torch.nn.functional.mse_loss(pred_next_ret, target)
        self.opt_an.zero_grad()
        loss.backward()
        self.opt_an.step()


if __name__ == "__main__":
    trainer = OnlineTrainer(speed=0.0)  # speed=0 -> run as fast as possible for a smoke test
    t0 = time.time()
    th = threading.Thread(target=trainer.run, daemon=True)
    th.start()
    time.sleep(15)
    trainer.stop()
    print("state:", trainer.get_state())
