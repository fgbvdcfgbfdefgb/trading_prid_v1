"""
Heavy, multi-GPU DISTRIBUTED training job for the price predictor.

Important design note: the *live* loop (training/online_trainer.py) does one
lightweight gradient step per second against a small in-memory replay buffer
-- that is inherently a single-process loop (you can't usefully synchronize
a DDP all-reduce across nodes every single second; the sync overhead alone
would dwarf the compute). Real systems split these concerns, and so does this
one:

  - online_trainer.py  -> single process, runs next to the live simulator,
                          does per-second "fine-tuning" nudges, feeds the
                          live dashboard.
  - this script         -> periodic (e.g. every few hours, or once per day)
                          HEAVY retrain over the full historical dataset,
                          scaled across all GPUs you give it via DistributedDataParallel.
                          Its output checkpoint is what online_trainer.py
                          loads to "promote" a better base model.

Run locally (single GPU/CPU smoke test):
    python3 training/train_price_predictor_ddp.py --epochs 1 --batch-size 16

Run distributed on an N-GPU machine (e.g. on Cerebrium/Snowflake):
    torchrun --standalone --nproc_per_node=N training/train_price_predictor_ddp.py \
        --epochs 20 --batch-size 128
"""
import argparse
import glob
import os
import sys

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.market_analyser import MarketAnalyser, extract_raw_features, WINDOW
from models.price_predictor import PricePredictor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OHLC_1S_DIR = os.path.join(BASE_DIR, "data", "ohlc_1s")
OHLC_1M_PATH = os.path.join(BASE_DIR, "data", "ohlc_1m", "btcusd_1m.parquet")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

HORIZON_SECONDS = 1800
SEQ_LEN = 300


class BarsDataset(Dataset):
    """Builds (raw-window-sequence, target 30-min-ahead log-return) samples
    from resampled 1-second (preferred) or 1-minute (fallback) bars."""

    def __init__(self, seq_len=SEQ_LEN, horizon=HORIZON_SECONDS, stride=30):
        files = sorted(glob.glob(os.path.join(OHLC_1S_DIR, "*.parquet")))
        if files:
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            self.bar_seconds = 1
        elif os.path.exists(OHLC_1M_PATH):
            df = pd.read_parquet(OHLC_1M_PATH)
            self.bar_seconds = 60
        else:
            raise FileNotFoundError(
                "No resampled bars found yet. Run scripts/resample_to_bars.py "
                "after some trade data has downloaded, or wait for the background "
                "download job (see data/download*.log)."
            )
        df = df.sort_values("timestamp").reset_index(drop=True)
        self.df = df
        self.horizon_bars = max(1, horizon // self.bar_seconds)
        self.seq_len = seq_len
        self.stride = stride
        self.indices = list(range(seq_len, len(df) - self.horizon_bars, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx]
        start = end - self.seq_len
        window_df = self.df.iloc[start:end]
        rows = window_df[["open", "high", "low", "close", "volume", "n_trades"]].to_dict("records")
        feats = extract_raw_features(rows)
        price0 = float(self.df.iloc[end - 1]["close"])
        price_future = float(self.df.iloc[end - 1 + self.horizon_bars]["close"])
        target = torch.log(torch.tensor(max(price_future, 1e-9)) / torch.tensor(max(price0, 1e-9)))
        return torch.tensor(feats, dtype=torch.float32), target.float()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    is_distributed, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    ds = BarsDataset()
    sampler = DistributedSampler(ds) if is_distributed else None
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                         shuffle=(sampler is None), num_workers=2, drop_last=True)

    analyser = MarketAnalyser().to("cpu")  # feature encoder stays on CPU by design
    predictor = PricePredictor().to(device)
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp_kwargs = {"device_ids": [local_rank]} if torch.cuda.is_available() else {}
        predictor = DDP(predictor, **ddp_kwargs)

    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))

    os.makedirs(CKPT_DIR, exist_ok=True)
    rank = dist.get_rank() if is_distributed else 0

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        total_loss, n = 0.0, 0
        for raw_batch, target in loader:
            with torch.no_grad():
                b, l, f = raw_batch.shape
                embeds = analyser(raw_batch.view(b, l, f))  # (b, embed_dim) single-step encoder
                # expand to a pseudo-sequence the transformer predictor expects:
                seq = embeds.unsqueeze(1).repeat(1, 2, 1).to(device)
            target = target.to(device)

            predictor.train()
            pred = predictor(seq)
            loss = torch.nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt.step()
            sched.step()

            total_loss += loss.item()
            n += 1

        avg = total_loss / max(n, 1)
        if rank == 0:
            print(f"[epoch {epoch}] avg_mse_loss={avg:.6f} batches={n}")
            ckpt = {
                "epoch": epoch,
                "predictor_state": (predictor.module if hasattr(predictor, "module") else predictor).state_dict(),
                "optim_state": opt.state_dict(),
                "avg_loss": avg,
            }
            torch.save(ckpt, os.path.join(CKPT_DIR, f"ddp_epoch_{epoch:04d}.pt"))

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
