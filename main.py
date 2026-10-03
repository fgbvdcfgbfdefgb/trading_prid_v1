"""
Cerebrium entrypoint (must live at the repo root, next to cerebrium.toml,
because Cerebrium's default Cortex runtime packages the directory containing
cerebrium.toml and exposes top-level functions from main.py as endpoints).

Local usage before touching Cerebrium at all:
    python3 training/train_price_predictor_ddp.py --epochs 1 --batch-size 16

Cerebrium usage (after `pip install cerebrium`, `cerebrium login` or saving a
service-account token, from this repo root):
    cerebrium deploy

Then call the deployed endpoint (Cerebrium gives you the URL + an inference
API key after deploy) with a JSON body like:
    {"epochs": 10, "batch_size": 128}
"""
import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def train(epochs: int = 5, batch_size: int = 64, lr: float = 0.0001, nproc_per_node: int = 1):
    """Launches training/train_price_predictor_ddp.py via torchrun on the GPU(s)
    this Cerebrium instance was granted. For a free-trial single-GPU plan,
    leave nproc_per_node=1 (it still uses the GPU, just not multi-process DDP)."""
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}",
        os.path.join(REPO_ROOT, "training", "train_price_predictor_ddp.py"),
        "--epochs", str(epochs), "--batch-size", str(batch_size), "--lr", str(lr),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def health():
    """Simple smoke-test endpoint: confirms the container has torch + CUDA
    visibility and that the repo's resampled data is present."""
    import torch
    data_dir = os.path.join(REPO_ROOT, "data", "ohlc_1m")
    has_1m = os.path.exists(os.path.join(data_dir, "btcusd_1m.parquet"))
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "has_1m_bars": has_1m,
    }


if __name__ == "__main__":
    # local smoke test
    print(health())
