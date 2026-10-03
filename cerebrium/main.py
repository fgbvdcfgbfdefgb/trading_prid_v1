"""
Cerebrium entrypoint. Cerebrium wraps whatever top-level functions you define
here as callable HTTP endpoints once deployed (`cerebrium deploy`). This
exposes one endpoint to kick off a (single-GPU on free trial, or
multi-GPU/DDP if your plan grants more) training run of the price predictor
on whatever data has been pushed to data/ in this repo.

Local usage before touching Cerebrium at all:
    python3 training/train_price_predictor_ddp.py --epochs 1 --batch-size 16

Cerebrium usage (after `pip install cerebrium` and `cerebrium login`, from the
repo root):
    cerebrium deploy cerebrium/cerebrium.toml

Then call the deployed endpoint (Cerebrium gives you the URL + an inference
API key after deploy) with a JSON body like:
    {"epochs": 10, "batch_size": 128}
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(epochs: int = 5, batch_size: int = 64, lr: float = 1e-4, nproc_per_node: int = 1):
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


if __name__ == "__main__":
    # local smoke test
    print(train(epochs=1, batch_size=16, nproc_per_node=1))
