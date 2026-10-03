# trading_prid_v1 — BTC/USD live market simulator + market analyser + price predictor

⚠️ **Disclaimer**: this is a research/engineering prototype, not a financial product.
No model here can *guarantee* stable or profitable trading predictions. The
"stability" measures below (EMA smoothing, clamping, gradient clipping) reduce
jitter in the output signal — they do not make the underlying forecast
accurate or safe to trade on. Use at your own risk, paper-trade first, and
don't point real capital at this without a lot more validation than a single
prototype build.

🔑 **About the GitHub token used to set this repo up**: it was pasted in plain
text in chat. Please rotate/revoke it (Settings → Developer settings →
Personal access tokens) once you've copied this repo to your own account,
even though you called it a test account — tokens shared in plaintext should
always be treated as burned.

## What's actually in this repo

| Path | What it is |
|---|---|
| `scripts/download_trades.py` | Resumable Kraken `/Trades` downloader (tick-level). Kraken's OHLC REST endpoint only keeps ~720 recent candles, so true second-level history has to be built from raw trades. Two jobs run from this one script: a **recent** job (last few days, catches up fast, then polls live) and a **backfill** job (from pair inception, 2013, runs for hours/days in the background). |
| `scripts/resample_to_bars.py` | Converts raw trade chunks into regular 1-second and 1-minute OHLCV bars (`data/ohlc_1s/*.parquet`, `data/ohlc_1m/btcusd_1m.parquet`). |
| `scripts/auto_push.sh` | Loop: resample → `git add/commit/push`, every N seconds, so the backfill's progress is continuously saved to GitHub instead of being lost if the job is interrupted. |
| `simulator/market_simulator.py` | Replays real 1-second bars one-per-wall-clock-second (`speed=1.0`), or as fast as possible for training (`speed=0`). Falls back to a clearly-labeled **synthetic** random-walk feed if real data hasn't landed yet. |
| `models/market_analyser.py` | **CPU** model. Turns a rolling 60-second raw window into a 16-dim "market state" embedding (volatility, momentum, volume-imbalance features through a small 1D-conv net), trained with a cheap self-supervised next-return objective. |
| `models/price_predictor.py` | **GPU** model, DDP-ready. Transformer encoder over a sequence of market-analyser embeddings → predicts the **log-return 30 minutes ahead**. Output is bounded (`tanh`-clamped) as a crude safety rail. |
| `training/online_trainer.py` | The live loop: simulator tick → analyser → predictor → smoothed prediction → log every single second → resolve 30-min-old predictions into (state, actual outcome) pairs → light online gradient step → checkpoint every 1800 ticks (**1 epoch = 30 simulated minutes**). |
| `training/train_price_predictor_ddp.py` | The **heavy**, periodic, multi-GPU **distributed** (`torch.distributed` / `DistributedDataParallel`, launched via `torchrun`) retrain job over the full historical dataset. See "Why two training scripts?" below. |
| `dashboard/` | FastAPI + WebSocket live dashboard: market price vs AI prediction chart (updates every second) + training "sliders" (epoch progress, loss, reward, replay buffer, and two **live, draggable** controls — learning rate and prediction-smoothing α — that actually feed back into the running trainer). |
| `cerebrium/` | Deployment config + entrypoint for running the heavy DDP job on Cerebrium's GPUs. |

## Why two training scripts? (online vs. distributed)

You asked for both "update every sec" *and* "distributed GPU training" — those
are in tension: you cannot usefully do a multi-GPU/multi-node DDP
all-reduce every single second; the synchronization overhead alone would
dwarf the compute, and real systems don't do this. So the design splits it
exactly how production trading-signal systems typically do:

- **`training/online_trainer.py`** — single process, runs continuously next
  to the live simulator. Every second it: makes a prediction, logs it, and
  (once enough 30-minute-old predictions have resolved against real
  outcomes) takes one small gradient step. This is what drives the live
  dashboard.
- **`training/train_price_predictor_ddp.py`** — a heavier job you run
  periodically (hourly/daily) on a GPU box (or several, via
  `torchrun --nproc_per_node=N`), scanning the *entire* historical dataset
  to retrain/refresh the base model. Its checkpoint is what you'd load into
  the online trainer to "promote" an improved model.

## Running it yourself

```bash
git clone https://github.com/fgbvdcfgbfdefgb/trading_prid_v1.git
cd trading_prid_v1
pip install -r requirements.txt

# 1. (optional) keep pulling data — these two already ran for a while in the
#    sandbox and pushed whatever they'd gathered to this repo under data/
python3 scripts/download_trades.py --checkpoint data/raw_trades/checkpoint_recent.json \
    --chunk-prefix trades_recent_ --since-days-ago 3 --poll-when-live 5 &
python3 scripts/download_trades.py --checkpoint data/raw_trades/checkpoint_backfill.json \
    --chunk-prefix trades_hist_ &

# 2. fold raw trades into 1s/1m bars
python3 scripts/resample_to_bars.py

# 3. live dashboard (market simulator + CPU analyser + GPU predictor + training, all in one)
python3 dashboard/server.py
# open http://localhost:8000 (or the sandbox's forwarded preview URL)

# 4. heavy distributed retrain (single box smoke test)
python3 training/train_price_predictor_ddp.py --epochs 1 --batch-size 16
# multi-GPU:
torchrun --standalone --nproc_per_node=N training/train_price_predictor_ddp.py --epochs 20
```

## Running the heavy training job on Cerebrium

I don't have a way to generate a personalized Cerebrium login/OAuth link for
your account from here (no Cerebrium integration on my end) — just use the
normal sign-in:

**https://dashboard.cerebrium.ai/login**

Once you're in, from the repo root:

```bash
pip install cerebrium
cerebrium login
cerebrium deploy cerebrium/cerebrium.toml
```

`cerebrium/cerebrium.toml` requests 1 GPU (typical for a free trial) and
`cerebrium/main.py` exposes a `train(epochs, batch_size, lr, nproc_per_node)`
endpoint that launches `training/train_price_predictor_ddp.py` via `torchrun`
on whatever GPU(s) your plan grants. Leave `nproc_per_node=1` unless your plan
actually gives you more than one GPU — `torchrun` will hang waiting for ranks
that never start otherwise.

## Data notes / "sec-sec" caveat

True second-resolution data for the *entire* history of BTC/USD only exists
if you build it yourself from raw trades (which is what this does) — no
exchange hands out a complete 2013-to-today 1-second candle file. That backfill
is tens of millions of trades and will take many hours of continuous,
politely-rate-limited downloading to complete; it's running as a resumable
background job (`scripts/download_trades.py --chunk-prefix trades_hist_`) that
you can stop/restart anytime without losing progress (see `data/raw_trades/checkpoint_backfill.json`).
The **recent** job (last few days) catches up in minutes and is what actually
drives the live simulator/dashboard meaningfully until the backfill matures.

## Stability mitigations (not guarantees)

- Predictor outputs a **log-return**, not a raw price (keeps scale sane across regimes)
- Output hard-clamped to ±5% over the 30-min horizon (`max_move` in `price_predictor.py`)
- Displayed prediction is an **EMA** of the raw model output (`EMA_ALPHA`, live-adjustable from the dashboard)
- Per-tick change is clamped (`MAX_STEP_CHANGE`) so one bad forward pass can't spike the signal
- Gradient clipping (`clip_grad_norm_`) on every online update
- Every single per-second prediction is logged (`training/logs/prediction_log.parquet`) so you can audit drift/divergence after the fact, and a checkpoint is saved every simulated 30-minute epoch (`checkpoints/epoch_*.pt`)

None of this makes the forecast *correct* — it only makes the signal well-behaved enough to reason about.
