"""
Live dashboard: runs the OnlineTrainer in a background thread and streams its
state to the browser every second over a WebSocket. The page shows:
  - a live line chart: real market price vs AI (smoothed) prediction
  - "sliders" panel that visualizes training in real time: epoch progress,
    loss (EMA), reward (EMA), replay-buffer fill, and two LIVE, draggable
    sliders (learning rate, EMA smoothing factor) that actually feed back
    into the running trainer.

Run:
    python3 dashboard/server.py
Then open the printed URL (binds 0.0.0.0:8000).
"""
import asyncio
import os
import sys
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.online_trainer import OnlineTrainer

app = FastAPI()
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

trainer = OnlineTrainer(speed=1.0)   # speed=1.0 -> true real-time, 1 tick/sec
_trainer_thread = threading.Thread(target=trainer.run, daemon=True)
_trainer_thread.start()


class ParamUpdate(BaseModel):
    name: str
    value: float


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/state")
def get_state():
    return trainer.get_state()


@app.post("/api/param")
def set_param(p: ParamUpdate):
    if p.name == "lr":
        trainer.set_lr(p.value)
    elif p.name == "ema_alpha":
        trainer.set_ema_alpha(p.value)
    return {"ok": True, "name": p.name, "value": p.value}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(trainer.get_state())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
