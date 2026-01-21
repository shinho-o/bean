# app.py (simple)
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Bean Moisture API (Simple)")

# in-memory DB: sensor_id -> list of records
DB: Dict[str, List[Dict[str, Any]]] = {}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure(sensor_id: str):
    if sensor_id not in DB:
        DB[sensor_id] = []


def add(sensor_id: str, rec: Dict[str, Any]):
    ensure(sensor_id)
    DB[sensor_id].append(rec)
    DB[sensor_id].sort(key=lambda x: x["ts"])


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


# -------------------
# Models (Request)
# -------------------
class LogRequest(BaseModel):
    sensor_id: str = Field(..., examples=["sensor_01"])
    moisture_percent: float = Field(..., ge=0, le=100)
    raw: Optional[float] = None
    ts: Optional[str] = None  # ISO8601, 없으면 서버시간


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/log")
def log_data(req: LogRequest):
    ts = now_utc() if req.ts is None else datetime.fromisoformat(req.ts.replace("Z", "+00:00"))
    rec = {
        "ts": ts,
        "moisture_percent": float(req.moisture_percent),
        "raw": None if req.raw is None else float(req.raw),
    }
    add(req.sensor_id, rec)
    return {"ok": True, "sensor_id": req.sensor_id, "saved": {**rec, "ts": iso(ts)}, "count": len(DB[req.sensor_id])}


@app.get("/latest")
def latest(sensor_id: str = "sensor_01"):
    """
    앱에서 GET으로 바로 호출 가능:
    /latest?sensor_id=sensor_01
    """
    ensure(sensor_id)
    if len(DB[sensor_id]) == 0:
        return {"ok": False, "error": "no_data"}

    last = DB[sensor_id][-1]

    # 더미 예측(아무거나 뜨게): 수분값으로 간단 규칙만
    m = float(last["moisture_percent"])
    if m >= 75:
        dummy = {"treat_pred": "overhydration", "stage_pred": "V1", "note": "dummy_rule(m>=75)"}
    elif m <= 25:
        dummy = {"treat_pred": "drought", "stage_pred": "V1", "note": "dummy_rule(m<=25)"}
    else:
        dummy = {"treat_pred": "normal", "stage_pred": "V1", "note": "dummy_rule(else)"}

    return {
        "ok": True,
        "sensor_id": sensor_id,
        "latest": {"ts": iso(last["ts"]), "moisture_percent": m, "raw": last["raw"]},
        "dummy_pred": dummy,
    }


@app.get("/history")
def history(sensor_id: str = "sensor_01", n: int = 200):
    """
    그래프용:
    /history?sensor_id=sensor_01&n=300
    """
    ensure(sensor_id)
    data = DB[sensor_id][-max(1, n):]
    out = [{"ts": iso(r["ts"]), "moisture_percent": r["moisture_percent"], "raw": r["raw"]} for r in data]
    return {"ok": True, "sensor_id": sensor_id, "n": len(out), "data": out}


@app.get("/seed")
def seed(sensor_id: str = "sensor_01", days: int = 7, step_minutes: int = 10):
    """
    시연용 더미 데이터 채우기(POST 필요 없음):
    /seed?sensor_id=sensor_01&days=7&step_minutes=10
    """
    ensure(sensor_id)
    end = now_utc()
    start = end - timedelta(days=max(1, days))
    step = timedelta(minutes=max(1, step_minutes))

    rng = np.random.default_rng(42)
    t = start
    count = 0
    while t <= end:
        # 0~100 사이 랜덤 + 약간의 일변화
        day_frac = (t - start).total_seconds() / 86400.0
        wave = 10.0 * np.sin(2 * np.pi * day_frac)
        val = 55.0 + wave + rng.normal(0, 6.0)
        val = float(np.clip(val, 0, 100))
        add(sensor_id, {"ts": t, "moisture_percent": val, "raw": None})
        count += 1
        t += step

    return {"ok": True, "sensor_id": sensor_id, "seeded": count}

