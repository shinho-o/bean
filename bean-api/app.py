# app.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# =========================
# CONFIG
# =========================
FEATURES_NUM = [
    "days_since_stage_start",
    "soil_mean_7d",
    "soil_min_7d",
    "soil_std_7d",
    "soil_trend_7d",
    "soil_missing",
    "soil_stress_intensity",
    "days_x_soil_mean",
]
CAT_COLS = ["stage", "treatment", "zone"]

STAGE_CATS = ["V1", "V5", "R1", "R3", "R5"]
TREAT_CATS = ["과습", "한발"]
DEFAULT_ZONE = "0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

PKL_LEN   = os.path.join(MODEL_DIR, "len_reg.pkl")
PKL_DIA   = os.path.join(MODEL_DIR, "dia_reg.pkl")
PKL_TREAT = os.path.join(MODEL_DIR, "treat_model.pkl")
PKL_STAGE = os.path.join(MODEL_DIR, "stage_model.pkl")


# =========================
# Load Models
# =========================
def load_model(pkl_path: str):
    obj = joblib.load(pkl_path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj

LEN_REG = load_model(PKL_LEN)
DIA_REG = load_model(PKL_DIA)
TREAT_MODEL = load_model(PKL_TREAT)
STAGE_MODEL = load_model(PKL_STAGE)


# =========================
# In-memory storage (과제용 빠른 방식)
# 레코드: {sensor_id, ts, moisture_percent, raw, stage, treatment, zone}
# =========================
DB: Dict[str, List[Dict[str, Any]]] = {}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_sensor(sensor_id: str):
    if sensor_id not in DB:
        DB[sensor_id] = []


def add_record(sensor_id: str, rec: Dict[str, Any]):
    ensure_sensor(sensor_id)
    DB[sensor_id].append(rec)
    DB[sensor_id].sort(key=lambda x: x["ts"])  # ts 기준 정렬


def get_history(sensor_id: str, since: datetime) -> List[Dict[str, Any]]:
    ensure_sensor(sensor_id)
    return [r for r in DB[sensor_id] if r["ts"] >= since]


def compute_7d_features(sensor_id: str) -> Dict[str, float]:
    """
    최근 7일치 moisture_percent 기반 피처 생성.
    - soil_mean_7d/min/std
    - soil_trend_7d: 시간에 대한 선형회귀 기울기
    - soil_missing: 기대 샘플 대비 누락 비율 (10분 간격 가정)
    - soil_stress_intensity: (임계치 기준) 스트레스 강도(간단 버전)
    - days_x_soil_mean: days_since_stage_start * soil_mean_7d
    """
    end = now_utc()
    start = end - timedelta(days=7)
    rows = get_history(sensor_id, start)

    if len(rows) == 0:
        # 데이터 없으면 전부 0으로
        return {
            "soil_mean_7d": 0.0,
            "soil_min_7d": 0.0,
            "soil_std_7d": 0.0,
            "soil_trend_7d": 0.0,
            "soil_missing": 1.0,
            "soil_stress_intensity": 0.0,
        }

    y = np.array([float(r["moisture_percent"]) for r in rows], dtype=float)

    mean_ = float(np.mean(y))
    min_ = float(np.min(y))
    std_ = float(np.std(y))

    # trend: x=시간(초)로 회귀
    t0 = rows[0]["ts"]
    x = np.array([(r["ts"] - t0).total_seconds() for r in rows], dtype=float)
    if len(rows) >= 2 and np.std(x) > 0:
        slope = float(np.polyfit(x, y, 1)[0])  # % per sec
        slope *= 86400  # % per day 로 보기 좋게 변환
    else:
        slope = 0.0

    # missing: 10분마다라면 7일 기대 샘플 수
    expected = int((7 * 24 * 60) / 10)  # 1008
    missing_ratio = float(max(0, expected - len(rows)) / expected)

    # stress_intensity (예: 너무 건조/과습 구간을 벌점으로)
    # 발표용 간단 버전: 30% 아래면 건조 스트레스, 80% 이상이면 과습 스트레스
    dry = np.clip(30 - y, 0, None)   # 0~30
    wet = np.clip(y - 80, 0, None)   # 0~20
    stress_intensity = float(np.mean(dry + wet) / 50.0)  # 0~1 근사

    return {
        "soil_mean_7d": mean_,
        "soil_min_7d": min_,
        "soil_std_7d": std_,
        "soil_trend_7d": slope,
        "soil_missing": missing_ratio,
        "soil_stress_intensity": stress_intensity,
    }


def align_input(row: dict) -> pd.DataFrame:
    for c in FEATURES_NUM:
        row.setdefault(c, 0.0)
    row.setdefault("stage", None)
    row.setdefault("treatment", None)
    row.setdefault("zone", DEFAULT_ZONE)

    X = pd.DataFrame([{c: row.get(c) for c in (FEATURES_NUM + CAT_COLS)}])

    for c in FEATURES_NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    # categorical 고정
    X["stage"] = pd.Categorical(X["stage"], categories=STAGE_CATS)
    X["treatment"] = pd.Categorical(X["treatment"], categories=TREAT_CATS)
    X["zone"] = X["zone"].astype("string").fillna(DEFAULT_ZONE).astype("category")
    return X


def predict_one_classifier(model, X: pd.DataFrame):
    out = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        out["proba"] = {str(k): float(v) for k, v in zip(model.classes_, proba)}
    out["pred"] = str(model.predict(X)[0])
    return out


def predict_one_regressor(model, X: pd.DataFrame):
    return float(model.predict(X)[0])


# =========================
# API
# =========================
app = FastAPI(title="Bean IoT Predict API")


class LogRequest(BaseModel):
    sensor_id: str = Field(..., examples=["sensor_01"])
    moisture_percent: float = Field(..., ge=0, le=100)
    raw: Optional[float] = None

    # 선택: ESP에서 같이 보내도 되고, 서버가 유지해도 됨
    stage: Optional[str] = None       # "V1" ...
    treatment: Optional[str] = None   # "과습"/"한발"
    zone: Optional[str] = DEFAULT_ZONE

    # 선택: 없으면 서버시간 사용
    ts: Optional[str] = None          # ISO8601


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/log")
def log_data(req: LogRequest):
    ts = now_utc() if req.ts is None else datetime.fromisoformat(req.ts.replace("Z", "+00:00"))
    rec = {
        "sensor_id": req.sensor_id,
        "ts": ts,
        "moisture_percent": float(req.moisture_percent),
        "raw": None if req.raw is None else float(req.raw),
        "stage": req.stage,
        "treatment": req.treatment,
        "zone": req.zone if req.zone is not None else DEFAULT_ZONE,
    }
    add_record(req.sensor_id, rec)

    return {
        "ok": True,
        "saved": {
            **{k: v for k, v in rec.items() if k != "ts"},
            "ts": rec["ts"].isoformat().replace("+00:00", "Z"),
        },
        "count": len(DB[req.sensor_id]),
    }


@app.get("/history")
def history(sensor_id: str, n: int = 200):
    ensure_sensor(sensor_id)
    data = DB[sensor_id][-max(1, n):]
    out = []
    for r in data:
        out.append({
            "ts": r["ts"].isoformat().replace("+00:00", "Z"),
            "moisture_percent": r["moisture_percent"],
            "raw": r["raw"],
        })
    return {"sensor_id": sensor_id, "n": len(out), "data": out}


class LatestRequest(BaseModel):
    sensor_id: str = "sensor_01"

    # stage/treatment/zone은 “앱에서 보내도 되고”,
    # “서버에 저장된 마지막 값 쓰고 싶으면” null로 보내도 됨.
    stage: Optional[str] = None
    treatment: Optional[str] = None
    zone: Optional[str] = DEFAULT_ZONE

    days_since_stage_start: float = 0.0


@app.post("/latest")
def latest_and_predict(req: LatestRequest):
    ensure_sensor(req.sensor_id)
    if len(DB[req.sensor_id]) == 0:
        return {"ok": False, "error": "no_data_for_sensor"}

    last = DB[req.sensor_id][-1]

    # 7일 피처 계산
    f7 = compute_7d_features(req.sensor_id)

    # cat 값 결정: 요청값 우선, 없으면 마지막 저장값 사용
    stage = req.stage if req.stage is not None else last.get("stage")
    treatment = req.treatment if req.treatment is not None else last.get("treatment")
    zone = req.zone if req.zone is not None else last.get("zone", DEFAULT_ZONE)

    row = {
        "days_since_stage_start": float(req.days_since_stage_start),
        "soil_mean_7d": f7["soil_mean_7d"],
        "soil_min_7d": f7["soil_min_7d"],
        "soil_std_7d": f7["soil_std_7d"],
        "soil_trend_7d": f7["soil_trend_7d"],
        "soil_missing": f7["soil_missing"],
        "soil_stress_intensity": f7["soil_stress_intensity"],
        "days_x_soil_mean": float(req.days_since_stage_start) * float(f7["soil_mean_7d"]),
        "stage": stage,
        "treatment": treatment,
        "zone": zone,
    }

    X = align_input(row)

    res = {
        "ok": True,
        "latest": {
            "ts": last["ts"].isoformat().replace("+00:00", "Z"),
            "moisture_percent": last["moisture_percent"],
            "raw": last["raw"],
        },
        "features_7d": f7,
        "input_used": {
            "stage": stage,
            "treatment": treatment,
            "zone": zone,
            "days_since_stage_start": req.days_since_stage_start,
        },
        "pred": {
            "treat": predict_one_classifier(TREAT_MODEL, X),
            "stage": predict_one_classifier(STAGE_MODEL, X),
            "stem_length_mm_pred": predict_one_regressor(LEN_REG, X),
            "stem_diameter_mm_pred": predict_one_regressor(DIA_REG, X),
        }
    }
    return res


# ====== 더미 7일치 데이터 채우기 ======
class SeedRequest(BaseModel):
    sensor_id: str = "sensor_01"
    # 10분 간격 7일 = 1008개. 너무 많으면 30분(336개)도 가능.
    step_minutes: int = 10
    base: float = 55.0          # 평균 수분(%)
    noise: float = 8.0          # 랜덤 변동
    daily_wave: float = 10.0    # 일변화(사인)
    stage: str = "V1"
    treatment: str = "과습"
    zone: str = DEFAULT_ZONE


@app.post("/seed7d")
def seed_7days(req: SeedRequest):
    end = now_utc()
    start = end - timedelta(days=7)
    step = timedelta(minutes=max(1, req.step_minutes))

    t = start
    rng = np.random.default_rng(42)

    count = 0
    while t <= end:
        # 일 변화
        day_frac = (t - start).total_seconds() / 86400.0
        wave = req.daily_wave * np.sin(2 * np.pi * day_frac)

        # 노이즈
        val = req.base + wave + rng.normal(0, req.noise)
        val = float(np.clip(val, 0, 100))

        rec = {
            "sensor_id": req.sensor_id,
            "ts": t,
            "moisture_percent": val,
            "raw": None,
            "stage": req.stage,
            "treatment": req.treatment,
            "zone": req.zone,
        }
        add_record(req.sensor_id, rec)
        count += 1
        t += step

    return {"ok": True, "sensor_id": req.sensor_id, "seeded": count}
