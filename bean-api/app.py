# app.py
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

PKL_LEN   = MODEL_DIR / "len_reg.pkl"
PKL_DIA   = MODEL_DIR / "dia_reg.pkl"
PKL_TREAT = MODEL_DIR / "treat_model.pkl"
PKL_STAGE = MODEL_DIR / "stage_model.pkl"

def load_model(pkl_path: Path):
    obj = joblib.load(pkl_path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj

def align_input(row: dict) -> pd.DataFrame:
    for c in FEATURES_NUM:
        row.setdefault(c, 0.0)

    row.setdefault("stage", None)
    row.setdefault("treatment", None)
    row.setdefault("zone", DEFAULT_ZONE)

    X = pd.DataFrame([{c: row.get(c) for c in (FEATURES_NUM + CAT_COLS)}])

    for c in FEATURES_NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    # category 고정
    X["stage"] = pd.Categorical(X["stage"], categories=STAGE_CATS)
    X["treatment"] = pd.Categorical(X["treatment"], categories=TREAT_CATS)

    X["zone"] = X["zone"].astype("string").fillna(DEFAULT_ZONE)
    X["zone"] = X["zone"].astype("category")

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

# 전역으로 모델을 "나중에" 채움
LEN_REG = None
DIA_REG = None
TREAT_MODEL = None
STAGE_MODEL = None

@app.on_event("startup")
def startup_load_models():
    global LEN_REG, DIA_REG, TREAT_MODEL, STAGE_MODEL

    missing = [p for p in [PKL_LEN, PKL_DIA, PKL_TREAT, PKL_STAGE] if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing model files: {[str(m) for m in missing]}")

    LEN_REG = load_model(PKL_LEN)
    DIA_REG = load_model(PKL_DIA)
    TREAT_MODEL = load_model(PKL_TREAT)
    STAGE_MODEL = load_model(PKL_STAGE)

class PredictRequest(BaseModel):
    days_since_stage_start: float = 0.0
    soil_mean_7d: float = 0.0
    soil_min_7d: float = 0.0
    soil_std_7d: float = 0.0
    soil_trend_7d: float = 0.0
    soil_missing: float = 0.0
    soil_stress_intensity: float = 0.0
    days_x_soil_mean: float = 0.0

    stage: str | None = None
    treatment: str | None = None
    zone: str | None = DEFAULT_ZONE

@app.get("/health")
def health():
    ok = all(m is not None for m in [LEN_REG, DIA_REG, TREAT_MODEL, STAGE_MODEL])
    return {"ok": ok}

@app.post("/predict")
def predict(req: PredictRequest):
    if any(m is None for m in [LEN_REG, DIA_REG, TREAT_MODEL, STAGE_MODEL]):
        raise HTTPException(status_code=500, detail="Models not loaded")

    X = align_input(req.model_dump())

    return {
        "treat": predict_one_classifier(TREAT_MODEL, X),
        "stage": predict_one_classifier(STAGE_MODEL, X),
        "stem_length_mm_pred": predict_one_regressor(LEN_REG, X),
        "stem_diameter_mm_pred": predict_one_regressor(DIA_REG, X),
    }

from datetime import datetime
from typing import Dict, Any

# 메모리 저장소(센서별 마지막 값)
LAST: Dict[str, Dict[str, Any]] = {}

class LogRequest(BaseModel):
    sensor_id: str = "sensor_01"
    moisture_percent: float
    raw: float | int

@app.post("/log")
def log_data(req: LogRequest):
    LAST[req.sensor_id] = {
        "sensor_id": req.sensor_id,
        "moisture_percent": float(req.moisture_percent),
        "raw": float(req.raw),
        "ts": datetime.utcnow().isoformat() + "Z",
    }
    return {"ok": True, "saved": LAST[req.sensor_id]}

