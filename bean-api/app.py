# app.py
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# =========================
# CONFIG (학습 때 쓰던 피처명 고정)
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

# 학습에 존재했던 범주 목록(너가 말한 unique 기반)
STAGE_CATS = ["V1", "V5", "R1", "R3", "R5"]
TREAT_CATS = ["과습", "한발"]

# zone은 학습 CSV에 있던 값들로 맞추는 게 베스트지만,
# 다른 데이터 적용할 거라면 zone은 고정값(예: "0")로 넣는 방식이 실용적.
# 여기서는 문자열로 받되, 없으면 "0" 넣음.
DEFAULT_ZONE = "0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

PKL_LEN   = os.path.join(MODEL_DIR, "len_reg.pkl")
PKL_DIA   = os.path.join(MODEL_DIR, "dia_reg.pkl")
PKL_TREAT = os.path.join(MODEL_DIR, "treat_model.pkl")
PKL_STAGE = os.path.join(MODEL_DIR, "stage_model.pkl")


def load_model(pkl_path: str):
    obj = joblib.load(pkl_path)
    # dict로 저장한 경우도 지원
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


LEN_REG = load_model(PKL_LEN)
DIA_REG = load_model(PKL_DIA)
TREAT_MODEL = load_model(PKL_TREAT)
STAGE_MODEL = load_model(PKL_STAGE)


def align_input(row: dict) -> pd.DataFrame:
    """
    입력 row(dict)를 학습 스키마(FEATURES_NUM + CAT_COLS)에 맞게 정렬 + dtype 고정.
    """
    # 1) 수치 피처 디폴트
    for c in FEATURES_NUM:
        row.setdefault(c, 0.0)

    # 2) cat 디폴트
    row.setdefault("stage", None)
    row.setdefault("treatment", None)
    row.setdefault("zone", DEFAULT_ZONE)

    # 3) DF 생성(컬럼 순서 고정)
    X = pd.DataFrame([{c: row.get(c) for c in (FEATURES_NUM + CAT_COLS)}])

    # 4) numeric 강제
    for c in FEATURES_NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    # 5) categorical 강제 + category 목록 고정 (중요!)
    # stage / treatment는 학습 범주 목록으로 맞춘다
    X["stage"] = pd.Categorical(X["stage"], categories=STAGE_CATS)
    X["treatment"] = pd.Categorical(X["treatment"], categories=TREAT_CATS)

    # zone은 학습 때 숫자였을 수도 있어서 문자열/숫자 다 받아주고 category로만 만든다
    # (학습 시 zone의 category 목록을 모르면 "카테고리 mismatch" 가능성이 있으니
    #  가장 안전한 방법은 zone을 아예 모델에서 빼고 재학습하는 것.
    #  일단은 category로만 캐스팅.)
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
    y = model.predict(X)[0]
    return float(y)


# =========================
# API
# =========================
app = FastAPI(title="Bean IoT Predict API")


class PredictRequest(BaseModel):
    # 필요한 것만 보내도 되게 기본값 설정
    days_since_stage_start: float = 0.0
    soil_mean_7d: float = 0.0
    soil_min_7d: float = 0.0
    soil_std_7d: float = 0.0
    soil_trend_7d: float = 0.0
    soil_missing: float = 0.0
    soil_stress_intensity: float = 0.0
    days_x_soil_mean: float = 0.0

    # cat
    stage: str | None = None
    treatment: str | None = None
    zone: str | None = DEFAULT_ZONE


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
def predict(req: PredictRequest):
    row = req.model_dump()
    X = align_input(row)

    res = {
        "treat": predict_one_classifier(TREAT_MODEL, X),
        "stage": predict_one_classifier(STAGE_MODEL, X),
        "stem_length_mm_pred": predict_one_regressor(LEN_REG, X),
        "stem_diameter_mm_pred": predict_one_regressor(DIA_REG, X),
    }
    return res
