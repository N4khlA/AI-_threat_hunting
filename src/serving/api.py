"""FastAPI inference service for the UNSW-NB15 threat detector.

Loads the trained artifacts at startup and reuses the *same* feature
transformation functions from ``src/features/`` (no reimplementation):

    * models/preprocess_pipeline.joblib  (imputer + encoder + scaler)
    * models/random_forest_best.joblib   (per-flow classifier)
    * models/lstm_best.keras             (sequence classifier)
    * models/preprocess_metadata.json    (feature names, target info)

Endpoints:
    GET  /health                 -> service + model status
    POST /predict                -> single raw flow record
    POST /predict/sequence       -> list of raw flow records, scored as
                                    windows by the LSTM (grouping via the
                                    same fallback logic as training)

Response of /predict:
    {prediction, confidence, attack_family_probs, method}

    attack_family_probs is an approximation: the Random Forest is a
    binary attack/normal model, so P(family f) is decomposed as
    P(attack) * P(family f | attack), where the conditional is the
    attack-family prior estimated on the training split. The most
    informative number remains `confidence` (binary attack probability).

Run:
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
    # or: python -m uvicorn src.serving.api:app --reload
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocess import DROP_COLUMNS  # noqa: E402
from src.models.lstm_model import (  # noqa: E402
    build_windows,
    detect_group_key,
    transform_flows,
    windows_to_sequences,
)
from src.serving.prediction_log import (  # noqa: E402
    init_db,
    log_prediction,
)

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RF_MODEL_PATH = MODELS_DIR / "random_forest_best.joblib"
PIPELINE_PATH = MODELS_DIR / "preprocess_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "preprocess_metadata.json"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_best.keras"

DEFAULT_WINDOW = 10
DEFAULT_STRIDE = 5


class FlowRecord(BaseModel):
    """A raw flow record; same fields as data/raw/. Extra fields allowed."""

    srcip: str | float | None = None
    sport: int | float | None = None
    dstip: str | float | None = None
    dsport: int | float | None = None
    proto: str | None = None
    state: str | None = None
    service: str | None = None
    dur: float | None = None
    sbytes: int | float | None = None
    dbytes: int | float | None = None
    sttl: int | None = None
    dttl: int | None = None
    sloss: int | None = None
    dloss: int | None = None
    Sload: float | None = None
    Dload: float | None = None
    Spkts: float | None = None
    Dpkts: float | None = None
    swin: int | None = None
    dwin: int | None = None
    stcpb: int | None = None
    dtcpb: int | None = None
    smeansz: int | float | None = None
    dmeansz: int | float | None = None
    trans_depth: int | None = None
    res_bdy_len: int | None = None
    Sjit: float | None = None
    Djit: float | None = None
    Stime: float | None = None
    Ltime: float | None = None
    Sintpkt: float | None = None
    Dintpkt: float | None = None
    tcprtt: float | None = None
    synack: float | None = None
    ackdat: float | None = None
    is_sm_ips_ports: int | None = None
    ct_state_ttl: int | None = None
    ct_flw_http_mthd: int | None = None
    is_ftp_login: int | None = None
    ct_ftp_cmd: int | None = None
    ct_srv_src: int | None = None
    ct_srv_dst: int | None = None
    ct_dst_ltm: int | None = None
    ct_src_ltm: int | None = None
    ct_src_dport_ltm: int | None = None
    ct_dst_sport_ltm: int | None = None
    ct_dst_src_ltm: int | None = None
    attack_cat: str | None = Field(default=None, exclude=True)
    Label: int | None = Field(default=None, exclude=True)

    model_config = {"extra": "allow"}


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    attack_family_probs: dict[str, float]
    method: str


class SequenceWindow(BaseModel):
    window_id: int
    group: str
    n_flows: int
    prediction: int
    confidence: float
    flow_positions: list[int]  # indices into the request's `flows` array


class SequenceRequest(BaseModel):
    flows: list[FlowRecord] = Field(min_length=1)


class SequenceResponse(BaseModel):
    windows: list[SequenceWindow]
    grouping: list[str]


# ---------------------------------------------------------------------------
# Startup: load artifacts once
# ---------------------------------------------------------------------------
def _load_artifacts():
    if not RF_MODEL_PATH.exists() or not PIPELINE_PATH.exists():
        raise RuntimeError(
            "model artifacts missing - run preprocess + random_forest first")
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    rf = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    lstm = None
    if LSTM_MODEL_PATH.exists():
        import tensorflow as tf

        lstm = tf.keras.models.load_model(LSTM_MODEL_PATH)
    family_priors = _family_priors()
    return {"pipeline": pipeline, "metadata": metadata, "rf": rf,
            "scaler": scaler, "lstm": lstm, "family_priors": family_priors}


def _family_priors() -> dict[str, float]:
    """P(family | attack) estimated on the original (pre-SMOTE) train split."""
    path = PROCESSED_DIR / "attack_cat_train.csv"
    if not path.exists():
        return {}
    cats = pd.read_csv(path, usecols=["attack_cat"]).iloc[:, 0]
    attacks = cats[cats != "Normal"]
    return attacks.value_counts(normalize=True).to_dict()


ARTIFACTS: dict | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ARTIFACTS
    ARTIFACTS = _load_artifacts()
    init_db()
    yield


app = FastAPI(title="UNSW-NB15 Threat Detector", version="1.0.0",
              lifespan=lifespan)
_START_TIME = time.time()


# ---------------------------------------------------------------------------
# Prediction helpers (pure, testable)
# ---------------------------------------------------------------------------

def predict_flow(rf, pipeline: dict, feature_names: list[str],
                 record: dict, family_priors: dict[str, float]) -> dict:
    """Single raw flow record -> {prediction, confidence, family_probs}."""
    df = pd.DataFrame([record])
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    X = transform_flows(df, pipeline, feature_names)
    proba = float(rf.predict_proba(X)[0, 1])
    confidence = max(proba, 1.0 - proba)
    family_probs = {f: proba * p for f, p in family_priors.items()}
    return {
        "prediction": int(proba >= 0.5),
        "confidence": float(confidence),
        "attack_family_probs": family_probs,
        "method": "random_forest",
    }


def score_sequence(lstm, pipeline: dict, feature_names: list[str],
                     records: list[dict], window: int, stride: int) -> dict:
    """List of raw flows -> LSTM window predictions (reuses training logic)."""
    df = pd.DataFrame(records)
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    if "Label" not in df.columns:
        df["Label"] = 0  # not needed for scoring; build_windows requires it
    group_cols = detect_group_key(df)
    windows = build_windows(df, group_cols, window, stride, label_mode="any")
    feature_matrix = transform_flows(df, pipeline, feature_names)
    X = windows_to_sequences(windows, feature_matrix, maxlen=window)
    proba = lstm.predict(X, verbose=0).ravel()
    result = []
    for i in range(len(windows)):
        win = windows.iloc[i]
        confidence = float(max(proba[i], 1.0 - proba[i]))
        result.append({
            "window_id": int(win["window_id"]),
            "group": win["group"],
            "n_flows": int(win["n_flows"]),
            "prediction": int(proba[i] >= 0.5),
            "confidence": confidence,
            "flow_positions": list(win["flow_index"]),
        })
    return {"windows": result, "grouping": group_cols}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "models": {
            "random_forest": RF_MODEL_PATH.name,
            "lstm": LSTM_MODEL_PATH.name if ARTIFACTS["lstm"] is not None else "not loaded",
        },
        "uptime_seconds": round(time.time() - _START_TIME, 1),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(record: FlowRecord) -> PredictResponse:
    if ARTIFACTS is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    out = predict_flow(ARTIFACTS["rf"], ARTIFACTS["pipeline"],
                       ARTIFACTS["metadata"]["feature_names"],
                       record.model_dump(), ARTIFACTS["family_priors"])
    dump = record.model_dump()
    if record.Label is not None:
        dump["Label"] = record.Label
    if record.attack_cat is not None:
        dump["attack_cat"] = record.attack_cat
    predicted_family = max(out["attack_family_probs"],
                           key=out["attack_family_probs"].get)
    log_prediction(
        model="rf",
        prediction=out["prediction"],
        confidence=out["confidence"],
        true_label=record.Label,
        predicted_family=predicted_family,
        true_family=record.attack_cat,
        n_flows=1,
        record_json=json.dumps(dump, default=str),
    )
    return PredictResponse(**out)


@app.post("/predict/sequence", response_model=SequenceResponse)
def predict_sequence(req: SequenceRequest) -> SequenceResponse:
    if ARTIFACTS is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if ARTIFACTS["lstm"] is None:
        raise HTTPException(status_code=503,
                            detail="LSTM model not loaded - train it first")
    records = [f.model_dump() for f in req.flows]
    out = score_sequence(ARTIFACTS["lstm"], ARTIFACTS["pipeline"],
                         ARTIFACTS["metadata"]["feature_names"],
                         records, DEFAULT_WINDOW, DEFAULT_STRIDE)
    labels = [f.Label for f in req.flows]
    families = [f.attack_cat for f in req.flows]
    for win in out["windows"]:
        win_labels = [labels[i] for i in win["flow_positions"]
                      if labels[i] is not None]
        known = len(win_labels) == len(win["flow_positions"])
        true_label = int(any(win_labels)) if known else None
        true_family = None
        if known and true_label == 1:
            true_family = next(
                (families[i] for i in win["flow_positions"]
                 if families[i] is not None), None)
        log_prediction(
            model="lstm",
            prediction=win["prediction"],
            confidence=win["confidence"],
            true_label=true_label,
            predicted_family=None,
            true_family=true_family,
            n_flows=win["n_flows"],
        )
    return SequenceResponse(**out)
