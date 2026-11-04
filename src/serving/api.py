"""FastAPI inference service for the UNSW-NB15 threat detector.

Loads the trained artifacts at startup - only after checking each one
against the SHA-256 pins in ``models/manifest.json`` - and reuses the
*same* feature transformation functions as training (no reimplementation).

Endpoints (roles from ``src/serving/security.py``):
    GET  /health               public  -> liveness + whether models loaded
    POST /predict              sensor  -> one raw flow -> verdict (+ alert)
    POST /predict/sequence     sensor  -> up to MAX_SEQUENCE_FLOWS flows,
                                          scored as windows by the LSTM
    POST /feedback             analyst -> ground-truth label for a logged
                                          prediction (the ONLY label path)

Response of /predict:
    {prediction_id, prediction, p_attack, threshold, model_version, alert}

    ``p_attack`` is the Random Forest's attack probability; ``prediction``
    is ``p_attack >= threshold``, where the threshold was chosen on the
    validation split for FPR <= 1% (pinned in ``models/operating_point.json``).
    ``alert`` (attacks only) carries severity, the family classifier's
    guess, the ATT&CK mapping, the features that drove the score and a
    triage note - see ``src/soc/alerts.py``.

Run:
    export IDS_API_KEYS="$(python -m src.serving.security new-key ...)"   # see README
    uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_model import (  # noqa: E402
    build_windows,
    detect_group_key,
    transform_flows,
    windows_to_sequences,
)
from src.models.manifest import load_verified_pickle, verify_manifest  # noqa: E402
from src.serving.prediction_log import (  # noqa: E402
    init_db,
    log_prediction,
    record_feedback,
)
from src.serving.security import (  # noqa: E402
    BodySizeLimitMiddleware,
    Principal,
    reload_config,
    require,
    security_headers,
)
from src.soc.alerts import build_alert  # noqa: E402
from src.soc.explain import path_contributions, top_drivers  # noqa: E402

DEFAULT_WINDOW = 10
DEFAULT_STRIDE = 5
MAX_SEQUENCE_FLOWS = 512
FAMILIES = ("Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
            "Reconnaissance", "Shellcode", "Worms")


def models_dir() -> Path:
    return Path(os.environ.get("IDS_MODELS_DIR", PROJECT_ROOT / "models"))


# ---------------------------------------------------------------------------
# Request / response schemas. Strict by design (OWASP API3/API4): unknown
# fields are rejected (that is what closes the label-poisoning path),
# NaN/inf are rejected, and every numeric field is bounded.
# ---------------------------------------------------------------------------
Count = Annotated[float, Field(ge=0, le=1e12)]
Small = Annotated[int, Field(ge=0, le=1_000_000)]
Ttl = Annotated[int, Field(ge=0, le=255)]
Port = Annotated[int, Field(ge=0, le=65535)]
Flag = Annotated[int, Field(ge=0, le=1)]
Token = Annotated[str, Field(min_length=1, max_length=16,
                             pattern=r"^[A-Za-z0-9_.\-]+$")]
Ip = Annotated[str, Field(max_length=45, pattern=r"^[0-9A-Fa-f.:]+$")]


class FlowRecord(BaseModel):
    """One raw flow record (UNSW-NB15 / Argus fields). Labels are NOT accepted."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    srcip: Ip | None = None
    sport: Port | None = None
    dstip: Ip | None = None
    dsport: Port | None = None
    proto: Token | None = None
    state: Token | None = None
    service: Token | None = None
    dur: Count | None = None
    sbytes: Count | None = None
    dbytes: Count | None = None
    sttl: Ttl | None = None
    dttl: Ttl | None = None
    sloss: Count | None = None
    dloss: Count | None = None
    Sload: Count | None = None
    Dload: Count | None = None
    Spkts: Count | None = None
    Dpkts: Count | None = None
    swin: Port | None = None
    dwin: Port | None = None
    stcpb: Count | None = None
    dtcpb: Count | None = None
    smeansz: Count | None = None
    dmeansz: Count | None = None
    trans_depth: Small | None = None
    res_bdy_len: Count | None = None
    Sjit: Count | None = None
    Djit: Count | None = None
    Stime: Count | None = None
    Ltime: Count | None = None
    Sintpkt: Count | None = None
    Dintpkt: Count | None = None
    tcprtt: Count | None = None
    synack: Count | None = None
    ackdat: Count | None = None
    is_sm_ips_ports: Flag | None = None
    ct_state_ttl: Small | None = None
    ct_flw_http_mthd: Small | None = None
    is_ftp_login: Small | None = None
    ct_ftp_cmd: Small | None = None
    ct_srv_src: Small | None = None
    ct_srv_dst: Small | None = None
    ct_dst_ltm: Small | None = None
    ct_src_ltm: Small | None = None
    ct_src_dport_ltm: Small | None = None
    ct_dst_sport_ltm: Small | None = None
    ct_dst_src_ltm: Small | None = None


class PredictResponse(BaseModel):
    prediction_id: int
    prediction: int
    p_attack: float
    threshold: float
    model_version: str
    alert: dict | None = None


class SequenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flows: list[FlowRecord] = Field(min_length=1, max_length=MAX_SEQUENCE_FLOWS)


class SequenceWindow(BaseModel):
    window_id: int
    group: str
    n_flows: int
    prediction: int
    p_attack: float
    flow_positions: list[int]  # indices into the request's `flows` array


class SequenceResponse(BaseModel):
    windows: list[SequenceWindow]
    grouping: list[str]


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: int = Field(ge=1)
    label: Literal[0, 1]
    attack_cat: Literal[FAMILIES] | None = None  # type: ignore[valid-type]


# ---------------------------------------------------------------------------
# Startup: verify then load artifacts once
# ---------------------------------------------------------------------------
REQUIRED = ["random_forest_best.joblib", "preprocess_pipeline.joblib",
            "preprocess_metadata.json"]
OPTIONAL = ["family_rf.joblib", "lstm_best.keras", "operating_point.json"]


def _load_artifacts() -> dict:
    mdir = models_dir()
    present = [n for n in REQUIRED + OPTIONAL if (mdir / n).exists()]
    missing = [n for n in REQUIRED if n not in present]
    if missing:
        raise RuntimeError(f"model artifacts missing: {missing} - run the pipeline first")
    # Refuse to load anything whose digest is not pinned. Pickles are
    # loaded from the exact bytes that were hashed (no TOCTOU window).
    manifest = verify_manifest(mdir, present)
    metadata = json.loads((mdir / "preprocess_metadata.json").read_text(encoding="utf-8"))
    # The alert threshold is security-relevant (threshold=1.0 blinds the
    # IDS), so it lives in models/ and is pinned like any other artifact.
    # The served RF may see a subset of the pipeline's features (hardened
    # model: no TTL); the subset is pinned with the threshold.
    operating = {}
    if "operating_point.json" in present:
        operating = json.loads((mdir / "operating_point.json").read_text(encoding="utf-8"))
    threshold = float(operating.get("threshold", 0.5))
    threshold = float(os.environ.get("IDS_ALERT_THRESHOLD", threshold))
    lstm = None
    if "lstm_best.keras" in present:
        import tensorflow as tf  # noqa: PLC0415

        lstm = tf.keras.models.load_model(mdir / "lstm_best.keras")
    return {
        "pipeline": load_verified_pickle(mdir, "preprocess_pipeline.joblib", manifest),
        "rf": load_verified_pickle(mdir, "random_forest_best.joblib", manifest),
        "family": (load_verified_pickle(mdir, "family_rf.joblib", manifest)
                   if "family_rf.joblib" in present else None),
        "lstm": lstm,
        "metadata": metadata,
        "threshold": threshold,
        "model_features": operating.get("features") or metadata["feature_names"],
        "hardened": bool(operating.get("hardening")),
        "model_version": manifest["files"]["random_forest_best.joblib"]["sha256"][:12],
    }


ARTIFACTS: dict | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ARTIFACTS
    reload_config()
    ARTIFACTS = _load_artifacts()
    init_db()
    yield
    ARTIFACTS = None


docs = os.environ.get("IDS_ENABLE_DOCS") == "1"
app = FastAPI(title="UNSW-NB15 Threat Detector", version="2.0.0", lifespan=lifespan,
              docs_url="/docs" if docs else None, redoc_url=None,
              openapi_url="/openapi.json" if docs else None)
app.add_middleware(BodySizeLimitMiddleware)
app.middleware("http")(security_headers)
_START_TIME = time.time()


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Do not echo the offending input back: it may be non-JSON-serialisable
    # (NaN crashed the default handler with a 500) and reflecting attacker
    # data into responses/logs is never needed to explain a 422.
    errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    # Never leak stack traces or file paths to clients (OWASP API8).
    print(f"[api] unhandled {type(exc).__name__}: {exc}", file=sys.stderr)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


# ---------------------------------------------------------------------------
# Prediction helpers (pure, testable)
# ---------------------------------------------------------------------------

def _frame(records: list[dict], feature_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    return df


def score_flow(art: dict, record: dict) -> dict:
    """Single raw flow -> p_attack, family probabilities and drivers."""
    names = art["metadata"]["feature_names"]
    X = pd.DataFrame(transform_flows(_frame([record], names), art["pipeline"], names),
                     columns=names)
    X_model = X[art["model_features"]]
    p_attack = float(art["rf"].predict_proba(X_model)[0, 1])
    family_probs: dict[str, float] = {}
    drivers: list[dict] = []
    if p_attack >= art["threshold"]:
        if art["family"] is not None:
            probs = art["family"].predict_proba(X)[0]
            family_probs = {str(c): float(p) for c, p in zip(art["family"].classes_, probs)}
        _, contrib = path_contributions(art["rf"], X_model.to_numpy())
        drivers = top_drivers(contrib[0], art["model_features"], record)
    return {"p_attack": p_attack, "family_probs": family_probs, "drivers": drivers}


def score_sequence(lstm, pipeline: dict, feature_names: list[str],
                   records: list[dict], window: int, stride: int) -> dict:
    """List of raw flows -> LSTM window predictions (reuses training logic)."""
    df = _frame(records, feature_names)
    df["Label"] = 0  # not used for scoring; build_windows requires the column
    group_cols = detect_group_key(df)
    windows = build_windows(df, group_cols, window, stride, label_mode="any")
    X = windows_to_sequences(windows, transform_flows(df, pipeline, feature_names),
                             maxlen=window)
    proba = lstm.predict(X, verbose=0).ravel()
    result = [{
        "window_id": int(win["window_id"]),
        "group": win["group"],
        "n_flows": int(win["n_flows"]),
        "prediction": int(proba[i] >= 0.5),
        "p_attack": float(proba[i]),
        "flow_positions": list(win["flow_index"]),
    } for i, (_, win) in enumerate(windows.iterrows())]
    return {"windows": result, "grouping": group_cols}


def _artifacts() -> dict:
    if ARTIFACTS is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return ARTIFACTS


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    # Public: liveness only. Model names/paths are not disclosed.
    return {"status": "ok" if ARTIFACTS is not None else "loading",
            "uptime_seconds": round(time.time() - _START_TIME, 1)}


@app.post("/predict", response_model=PredictResponse)
def predict(record: FlowRecord,
            who: Principal = Depends(require("sensor"))) -> PredictResponse:
    art = _artifacts()
    raw = record.model_dump(exclude_none=True)
    out = score_flow(art, raw)
    prediction = int(out["p_attack"] >= art["threshold"])
    family = (max(out["family_probs"], key=out["family_probs"].get)
              if out["family_probs"] else None)
    prediction_id = log_prediction(
        model="rf", prediction=prediction, p_attack=out["p_attack"],
        predicted_family=family, n_flows=1, sensor=who.name,
        record_json=json.dumps(raw))
    alert = build_alert(prediction_id, raw, out["p_attack"], out["family_probs"],
                        out["drivers"], art["threshold"], art["model_version"])
    if alert and art["hardened"]:
        alert["ml"]["model"] = "random_forest_hardened"
    return PredictResponse(prediction_id=prediction_id, prediction=prediction,
                           p_attack=out["p_attack"], threshold=art["threshold"],
                           model_version=art["model_version"], alert=alert)


@app.post("/predict/sequence", response_model=SequenceResponse)
def predict_sequence(req: SequenceRequest,
                     who: Principal = Depends(require("sensor"))) -> SequenceResponse:
    art = _artifacts()
    if art["lstm"] is None:
        raise HTTPException(status_code=503, detail="sequence model not loaded")
    records = [f.model_dump(exclude_none=True) for f in req.flows]
    out = score_sequence(art["lstm"], art["pipeline"], art["metadata"]["feature_names"],
                         records, DEFAULT_WINDOW, DEFAULT_STRIDE)
    for win in out["windows"]:
        log_prediction(model="lstm", prediction=win["prediction"],
                       p_attack=win["p_attack"], n_flows=win["n_flows"],
                       sensor=who.name)
    return SequenceResponse(**out)


@app.post("/feedback")
def feedback(fb: Feedback, who: Principal = Depends(require("analyst"))) -> dict:
    """Analyst verdict for a logged prediction - the only way labels enter."""
    if fb.label == 1 and fb.attack_cat is None:
        raise HTTPException(status_code=422, detail="attack_cat required when label=1")
    if not record_feedback(fb.prediction_id, fb.label,
                           fb.attack_cat if fb.label == 1 else "Normal", who.name):
        raise HTTPException(status_code=404, detail="unknown prediction_id")
    return {"status": "recorded", "prediction_id": fb.prediction_id,
            "labeled_by": who.name}
