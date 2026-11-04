"""Shared fixtures: a tiny but real artifact set built from synthetic flows.

The API tests used to skip whenever the trained models were missing,
which meant CI (no dataset) never exercised the service at all. This
fixture fits the *real* preprocessing code, a small Random Forest, a
family classifier and an untrained LSTM on synthetic UNSW-shaped data,
pins them in a manifest, and points the API at them via env vars.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAMILIES = ["DoS", "Exploits", "Reconnaissance"]
NUMERIC = ["dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "Sload",
           "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
           "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
           "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
           "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
           "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
           "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
           "ct_dst_src_ltm"]


def synthetic_flows(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """UNSW-shaped flows where attacks send more bytes at lower TTL."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    df = pd.DataFrame({c: rng.integers(0, 50, n).astype(float) for c in NUMERIC})
    df["sbytes"] = np.where(label == 1, rng.integers(5_000, 50_000, n),
                            rng.integers(100, 2_000, n)).astype(float)
    df["sttl"] = np.where(label == 1, 254, 31).astype(float)
    df["Spkts"] = rng.integers(1, 100, n).astype(float)
    df["dur"] = rng.random(n) + 0.01
    df["proto"] = rng.choice(["tcp", "udp"], n)
    df["service"] = rng.choice(["-", "http", "dns"], n)
    df["state"] = rng.choice(["FIN", "INT", "CON"], n)
    df["Label"] = label
    df["attack_cat"] = np.where(label == 1, rng.choice(FAMILIES, n), "Normal")
    return df


def benign_record() -> dict:
    return {"proto": "tcp", "service": "http", "state": "FIN", "dur": 0.5,
            "sbytes": 500.0, "dbytes": 4000.0, "sttl": 31, "dttl": 29,
            "Spkts": 8.0, "Dpkts": 10.0}


def attack_record() -> dict:
    return {**benign_record(), "sbytes": 40_000.0, "sttl": 254}


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory) -> Path:
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

    from src.features.preprocess import build_splits  # noqa: PLC0415
    from src.models.lstm_model import build_model, transform_flows  # noqa: PLC0415
    from src.models.manifest import write_manifest  # noqa: PLC0415

    out = tmp_path_factory.mktemp("models")
    df = synthetic_flows()
    idx = df.index
    parts = {"train": idx[:400], "val": idx[400:500], "test": idx[500:]}
    data = build_splits(df, parts, imbalance="none")
    names = data["X_train"].columns.tolist()
    pipeline = {"imputers": data["imputers"], "encoder": data["encoder_state"],
                "scaler": data["scaler"]}
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(
        data["X_train"], data["y_train"])
    attacks = df.loc[parts["train"]].query("Label == 1")
    family = RandomForestClassifier(n_estimators=20, random_state=0).fit(
        transform_flows(attacks, pipeline, names), attacks["attack_cat"])
    lstm = build_model(maxlen=10, n_features=len(names), units=4, layers=1,
                       dropout=0.0, learning_rate=1e-3)

    joblib.dump(pipeline, out / "preprocess_pipeline.joblib")
    joblib.dump(rf, out / "random_forest_best.joblib")
    joblib.dump(family, out / "family_rf.joblib")
    lstm.save(out / "lstm_best.keras")
    (out / "operating_point.json").write_text(json.dumps({"threshold": 0.5}),
                                              encoding="utf-8")
    (out / "preprocess_metadata.json").write_text(json.dumps(
        {"feature_names": names, "split_protocol": "synthetic"}), encoding="utf-8")
    write_manifest(out)
    return out
