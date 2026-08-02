"""API endpoint tests using FastAPI's TestClient.

Skipped when the model artifacts are missing (run the pipeline first).
The prediction log is redirected to a temp file via PREDICTION_LOG_PATH
so the real reports/prediction_log.sqlite is never touched.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "random_forest_best.joblib").exists()
    or not (MODELS_DIR / "lstm_best.keras").exists(),
    reason="model artifacts missing - run preprocess/random_forest/lstm first",
)

FAMILIES = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
            "Generic", "Reconnaissance", "Shellcode", "Worms"]


def build_record(label: int, family: str, **overrides) -> dict:
    """A raw flow record with every feature the pipeline expects."""
    metadata = json.loads(
        (MODELS_DIR / "preprocess_metadata.json").read_text(encoding="utf-8"))
    record = {feature: 0 for feature in metadata["feature_names"]}
    record.update({
        "proto": "tcp", "service": "http", "state": "FIN",
        "sport": 443, "dsport": 80, "sttl": 64, "dttl": 128,
        "Label": label, "attack_cat": family,
    })
    record.update(overrides)
    return record


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from src.serving import api  # noqa: PLC0415

    log_db = tmp_path_factory.mktemp("log") / "prediction_log.sqlite"
    os.environ["PREDICTION_LOG_PATH"] = str(log_db)
    with TestClient(api.app) as test_client:
        test_client.log_db = log_db
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models"]["random_forest"].endswith(".joblib")


def test_predict_single_flow(client):
    record = build_record(label=1, family="Generic")
    response = client.post("/predict", json=record)
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "random_forest"
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["attack_family_probs"]) == set(FAMILIES)
    # family probs = P(family|attack) * P(attack), so they sum to the
    # model's attack probability (== confidence for attack predictions)
    probs = body["attack_family_probs"]
    attack_prob = (body["confidence"] if body["prediction"] == 1
                   else 1.0 - body["confidence"])
    assert abs(sum(probs.values()) - attack_prob) < 1e-6


def test_predict_accepts_partial_record(client):
    """Missing optional fields must not crash: they are imputed at serving."""
    response = client.post("/predict", json={
        "proto": "udp", "service": "dns", "state": "INT",
        "sbytes": 100, "dbytes": 200,
    })
    assert response.status_code == 200
    assert response.json()["prediction"] in (0, 1)


def test_predict_sequence_windows(client):
    flows = [build_record(label=i % 2, family="Generic" if i % 2 else "Normal")
             for i in range(14)]
    response = client.post("/predict/sequence", json={"flows": flows})
    assert response.status_code == 200
    body = response.json()
    assert body["grouping"], "grouping must be non-empty"
    windows = body["windows"]
    assert windows, "at least one window expected"
    ids = [w["window_id"] for w in windows]
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    for win in windows:
        assert win["prediction"] in (0, 1)
        assert 0.0 <= win["confidence"] <= 1.0
        assert win["n_flows"] >= 1
        assert all(0 <= p < 14 for p in win["flow_positions"])


def test_predict_logs_to_sqlite(client):
    record = build_record(label=1, family="Exploits")
    response = client.post("/predict", json=record)
    assert response.status_code == 200
    assert Path(client.log_db).exists()
    with sqlite3.connect(client.log_db) as conn:
        rows = conn.execute(
            "SELECT model, prediction, true_label, true_family,"
            "       record_json FROM predictions WHERE model='rf'"
            " ORDER BY id DESC LIMIT 1").fetchall()
    assert len(rows) == 1
    model, prediction, true_label, true_family, record_json = rows[0]
    assert model == "rf"
    assert true_label == 1
    assert true_family == "Exploits"  # ground truth carried by the request
    assert '"Label": 1' in record_json


def test_sequence_predictions_are_logged_with_true_labels(client):
    flows = [build_record(label=1, family="Generic" if i % 2 else "Normal")
             for i in range(14)]
    response = client.post("/predict/sequence", json={"flows": flows})
    assert response.status_code == 200
    n_windows = len(response.json()["windows"])
    with sqlite3.connect(client.log_db) as conn:
        lstm_rows = conn.execute(
            "SELECT prediction, true_label, n_flows FROM predictions"
            " WHERE model='lstm' ORDER BY id DESC LIMIT ?",
            (n_windows,)).fetchall()
    assert len(lstm_rows) == n_windows
    assert any(row[1] == 1 for row in lstm_rows), \
        "windows covering attack flows must carry true_label=1"
