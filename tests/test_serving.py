"""API tests: behaviour, security controls and the closed poisoning path.

Runs against the synthetic artifact set from ``conftest.py`` (no dataset
needed). Every security control in docs/threat_model.md that lives in
the API has at least one test here.
"""

from __future__ import annotations

import json
import shutil
import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving import api
from src.serving.security import hash_key
from tests.conftest import attack_record, benign_record

SENSOR_KEY = "sensor-test-key-0123456789"
ANALYST_KEY = "analyst-test-key-0123456789"
SENSOR = {"X-API-Key": SENSOR_KEY}
ANALYST = {"X-API-Key": ANALYST_KEY}


def _env(monkeypatch, models, tmp_path, **extra):
    monkeypatch.setenv("IDS_MODELS_DIR", str(models))
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "log.sqlite"))
    monkeypatch.setenv("IDS_API_KEYS", ",".join([
        f"sensor-1:sensor:{hash_key(SENSOR_KEY)}",
        f"alice:analyst:{hash_key(ANALYST_KEY)}"]))
    monkeypatch.setenv("IDS_RATE_PER_MIN", "6000")
    monkeypatch.setenv("IDS_RATE_BURST", "1000")
    for key, value in extra.items():
        monkeypatch.setenv(key, str(value))


@pytest.fixture
def client(artifacts_dir, tmp_path, monkeypatch):
    _env(monkeypatch, artifacts_dir, tmp_path)
    with TestClient(api.app) as c:
        c.log_db = tmp_path / "log.sqlite"
        yield c


# --- behaviour -------------------------------------------------------------

def test_health_is_public_and_discloses_nothing(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert set(r.json()) == {"status", "uptime_seconds"}
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_attack_flow_produces_alert_with_attack_mapping(client):
    r = client.post("/predict", json=attack_record(), headers=SENSOR)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1 and body["p_attack"] >= body["threshold"]
    alert = body["alert"]
    assert alert["event"]["severity"] in {"low", "medium", "high"}
    assert alert["ml"]["family"] in {"DoS", "Exploits", "Reconnaissance"}
    assert alert["threat"]["technique"], "mapped families carry ATT&CK ids"
    assert alert["ml"]["drivers"][0]["feature"] in {"sbytes", "sttl"}
    assert "Next:" in alert["triage_note"]


def test_benign_flow_has_no_alert(client):
    body = client.post("/predict", json=benign_record(), headers=SENSOR).json()
    assert body["prediction"] == 0 and body["alert"] is None


def test_partial_record_is_imputed(client):
    r = client.post("/predict", json={"proto": "udp", "sbytes": 100}, headers=SENSOR)
    assert r.status_code == 200


def test_sequence_scoring(client):
    flows = [benign_record() for _ in range(14)]
    r = client.post("/predict/sequence", json={"flows": flows}, headers=SENSOR)
    assert r.status_code == 200
    wins = r.json()["windows"]
    assert wins and all(0 <= p < 14 for w in wins for p in w["flow_positions"])


# --- authentication / authorization (OWASP API2, API5) ---------------------

@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_predict_requires_valid_key(client, headers):
    r = client.post("/predict", json=benign_record(), headers=headers)
    assert r.status_code == 401


def test_fails_closed_without_configured_keys(artifacts_dir, tmp_path, monkeypatch):
    _env(monkeypatch, artifacts_dir, tmp_path, IDS_API_KEYS="")
    with TestClient(api.app) as c:
        assert c.post("/predict", json=benign_record(), headers=SENSOR).status_code == 401


def test_sensor_cannot_submit_labels(client):
    pid = client.post("/predict", json=benign_record(), headers=SENSOR).json()["prediction_id"]
    r = client.post("/feedback", json={"prediction_id": pid, "label": 0}, headers=SENSOR)
    assert r.status_code == 403


# --- the v1 poisoning path is closed (MITRE ATLAS AML.T0020) ---------------

@pytest.mark.parametrize("field", [{"Label": 0}, {"attack_cat": "Normal"}])
def test_labels_in_scoring_payload_are_rejected(client, field):
    r = client.post("/predict", json={**attack_record(), **field}, headers=SENSOR)
    assert r.status_code == 422


def test_analyst_feedback_is_the_only_label_path_and_is_audited(client):
    pid = client.post("/predict", json=attack_record(), headers=SENSOR).json()["prediction_id"]
    with sqlite3.connect(client.log_db) as conn:
        assert conn.execute("SELECT true_label FROM predictions WHERE id=?",
                            (pid,)).fetchone()[0] is None
    r = client.post("/feedback", headers=ANALYST,
                    json={"prediction_id": pid, "label": 1, "attack_cat": "DoS"})
    assert r.status_code == 200 and r.json()["labeled_by"] == "alice"
    with sqlite3.connect(client.log_db) as conn:
        row = conn.execute("SELECT true_label, true_family, label_source, labeled_by,"
                           " sensor FROM predictions WHERE id=?", (pid,)).fetchone()
    assert row == (1, "DoS", "analyst", "alice", "sensor-1")


def test_feedback_on_unknown_prediction_is_404(client):
    r = client.post("/feedback", headers=ANALYST,
                    json={"prediction_id": 999_999, "label": 0})
    assert r.status_code == 404


def test_attack_feedback_needs_a_family(client):
    pid = client.post("/predict", json=attack_record(), headers=SENSOR).json()["prediction_id"]
    r = client.post("/feedback", headers=ANALYST, json={"prediction_id": pid, "label": 1})
    assert r.status_code == 422


# --- input validation (OWASP API3 / API8) ----------------------------------

@pytest.mark.parametrize("bad", [
    {"sbytes": -1},                     # negative counter
    {"sttl": 300},                      # TTL out of range
    {"dsport": 70000},                  # port out of range
    {"proto": "tcp; DROP TABLE x"},     # token pattern
    {"service": "x" * 100},             # length cap
    {"surprise": "field"},              # unknown field (log poisoning)
])
def test_invalid_records_are_rejected(client, bad):
    r = client.post("/predict", json={**benign_record(), **bad}, headers=SENSOR)
    assert r.status_code == 422


def test_nan_and_infinity_are_rejected(client):
    body = json.dumps(benign_record())[:-1] + ', "dur": NaN}'
    r = client.post("/predict", content=body, headers={
        **SENSOR, "Content-Type": "application/json"})
    assert r.status_code == 422


# --- resource consumption (OWASP API4, ATLAS AML.T0029) --------------------

def test_sequence_length_is_capped(client):
    flows = [benign_record()] * (api.MAX_SEQUENCE_FLOWS + 1)
    r = client.post("/predict/sequence", json={"flows": flows}, headers=SENSOR)
    assert r.status_code in (413, 422)


def test_oversized_body_is_rejected(artifacts_dir, tmp_path, monkeypatch):
    _env(monkeypatch, artifacts_dir, tmp_path, IDS_MAX_BODY_BYTES=1024)
    with TestClient(api.app) as c:
        flows = [benign_record()] * 20
        r = c.post("/predict/sequence", json={"flows": flows}, headers=SENSOR)
        assert r.status_code == 413


def test_rate_limit_returns_429_with_retry_after(artifacts_dir, tmp_path, monkeypatch):
    _env(monkeypatch, artifacts_dir, tmp_path, IDS_RATE_PER_MIN=1, IDS_RATE_BURST=3)
    with TestClient(api.app) as c:
        codes = [c.post("/predict", json=benign_record(), headers=SENSOR).status_code
                 for _ in range(5)]
        assert codes[:3] == [200, 200, 200] and codes[-1] == 429
        r = c.post("/predict", json=benign_record(), headers=SENSOR)
        assert int(r.headers["Retry-After"]) >= 1


# --- artifact integrity (ATLAS AML.T0010 / pickle RCE) ---------------------

def test_tampered_model_refuses_to_load(artifacts_dir, tmp_path, monkeypatch):
    tampered = tmp_path / "models"
    shutil.copytree(artifacts_dir, tampered)
    with open(tampered / "random_forest_best.joblib", "ab") as handle:
        handle.write(b"\x00")  # one byte appended
    _env(monkeypatch, tampered, tmp_path)
    with pytest.raises(Exception, match="sha256"):
        with TestClient(api.app):
            pass


def test_tampered_threshold_refuses_to_load(artifacts_dir, tmp_path, monkeypatch):
    """Raising the alert threshold silently blinds the IDS: it is pinned too."""
    tampered = tmp_path / "models"
    shutil.copytree(artifacts_dir, tampered)
    (tampered / "operating_point.json").write_text('{"threshold": 1.0}')
    _env(monkeypatch, tampered, tmp_path)
    with pytest.raises(Exception, match="operating_point.json"):
        with TestClient(api.app):
            pass


def test_swap_after_verification_is_not_loaded(artifacts_dir, tmp_path):
    """load_verified_pickle hashes the bytes it unpickles (no TOCTOU)."""
    from src.models.manifest import IntegrityError, load_verified_pickle

    models = tmp_path / "models"
    shutil.copytree(artifacts_dir, models)
    manifest = json.loads((models / "manifest.json").read_text())
    (models / "family_rf.joblib").write_bytes(b"not the pinned bytes")
    with pytest.raises(IntegrityError):
        load_verified_pickle(models, "family_rf.joblib", manifest)


# --- hardened served model (no TTL features) -------------------------------

@pytest.fixture
def hardened_dir(artifacts_dir, tmp_path):
    """Copy of the synthetic artifacts whose RF ignores the TTL features."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    from src.models.hardening import model_features
    from src.models.lstm_model import transform_flows
    from src.models.manifest import write_manifest
    from tests.conftest import synthetic_flows

    out = tmp_path / "hardened"
    shutil.copytree(artifacts_dir, out)
    names = json.loads((out / "preprocess_metadata.json").read_text())["feature_names"]
    feats = model_features(names)
    pipeline = joblib.load(out / "preprocess_pipeline.joblib")
    train = synthetic_flows(400, seed=11)
    X = pd.DataFrame(transform_flows(train, pipeline, names), columns=names)[feats]
    joblib.dump(RandomForestClassifier(n_estimators=20, random_state=0).fit(X, train["Label"]),
                out / "random_forest_best.joblib")
    (out / "operating_point.json").write_text(json.dumps(
        {"threshold": 0.5, "features": feats, "hardening": {"dropped_features": ["sttl"]}}))
    write_manifest(out)
    return out


def test_hardened_model_verdict_ignores_ttl(hardened_dir, tmp_path, monkeypatch):
    """The TTL skeleton key (reports/evasion.md) must not move the score."""
    _env(monkeypatch, hardened_dir, tmp_path)
    with TestClient(api.app) as c:
        scores = {ttl: c.post("/predict", json={**attack_record(), "sttl": ttl},
                              headers=SENSOR).json()
                  for ttl in (31, 62, 254)}
    assert len({round(r["p_attack"], 9) for r in scores.values()}) == 1
    assert scores[31]["prediction"] == 1
    drivers = {d["feature"] for d in scores[31]["alert"]["ml"]["drivers"]}
    assert not drivers & {"sttl", "dttl", "ct_state_ttl"}
