"""Attack harness tests: perturbations respect the threat model."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.attacks.evasion import Scored, perturb, random_search
from src.attacks.poisoning import poison_rows
from tests.conftest import synthetic_flows


@pytest.fixture
def flows() -> pd.DataFrame:
    return synthetic_flows(50).query("Label == 1").reset_index(drop=True)


def test_zero_budget_is_identity(flows):
    out = perturb(flows, 0, 0, 0)
    for col in ("sbytes", "Spkts", "dur", "Sload", "Dload", "smeansz",
                "Sintpkt", "Dintpkt", "sttl"):
        np.testing.assert_allclose(out[col], flows[col])


def test_perturbations_only_add_attacker_cost(flows):
    out = perturb(flows, pad=0.5, pkts=1.0, slow=2.0)
    assert (out["sbytes"] >= flows["sbytes"]).all()
    assert (out["Spkts"] >= flows["Spkts"]).all()
    assert (out["dur"] >= flows["dur"]).all()
    # destination side is not attacker-controlled
    for col in ("dbytes", "Dpkts", "dttl"):
        np.testing.assert_array_equal(out[col], flows[col])


def test_derived_fields_stay_consistent(flows):
    out = perturb(flows, pad=1.0, pkts=0.0, slow=1.0)
    np.testing.assert_allclose(out["smeansz"], flows["smeansz"] * 2)  # 2x bytes, same pkts
    # 2x bytes over 2x time -> same source load
    np.testing.assert_allclose(out["Sload"], flows["Sload"], rtol=1e-9)
    np.testing.assert_allclose(out["Dload"], flows["Dload"] / 2)


def test_ttl_rewrite_is_opt_in(flows):
    assert (perturb(flows, 0, 0, 0)["sttl"] == flows["sttl"]).all()
    assert (perturb(flows, 0, 0, 0, ttl=31)["sttl"] == 31).all()


def test_random_search_evades_a_ttl_only_model(artifacts_dir, flows):
    """The synthetic RF keys on sttl/sbytes; TTL rewriting must beat it
    while a zero budget without TTL must not."""
    pipeline = joblib.load(artifacts_dir / "preprocess_pipeline.joblib")
    names = json.loads((artifacts_dir / "preprocess_metadata.json").read_text())["feature_names"]
    rf = joblib.load(artifacts_dir / "random_forest_best.joblib")
    scorer = Scored(rf, names, ["sttl"], 0.5)
    # refit on sttl alone so the toy model is TTL-only
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

    from src.models.lstm_model import transform_flows  # noqa: PLC0415
    train = synthetic_flows(400, seed=3)
    X = transform_flows(train, pipeline, names)
    scorer.model = RandomForestClassifier(n_estimators=10, random_state=0).fit(
        X[:, scorer.idx], train["Label"])
    assert not random_search(flows, scorer, pipeline, names, 0.0, 3, False, 0).any()
    assert random_search(flows, scorer, pipeline, names, 0.0, 3, True, 0).mean() > 0.9


def test_poison_rows_are_target_family_labelled_benign():
    X = pd.DataFrame({"a": range(10)})
    fam = pd.Series(["DoS"] * 5 + ["Backdoor"] * 5)
    rows, labels = poison_rows(X, fam, "Backdoor", 7, seed=0)
    assert len(rows) == 7 and (labels == 0).all()
    assert set(rows["a"]) <= {5, 6, 7, 8, 9}
    assert poison_rows(X, fam, "Worms", 3, seed=0)[0].empty


# --- hardening recipe (src/models/hardening.py) ----------------------------

def test_model_features_drop_ttl_only():
    from src.models.hardening import TTL_FEATURES, model_features

    names = ["sbytes", "sttl", "dttl", "ct_state_ttl", "proto"]
    assert model_features(names) == ["sbytes", "proto"]
    assert model_features(names, hardened=False) == names
    assert not set(model_features(names)) & set(TTL_FEATURES)


def test_harden_training_set_adds_labelled_attack_copies(artifacts_dir):
    from src.models.hardening import harden_training_set
    from src.models.lstm_model import transform_flows

    pipeline = joblib.load(artifacts_dir / "preprocess_pipeline.joblib")
    names = json.loads((artifacts_dir / "preprocess_metadata.json").read_text())["feature_names"]
    raw = synthetic_flows(200, seed=7)
    X = pd.DataFrame(transform_flows(raw, pipeline, names), columns=names)
    X_h, y_h, info = harden_training_set(X, raw["Label"], raw, pipeline, names, seed=0)
    n_attacks = int((raw["Label"] == 1).sum())
    assert len(X_h) == len(X) + n_attacks and len(y_h) == len(X_h)
    assert (y_h.iloc[len(X):] == 1).all()
    assert "sttl" not in X_h.columns
    assert info["adversarial_training"]["rows"] == n_attacks
    # deterministic: same seed, same augmentation (retrain parity relies on it)
    X_h2, _, _ = harden_training_set(X, raw["Label"], raw, pipeline, names, seed=0)
    pd.testing.assert_frame_equal(X_h, X_h2)
