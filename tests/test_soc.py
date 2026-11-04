"""SOC outputs: exact explanations, severity, ATT&CK mapping, alert shape."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.soc.alerts import artifact_share, build_alert, severity
from src.soc.attack_mapping import ATTACK_MAP, techniques_for
from src.soc.explain import path_contributions, top_drivers

UNSW_FAMILIES = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
                 "Reconnaissance", "Shellcode", "Worms"]


@pytest.fixture(scope="module")
def forest():
    rng = np.random.default_rng(0)
    X = rng.random((400, 5))
    y = (X[:, 0] + 0.5 * X[:, 3] > 0.8).astype(int)
    return RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y), X


def test_contributions_reconstruct_predict_proba_exactly(forest):
    model, X = forest
    bias, contrib = path_contributions(model, X[:50])
    np.testing.assert_allclose(bias + contrib.sum(axis=1),
                               model.predict_proba(X[:50])[:, 1], atol=1e-9)


def test_contributions_find_the_real_signal(forest):
    model, X = forest
    _, contrib = path_contributions(model, X[:200])
    ranking = np.argsort(-np.abs(contrib).mean(axis=0))
    assert set(ranking[:2]) == {0, 3}  # the two features y depends on


def test_top_drivers_signed_and_sorted():
    drivers = top_drivers(np.array([0.1, -0.3, 0.0, 0.2]), ["a", "b", "c", "d"],
                          {"b": 7}, k=3)
    assert [d["feature"] for d in drivers] == ["b", "d", "a"]
    assert drivers[0]["direction"] == "towards benign" and drivers[0]["value"] == 7


@pytest.mark.parametrize("p, sev", [(0.3, None), (0.6, "low"), (0.85, "medium"),
                                    (0.99, "high")])
def test_severity_bands(p, sev):
    assert severity(p, threshold=0.5) == sev


def test_severity_is_driven_by_attack_probability_not_confidence():
    """v1 used max(p, 1-p): a confident *benign* verdict scored HIGH."""
    assert severity(0.02) is None


def test_every_unsw_family_has_an_explicit_mapping():
    assert set(ATTACK_MAP) == set(UNSW_FAMILIES)
    for fam, entry in ATTACK_MAP.items():
        assert entry["fidelity"] in {"high", "medium", "low", "none"}
        assert entry["next_steps"], fam
        for tid, _ in entry["techniques"]:
            assert tid.startswith("T1") and len(tid) in (5, 9), tid


def test_generic_is_unmapped_rather_than_forced():
    assert techniques_for("Generic")["techniques"] == []
    assert techniques_for(None)["fidelity"] == "none"


def test_alert_flags_ttl_artifact_driven_scores():
    drivers = [{"feature": "sttl", "contribution": 0.3, "value": 254,
                "direction": "towards attack"},
               {"feature": "sbytes", "contribution": 0.1, "value": 9e4,
                "direction": "towards attack"}]
    assert artifact_share(drivers) == pytest.approx(0.75)
    alert = build_alert(7, {"proto": "tcp", "service": "http", "state": "FIN"},
                        0.97, {"DoS": 0.8, "Exploits": 0.2}, drivers)
    assert alert["ml"]["artifact_driven"] is True
    assert "testbed artifact" in alert["triage_note"]
    assert alert["threat"]["technique"][0]["id"] == "T1498"
    assert alert["event"]["reference"] == "prediction:7"


def test_no_alert_below_threshold():
    assert build_alert(1, {}, 0.4, {}, [], threshold=0.5) is None
