"""Tests for the retraining script: compare-and-replace guard,
pseudo-label handling, and deduplication of harvested predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.retrain import (  # noqa: E402
    LOG_COLUMNS,
    apply_labels,
    dedupe_new_rows,
    should_replace,
)

OLD = {"accuracy": 0.9000, "f1_macro": 0.8800}


def test_should_replace_when_accuracy_improves():
    new = {"accuracy": 0.9100, "f1_macro": 0.8800}
    replace, reason = should_replace(OLD, new, min_gain=0.001)
    assert replace is True
    assert "accuracy" in reason


def test_should_refuse_when_worse():
    new = {"accuracy": 0.8900, "f1_macro": 0.8700}
    replace, reason = should_replace(OLD, new, min_gain=0.001)
    assert replace is False
    assert "not better" in reason
    assert "0.9000" in reason and "0.8900" in reason  # prints both scores


def test_should_refuse_on_tie_without_f1_gain():
    new = {"accuracy": 0.9004, "f1_macro": 0.8790}  # within min_gain, f1 lower
    replace, _ = should_replace(OLD, new, min_gain=0.001)
    assert replace is False


def test_should_replace_on_tie_with_f1_gain():
    new = {"accuracy": 0.9004, "f1_macro": 0.9000}  # within min_gain, f1 better
    replace, reason = should_replace(OLD, new, min_gain=0.001)
    assert replace is True
    assert "F1-macro" in reason


def test_should_replace_respects_min_gain():
    new = {"accuracy": 0.9008, "f1_macro": 0.8800}
    assert should_replace(OLD, new, min_gain=0.001)[0] is False
    assert should_replace(OLD, new, min_gain=0.0005)[0] is True


def _log_row(true_label, confidence, prediction, family=None, source=None):
    return {
        "ts": 1.0, "model": "rf", "prediction": prediction,
        "confidence": confidence, "true_label": true_label,
        "predicted_family": family, "true_family": None,
        "n_flows": 1,
        "record_json": json.dumps({"proto": "tcp", "sbytes": 100}),
        "p_attack": confidence if prediction == 1 else 1 - confidence,
        "label_source": source if source else ("analyst" if true_label is not None else None),
        "labeled_by": None,
    }


def test_apply_labels_confirmed_and_pseudo():
    df = pd.DataFrame([
        _log_row(true_label=1, confidence=0.99, prediction=1, family="Generic"),
        _log_row(true_label=None, confidence=0.99, prediction=1, family="Fuzzers"),
        _log_row(true_label=None, confidence=0.50, prediction=1, family="DoS"),
    ], columns=LOG_COLUMNS)
    flows, stats = apply_labels(df, min_confidence=0.95, labeled_only=False)
    assert len(flows) == 2
    assert stats == {"confirmed": 1, "pseudo": 1, "dropped_low_confidence": 1,
                     "dropped_untrusted_label": 0}
    labels = sorted(int(f["Label"]) for f in flows.to_dict("records"))
    assert labels == [1, 1]
    pseudo = flows.iloc[1]
    assert pseudo["attack_cat"] == "Fuzzers"


def test_apply_labels_labeled_only():
    df = pd.DataFrame([
        _log_row(true_label=1, confidence=0.50, prediction=1),
        _log_row(true_label=None, confidence=0.99, prediction=1),
    ], columns=LOG_COLUMNS)
    flows, stats = apply_labels(df, min_confidence=0.95, labeled_only=True)
    assert len(flows) == 1
    assert stats["confirmed"] == 1 and stats["pseudo"] == 0
    assert flows.iloc[0]["Label"] == 1


def test_apply_labels_normal_pseudo_label():
    df = pd.DataFrame([
        _log_row(true_label=None, confidence=0.98, prediction=0),
    ], columns=LOG_COLUMNS)
    flows, _ = apply_labels(df, min_confidence=0.95, labeled_only=False)
    assert int(flows.iloc[0]["Label"]) == 0
    assert flows.iloc[0]["attack_cat"] == "Normal"


def test_dedupe_new_rows_drops_exact_duplicates():
    base = pd.DataFrame([{"proto": "tcp", "sbytes": 100, "Label": 1},
                         {"proto": "udp", "sbytes": 200, "Label": 0}])
    new = pd.DataFrame([{"proto": "tcp", "sbytes": 100, "Label": 1},
                        {"proto": "tcp", "sbytes": 999, "Label": 1}])
    kept = dedupe_new_rows(new, base)
    assert len(kept) == 1
    assert kept.iloc[0]["sbytes"] == 999


def test_dedupe_new_rows_handles_empty():
    empty = pd.DataFrame(columns=["proto", "sbytes"])
    assert dedupe_new_rows(empty, pd.DataFrame()).empty


def test_dedupe_matches_nan_base_columns_despite_dtype_mismatch():
    """Base CSV columns are float64 (all-NaN); logged JSON rows carry None
    (object dtype). Exact duplicates must still be dropped."""
    base = pd.DataFrame({
        "srcip": pd.Series([float("nan"), float("nan")], dtype=float),
        "proto": ["tcp", "udp"],
        "sbytes": [100, 200],
    })
    new = pd.DataFrame({
        "srcip": [None, None],
        "proto": ["tcp", "tcp"],
        "sbytes": [100, 999],
    })
    kept = dedupe_new_rows(new, base)
    assert len(kept) == 1
    assert kept.iloc[0]["sbytes"] == 999


def test_apply_labels_ignores_labels_not_from_an_analyst():
    """Regression for the v1 poisoning path: a label that did not come
    through the authenticated /feedback endpoint must never be trained on."""
    df = pd.DataFrame([
        _log_row(true_label=0, confidence=0.99, prediction=1, source="sensor"),
        _log_row(true_label=0, confidence=0.99, prediction=1, source=None)
        | {"label_source": None},
        _log_row(true_label=1, confidence=0.99, prediction=1, family="DoS"),
    ], columns=LOG_COLUMNS)
    flows, stats = apply_labels(df, min_confidence=0.95, labeled_only=True)
    assert stats["dropped_untrusted_label"] == 2
    assert stats["confirmed"] == 1 and len(flows) == 1


def test_guard_refuses_targeted_family_regression():
    """Aggregate accuracy improves, but one family's recall collapses:
    exactly what a targeted poisoning attack looks like."""
    old = {**OLD, "family_recall": {"Backdoor": 0.95, "DoS": 0.90}}
    new = {"accuracy": 0.9200, "f1_macro": 0.9000,
           "family_recall": {"Backdoor": 0.40, "DoS": 0.91}}
    replace, reason = should_replace(old, new, min_gain=0.001)
    assert replace is False
    assert "Backdoor" in reason and "poisoning" in reason


def test_guard_tolerates_small_family_noise():
    old = {**OLD, "family_recall": {"Backdoor": 0.95}}
    new = {"accuracy": 0.9100, "f1_macro": 0.8900, "family_recall": {"Backdoor": 0.94}}
    assert should_replace(old, new, min_gain=0.001)[0] is True
