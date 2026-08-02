"""Tests for the evaluation and leakage-check modules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.leakage_check import prepare, scan  # noqa: E402
from src.evaluation.evaluate import per_family_f1  # noqa: E402
from src.evaluation.per_class_metrics import per_family_table  # noqa: E402


@pytest.fixture
def mini_labels() -> tuple[pd.Series, pd.Series, pd.Series]:
    y = pd.Series([0] * 60 + [1] * 40)
    pred = pd.Series([0] * 54 + [1] * 6 + [1] * 38 + [0] * 2)
    fam = pd.Series(
        ["Normal"] * 60
        + ["Generic"] * 20 + ["Worms"] * 10 + ["Fuzzers"] * 10
    )
    return y, pred, fam


def test_per_family_table_counts(mini_labels):
    y, pred, fam = mini_labels
    table = per_family_table(y, pred, fam)
    rows = table.set_index("family")
    assert rows.loc["Normal", "detection_rate"] == 0.0     # no attack records
    assert rows.loc["Normal", "false_positive_rate"] == pytest.approx(0.1)  # 6/60
    assert rows.loc["Generic", "detection_rate"] == 1.0
    assert rows.loc["Worms", "detection_rate"] == 1.0
    assert rows.loc["Worms", "records"] == 10
    assert rows.loc["Normal", "attack_records"] == 0
    assert (table["records"] == [60, 20, 10, 10]).all()


@pytest.fixture
def leaky_df() -> pd.DataFrame:
    rng = np.random.RandomState(3)
    n = 400
    label = rng.randint(0, 2, n)
    return pd.DataFrame(
        {
            "srcip": [f"1.2.3.{rng.randint(1, 255)}" for _ in range(n)],
            "Label": label,
            "attack_cat": ["Normal" if l == 0 else "DoS" for l in label],
            "copy_of_label": label,          # a would-be leak: equals the label
            "noise": rng.randn(n),
            "proto": rng.choice(["tcp", "udp"], n),
        }
    )


def test_prepare_drops_identity_and_keeps_no_labels(leaky_df):
    X, y, dropped = prepare(leaky_df)
    assert "srcip" in dropped
    assert "Label" not in X.columns and "attack_cat" not in X.columns
    assert set(X.columns) == {"copy_of_label", "noise", "proto"}
    assert (y.values == leaky_df["Label"].values).all()


def test_scan_flags_exact_label_copy(leaky_df):
    X, y, _ = prepare(leaky_df)
    scores = scan(X, y, sample=400, seed=0)
    flagged = scores.set_index("feature").loc["copy_of_label", "flag"]
    assert flagged == "LEAK-LIKE"
    assert scores.set_index("feature").loc["noise", "flag"] == ""
    assert scores.set_index("feature").loc["copy_of_label", "auc"] == pytest.approx(1.0)


def test_per_family_f1_one_vs_rest(mini_labels):
    y, pred, fam = mini_labels
    table = per_family_f1(y.to_numpy(), pred.to_numpy(), fam)
    rows = table.set_index("family")
    # Fuzzers rows are indices 90-99; pred=1 for 90-97 only -> 8/10 caught
    assert rows.loc["Fuzzers", "recall"] == pytest.approx(0.8)
    assert rows.loc["Generic", "recall"] == pytest.approx(1.0)
    assert rows.loc["Worms", "support"] == 10
    assert rows.loc["Normal", "support"] == 0                   # no attack rows
    # Worms all caught; 44 attack flags total, 10 of them Worms -> 10/44
    assert rows.loc["Worms", "precision"] == pytest.approx(10 / 44)
    assert rows.loc["Worms", "recall"] == pytest.approx(1.0)
