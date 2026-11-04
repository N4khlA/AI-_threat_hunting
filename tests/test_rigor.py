"""Split protocols, SOC metrics and the LSTM window split: the claims
behind the README's numbers, as falsifiable tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splits import assert_disjoint, feature_columns, split_positions
from src.evaluation.metrics import (
    base_rate_table,
    binary_report,
    bootstrap_ci,
    per_family_recall,
    rates,
    threshold_for_fpr,
)
from src.models.lstm_model import build_windows, shared_flows, temporal_holdout


def _table(n_train: int = 400, n_test: int = 200, dup_share: float = 0.4,
           seed: int = 0) -> pd.DataFrame:
    """Combined-style table: first n_train rows = 'official training file'."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    df = pd.DataFrame({"a": rng.integers(0, 10_000, n), "b": rng.random(n),
                       "proto": rng.choice(["tcp", "udp"], n)})
    df["Label"] = rng.integers(0, 2, n)
    df["attack_cat"] = np.where(df["Label"] == 1, "DoS", "Normal")
    # make dup_share of the rows exact copies of earlier rows
    k = int(dup_share * n)
    src = rng.integers(0, n, k)
    dst = rng.choice(np.arange(n), k, replace=False)
    df.iloc[dst] = df.iloc[src].to_numpy()
    df["Label"] = df["Label"].astype(int)
    return df


def _cross_split_twins(df, parts) -> int:
    cols = feature_columns(df)
    train = set(map(tuple, df.loc[parts["train"], cols].astype(str).to_numpy()))
    return sum(tuple(r) in train for r in df.loc[parts["test"], cols].astype(str).to_numpy())


def test_random_split_leaks_duplicates_across_the_boundary():
    df = _table()
    parts, _ = split_positions(df, "random", seed=1, n_official_train=400)
    assert_disjoint(parts)  # disjoint *positions* ...
    assert _cross_split_twins(df, parts) > 0  # ... but identical *flows*


def test_official_novel_has_no_test_flow_seen_in_training():
    df = _table()
    parts, stats = split_positions(df, "official_novel", n_official_train=400)
    assert_disjoint(parts)
    assert _cross_split_twins(df, parts) == 0
    assert parts["test"].min() >= 400  # test only from the testing file
    assert parts["train"].max() < 400 and parts["val"].max() < 400
    assert stats["dropped_test_seen_in_train"] > 0


def test_official_keeps_whole_test_file():
    df = _table()
    parts, _ = split_positions(df, "official", n_official_train=400)
    assert list(parts["test"]) == list(range(400, 600))


def test_unknown_protocol_is_rejected():
    with pytest.raises(ValueError):
        split_positions(_table(), "kfold")


# --- metrics ---------------------------------------------------------------

def test_rates_on_known_confusion():
    r = rates(np.array([1, 1, 0, 0, 0]), np.array([1, 0, 1, 0, 0]))
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 2)
    assert r["tpr"] == 0.5 and r["fpr"] == pytest.approx(1 / 3)


def test_base_rate_precision_matches_bayes():
    row = base_rate_table(tpr=0.9, fpr=0.01, prevalences=(0.001,),
                          flows_per_day=1_000_000)[0]
    expected = 0.9 * 0.001 / (0.9 * 0.001 + 0.01 * 0.999)
    assert row["alert_precision"] == pytest.approx(expected)
    assert row["false_alerts_per_day"] == round(0.01 * 0.999 * 1_000_000)
    assert row["alert_precision"] < 0.1  # 99% "accurate" IDS, >90% false alerts


def test_threshold_for_fpr_respects_budget():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    thr = threshold_for_fpr(y, p, 0.05)
    assert rates(y, p >= thr)["fpr"] <= 0.05 + 1e-9


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    point = binary_report(y, p)["roc_auc"]
    lo, hi = bootstrap_ci(y, p, ("roc_auc",), n_boot=100)["roc_auc"]
    assert lo <= point <= hi and hi - lo < 0.2


def test_per_family_recall_ignores_normal():
    out = per_family_recall(np.array([1, 0, 1, 1]), np.array(["DoS", "DoS", "Worms", "Normal"]))
    assert out == {"DoS": 0.5, "Worms": 1.0}


# --- LSTM window split -----------------------------------------------------

def _flows(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"proto": rng.choice(["tcp", "udp"], n),
                         "service": rng.choice(["-", "dns"], n),
                         "Label": rng.integers(0, 2, n)})


def test_random_window_split_shares_flows():
    """Documents the legacy leak: overlapping windows split at random."""
    w = build_windows(_flows(), ["proto", "service"], window=10, stride=5, label_mode="any")
    rng = np.random.default_rng(0)
    mask = rng.random(len(w)) < 0.7
    assert shared_flows(w[mask], w[~mask]) > 0


def test_temporal_holdout_shares_no_flows():
    w = build_windows(_flows(), ["proto", "service"], window=10, stride=5, label_mode="any")
    early, late = temporal_holdout(w, 0.15)
    assert len(early) and len(late)
    assert shared_flows(early, late) == 0


def test_serving_transform_matches_training_transform():
    """Train/serve skew guard: the API's transform_flows must reproduce
    preprocess.build_splits exactly (it once misrouted JSON nulls)."""
    from src.features.preprocess import build_splits
    from src.models.lstm_model import transform_flows
    from tests.conftest import synthetic_flows

    df = synthetic_flows(300, seed=5)
    parts = {"train": df.index[:200], "val": df.index[200:250], "test": df.index[250:]}
    data = build_splits(df, parts, imbalance="none")
    pipeline = {"imputers": data["imputers"], "encoder": data["encoder_state"],
                "scaler": data["scaler"]}
    names = data["X_train"].columns.tolist()
    served = transform_flows(df.loc[parts["test"]], pipeline, names)
    np.testing.assert_allclose(served, data["X_test"][names].to_numpy(), atol=1e-5)
