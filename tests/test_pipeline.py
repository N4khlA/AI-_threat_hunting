"""Smoke/unit tests for the feature engineering and preprocessing modules.

Run from the project root:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.extract import (  # noqa: E402
    EXTRACTORS,
    byte_counts,
    extract_features,
    forward_backward_ratios,
    inter_arrival_stats,
    packet_size_stats,
    payload_entropy,
    port_features,
    protocol_one_hot,
    shannon_entropy,
    duration_features,
)
from src.features.preprocess import (  # noqa: E402
    encode_categorical,
    impute,
    scale,
    stratified_split,
)


@pytest.fixture
def raw_flows() -> pd.DataFrame:
    """A tiny synthetic flow table resembling UNSW-NB15 structure."""
    return pd.DataFrame(
        {
            "spkts": [10, 100, 1], "dpkts": [10, 50, 0],
            "sbytes": [1000, 50000, 64], "dbytes": [9000, 5000, 0],
            "Sintpkt": [0.1, 0.01, np.nan], "Dintpkt": [0.2, 0.02, np.nan],
            "Sjit": [0.01, 0.001, 0.0], "Djit": [0.02, 0.002, 0.0],
            "dur": [1.0, 5.0, 0.0], "proto": ["tcp", "udp", "icmp"],
            "sport": [443, 1024, 0], "dsport": [50000, 53, 8],
            "Sload": [1000, 10000, 64], "Dload": [9000, 1000, 0],
            "smeansz": [100, 500, 64], "dmeansz": [900, 100, np.nan],
        }
    )


# ---------------------------------------------------------------------------
# extract.py
# ---------------------------------------------------------------------------

def test_extract_features_returns_new_columns(raw_flows):
    augmented, names = extract_features(raw_flows)
    assert len(names) > 20
    assert set(names).issubset(augmented.columns)
    assert augmented.index.equals(raw_flows.index)


def test_packet_size_stats(raw_flows):
    out = packet_size_stats(raw_flows)
    assert {"pkt_size_mean", "pkt_size_min", "pkt_size_max", "pkt_size_std"} <= set(out.columns)
    assert (out["pkt_size_mean"] >= 0).all()


def test_inter_arrival_stats(raw_flows):
    out = inter_arrival_stats(raw_flows)
    assert {"iat_mean", "iat_std", "jit_mean"} <= set(out.columns)
    assert out["iat_mean"].iloc[0] == pytest.approx(0.15)


def test_byte_counts_and_duration(raw_flows):
    out = byte_counts(raw_flows)
    assert out["total_bytes"].iloc[0] == 10_000
    assert out["bytes_per_pkt"].iloc[0] == pytest.approx(500.0)
    dur = duration_features(out.merge(raw_flows[["dur"]], left_index=True, right_index=True))
    assert dur["duration"].iloc[0] == 1.0


def test_port_features(raw_flows):
    out = port_features(raw_flows)
    assert {"sport", "dsport", "sport_well_known", "dsport_well_known"} <= set(out.columns)
    assert out["sport_well_known"].iloc[0] == 1  # 443
    assert out["dsport_well_known"].iloc[0] == 0  # 50000


def test_protocol_one_hot(raw_flows):
    out = protocol_one_hot(raw_flows)
    assert list(out.columns) == ["proto_tcp", "proto_udp", "proto_icmp", "proto_other"]
    assert out.sum(axis=1).eq(1).all()


def test_forward_backward_ratios(raw_flows):
    out = forward_backward_ratios(raw_flows)
    assert out["fwd_byte_ratio"].iloc[0] == pytest.approx(1000 / 10000)
    assert out["fwd_load_ratio"].iloc[0] == pytest.approx(1000 / 10000)


def test_shannon_entropy():
    assert shannon_entropy(b"\x00\x00\x00\x00") == pytest.approx(0.0)
    assert shannon_entropy(bytes(range(256))) == pytest.approx(8.0)
    assert np.isnan(shannon_entropy(b""))


def test_payload_entropy_missing_column(raw_flows):
    out = payload_entropy(raw_flows)  # no 'payload' column in synthetic data
    assert out.empty or out["payload_entropy"].isna().all()


def test_extractors_all_run(raw_flows):
    for name, extractor in EXTRACTORS:
        out = extractor(raw_flows)
        assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# preprocess.py
# ---------------------------------------------------------------------------

@pytest.fixture
def messy_df() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 120
    return pd.DataFrame(
        {
            "a": rng.randn(n),
            "b": np.where(rng.rand(n) < 0.3, np.nan, rng.rand(n) * 10),
            "cat": np.random.choice(["x", "y", "z"], n),
            "proto": np.random.choice(["tcp", "udp"], n),
        }
    )


def test_stratified_split_preserves_balance(messy_df):
    y = pd.Series([0] * 100 + [1] * 20, index=messy_df.index)
    Xtr, Xva, Xte, ytr, yva, yte = stratified_split(messy_df, y, seed=42)
    for part in (ytr, yva, yte):
        assert part.min() >= 0 and part.max() <= 1
    ratios = [part.mean() for part in (ytr, yva, yte)]
    assert max(ratios) - min(ratios) < 0.1
    assert len(Xtr) / len(y) == pytest.approx(0.70, abs=0.01)
    assert len(Xva) / len(y) == pytest.approx(0.15, abs=0.01)


def test_impute_fills_missing(messy_df):
    Xtr, Xva, Xte, imputers = impute(*[f.copy() for f in (messy_df,) * 3])
    assert not Xtr["b"].isna().any()
    assert Xtr[["a", "b"]].equals(Xva[["a", "b"]].astype(Xtr[["a", "b"]].dtypes)
                                  if Xva["b"].isna().sum() == 0 else Xtr[["a", "b"]])


def test_encode_categorical_label_mode(messy_df):
    Xtr, Xva, Xte, state = encode_categorical(
        *[f.copy() for f in (messy_df,) * 3], mode="label")
    assert Xtr["cat"].dtype.kind in "ifu"
    assert state["mode"] == "label"
    assert set(state["objects"]) == {"cat", "proto"}


def test_encode_categorical_onehot_mode(messy_df):
    Xtr, Xva, Xte, state = encode_categorical(
        *[f.copy() for f in (messy_df,) * 3], mode="onehot")
    assert not {"cat", "proto"} <= set(Xtr.columns)
    assert {"cat_x", "cat_y", "cat_z", "proto_tcp", "proto_udp"} <= set(Xtr.columns)
    assert state["mode"] == "onehot"


def test_scale_standardizes(messy_df):
    Xtr, Xva, Xte, scaler = scale(*[f.copy() for f in (messy_df,) * 3])
    for col in ["a", "b"]:
        assert np.allclose(Xtr[col].mean(), 0, atol=1e-9)
        assert np.allclose(Xtr[col].std(ddof=0), 1, atol=1e-9)
    assert hasattr(scaler, "mean_")
