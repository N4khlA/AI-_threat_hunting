"""Tests for LSTM sequence construction (grouping, windows, label modes)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.lstm_model import (  # noqa: E402
    build_windows,
    detect_group_key,
    windows_to_sequences,
)


@pytest.fixture
def canonical_flows() -> pd.DataFrame:
    """Flows with real srcip/dstip/Stime (canonical schema)."""
    return pd.DataFrame(
        {
            "srcip": ["1.1.1.1"] * 7 + ["2.2.2.2"] * 3,
            "dstip": ["9.9.9.9"] * 7 + ["8.8.8.8"] * 3,
            "Stime": [1, 3, 2, 5, 4, 6, 7, 10, 9, 8],  # deliberately unsorted
            "proto": ["tcp"] * 10,
            "Label":   [0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
        }
    )


@pytest.fixture
def keyless_flows() -> pd.DataFrame:
    """Cleaned-variant schema: no IPs, no ports, no timestamps."""
    return pd.DataFrame(
        {
            "proto": ["tcp"] * 5 + ["udp"] * 5,
            "service": ["http"] * 5 + ["dns"] * 5,
            "Label": [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        }
    )


def test_detect_group_key_prefers_pair(canonical_flows):
    assert detect_group_key(canonical_flows) == ["srcip", "dstip"]


def test_detect_group_key_falls_back_to_service(keyless_flows):
    assert detect_group_key(keyless_flows) == ["proto", "service"]


def test_detect_group_key_forced_missing_key(keyless_flows):
    with pytest.raises(SystemExit):
        detect_group_key(keyless_flows, forced="pair")


def test_build_windows_sorts_by_time_and_slides(canonical_flows):
    windows = build_windows(canonical_flows, ["srcip", "dstip"],
                            window=3, stride=2, label_mode="any")
    # group 1 has 7 flows -> windows start at 0,2,4,6  (4 windows)
    # group 2 has 3 flows -> windows start at 0,2      (2 windows)
    assert len(windows) == 6
    g1 = windows[windows["group"] == "('1.1.1.1', '9.9.9.9')"]
    assert g1["n_flows"].tolist() == [3, 3, 3, 1]
    # first window covers the 3 earliest flows (Stime 1,2,3 -> labels 0,0,1)
    assert windows.iloc[0]["label"] == 1
    assert windows.iloc[0]["flow_index"] == [0, 2, 1]


def test_window_label_modes(canonical_flows):
    df = canonical_flows
    any_win = build_windows(df, ["srcip", "dstip"], window=3, stride=1, label_mode="any")
    maj_win = build_windows(df, ["srcip", "dstip"], window=3, stride=1, label_mode="majority")
    last_win = build_windows(df, ["srcip", "dstip"], window=3, stride=1, label_mode="last")
    assert any_win["label"].sum() >= maj_win["label"].sum()
    assert maj_win["label"].sum() >= last_win["label"].sum()


def test_majority_tie_goes_attack():
    df = pd.DataFrame({
        "srcip": ["a"] * 4, "dstip": ["b"] * 4, "Stime": [1, 2, 3, 4],
        "Label": [1, 0, 1, 0],
    })
    windows = build_windows(df, ["srcip", "dstip"], window=4, stride=4,
                            label_mode="majority")
    assert windows.iloc[0]["label"] == 1  # 2 vs 2 -> attack


def test_windows_to_sequences_pads_and_truncates():
    windows = pd.DataFrame({
        "flow_index": [[0, 1], [2, 3, 4], [5]],  # lengths 2, 3, 1
        "label": [0, 1, 0],
    })
    features = np.arange(18, dtype=np.float32).reshape(6, 3)
    seqs = windows_to_sequences(windows, features, maxlen=2)
    assert seqs.shape == (3, 2, 3)
    assert seqs[0][0].tolist() == features[0].tolist()   # pre-padding: kept tail
    assert seqs[1][0].tolist() == features[3].tolist()   # truncate 'pre' keeps tail
    assert seqs[2][0].tolist() == [0.0, 0.0, 0.0]        # padded with zeros
