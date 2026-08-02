"""LSTM sequence-windowing: exact window counts and label distributions
on a small synthetic flow set (no TensorFlow needed - pure pandas)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.lstm_model import build_windows  # noqa: E402


@pytest.fixture
def synthetic_flows() -> pd.DataFrame:
    """One group of 13 flows with a known attack pattern.

    Labels (13): 0 0 0 0 1 | 0 0 0 0 1 | 0 0 0
    """
    labels = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    return pd.DataFrame({
        "proto": ["tcp"] * 13,
        "service": ["http"] * 13,
        "Label": labels,
    })


def test_window_count_follows_stride(synthetic_flows):
    """window=5, stride=3 over 13 flows -> starts at 0,3,6,9,12 -> 5 windows."""
    windows = build_windows(synthetic_flows, ["proto", "service"],
                            window=5, stride=3, label_mode="any")
    assert len(windows) == 5
    assert windows["n_flows"].tolist() == [5, 5, 5, 4, 1]
    assert windows["start"].tolist() == [0, 3, 6, 9, 12]


def test_window_count_sums_over_groups():
    """Two groups of 13 and 7 flows, stride 3 -> 5 + 3 = 8 windows."""
    df = pd.DataFrame({
        "proto": ["tcp"] * 13 + ["udp"] * 7,
        "service": ["http"] * 13 + ["dns"] * 7,
        "Label": [0] * 20,
    })
    windows = build_windows(df, ["proto", "service"], window=5, stride=3,
                            label_mode="any")
    assert len(windows) == 8


def test_window_label_distribution_any(synthetic_flows):
    """'any' mode: attack if at least one flow in the window is an attack.

    window=5, stride=3 windows over
        0 0 0 0 1 | 0 0 0 0 1 | 0 0 0
    -> [0:5]=[0,0,0,0,1]->1  [3:8]=[0,1,0,0,0]->1  [6:11]=[0,0,0,1,0]->1
       [9:13]=[1,0,0,0]->1   [12:13]=[0]->0
    """
    windows = build_windows(synthetic_flows, ["proto", "service"],
                            window=5, stride=3, label_mode="any")
    assert windows["label"].tolist() == [1, 1, 1, 1, 0]


def test_window_label_distribution_majority():
    """'majority' mode: attack iff attacks outnumber normals in the window.

    Labels: 1 1 1 | 0 0 1 | 1 1 0 | 0 1 0 | 1   (window=3, stride=3)
    ->      [1,1,1]->1  [0,0,1]->0  [1,1,0]->1  [0,1,0]->0  [1]->1
    """
    df = pd.DataFrame({
        "proto": ["tcp"] * 13,
        "service": ["http"] * 13,
        "Label": [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
    })
    windows = build_windows(df, ["proto", "service"], window=3, stride=3,
                            label_mode="majority")
    assert windows["label"].tolist() == [1, 0, 1, 0, 1]


def test_windows_cover_every_flow_once_per_position(synthetic_flows):
    """Each flow index appears in at least one window's flow_index."""
    windows = build_windows(synthetic_flows, ["proto", "service"],
                            window=5, stride=3, label_mode="any")
    covered = set()
    for _, win in windows.iterrows():
        covered.update(win["flow_index"])
    assert covered == set(range(13))


def test_window_labels_match_hand_computed_any_mode():
    """Sanity: hand-computed 'any' labels for a tiny mixed window."""
    df = pd.DataFrame({
        "proto": ["tcp"] * 6,
        "service": ["http"] * 6,
        "Label": [0, 0, 1, 0, 0, 0],
    })
    windows = build_windows(df, ["proto", "service"], window=3, stride=2,
                            label_mode="any")
    # starts 0,2,4 -> [0,0,1]->1  [1,0,0]->1  [0,0]->0
    assert windows["label"].tolist() == [1, 1, 0]
