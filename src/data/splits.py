"""Leakage-aware split protocols for UNSW-NB15.

Why this module exists
----------------------
The combined UNSW-NB15 table (official training + testing CSVs) contains
~40% exact-duplicate flow records. A random 70/15/15 re-split therefore
puts byte-identical flows on both sides of the train/test boundary, and
the model is partly graded on rows it memorised. The official partition
(training-set -> testing-set) was captured on different days and is the
protocol the literature reports against.

Protocols (all return raw-file positions, so retrain.py and the serving
simulator can reconstruct the exact partitions):

    random         legacy behaviour: stratified random 70/15/15 re-split
                   of the combined table (duplicates cross the split)
    random_dedup   same, after dropping exact duplicate rows first
    official       official training file -> train/val, official testing
                   file -> test (train de-duplicated, test untouched)
    official_novel official, plus the test set is de-duplicated and every
                   test row whose feature vector also occurs in the
                   training file is removed. This is the headline
                   protocol: every test flow is unseen.

The combined CSV is ``concat([training, testing])`` with a fresh index
(``download_dataset.combine``), so position < OFFICIAL_TRAIN_ROWS means
"came from the official training file".
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

OFFICIAL_TRAIN_ROWS = 175_341
OFFICIAL_TEST_ROWS = 82_332

LABEL_COLUMNS = ("Label", "attack_cat")
# Never part of a flow's identity for duplicate detection.
NON_FEATURE_COLUMNS = ("id",) + LABEL_COLUMNS

PROTOCOLS = ("random", "random_dedup", "official", "official_novel")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns that describe the flow itself (labels and row ids excluded)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def _dedupe(df: pd.DataFrame) -> pd.Index:
    """Positions of the first occurrence of every (features, label) row."""
    cols = feature_columns(df) + [c for c in ("Label",) if c in df.columns]
    return df.index[~df.duplicated(subset=cols, keep="first")]


def _stratified(index: pd.Index, y: pd.Series, test_size: float, seed: int
                ) -> tuple[pd.Index, pd.Index]:
    a, b = train_test_split(index, test_size=test_size,
                            stratify=y.loc[index], random_state=seed)
    return pd.Index(a), pd.Index(b)


def split_positions(df: pd.DataFrame, protocol: str, seed: int = 42,
                    target: str = "Label",
                    n_official_train: int = OFFICIAL_TRAIN_ROWS
                    ) -> tuple[dict[str, pd.Index], dict]:
    """Return ({train, val, test} -> raw positions, stats) for ``protocol``."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")
    y = df[target]
    stats: dict = {"protocol": protocol, "rows_in": int(len(df))}

    if protocol.startswith("random"):
        pool = _dedupe(df) if protocol == "random_dedup" else df.index
        stats["dropped_duplicates"] = int(len(df) - len(pool))
        train, tmp = _stratified(pool, y, 0.30, seed)
        val, test = _stratified(tmp, y, 0.50, seed)
    else:
        if len(df) < n_official_train:
            raise ValueError("table is smaller than the official training file; "
                             "official protocols need the combined CSV")
        train_file = df.iloc[:n_official_train]
        test_file = df.iloc[n_official_train:]
        train_pool = _dedupe(train_file)
        stats["dropped_train_duplicates"] = int(len(train_file) - len(train_pool))
        train, val = _stratified(train_pool, y, 0.15, seed)
        test = test_file.index
        if protocol == "official_novel":
            test = _dedupe(test_file)
            stats["dropped_test_duplicates"] = int(len(test_file) - len(test))
            cols = feature_columns(df)
            seen = pd.MultiIndex.from_frame(train_file[cols].astype(str))
            candidate = pd.MultiIndex.from_frame(df.loc[test, cols].astype(str))
            novel = ~candidate.isin(seen)
            stats["dropped_test_seen_in_train"] = int((~novel).sum())
            test = test[novel]

    parts = {"train": pd.Index(train).sort_values(),
             "val": pd.Index(val).sort_values(),
             "test": pd.Index(test).sort_values()}
    for name, idx in parts.items():
        stats[f"{name}_rows"] = int(len(idx))
        stats[f"{name}_attack_share"] = round(float(y.loc[idx].astype(float).mean()), 4) \
            if target == "Label" else None
    return parts, stats


def assert_disjoint(parts: dict[str, pd.Index]) -> None:
    """Raise if any raw position appears in more than one partition."""
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = parts[a].intersection(parts[b])
            if len(shared):
                raise AssertionError(f"{a} and {b} share {len(shared)} rows")
