"""Preprocess raw flow records into ML-ready train/val/test splits.

Pipeline (all fitted on the training split only, then applied to
validation/test to avoid leakage):

    1. Drop low-information columns (IPs, timestamps, ids).
    2. Leakage-aware split (src/data/splits.py): by default the official
       training file -> train/val and the official testing file -> test,
       with duplicates removed and test rows already seen in training
       dropped. ``--split random`` reproduces the legacy re-split.
    3. Impute missing values (median for numeric, most-frequent for
       categorical).
    4. Encode categorical features (label encoding or one-hot).
    5. Standardize numeric features with a fitted StandardScaler.
    6. Address class imbalance: SMOTE oversampling and/or class weights
       (class weights are recorded in metadata for the model script).

Outputs
-------
    data/processed/X_train.csv, X_val.csv, X_test.csv
    data/processed/y_train.csv, y_val.csv, y_test.csv
    models/scaler.joblib                  (fitted StandardScaler)
    models/preprocess_pipeline.joblib     (imputer + encoders + scaler)
    models/preprocess_metadata.json       (mode flags, feature names)

Usage:
    python src/features/preprocess.py
    python src/features/preprocess.py --imbalance smote
    python src/features/preprocess.py --imbalance weights --categorical onehot
    python src/features/preprocess.py --target multiclass   # predict attack_cat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import PROTOCOLS, assert_disjoint, split_positions  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_INPUT = RAW_DIR / "UNSW_NB15_combined.csv"

# Columns that carry identity/timing information, not predictive signal.
DROP_COLUMNS = ["srcip", "dstip", "Stime", "Ltime", "id"]

# Categorical feature columns expected in UNSW-NB15.
CATEGORICAL_COLUMNS = ["proto", "state", "service"]

# Numeric columns known to exist; anything numeric not listed elsewhere is kept.
TARGET_BINARY = "Label"
TARGET_MULTICLASS = "attack_cat"


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/data/download_dataset.py` first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {df.shape[0]:,} records x {df.shape[1]} columns")
    return df


def drop_low_value(df: pd.DataFrame) -> pd.DataFrame:
    dropped = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"Dropped low-information columns: {dropped}")
    return df


def stratified_split(
    X: pd.DataFrame, y: pd.Series, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """70/15/15 stratified split. Ordering: train -> (val, test)."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    for name, y_part in (("train", y_train), ("val", y_val), ("test", y_test)):
        counts = y_part.value_counts(normalize=True).round(3).to_dict()
        print(f"  split {name}: {len(y_part):,} rows, class balance {counts}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def impute(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """Median-impute numeric cols, most-frequent-impute categorical cols."""
    num_cols = [c for c in X_train.columns if _is_numeric(X_train[c])]
    cat_cols = [c for c in X_train.columns if not _is_numeric(X_train[c])]
    imputers: dict[str, SimpleImputer] = {}
    for group, cols, strategy in (("numeric", num_cols, "median"),
                                  ("categorical", cat_cols, "most_frequent")):
        if not cols:
            continue
        imp = SimpleImputer(strategy=strategy, keep_empty_features=True)
        imp.fit(X_train[cols])
        imputers[group] = imp
        for frame, name in ((X_train, "train"), (X_val, "val"), (X_test, "test")):
            frame[cols] = imp.transform(frame[cols])
        print(f"Imputed {len(cols)} {group} columns (strategy={strategy})")
    return X_train, X_val, X_test, imputers


def encode_categorical(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Label- or one-hot-encode categorical columns (fit on train)."""
    cat_cols = [c for c in X_train.columns if not _is_numeric(X_train[c])]
    if not cat_cols:
        return X_train, X_val, X_test, {"mode": mode, "columns": [], "objects": {}}
    if mode == "label":
        objects: dict[str, LabelEncoder] = {}
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(X_train[col].astype(str))
            objects[col] = le
            for frame in (X_train, X_val, X_test):
                frame[col] = frame[col].astype(str).map(
                    lambda v: le.transform([v])[0] if v in le.classes_ else -1
                )
        print(f"Label-encoded {cat_cols}")
        encoder_state = {"mode": mode, "columns": cat_cols, "objects": objects}
    else:  # onehot
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        oh.fit(X_train[cat_cols].astype(str))
        for frame, name in ((X_train, "train"), (X_val, "val"), (X_test, "test")):
            encoded = pd.DataFrame(
                oh.transform(frame[cat_cols].astype(str)),
                columns=oh.get_feature_names_out(cat_cols),
                index=frame.index,
            )
            frame.drop(columns=cat_cols, inplace=True)
            frame[encoded.columns] = encoded
        print(f"One-hot encoded {cat_cols} -> {oh.get_feature_names_out(cat_cols).tolist()}")
        encoder_state = {"mode": mode, "columns": cat_cols, "objects": {"onehot": oh}}
    return X_train, X_val, X_test, encoder_state


def scale(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize numeric features; fit on train only."""
    num_cols = [c for c in X_train.columns if _is_numeric(X_train[c])]
    scaler = StandardScaler()
    scaler.fit(X_train[num_cols])
    X_train[num_cols] = scaler.transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    print(f"Standardized {len(num_cols)} numeric features "
          f"(mean 0, unit variance) on train only")
    return X_train, X_val, X_test, scaler


def oversample(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTE oversampling to balance the training set."""
    sampler = SMOTE(random_state=seed)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"SMOTE: training set {len(y_train):,} -> {len(y_res):,} "
          f"(minority now {y_res.value_counts().min():,})")
    return X_res, y_res


def encode_target(
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, multiclass: bool
) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    """Encode target into 0..K integers; return (y_train, y_val, y_test, info)."""
    le = LabelEncoder()
    le.fit(y_train.astype(str))
    encoded = [le.transform(part.astype(str)) for part in (y_train, y_val, y_test)]
    return (
        pd.Series(encoded[0], index=y_train.index, name=y_train.name),
        pd.Series(encoded[1], index=y_val.index, name=y_val.name),
        pd.Series(encoded[2], index=y_test.index, name=y_test.name),
        {"classes": le.classes_.tolist(), "type": "multiclass" if multiclass else "binary"},
    )


def save_artifacts(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    scaler: StandardScaler, imputers: dict, encoder_state: dict,
    metadata: dict,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    for name, series in (("y_train", y_train), ("y_val", y_val), ("y_test", y_test)):
        series.to_csv(PROCESSED_DIR / f"{name}.csv", index=False, header=["label"])
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(
        {"imputers": imputers, "encoder": encoder_state, "scaler": scaler},
        MODELS_DIR / "preprocess_pipeline.joblib",
    )
    (MODELS_DIR / "preprocess_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print(f"[ok] splits -> {PROCESSED_DIR}")
    print(f"[ok] scaler -> {MODELS_DIR / 'scaler.joblib'}")
    print(f"[ok] pipeline -> {MODELS_DIR / 'preprocess_pipeline.joblib'}")
    print(f"[ok] metadata -> {MODELS_DIR / 'preprocess_metadata.json'}")


def build_splits(
    df: pd.DataFrame, parts: dict[str, pd.Index], target: str = "binary",
    categorical: str = "label", imbalance: str = "smote", seed: int = 42,
    drop_features: tuple[str, ...] = (),
) -> dict:
    """Fit the pipeline on ``parts['train']`` and transform every split.

    ``parts`` maps train/val/test to raw-file positions (see
    ``src/data/splits.py``). Returns the transformed splits, the per-split
    attack_cat labels and the fitted pipeline objects.
    """
    df = drop_low_value(df)
    target_col = TARGET_MULTICLASS if target == "multiclass" else TARGET_BINARY
    attack_cat = df[TARGET_MULTICLASS] if TARGET_MULTICLASS in df.columns else None
    # Drop both label columns from X: the other label leaks the target,
    # and the target itself must never appear as a feature.
    drop_cols = {TARGET_BINARY, TARGET_MULTICLASS, *drop_features}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]

    X_train, X_val, X_test = (X.loc[parts[k]].copy() for k in ("train", "val", "test"))
    y_train, y_val, y_test = (y.loc[parts[k]] for k in ("train", "val", "test"))
    cats = ({k: attack_cat.loc[parts[k]] for k in ("train", "val", "test")}
            if attack_cat is not None else {})

    X_train, X_val, X_test, imputers = impute(X_train, X_val, X_test)
    X_train, X_val, X_test, encoder_state = encode_categorical(
        X_train, X_val, X_test, categorical)
    X_train, X_val, X_test, scaler = scale(X_train, X_val, X_test)
    if imbalance == "smote":
        X_train, y_train = oversample(X_train, y_train, seed)
    y_train, y_val, y_test, target_info = encode_target(
        y_train, y_val, y_test, target == "multiclass")
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "attack_cat": cats, "imputers": imputers,
        "encoder_state": encoder_state, "scaler": scaler,
        "target_info": target_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--imbalance", choices=["none", "smote", "weights"],
                        default="smote",
                        help="how to handle class imbalance (default: smote)")
    parser.add_argument("--categorical", choices=["label", "onehot"], default="label",
                        help="categorical encoding strategy (default: label)")
    parser.add_argument("--target", choices=["binary", "multiclass"], default="binary",
                        help="predict Label (binary) or attack_cat (multiclass)")
    parser.add_argument("--split", choices=list(PROTOCOLS), default="official_novel",
                        help="split protocol (default: official_novel; "
                             "'random' reproduces the legacy leaky re-split)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_raw(args.input)
    target_col = TARGET_MULTICLASS if args.target == "multiclass" else TARGET_BINARY
    parts, split_stats = split_positions(df, args.split, args.seed, target_col)
    assert_disjoint(parts)
    for name in ("train", "val", "test"):
        print(f"  split {name}: {split_stats[f'{name}_rows']:,} rows, "
              f"attack share {split_stats[f'{name}_attack_share']}")
    print(f"Split protocol {args.split}: {split_stats}")

    # Raw-file positions of every split. test_positions drives the serving
    # simulator and cross-model alignment; train/val positions let the
    # retraining script reconstruct the exact original partitions.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, idx in parts.items():
        pd.DataFrame({"position": idx}).to_csv(
            PROCESSED_DIR / f"{name}_positions.csv", index=False)
    print(f"Saved split positions -> {PROCESSED_DIR}")

    out = build_splits(df, parts, args.target, args.categorical,
                       args.imbalance, args.seed)
    # Keep the multi-class family label alongside the splits so evaluation
    # can break down detection rates per attack family.
    for name, series in out["attack_cat"].items():
        series.to_csv(PROCESSED_DIR / f"attack_cat_{name}.csv",
                      index=False, header=["attack_cat"])
    if out["attack_cat"]:
        print("Saved per-split attack_cat labels for per-family evaluation")

    metadata = {
        "imbalance": args.imbalance,
        "use_class_weight": args.imbalance == "weights",
        "categorical_encoding": args.categorical,
        "target": args.target,
        "target_info": out["target_info"],
        "feature_names": out["X_train"].columns.tolist(),
        "n_features": len(out["X_train"].columns),
        "split_protocol": args.split,
        "split_stats": split_stats,
        "split_sizes": {k: int(len(out[f"X_{k}"])) for k in ("train", "val", "test")},
        "seed": args.seed,
    }
    save_artifacts(
        out["X_train"], out["y_train"], out["X_val"], out["y_val"],
        out["X_test"], out["y_test"], out["scaler"], out["imputers"],
        out["encoder_state"], metadata,
    )


if __name__ == "__main__":
    main()
