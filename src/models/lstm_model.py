"""Bidirectional LSTM for sequence-based network threat detection.

UNSW-NB15 ships pre-aggregated flow records, so there is no packet-level
ordering *within* a flow. This script instead builds *inter-flow*
sequences: flows are grouped by connection identity, ordered by time,
and cut into sliding windows that the LSTM classifies as attack/normal.

Grouping (auto-detected, in order of preference):
    1. (srcip, dstip)        - needs the canonical UNSW-NB15 files
    2. srcip alone           - fallback if pair grouping leaves too few
                               sequences (per the spec)
    3. (sport, dsport, proto) - some mirrors keep ports but drop IPs
    4. (proto, service)      - last resort; used by the "cleaned" 45-col
                               variant, which has NO connection identity.
                               In that case flow order (file order, which
                               mirrors capture order) is used as the
                               temporal axis and a loud warning is printed.

Windows: sliding window of `--window` (default 10) flows, stride
`--stride` (default 5), sorted by Stime when present. Partial trailing
windows are kept and padded with keras pad_sequences to `--maxlen`.

Window label: `--label-mode any|majority|last` (default any = malicious
if any flow in the window is an attack).

Per-flow features are transformed with the SAME fitted pipeline saved by
``src/features/preprocess.py`` (imputer + encoder + scaler +
models/preprocess_pipeline.joblib), so the LSTM consumes the exact same
feature space as the Random Forest baseline.

Split: stratified 70/15/15 on the *window* labels (same logic as the
tabular pipeline). NOTE: this splits windows, not connection groups, so
windows of the same connection pair can appear in more than one split.

Architecture: dense projection -> 1-2 bidirectional LSTM layers
(64-128 units, dropout 0.3-0.5) -> sigmoid head. Early stopping on
validation loss; best checkpoint saved to models/lstm_best.keras.

Usage:
    python src/models/lstm_model.py                 # full run
    python src/models/lstm_model.py --fast          # smoke test
    python src/models/lstm_model.py --label-mode majority --window 20
    python src/models/lstm_model.py --group srcip   # force a grouping tier

Outputs: models/lstm_best.keras, reports/lstm_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_INPUT = RAW_DIR / "UNSW_NB15_combined.csv"
PIPELINE_PATH = MODELS_DIR / "preprocess_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "preprocess_metadata.json"
MODEL_OUT = MODELS_DIR / "lstm_best.keras"

GROUPING_TIERS = {
    "pair": ["srcip", "dstip"],
    "srcip": ["srcip"],
    "ports": ["sport", "dsport", "proto"],
    "service": ["proto", "service"],
}

DROP_COLUMNS = ["srcip", "dstip", "Stime", "Ltime", "id", "attack_cat", "Label"]
ORDER_COLUMNS = ["Stime", "Ltime"]


# ---------------------------------------------------------------------------
# Sequence construction (pure pandas/numpy; no TF needed -> unit-testable)
# ---------------------------------------------------------------------------

def detect_group_key(df: pd.DataFrame, forced: str | None = None) -> list[str]:
    """Pick the best available grouping tier; force one with ``forced``."""
    if forced is not None:
        cols = GROUPING_TIERS[forced]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            sys.exit(f"[!] --group {forced} needs columns {missing} "
                     f"which are not in the data")
        return cols

    def usable(cols: list[str]) -> bool:
        if any(c not in df.columns for c in cols):
            return False
        if not df[cols].notna().any().any():  # all-NaN (cleaned variant)
            return False
        # At least 2 distinct values: a constant column is no grouping key.
        return df[cols].nunique().min() >= 2

    for tier in ("pair", "srcip", "ports", "service"):
        if usable(GROUPING_TIERS[tier]):
            return GROUPING_TIERS[tier]
    return GROUPING_TIERS["service"]


def warn_grouping(key: list[str], df: pd.DataFrame) -> None:
    print("=" * 72)
    print(f"Grouping flows by: {key}")
    if not {"srcip", "dstip"}.intersection(key):
        print("WARNING: no real connection identity (srcip/dstip) in this data.")
        print("The canonical UNSW-NB15 CSVs (from UNSW SharePoint) are needed")
        print("for true per-host sequences; see download_dataset.py manual steps.")
        if "Stime" in df.columns and df["Stime"].notna().any():
            print("(sorted by Stime)")
        else:
            print("WARNING: no Stime either - using file order (= capture order)")
    print("=" * 72)


def build_windows(
    df: pd.DataFrame, group_cols: list[str], window: int, stride: int,
    label_mode: str,
) -> pd.DataFrame:
    """Slide windows over time-ordered per-group flows.

    Returns a DataFrame with one row per window:
        window_id, group, start, end, n_flows, label, flow_index (list)
    """
    order = [c for c in ORDER_COLUMNS if c in df.columns and df[c].notna().any()]
    if order:
        df = df.sort_values(group_cols + order, kind="mergesort")
    else:
        df = df.sort_values(group_cols, kind="mergesort")  # stable: file order

    records = []
    for group, grp in df.groupby(group_cols, sort=False):
        idx = grp.index.tolist()
        labels = grp["Label"].astype(int).tolist()
        for start in range(0, len(idx), stride):
            end = min(start + window, len(idx))
            seg = labels[start:end]
            if label_mode == "any":
                label = int(max(seg))
            elif label_mode == "majority":
                label = 1 if sum(seg) >= (len(seg) + 1) // 2 else 0
            else:  # last
                label = seg[-1]
            records.append({
                "window_id": len(records),
                "group": str(group),
                "start": start,
                "end": end,
                "n_flows": len(seg),
                "label": label,
                "flow_index": idx[start:end],
            })
    return pd.DataFrame(records)


def windows_to_sequences(
    windows: pd.DataFrame, feature_matrix: np.ndarray, maxlen: int
) -> np.ndarray:
    """Materialize window flow indices into a padded (n, maxlen, n_feat) array."""
    from keras.preprocessing.sequence import pad_sequences  # noqa: PLC0415

    sequences = [feature_matrix[list(indices)] for indices in windows["flow_index"]]
    return pad_sequences(
        sequences, maxlen=maxlen, dtype="float32",
        padding="pre", truncating="pre", value=0.0,
    )


# ---------------------------------------------------------------------------
# Feature transformation with the saved preprocess pipeline
# ---------------------------------------------------------------------------

def transform_flows(df: pd.DataFrame, pipeline: dict, feature_names: list[str]
                    ) -> np.ndarray:
    """Replay preprocess.py on raw flows: drop -> impute -> encode -> scale."""
    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).copy()

    # Use the imputers' fitted column lists instead of dtype inference:
    # JSON-serialized records turn all-NaN numeric fields into object dtype,
    # which would otherwise misroute them to the categorical branch.
    numeric = [c for c in pipeline["imputers"]["numeric"].feature_names_in_
               if c in X.columns]
    categorical = [c for c in pipeline["imputers"]["categorical"].feature_names_in_
                   if c in X.columns]
    missing = [c for c in feature_names
               if c not in numeric and c not in categorical and c not in X.columns]
    if missing:
        raise ValueError(f"columns absent from incoming record: {missing}")

    if numeric:
        X[numeric] = pipeline["imputers"]["numeric"].transform(
            X[numeric].astype(float))
    if categorical:
        X[categorical] = pipeline["imputers"]["categorical"].transform(
            X[categorical])

    encoder = pipeline["encoder"]
    if encoder["mode"] == "label":
        for col, le in encoder["objects"].items():
            X[col] = X[col].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )
    else:
        raise NotImplementedError(
            "onehot encoding mode is not supported for LSTM; re-run "
            "preprocess.py with --categorical label"
        )

    scale_cols = [c for c in feature_names if c in X.columns]
    X[scale_cols] = pipeline["scaler"].transform(X[scale_cols].astype(float))

    missing_cols = [c for c in feature_names if c not in X.columns]
    if missing_cols:
        raise ValueError(f"feature_names missing after transform: {missing_cols}")
    return X[feature_names].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(maxlen: int, n_features: int, units: int, layers: int,
                dropout: float, learning_rate: float):
    """Dense projection -> 1-2 bidirectional LSTMs -> sigmoid head."""
    import tensorflow as tf  # noqa: PLC0415

    inputs = tf.keras.Input(shape=(maxlen, n_features))
    x = tf.keras.layers.Dense(units, activation="relu")(inputs)
    for i in range(layers):
        return_sequences = i < layers - 1
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units, return_sequences=return_sequences, dropout=dropout,
            )
        )(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    units: int, layers: int, dropout: float, epochs: int, batch_size: int,
    patience: int, learning_rate: float, seed: int, model_path: Path,
):
    """Train with early stopping on val loss; return the best model + history."""
    import tensorflow as tf  # noqa: PLC0415

    tf.random.set_seed(seed)
    model = build_model(X_train.shape[1], X_train.shape[2],
                        units, layers, dropout, learning_rate)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]
    start = time.time()
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1,
    )
    print(f"Training took {(time.time() - start) / 60:.1f} min")
    return model, history


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: PLC0415

    proba = model.predict(X_test, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--maxlen", type=int, default=None,
                        help="pad/truncate length (default: --window)")
    parser.add_argument("--label-mode", choices=["any", "majority", "last"],
                        default="any")
    parser.add_argument("--group", choices=list(GROUPING_TIERS), default=None,
                        help="force a grouping tier (default: auto-detect)")
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2, choices=[1, 2])
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true",
                        help="small units/layers/epochs + capped windows (smoke test)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap the number of windows built (debugging)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    if args.fast:
        args.units = min(args.units, 32)
        args.layers = min(args.layers, 1)
        args.epochs = min(args.epochs, 5)
        args.patience = min(args.patience, 2)

    if not args.input.exists():
        sys.exit(f"[!] {args.input} not found - run src/data/download_dataset.py first")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} flows x {len(df.columns)} columns")

    group_cols = detect_group_key(df, args.group)
    warn_grouping(group_cols, df)
    windows = build_windows(df, group_cols, args.window, args.stride, args.label_mode)
    if args.max_windows:
        windows = windows.iloc[: args.max_windows]
    n_attack = int(windows["label"].sum())
    print(f"\nBuilt {len(windows):,} windows (window={args.window}, "
          f"stride={args.stride}, label={args.label_mode})")
    print(f"Window label distribution: normal={len(windows) - n_attack:,} "
          f"({(1 - n_attack / len(windows)):.1%}), "
          f"attack={n_attack:,} ({n_attack / len(windows):.1%})")

    pipeline = joblib.load(PIPELINE_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_names = metadata["feature_names"]
    feature_matrix = transform_flows(df, pipeline, feature_names)
    print(f"Feature vectors: {feature_matrix.shape[1]} dims (per flow)")

    maxlen = args.maxlen or args.window
    X = windows_to_sequences(windows, feature_matrix, maxlen)
    y = windows["label"].to_numpy()

    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    positions = np.arange(len(X))
    pos_train, pos_tmp, y_train, y_tmp = train_test_split(
        positions, y, test_size=0.30, stratify=y, random_state=args.seed)
    pos_val, pos_test, y_val, y_test = train_test_split(
        pos_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=args.seed)
    X_train, X_val, X_test = X[pos_train], X[pos_val], X[pos_test]
    for name, y_part in (("train", y_train), ("val", y_val), ("test", y_test)):
        print(f"  {name}: {len(y_part):,} windows, "
              f"attack {y_part.mean():.1%}")

    # Persist the held-out window test set (features, labels, and the raw
    # flow positions each window covers) so src/evaluation/evaluate.py can
    # align LSTM window predictions with per-flow Random Forest predictions.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PROCESSED_DIR / "lstm_test.npz",
        X_test=X_test, y_test=y_test,
        flow_indices=np.array(windows.iloc[pos_test]["flow_index"], dtype=object),
    )
    print(f"[ok] window test set -> {PROCESSED_DIR / 'lstm_test.npz'} "
          f"({len(y_test):,} windows)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model, history = train_model(
        X_train, y_train, X_val, y_val, args.units, args.layers, args.dropout,
        args.epochs, args.batch_size, args.patience, args.learning_rate,
        args.seed, MODEL_OUT,
    )
    print(f"[ok] best model -> {MODEL_OUT}")

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))
    results = {"val_loss": best_val_loss, "best_epoch": best_epoch}
    results["test"] = evaluate(model, X_test, y_test)
    print(f"Best epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"TEST: accuracy={results['test']['accuracy']:.4f}  "
          f"f1={results['test']['f1']:.4f}  auc={results['test']['auc']:.4f}")

    results.update({
        "window": args.window, "stride": args.stride, "maxlen": maxlen,
        "label_mode": args.label_mode, "group": group_cols,
        "n_windows": int(len(windows)),
        "window_label_dist": {
            "normal": int(len(windows) - n_attack), "attack": n_attack},
        "units": args.units, "layers": args.layers, "dropout": args.dropout,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "seed": args.seed,
    })
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "lstm_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"[ok] results -> {REPORTS_DIR / 'lstm_results.json'}")


if __name__ == "__main__":
    main()
