"""Retrain the Random Forest on base data + freshly logged predictions.

Every prediction served by the API is stored (with its raw flow record)
in ``reports/prediction_log.sqlite``. This script harvests those rows as
additional training material, re-runs the full preprocessing pipeline,
trains a new Random Forest, and only replaces
``models/random_forest_best.joblib`` if the new model *beats the old one
on the exact same held-out test set* (fixed via
``data/processed/test_positions.csv``). Otherwise it refuses and prints
why.

Label handling for the harvested rows:
    * confirmed labels   - only rows an *analyst* labelled through the
                           authenticated ``/feedback`` endpoint
                           (``label_source == 'analyst'``). Labels from any
                           other origin are dropped: the v1 API took
                           ``Label`` from the sensor payload, which was an
                           unauthenticated training-data poisoning path
                           (see src/attacks/poisoning.py).
    * pseudo-labels      - OFF by default; ``--pseudo-labels`` uses rows
                           with confidence >= ``--min-confidence`` and the
                           prediction as the label. Self-training on its own
                           output lets an evasion that once scored "benign
                           with high confidence" teach the model it is
                           benign, so keep it off unless you need it.
    * deduplication      - rows that exactly duplicate a base-dataset row
                           are dropped first (the traffic simulator replays
                           held-out test flows, which must never leak into
                           training).

The guard also refuses a model whose recall on any attack family drops by
more than ``--max-family-drop`` (default 0.02): targeted poisoning hides
inside aggregate accuracy, so aggregate metrics alone are not a guard.

Usage:
    python src/models/retrain.py                        # analyst labels only
    python src/models/retrain.py --pseudo-labels        # + pseudo-labels
    python src/models/retrain.py --dry-run              # evaluate, don't write
    python src/models/retrain.py --force                # replace regardless
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocess import (  # noqa: E402
    TARGET_BINARY,
    TARGET_MULTICLASS,
    drop_low_value,
    encode_categorical,
    encode_target,
    impute,
    load_raw,
    oversample,
    save_artifacts,
    scale,
)
from src.evaluation.metrics import per_family_recall, threshold_for_fpr  # noqa: E402
from src.models.hardening import harden_training_set, model_features  # noqa: E402
from src.models.manifest import write_manifest  # noqa: E402
from src.serving.prediction_log import load_predictions  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_INPUT = RAW_DIR / "UNSW_NB15_combined.csv"
MODEL_PATH = MODELS_DIR / "random_forest_best.joblib"
RF_RESULTS = REPORTS_DIR / "random_forest_results.json"
METADATA_PATH = MODELS_DIR / "preprocess_metadata.json"
OPERATING_PATH = MODELS_DIR / "operating_point.json"

LOG_COLUMNS = ["ts", "model", "prediction", "confidence", "true_label",
               "predicted_family", "true_family", "n_flows", "record_json",
               "p_attack", "label_source", "labeled_by"]

FALLBACK_RF_PARAMS = {"n_estimators": 200, "max_depth": 25,
                      "min_samples_split": 2}


# ---------------------------------------------------------------------------
# Harvesting logged predictions (pure functions -> unit-testable)
# ---------------------------------------------------------------------------

def load_logged_rows(path: Path | None = None) -> pd.DataFrame:
    """Rows from the prediction log that carry a raw flow record (rf rows)."""
    rows = load_predictions(path, extended=True)
    if not rows:
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    df = df[df["record_json"].notna()]
    return df


def apply_labels(
    log_df: pd.DataFrame, min_confidence: float, labeled_only: bool
) -> tuple[pd.DataFrame, dict]:
    """Turn logged rows into flow records with Label + attack_cat.

    Returns (flows, stats) where flows has the raw record fields plus
    ``Label``/``attack_cat``. Rows that are neither confirmed nor
    confidently predicted are dropped.
    """
    stats = {"confirmed": 0, "pseudo": 0, "dropped_low_confidence": 0,
             "dropped_untrusted_label": 0}
    flows = []
    for _, row in log_df.iterrows():
        record = json.loads(row["record_json"])
        trusted = row.get("label_source") == "analyst"
        if pd.notna(row["true_label"]) and not trusted:
            stats["dropped_untrusted_label"] += 1
            continue
        if pd.notna(row["true_label"]):
            record["Label"] = int(row["true_label"])
            if pd.notna(row["true_family"]):
                record["attack_cat"] = row["true_family"]
            stats["confirmed"] += 1
            flows.append(record)
        elif not labeled_only and row["confidence"] >= min_confidence:
            record["Label"] = int(row["prediction"])
            record["attack_cat"] = (
                row["predicted_family"] if row["prediction"] == 1 else "Normal")
            stats["pseudo"] += 1
            flows.append(record)
        else:
            stats["dropped_low_confidence"] += 1
    if flows:
        return pd.DataFrame(flows), stats
    return pd.DataFrame(), stats


def dedupe_new_rows(new_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows of ``new_df`` that exactly match a base-dataset row."""
    if new_df.empty or base_df.empty:
        return new_df
    common = [c for c in base_df.columns if c in new_df.columns]
    if not common:
        return new_df
    # Coerce the harvested rows to the base column dtypes first: JSON records
    # carry None where the CSV carries NaN, so all-NaN base columns (e.g.
    # srcip) are float64 there but object here, which would break the join.
    a = new_df.copy()
    for col in common:
        if pd.api.types.is_float_dtype(base_df[col]):
            a[col] = pd.to_numeric(a[col], errors="coerce")
        elif pd.api.types.is_object_dtype(base_df[col]):
            a[col] = a[col].where(pd.notna(a[col]), None)
    b = base_df[common].copy()
    for col in common:
        if pd.api.types.is_object_dtype(b[col]):
            b[col] = b[col].where(pd.notna(b[col]), None)
    merged = a.merge(b, how="left", indicator=True, on=common)
    kept = merged["_merge"] == "left_only"
    return new_df.loc[merged.index[kept]]


# ---------------------------------------------------------------------------
# Compare-and-replace guard (pure function -> unit-testable)
# ---------------------------------------------------------------------------

def should_replace(old: dict, new: dict, min_gain: float = 0.001,
                   max_family_drop: float = 0.02) -> tuple[bool, str]:
    """Decide whether ``new`` metrics beat ``old`` on the same test set.

    Metrics dicts carry ``accuracy`` and ``f1_macro`` (and optionally
    ``family_recall``: {family: recall}). The new model wins on accuracy
    (beyond ``min_gain``), or on F1-macro when accuracy is within
    ``min_gain`` of a tie - but never if any family's recall drops by
    more than ``max_family_drop`` (targeted-poisoning tripwire).
    """
    regressions = {
        fam: (rec, new["family_recall"].get(fam, 0.0))
        for fam, rec in old.get("family_recall", {}).items()
        if "family_recall" in new
        and rec - new["family_recall"].get(fam, 0.0) > max_family_drop
    }
    if regressions:
        worst = ", ".join(f"{f} {a:.3f}->{b:.3f}" for f, (a, b) in regressions.items())
        return False, (f"family recall regression beyond {max_family_drop}: {worst} "
                       "(possible targeted poisoning)")
    acc_gain = new["accuracy"] - old["accuracy"]
    f1_gain = new["f1_macro"] - old["f1_macro"]
    if acc_gain > min_gain:
        reason = (f"new accuracy {new['accuracy']:.4f} beats old "
                  f"{old['accuracy']:.4f} (gain +{acc_gain:.4f} > {min_gain})")
        return True, reason
    if abs(acc_gain) <= min_gain and f1_gain > min_gain:
        reason = (f"accuracy tied ({old['accuracy']:.4f} vs "
                  f"{new['accuracy']:.4f}, within {min_gain}); F1-macro "
                  f"improves +{f1_gain:.4f} ({old['f1_macro']:.4f} -> "
                  f"{new['f1_macro']:.4f})")
        return True, reason
    reason = (f"new model is not better: accuracy {old['accuracy']:.4f} -> "
              f"{new['accuracy']:.4f} (gain {acc_gain:+.4f}, need > "
              f"{min_gain}); F1-macro {old['f1_macro']:.4f} -> "
              f"{new['f1_macro']:.4f} (gain {f1_gain:+.4f})")
    return False, reason


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series,
             families: pd.Series | None = None) -> dict:
    """Accuracy + F1-macro (+ per-family recall) on a transformed test set."""
    pred = model.predict(X_test)
    out = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
    }
    if families is not None:
        out["family_recall"] = per_family_recall(pred, families.to_numpy())
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rf(X_train: pd.DataFrame, y_train: pd.Series, params: dict,
             class_weight, n_jobs: int, seed: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        random_state=seed,
        n_jobs=n_jobs,
        class_weight=class_weight,
        **params,
    )
    print(f"\nTraining Random Forest {params} on {len(y_train):,} rows ...")
    start = time.time()
    model.fit(X_train, y_train)
    print(f"  fitted in {(time.time() - start) / 60:.1f} min")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pseudo-labels", action="store_true",
                        help="also use high-confidence predictions as labels "
                             "(off by default: self-training amplifies evasion)")
    parser.add_argument("--labeled-only", action="store_true",
                        help="(default; kept for backwards compatibility)")
    parser.add_argument("--max-family-drop", type=float, default=0.02,
                        help="refuse if any family's recall drops more than this")
    parser.add_argument("--min-confidence", type=float, default=0.95,
                        help="confidence threshold for pseudo-labels (0.95)")
    parser.add_argument("--min-gain", type=float, default=0.001,
                        help="minimum test-accuracy gain to replace the model")
    parser.add_argument("--force", action="store_true",
                        help="replace the model even if it does not win")
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate everything but write nothing")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print("[!] no existing model - nothing to compare against; "
              "run src/models/random_forest.py first")
        sys.exit(1)
    for name in ("train_positions", "val_positions", "test_positions"):
        if not (PROCESSED_DIR / f"{name}.csv").exists():
            print(f"[!] missing data/processed/{name}.csv - "
                  "run src/features/preprocess.py first")
            sys.exit(1)

    base = load_raw(args.input)

    log_df = load_logged_rows()
    new_flows, stats = apply_labels(log_df, args.min_confidence,
                                    labeled_only=not args.pseudo_labels)
    print(f"\nHarvested {len(log_df)} logged flow predictions "
          f"({stats['confirmed']} confirmed, {stats['pseudo']} pseudo, "
          f"{stats['dropped_low_confidence']} dropped for low confidence, "
          f"{stats['dropped_untrusted_label']} dropped: label not from an analyst)")
    new_flows = dedupe_new_rows(new_flows, base)
    print(f"After deduplication against the base dataset: "
          f"{len(new_flows)} new training rows")

    # Reconstruct the EXACT original partitions from the saved raw-file
    # positions (train_positions/val_positions/test_positions.csv) and
    # append only the harvested rows to the training partition. With no
    # new rows this reproduces the original pipeline bit-for-bit, so the
    # old and new models are compared on identical data.
    train_pos = pd.read_csv(PROCESSED_DIR / "train_positions.csv")["position"]
    val_pos = pd.read_csv(PROCESSED_DIR / "val_positions.csv")["position"]
    test_pos = pd.read_csv(PROCESSED_DIR / "test_positions.csv")["position"]

    new_flows = new_flows.reindex(columns=base.columns)
    n_train = len(train_pos) + len(new_flows)
    n_val = len(val_pos)
    full = pd.concat(
        [base.loc[train_pos], new_flows.reset_index(drop=True),
         base.loc[val_pos], base.loc[test_pos]],
        ignore_index=True)
    print(f"Reconstructed splits: train={n_train:,} (incl. {len(new_flows):,} "
          f"new)  val={n_val:,}  test={len(test_pos):,}")

    df = drop_low_value(full)
    attack_cat = df[TARGET_MULTICLASS] if TARGET_MULTICLASS in df.columns else None
    drop_cols = {TARGET_BINARY, TARGET_MULTICLASS}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET_BINARY]

    X_train, X_val, X_test = X.iloc[:n_train], X.iloc[n_train:n_train + n_val], X.iloc[n_train + n_val:]
    y_train, y_val, y_test = y.iloc[:n_train], y.iloc[n_train:n_train + n_val], y.iloc[n_train + n_val:]

    # Capture per-split family labels while the split indices are still valid
    # (SMOTE later reindexes the training part).
    attack_cat_splits = {}
    if attack_cat is not None:
        for name, y_part in (("train", y_train), ("val", y_val),
                             ("test", y_test)):
            attack_cat_splits[name] = attack_cat.iloc[y_part.index]

    X_train, X_val, X_test, imputers = impute(X_train, X_val, X_test)
    X_train, X_val, X_test, encoder_state = encode_categorical(
        X_train, X_val, X_test, "label")
    X_train, X_val, X_test, scaler = scale(X_train, X_val, X_test)
    X_train, y_train = oversample(X_train, y_train, args.seed)
    y_train, y_val, y_test, target_info = encode_target(
        y_train, y_val, y_test, multiclass=False)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    class_weight = ("balanced" if metadata.get("use_class_weight", False)
                    else None)
    rf_params = FALLBACK_RF_PARAMS
    if RF_RESULTS.exists():
        saved = json.loads(RF_RESULTS.read_text(encoding="utf-8"))
        if saved.get("best_params"):
            # Reuse the EXACT grid-search config the deployed model was
            # trained with. max_depth=None is a legitimate search result
            # ("unlimited") and must be preserved, not dropped.
            best = {k: v for k, v in saved["best_params"].items()
                    if k in FALLBACK_RF_PARAMS}
            rf_params = {**FALLBACK_RF_PARAMS, **best}

    # Same recipe as the served model (src/models/hardening.py): if it was
    # hardened, the candidate is too, so a retrain cannot silently bring
    # back the TTL skeleton key.
    operating = (json.loads(OPERATING_PATH.read_text(encoding="utf-8"))
                 if OPERATING_PATH.exists() else {})
    names = X_train.columns.tolist()
    old_features = operating.get("features") or names
    hardening = None
    X_fit, y_fit = X_train, y_train
    if operating.get("hardening"):
        pipeline = {"imputers": imputers, "encoder": encoder_state, "scaler": scaler}
        X_fit, y_fit, hardening = harden_training_set(
            X_train, y_train, full.iloc[:n_train], pipeline, names, args.seed)
    new_features = model_features(names, hardened=hardening is not None)

    new_model = train_rf(X_fit[new_features], y_fit, rf_params, class_weight,
                         args.n_jobs, args.seed)

    old_model = joblib.load(MODEL_PATH)  # nosec B301
    test_families = attack_cat_splits.get("test")
    old_metrics = evaluate(old_model, X_test[old_features], y_test, test_families)
    new_metrics = evaluate(new_model, X_test[new_features], y_test, test_families)
    print(f"\nOld model on held-out test: "
          f"accuracy={old_metrics['accuracy']:.4f}  "
          f"f1_macro={old_metrics['f1_macro']:.4f}")
    print(f"New model on held-out test: "
          f"accuracy={new_metrics['accuracy']:.4f}  "
          f"f1_macro={new_metrics['f1_macro']:.4f}")

    replace, reason = should_replace(old_metrics, new_metrics, args.min_gain,
                                     args.max_family_drop)
    print(f"\nDecision: {'REPLACE' if replace else 'KEEP OLD'} - {reason}")

    result = {
        "harvested_logged_rows": int(len(log_df)),
        "confirmed": stats["confirmed"], "pseudo": stats["pseudo"],
        "new_training_rows_after_dedupe": int(len(new_flows)),
        "old_metrics": old_metrics, "new_metrics": new_metrics,
        "replace": bool(replace), "reason": reason,
        "min_gain": args.min_gain,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "retrain_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"[ok] retrain results -> {REPORTS_DIR / 'retrain_results.json'}")

    if not (replace or args.force):
        print("\nRefusing to replace models/random_forest_best.joblib.")
        sys.exit(1)
    if args.dry_run:
        print("\n[dry-run] no artifacts were written.")
        return

    joblib.dump(new_model, MODEL_PATH)
    print(f"[ok] replaced -> {MODEL_PATH}")

    if attack_cat_splits:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name, series in attack_cat_splits.items():
            series.to_csv(PROCESSED_DIR / f"attack_cat_{name}.csv",
                          index=False, header=["attack_cat"])

    new_metadata = dict(metadata)
    new_metadata.update({
        "retrained": True,
        "n_retrain_rows": int(len(new_flows)),
        "feature_names": X_train.columns.tolist(),
        "n_features": int(len(X_train.columns)),
    })
    save_artifacts(
        X_train, y_train, X_val, y_val, X_test, y_test,
        scaler, imputers, encoder_state, new_metadata,
    )
    threshold = threshold_for_fpr(
        y_val, new_model.predict_proba(X_val[new_features])[:, 1], 0.01)
    OPERATING_PATH.write_text(json.dumps(
        {**operating, "threshold": threshold, "features": new_features,
         "hardening": hardening}, indent=2), encoding="utf-8")
    write_manifest(MODELS_DIR)
    print("\n[done] models + pipeline replaced. The API serves the new "
          "model on next restart.")


if __name__ == "__main__":
    main()
