"""Attack-family classifier: which kind of attack is this flagged flow?

The binary Random Forest answers "attack or not". Earlier versions of
the API then reported ``P(attack) * P(family)`` with *fixed training
priors*, which gave every flow the same family ranking. This model is a
real per-flow estimate: a multi-class Random Forest trained only on the
attack rows of the training split (same fitted preprocessing pipeline,
no SMOTE), used to pick the ATT&CK mapping of an alert.

Evaluated on the attack rows of the held-out test split. UNSW-NB15's
DoS / Exploits / Analysis / Backdoor families share near-identical flow
statistics, so expect those to be confused; the report says by how much.

Usage:
    python src/models/family_classifier.py

Outputs: models/family_rf.joblib, reports/family_classifier.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocess import DEFAULT_INPUT, load_raw  # noqa: E402
from src.models.lstm_model import transform_flows  # noqa: E402
from src.models.manifest import write_manifest  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "family_rf.joblib"


def attack_rows(raw: pd.DataFrame, positions: pd.Series, pipeline: dict,
                feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    part = raw.loc[positions]
    part = part[part["attack_cat"] != "Normal"]
    X = pd.DataFrame(transform_flows(part, pipeline, feature_names),
                     columns=feature_names, index=part.index)
    return X, part["attack_cat"].astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = load_raw(args.input)
    pipeline = joblib.load(MODELS_DIR / "preprocess_pipeline.joblib")
    metadata = json.loads((MODELS_DIR / "preprocess_metadata.json")
                          .read_text(encoding="utf-8"))
    names = metadata["feature_names"]
    pos = {k: pd.read_csv(PROCESSED_DIR / f"{k}_positions.csv")["position"]
           for k in ("train", "test")}
    X_train, y_train = attack_rows(raw, pos["train"], pipeline, names)
    X_test, y_test = attack_rows(raw, pos["test"], pipeline, names)
    print(f"attack rows: train={len(y_train):,} test={len(y_test):,}")

    model = RandomForestClassifier(n_estimators=args.n_estimators,
                                   min_samples_split=5, class_weight="balanced",
                                   n_jobs=-1, random_state=args.seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = sorted(y_train.unique())
    report = classification_report(y_test, pred, labels=labels,
                                   output_dict=True, zero_division=0)
    result = {
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "accuracy": float(report["accuracy"]) if "accuracy" in report
        else float((pred == y_test.to_numpy()).mean()),
        "per_family": {f: {k: round(float(report[f][k]), 4)
                           for k in ("precision", "recall", "f1-score", "support")}
                       for f in labels},
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
    }
    print(classification_report(y_test, pred, labels=labels, zero_division=0))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    write_manifest(MODELS_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "family_classifier.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"[ok] {MODEL_PATH}  macro-F1={result['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
