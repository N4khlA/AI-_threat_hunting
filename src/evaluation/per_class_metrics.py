"""Per-class and per-attack-family evaluation of the trained classifier.

Loads the test split plus the per-split ``attack_cat`` labels saved by
``src/features/preprocess.py`` and reports:

    1. the full classification report (per-class precision/recall/F1 for
       the binary normal-vs-attack model),
    2. per-attack-family detection rates: for each family (Normal,
       Generic, Exploits, ..., Worms) the fraction of its records the
       binary model flags as an attack, plus the false-positive rate on
       Normal traffic.

Rare families (Shellcode, Worms) often dominate the F1-macro penalty;
this report makes that visible before investing in a more complex model.

Usage:
    python src/evaluation/per_class_metrics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_MODEL = MODELS_DIR / "random_forest_best.joblib"


def load_test_data(processed_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv", usecols=["label"]).iloc[:, 0]
    return X_test, y_test


def load_attack_cat(processed_dir: Path) -> pd.Series | None:
    path = processed_dir / "attack_cat_test.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=["attack_cat"]).iloc[:, 0]


def per_family_table(
    y_test: pd.Series, pred: pd.Series, families: pd.Series
) -> pd.DataFrame:
    """Detection rate + FP rate per attack family (alignment by position)."""
    rows = []
    for family, group in pd.DataFrame(
        {"actual": y_test.values, "pred": pred.values, "family": families.values}
    ).groupby("family", sort=True):
        n = len(group)
        positives = int((group["actual"] == 1).sum())
        detected = int(((group["pred"] == 1) & (group["actual"] == 1)).sum())
        fpr = ((group["pred"] == 1) & (group["actual"] == 0)).sum() / max(n - positives, 1)
        rows.append({
            "family": family,
            "records": n,
            "attack_records": positives,
            "detected_as_attack": detected,
            "detection_rate": detected / max(positives, 1),
            "false_positive_rate": fpr,
        })
    table = pd.DataFrame(rows).sort_values("records", ascending=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="path to the trained classifier (joblib)")
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR,
                        help="directory with processed splits")
    args = parser.parse_args()

    X_test, y_test = load_test_data(args.processed)
    pred = pd.Series(joblib.load(args.model).predict(X_test), index=y_test.index)

    print("\n" + "=" * 66)
    print("PER-CLASS REPORT (binary: 0 = normal, 1 = attack)")
    print("=" * 66)
    report_text = classification_report(y_test, pred, digits=4)
    print(report_text)

    families = load_attack_cat(args.processed)
    if families is None:
        print("[!] attack_cat_test.csv not found; re-run preprocess to "
              "enable per-family breakdown.", file=sys.stderr)
        return

    table = per_family_table(y_test, pred, families)
    print("=" * 66)
    print("PER-ATTACK-FAMILY DETECTION (binary model)")
    print("=" * 66)
    with pd.option_context("display.max_rows", None, "display.width", 120,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))
    print("\nNote: 'detection_rate' = fraction of that family flagged as an")
    print("attack; for Normal it equals the false-positive rate.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "per_class_metrics.txt").write_text(
        report_text + "\n" + table.to_string(index=False), encoding="utf-8")
    (REPORTS_DIR / "per_class_metrics.json").write_text(
        json.dumps({
            "classification_report": report_text,
            "per_family": table.to_dict(orient="records"),
        }, indent=2), encoding="utf-8")
    print(f"\n[ok] -> {REPORTS_DIR / 'per_class_metrics.txt'}")


if __name__ == "__main__":
    main()
