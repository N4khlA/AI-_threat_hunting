"""Exploratory data analysis for the UNSW-NB15 dataset.

This is the scripted equivalent of ``notebooks/01_eda.ipynb``. It loads the
raw dataset, prints the class distribution (binary ``Label`` and multi-class
``attack_cat``), summarizes the feature types, and reports missing values.

Usage:
    python src/data/explore.py                          # uses UNSW_NB15_combined.csv
    python src/data/explore.py --input data/raw/UNSW_NB15_training-set.csv
    python src/data/explore.py --no-plot                # skip figures, save text report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless-safe figure generation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))  # allow `import src...` from any cwd

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_INPUT = RAW_DIR / "UNSW_NB15_combined.csv"


def load_data(path: Path) -> pd.DataFrame:
    """Load a UNSW-NB15 CSV and report its shape."""
    df = pd.read_csv(path)
    print(f"\nLoaded {path.name}: {df.shape[0]:,} records x {df.shape[1]} features")
    return df


def class_distribution(df: pd.DataFrame) -> None:
    """Print binary (Label) and multi-class (attack_cat) distributions."""
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION - binary label (0 = normal, 1 = attack)")
    print("=" * 70)
    label_counts = df["Label"].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  {int(label):>8}  {count:>10,}  ({count / len(df):.2%})")

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION - attack categories (attack_cat)")
    print("=" * 70)
    if "attack_cat" in df.columns:
        cat_counts = df["attack_cat"].value_counts(dropna=False)
        for cat, count in cat_counts.items():
            print(f"  {str(cat):>18}  {count:>10,}  ({count / len(df):.2%})")
    else:
        print("  (attack_cat column not present)")

    print(f"\nImbalance ratio (normal:attack): 1 : {label_counts.max() / label_counts.min():.2f}")


def feature_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tabulate each column's dtype, role, and a hint about its nature."""
    print("\n" + "=" * 70)
    print("FEATURE TYPE SUMMARY")
    print("=" * 70)
    summary = []
    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=False)
        if pd.api.types.is_numeric_dtype(series):
            kind = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            kind = "datetime"
        else:
            kind = "categorical"
        if n_unique <= 2:
            role = "binary"
        elif kind == "categorical" and n_unique <= 20:
            role = "low-cardinality"
        elif kind == "categorical":
            role = "high-cardinality"
        else:
            role = kind
        summary.append(
            {"column": col, "dtype": str(series.dtype), "kind": kind, "role": role,
             "n_unique": n_unique, "missing": int(series.isna().sum())}
        )
    table = pd.DataFrame(summary).sort_values(["kind", "role", "column"])
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(table.to_string(index=False))
    return table


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count missing values per column and flag columns with any."""
    print("\n" + "=" * 70)
    print("MISSING VALUE REPORT")
    print("=" * 70)
    missing = (
        pd.DataFrame({"column": df.columns})
        .assign(
            missing=df.isna().sum().values,
            pct=(df.isna().mean() * 100).round(2).values,
        )
        .query("missing > 0")
        .sort_values("missing", ascending=False)
    )
    if missing.empty:
        print("  No missing values found.")
    else:
        print(missing.to_string(index=False))
    return missing


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> Path:
    """Save class distribution and numeric-feature distribution figures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    label_counts = df["Label"].value_counts().sort_index()
    axes[0].bar(["Normal (0)", "Attack (1)"], label_counts.values,
                color=["tab:green", "tab:red"])
    axes[0].set_title("Binary label distribution")
    axes[0].set_ylabel("records")

    if "attack_cat" in df.columns:
        cats = df["attack_cat"].value_counts(dropna=False)
        axes[1].barh(cats.index[::-1], cats.values[::-1], color="tab:blue")
        axes[1].set_title("Attack categories")
        axes[1].set_xscale("log")
    fig.tight_layout()
    dist_path = out_dir / "class_distribution.png"
    fig.savefig(dist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] saved {dist_path.name}")

    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns
                    if c not in {"Label"}]
    sample_cols = numeric_cols[:6]
    if sample_cols:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, col in zip(axes.ravel(), sample_cols):
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            values = values[values < np.percentile(values, 99)]  # trim outliers
            ax.hist(values, bins=60, color="tab:purple")
            ax.set_title(col)
        fig.suptitle("Numeric feature distributions (99th-percentile trimmed)")
        fig.tight_layout()
        hist_path = out_dir / "numeric_distributions.png"
        fig.savefig(hist_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [ok] saved {hist_path.name}")
    return dist_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="path to the raw CSV (default: data/raw/UNSW_NB15_combined.csv)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip generating figures")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[!] {args.input} not found. Run first:\n"
              f"    python src/data/download_dataset.py", file=sys.stderr)
        sys.exit(1)

    df = load_data(args.input)

    type_table = feature_type_summary(df)
    missing_table = missing_value_report(df)
    class_distribution(df)

    # Compact per-kind tallies for the text report
    kind_counts = type_table["kind"].value_counts().to_dict()
    total_missing = int(missing_table["missing"].sum()) if not missing_table.empty else 0
    report = "\n".join([
        "=" * 70,
        "UNSW-NB15 EDA SUMMARY",
        "=" * 70,
        f"records: {len(df):,}   features: {len(df.columns)}",
        f"feature kinds: {kind_counts}",
        f"total missing values: {total_missing:,}",
        "",
        "class distribution (binary Label):",
        *[f"  {int(k)}: {v:,} ({v / len(df):.2%})"
          for k, v in df['Label'].value_counts().sort_index().items()],
    ])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "eda_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[ok] text summary -> {report_path}")

    if not args.no_plot:
        plot_distributions(df, REPORTS_DIR)


if __name__ == "__main__":
    main()
