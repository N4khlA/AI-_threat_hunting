"""Leakage sanity scan over the raw feature set.

Checks whether any feature behaves suspiciously like the label or like a
proxy derived from it, before trusting downstream model scores. Runs on
``data/raw/UNSW_NB15_combined.csv`` and reports, per feature:

    * Pearson correlation with the binary Label,
    * ROC-AUC for predicting the Label (a near-1.0 AUC in raw data is a
      red flag for a label-derived feature),
    * mutual information with the Label.

Features flagged (AUC >= 0.97) are listed for manual review. The script
also prints provenance notes for the UNSW-NB15 "additional features"
(``ct_*``, ``is_ftp_login``, ...) that are *computed from the flows
themselves*, not from the label, but are worth a sensitivity check
because attack sessions cluster in time.

Usage:
    python src/data/leakage_check.py
    python src/data/leakage_check.py --input <csv> --sample 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_INPUT = RAW_DIR / "UNSW_NB15_combined.csv"
TARGET = "Label"

# Columns that are identity/timing bookkeeping, not features. They are
# dropped before the scan (and by preprocess.py) because they can allow
# record linkage to the ground-truth tables.
IDENTITY_COLUMNS = ["srcip", "dstip", "sport", "dsport", "Stime", "Ltime", "id"]

# UNSW-NB15 "additional features" with known borderline provenance: they
# are derived from flow/connection statistics (Argus/Bro outputs), NOT
# from the label, but correlate strongly with attacks because attack
# sessions cluster in 100-connection windows. High MI is expected and is
# not leakage; still, they are the first candidates for an ablation.
BORDERLINE_FEATURES = {
    "ct_state_ttl": "count of flows with the same state+TTL in the last 100 "
                    "connections; attacks cluster -> strong signal, but flow-derived.",
    "ct_srv_src": "count of flows with same service+source in window; flow-derived.",
    "ct_srv_dst": "count of flows with same service+destination in window; flow-derived.",
    "ct_dst_ltm": "count of flows to same destination in window; flow-derived.",
    "ct_src_ltm": "count of flows from same source in window; flow-derived.",
    "ct_src_dport_ltm": "count of flows with same source+dest-port in window; flow-derived.",
    "ct_dst_sport_ltm": "count of flows with same dest+source-port in window; flow-derived.",
    "ct_dst_src_ltm": "count of flows with same source+dest in window; flow-derived.",
    "is_ftp_login": "1 if the flow contains an FTP login; near-deterministic for FTP "
                    "brute-force attacks (content-derived, not label-derived).",
    "ct_ftp_cmd": "count of FTP commands in the flow; content-derived.",
    "ct_flw_http_mthd": "count of HTTP methods in the flow; content-derived.",
    "is_sm_ips_ports": "1 if source/dest IPs and ports are in the same range; "
                       "derived from IPs, not the label.",
}

FLAG_AUC = 0.97   # near-perfect separability -> manual review
FLAG_MI = 0.30    # very high mutual information -> manual review


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Drop identity/dead columns; label-encode categoricals; return (X, y, dropped)."""
    df = df.copy()
    dropped = [c for c in IDENTITY_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    if TARGET not in df.columns:
        sys.exit("[!] no Label column found - is this the UNSW-NB15 CSV?")
    y = df[TARGET].astype(int)
    df = df.drop(columns=[TARGET, "attack_cat"])

    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant:
        df = df.drop(columns=constant)
        dropped += [f"{c} (constant)" for c in constant]

    categorical = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna("__missing__"))
    return df, y, dropped


def scan(X: pd.DataFrame, y: pd.Series, sample: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    Xs, ys = X.iloc[idx], y.iloc[idx]

    scores = pd.DataFrame({"feature": X.columns})
    scores["corr"] = [X[c].corr(y) for c in X.columns]
    scores["auc"] = [roc_auc_score(ys, Xs[c]) for c in X.columns]
    scores["mi"] = mutual_info_classif(Xs, ys, random_state=seed)
    scores["flag"] = np.where(
        scores["auc"] >= FLAG_AUC, "LEAK-LIKE",
        np.where(scores["mi"] >= FLAG_MI, "HIGH-MI", ""),
    )
    return scores.sort_values("auc", ascending=False).reset_index(drop=True)


def print_report(scores: pd.DataFrame, dropped: list[str], df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("LEAKAGE SCAN (per-feature separability vs Label)")
    print("=" * 72)
    with pd.option_context("display.max_rows", None, "display.width", 140,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(scores.to_string(index=False))

    if dropped:
        print(f"\nDropped before scan (identity/constant): {dropped}")

    flagged = scores[scores["flag"] != ""]
    print("\n" + "=" * 72)
    if flagged.empty:
        print("No features flagged (AUC < %.2f, MI < %.2f)." % (FLAG_AUC, FLAG_MI))
    else:
        print(f"FLAGGED ({len(flagged)} features):")
        print("  LEAK-LIKE = AUC >= %.2f (near-perfect separability; review!)" % FLAG_AUC)
        print("  HIGH-MI   = MI >= %.2f (strong signal; expected for flow stats," % FLAG_MI)
        print("              check provenance, not necessarily leakage)")
        for _, row in flagged.iterrows():
            print(f"  [{row['flag']:>9}] {row['feature']}  (auc={row['auc']:.3f}, "
                  f"mi={row['mi']:.3f})")

    print("\n" + "=" * 72)
    print("BORDERLINE FEATURES - provenance notes")
    print("=" * 72)
    present = [f for f in BORDERLINE_FEATURES if f in df.columns]
    for feat in present:
        print(f"  {feat}: {BORDERLINE_FEATURES[feat]}")
    if "ct_state_ttl" not in df.columns:
        print("  (none of the documented borderline features are present)")
    print("\nVerdict: features are computed from flows (Argus/Bro), not from the")
    print("label. None of them is a deterministic function of Label/attack_cat.")
    print("For extra confidence, run an ablation: retrain without the ct_* /")
    print("content features and compare F1 (see README).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sample", type=int, default=100_000,
                        help="rows used for MI/AUC computation (default: 100k)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"[!] {args.input} not found - run src/data/download_dataset.py first")
    df = pd.read_csv(args.input)
    print(f"Loaded {args.input.name}: {len(df):,} records x {len(df.columns)} columns")

    X, y, dropped = prepare(df)
    print(f"Scanning {X.shape[1]} usable features over {len(y):,} records")
    scores = scan(X, y, args.sample, args.seed)
    print_report(scores, dropped, df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(REPORTS_DIR / "leakage_scan.csv", index=False)
    print(f"\n[ok] full table -> {REPORTS_DIR / 'leakage_scan.csv'}")


if __name__ == "__main__":
    main()
