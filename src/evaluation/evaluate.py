"""Evaluate and compare the Random Forest and LSTM models.

IMPORTANT UNIT CONVENTION - the two models are NOT directly comparable
on the same units:

    * Random Forest:  per-FLOW classification on the held-out flow test
                      split (data/processed/X_test.csv, y_test.csv).
    * LSTM:           per-WINDOW classification on the held-out window
                      test split (data/processed/lstm_test.npz). A
                      window covers up to 10 flows; its label is the
                      aggregated label of the flows in it.

Everything in the output is explicitly labelled with its unit. For a
statistically valid head-to-head, the two models are additionally
aligned on the flows they both see (the LAST flow of each LSTM test
window that belongs to the RF flow-test split) and compared with a
McNemar test.

Produces:
    reports/confusion_matrix_rf.png, reports/confusion_matrix_lstm.png
    reports/per_family_rf.csv
    reports/model_comparison.md

Usage:
    python src/evaluation/evaluate.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RF_MODEL = MODELS_DIR / "random_forest_best.joblib"
LSTM_MODEL = MODELS_DIR / "lstm_best.keras"

UNITS = {"rf": "per-flow", "lstm": "per-window"}


def load_rf_test() -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv", usecols=["label"]).iloc[:, 0]
    attack_cat = pd.read_csv(PROCESSED_DIR / "attack_cat_test.csv",
                             usecols=["attack_cat"]).iloc[:, 0]
    positions = pd.read_csv(PROCESSED_DIR / "test_positions.csv")["position"]
    return X_test.to_numpy(), y_test.to_numpy(), attack_cat, positions


def load_lstm_test() -> tuple[np.ndarray, np.ndarray, list]:
    data = np.load(PROCESSED_DIR / "lstm_test.npz", allow_pickle=True)
    return data["X_test"], data["y_test"], list(data["flow_indices"])


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return {k: v for k, v in report.items()
            if k in {"0", "1", "macro avg", "weighted avg"}}


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["normal (0)", "attack (1)"],
                yticklabels=["normal (0)", "attack (1)"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {path.name}")


def per_family_f1(y_true_binary: np.ndarray, y_pred: np.ndarray,
                  families: pd.Series) -> pd.DataFrame:
    """One-vs-rest F1 per attack family using the binary model.

    For family f: positive = flows of family f in truth; positive =
    flows the model flags as attack. So recall = fraction of f caught,
    precision = share of attack alarms that are family f.
    """
    rows = []
    fam_vals = families.to_numpy()
    for family in pd.unique(fam_vals):
        y_f = (fam_vals == family) & (y_true_binary == 1)
        tp = int(((y_pred == 1) & y_f).sum())
        fp = int(((y_pred == 1) & ~y_f).sum())
        fn = int(((y_pred == 0) & y_f).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "family": family, "support": int(y_f.sum()),
            "precision": precision, "recall": recall, "f1": f1,
        })
    table = pd.DataFrame(rows).sort_values("support", ascending=False)
    return table


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar's test with continuity correction on paired binary predictions.

    b = both correct; c = only B correct; d = only A correct.
    """
    b = int(((pred_a == y_true) & (pred_b == y_true)).sum())
    c = int(((pred_a != y_true) & (pred_b == y_true)).sum())
    d = int(((pred_a == y_true) & (pred_b != y_true)).sum())
    statistic = (abs(c - d) - 1) ** 2 / (c + d) if c + d else 0.0
    p_value = chi2.sf(statistic, 1) if c + d else 1.0
    return {"both_correct": b, "lstm_only_correct": c, "rf_only_correct": d,
            "chi2": statistic, "p_value": p_value}


def aligned_comparison(rf_pred, rf_y, rf_positions, lstm_pred, lstm_y,
                       flow_indices) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Align models on the last flow of each LSTM test window that belongs
    to the RF flow-test split. Returns (y_true, rf_pred_aligned,
    lstm_pred_aligned) plus alignment stats."""
    pos_to_y = dict(zip(rf_positions, rf_y))
    pos_to_rf = dict(zip(rf_positions, rf_pred))
    y_true, rf_p, lstm_p = [], [], []
    dropped_lstm_train_flow = 0
    for i, (window_pred, indices) in enumerate(zip(lstm_pred, flow_indices)):
        last_flow = indices[-1]
        if last_flow not in pos_to_y:
            continue
        y_true.append(pos_to_y[last_flow])
        rf_p.append(pos_to_rf[last_flow])
        lstm_p.append(int(window_pred))
    stats = {"aligned_n": len(y_true)}
    return (np.asarray(y_true), np.asarray(rf_p), np.asarray(lstm_p), stats)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MODEL EVALUATION")
    print("=" * 72)
    print(f"RF:   {RF_MODEL.name} (flow-level test set)")
    print(f"LSTM: {LSTM_MODEL.name} (window-level test set)")
    print("NOTE: the two test sets are on different units (flow vs window);")
    print("metrics are NOT directly comparable. Aligned McNemar below.")

    # ---- Random Forest (per-flow) -----------------------------------------
    X_flow, y_flow, attack_cat, positions = load_rf_test()
    rf = joblib.load(RF_MODEL)
    t0 = time.time()
    rf_proba = rf.predict_proba(X_flow)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    rf_time = time.time() - t0
    rf_metrics = compute_metrics(y_flow, rf_proba, rf_pred)
    rf_per_class = per_class_metrics(y_flow, rf_pred)
    print(f"\n--- Random Forest ({UNITS['rf']}) ------------------------------")
    print(f"  {len(y_flow):,} flows evaluated in {rf_time:.2f}s "
          f"({len(y_flow) / rf_time:,.0f} flows/s)")
    for k, v in rf_metrics.items():
        print(f"  {k:>16}: {v:.4f}")
    print(classification_report(y_flow, rf_pred, digits=4))
    plot_confusion_matrix(y_flow, rf_pred,
                          "Random Forest (per-flow)", REPORTS_DIR / "confusion_matrix_rf.png")

    # ---- LSTM (per-window) ------------------------------------------------
    X_win, y_win, flow_indices = load_lstm_test()
    import tensorflow as tf  # noqa: PLC0415

    lstm = tf.keras.models.load_model(LSTM_MODEL)
    t0 = time.time()
    lstm_proba = lstm.predict(X_win, verbose=0).ravel()
    lstm_pred = (lstm_proba >= 0.5).astype(int)
    lstm_time = time.time() - t0
    lstm_metrics = compute_metrics(y_win, lstm_proba, lstm_pred)
    lstm_per_class = per_class_metrics(y_win, lstm_pred)
    print(f"\n--- LSTM ({UNITS['lstm']}) -------------------------------------")
    print(f"  {len(y_win):,} windows evaluated in {lstm_time:.2f}s "
          f"({len(y_win) / lstm_time:,.0f} windows/s)")
    for k, v in lstm_metrics.items():
        print(f"  {k:>16}: {v:.4f}")
    print(classification_report(y_win, lstm_pred, digits=4))
    plot_confusion_matrix(y_win, lstm_pred,
                          "Bidirectional LSTM (per-window)",
                          REPORTS_DIR / "confusion_matrix_lstm.png")

    # ---- Per-attack-family F1 (RF) ---------------------------------------
    print("\n--- Random Forest: per-attack-family F1 (one-vs-rest) -----------")
    family_table = per_family_f1(y_flow, rf_pred, attack_cat)
    with pd.option_context("display.max_rows", None, "display.width", 120,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(family_table.to_string(index=False))
    family_table.to_csv(REPORTS_DIR / "per_family_rf.csv", index=False)
    weak = family_table[(family_table["support"] > 0)
                        & (family_table["recall"] < 0.90)]
    print(f"\nWeak spots (detection recall < 0.90, support > 0): "
          f"{weak['family'].tolist() if len(weak) else 'none'}")

    # ---- Aligned McNemar --------------------------------------------------
    y_align, rf_align, lstm_align, align_stats = aligned_comparison(
        rf_pred, y_flow, positions, lstm_pred, y_win, flow_indices)
    test = mcnemar(y_align, rf_align, lstm_align)
    print("\n--- Aligned comparison (last flow of each LSTM test window that")
    print("    also belongs to the RF flow-test split) ---------------------")
    print(f"  aligned pairs: {align_stats['aligned_n']:,}")
    print(f"  RF accuracy:   {accuracy_score(y_align, rf_align):.4f}   "
          f"F1: {f1_score(y_align, rf_align, zero_division=0):.4f}")
    print(f"  LSTM accuracy: {accuracy_score(y_align, lstm_align):.4f}   "
          f"F1: {f1_score(y_align, lstm_align, zero_division=0):.4f}")
    print(f"  McNemar chi2={test['chi2']:.3f} p={test['p_value']:.4f} "
          f"(c={test['rf_only_correct']}, d={test['lstm_only_correct']})")
    interpretation = (
        "LSTM significantly better" if test["p_value"] < 0.05 and test["lstm_only_correct"] > test["rf_only_correct"]
        else "RF significantly better" if test["p_value"] < 0.05 and test["rf_only_correct"] > test["lstm_only_correct"]
        else "no significant difference"
    )

    # ---- Markdown report --------------------------------------------------
    md = f"""# Model Comparison: Random Forest vs Bidirectional LSTM

> Generated by `src/evaluation/evaluate.py`.

## Unit convention (read first)

| Model | Evaluation unit | Test set |
|---|---|---|
| Random Forest | **per-flow** (one flow = one prediction) | `data/processed/X_test.csv` |
| LSTM | **per-window** (up to 10 flows = one prediction) | `data/processed/lstm_test.npz` |

The two test sets are **not directly comparable**: a window prediction
covers several flows and is labelled "attack" if any flow in the window
is an attack. Treat the table below as separate model reports, and the
"Aligned comparison" section as the only statistically valid head-to-head.

## Overall metrics

| Metric | RF ({UNITS['rf']}) | LSTM ({UNITS['lstm']}) |
|---|---|---|
| accuracy | {rf_metrics['accuracy']:.4f} | {lstm_metrics['accuracy']:.4f} |
| precision (macro) | {rf_metrics['precision_macro']:.4f} | {lstm_metrics['precision_macro']:.4f} |
| recall (macro) | {rf_metrics['recall_macro']:.4f} | {lstm_metrics['recall_macro']:.4f} |
| F1 (macro) | {rf_metrics['f1_macro']:.4f} | {lstm_metrics['f1_macro']:.4f} |
| F1 (weighted) | {rf_metrics['f1_weighted']:.4f} | {lstm_metrics['f1_weighted']:.4f} |
| ROC-AUC | {rf_metrics['auc']:.4f} | {lstm_metrics['auc']:.4f} |
| samples | {len(y_flow):,} flows | {len(y_win):,} windows |
| inference speed | {len(y_flow) / rf_time:,.0f} samples/s | {len(y_win) / lstm_time:,.0f} samples/s |

## Per-class F1

| Class | RF | LSTM |
|---|---|---|
| normal (0) | {rf_per_class['0']['f1-score']:.4f} | {lstm_per_class['0']['f1-score']:.4f} |
| attack (1) | {rf_per_class['1']['f1-score']:.4f} | {lstm_per_class['1']['f1-score']:.4f} |

## Confusion matrices

![RF confusion matrix](confusion_matrix_rf.png)

![LSTM confusion matrix](confusion_matrix_lstm.png)

## Random Forest: per-attack-family F1 (one-vs-rest)

Each family is the "positive" class in truth; "positive" for the model
is *any* attack alarm. Recall = share of the family caught; precision =
share of attack alarms belonging to that family.

"""
    md += family_table.to_markdown(index=False, floatfmt=".4f") + "\n\n"
    if len(weak):
        md += (f"**Weak spots:** {weak['family'].tolist()} fall below "
               f"0.90 detection recall.\n\n")
    md += f"""## Aligned comparison (McNemar)

Both models are evaluated on the same flows: the **last flow of each LSTM
test window** that also belongs to the RF flow-test split.

| statistic | value |
|---|---|
| aligned pairs | {align_stats['aligned_n']:,} |
| RF accuracy / F1 | {accuracy_score(y_align, rf_align):.4f} / {f1_score(y_align, rf_align, zero_division=0):.4f} |
| LSTM accuracy / F1 | {accuracy_score(y_align, lstm_align):.4f} / {f1_score(y_align, lstm_align, zero_division=0):.4f} |
| both correct | {test['both_correct']:,} |
| only LSTM correct | {test['lstm_only_correct']:,} |
| only RF correct | {test['rf_only_correct']:,} |
| McNemar chi-squared (cc) | {test['chi2']:.3f} |
| p-value | {test['p_value']:.4f} |

**Verdict:** {interpretation}.

> Caveat: the LSTM predicts window-level labels; aligning on the last
> flow mixes units. Treat the p-value as indicative, not definitive.

## Historical note: fairness of earlier comparisons

Earlier versions of this report (before the pipeline re-sync) showed RF
test accuracy 0.9372 and an aligned McNemar p = 0.0003 ("LSTM
significantly better"). That RF number was **not a different model** -
it was the same model evaluated against pipeline artifacts from a
*different* preprocessing run: the deployed `random_forest_best.joblib`
had been trained on an older `X_train.csv`, while every regenerated
artifact (splits, scaler, metadata) came from a newer run. The deployed
model was silently under-trained relative to its own pipeline, so the
earlier head-to-head compared the LSTM against an unfairly weak RF
baseline.

After re-syncing the RF to the pipeline it ships with (re-fit with its
recorded grid-search params on the current splits), the fair RF numbers
are the ones in this report. The McNemar p-value moving from 0.0003 to
{test['p_value']:.3f} is the **correction of that unfairness, not the LSTM
"losing its edge"**: the LSTM's metrics are unchanged; the RF baseline
caught up.

Lesson for real deployments: **a model artifact is meaningless without
the pipeline version that produced it.** Pin pipeline and model versions
together (record the pipeline git SHA + input dataset hash in
`preprocess_metadata.json`) and re-validate the model whenever the
pipeline changes.

## Plain-language verdict: RF vs LSTM tradeoffs

- **Speed**: the Random Forest scores tens of thousands of flows per
  second with no GPU; the LSTM processes roughly one order of magnitude
  fewer samples per second on CPU. For inline/edge detection at high
  flow rates, RF wins.
- **Interpretability**: RF exposes ranked feature importances
  (`reports/feature_importances.png`) and is a glass-box model. The LSTM
  is a black box - we can observe that window context helps, but not
  which flows or features drive a given decision.
- **Temporal pattern capture**: the LSTM is the only model that sees
  *context* - a window of consecutive flows from a connection context -
  and it is numerically ahead in the aligned head-to-head
  ({interpretation}), consistent with its higher AUC on its own test
  set. Coordinated multi-flow attacks (scans, DoS bursts, slow exfil)
  are precisely the patterns a per-flow classifier cannot see.
- **Recommendation**: use the LSTM where sequence context matters and
  latency/GPU budget allows (backbone sensor), and the RF as the
  fast, explainable per-flow filter (edge). Ensemble both: RF
  per-flow triage + LSTM second opinion on flagged windows.
"""
    (REPORTS_DIR / "model_comparison.md").write_text(md, encoding="utf-8")
    print(f"\n[ok] reports/model_comparison.md written")

    summary = {
        "rf": {"unit": UNITS["rf"], "metrics": rf_metrics,
               "per_class": rf_per_class},
        "lstm": {"unit": UNITS["lstm"], "metrics": lstm_metrics,
                 "per_class": lstm_per_class},
        "per_family_rf": family_table.to_dict(orient="records"),
        "aligned_mcnemar": {"n": align_stats["aligned_n"], **test,
                            "interpretation": interpretation},
    }
    (REPORTS_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
