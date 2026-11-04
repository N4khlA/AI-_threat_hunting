"""Detection metrics that answer SOC questions, not leaderboard ones.

Accuracy on UNSW-NB15's test set (55-68% attacks) says little about a
real network, where well under 1% of flows are malicious. These helpers
report the operating point (TPR/FPR at a threshold chosen on the
validation split), bootstrap confidence intervals, and what a given
FPR means in analyst workload at a realistic base rate.

All functions are pure (numpy in, dict out) and unit-tested.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """TPR (recall on attacks), FPR (benign flows alerted on), precision."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": tp / (tp + fn) if tp + fn else float("nan"),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
    }


def binary_report(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
                  ) -> dict:
    """Threshold-free (AUCs) + operating-point metrics for P(attack)."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "attack_share": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    out.update({k: v for k, v in rates(y_true, pred).items()
                if k in ("tpr", "fpr", "precision")})
    return out


def threshold_for_fpr(y_true: np.ndarray, proba: np.ndarray, max_fpr: float
                      ) -> float:
    """Lowest threshold whose FPR on (y_true, proba) is <= ``max_fpr``.

    Pick it on the validation split, then report test metrics at it, so
    the operating point is never tuned on the data it is graded on.
    """
    fpr, _, thresholds = roc_curve(np.asarray(y_true).astype(int), proba)
    ok = np.where(fpr <= max_fpr)[0]
    # roc_curve's first threshold is +inf (predict nothing); clip to 1.0
    return float(min(thresholds[ok[-1]], 1.0))


def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray,
                 metrics: tuple[str, ...] = ("f1_macro", "roc_auc", "pr_auc"),
                 threshold: float = 0.5, n_boot: int = 200, seed: int = 0,
                 alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for ``binary_report`` metrics (one pass)."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue  # single-class resample: AUC undefined
        report = binary_report(y_true[idx], proba[idx], threshold)
        for m in metrics:
            values[m].append(report[m])
    out = {}
    for m, vals in values.items():
        lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[m] = (float(lo), float(hi))
    return out


def base_rate_table(tpr: float, fpr: float,
                    prevalences: tuple[float, ...] = (0.5, 0.1, 0.01, 0.001),
                    flows_per_day: int = 1_000_000) -> list[dict]:
    """Alert precision and daily alert volume at realistic attack rates.

    precision(pi) = TPR*pi / (TPR*pi + FPR*(1-pi))  (Bayes; the base-rate
    fallacy that sinks most IDS deployments).
    """
    rows = []
    for pi in prevalences:
        true_alerts = tpr * pi * flows_per_day
        false_alerts = fpr * (1 - pi) * flows_per_day
        total = true_alerts + false_alerts
        rows.append({
            "attack_prevalence": pi,
            "alert_precision": true_alerts / total if total else float("nan"),
            "true_alerts_per_day": round(true_alerts),
            "false_alerts_per_day": round(false_alerts),
            "flows_per_day": flows_per_day,
        })
    return rows


def per_family_recall(y_pred: np.ndarray, families: np.ndarray) -> dict[str, float]:
    """Share of each attack family's flows that raised an alert."""
    y_pred = np.asarray(y_pred).astype(int)
    families = np.asarray(families).astype(str)
    out = {}
    for fam in sorted(set(families) - {"Normal"}):
        mask = families == fam
        out[fam] = float(y_pred[mask].mean())
    return out
