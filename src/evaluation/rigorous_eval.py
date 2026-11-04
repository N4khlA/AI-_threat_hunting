"""Same model, five evaluation protocols: how much of the score is real?

(Plus a sixth run: the hardened model the API serves, on the headline
protocol, so the SOC numbers describe what is actually deployed.)

Trains the Random Forest (the grid-searched params from
``reports/random_forest_results.json``) under each split protocol from
``src/data/splits.py`` plus a TTL-artifact ablation, and reports for each:

    * accuracy / macro-F1 / ROC-AUC / PR-AUC with bootstrap 95% CIs,
    * TPR and FPR at 0.5 and at a threshold picked on *validation* for
      FPR <= 1%,
    * per-attack-family recall,
    * for the headline protocol: alert precision and false alerts/day at
      realistic base rates (1M flows/day, 1% / 0.1% attacks).

Why the TTL ablation: in the UNSW-NB15 testbed, attacker and victim
hosts sit at fixed hop distances, so sttl/dttl/ct_state_ttl separate the
classes for reasons that do not transfer to another network (sttl=31 is
benign in 100% of rows). An attacker also controls their initial TTL.

Usage:
    python src/evaluation/rigorous_eval.py            # ~5 min on 8 cores
    python src/evaluation/rigorous_eval.py --n-boot 50 --fast

Outputs: reports/rigorous_eval.json, reports/rigorous_eval.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import assert_disjoint, split_positions  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    base_rate_table,
    binary_report,
    bootstrap_ci,
    per_family_recall,
    threshold_for_fpr,
)
from src.features.preprocess import DEFAULT_INPUT, build_splits, load_raw  # noqa: E402
from src.models.hardening import adversarial_rows  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"
from src.models.hardening import TTL_FEATURES  # noqa: E402
DEFAULT_RF = {"n_estimators": 100, "max_depth": None, "min_samples_split": 5}

# (name, split protocol, dropped features, adversarial training, description)
RUNS = [
    ("random_resplit", "random", (), False,
     "legacy: combined table, random 70/15/15 (what the old README reported)"),
    ("random_resplit_dedup", "random_dedup", (), False,
     "random re-split after removing exact duplicates"),
    ("official", "official", (), False,
     "official training file -> official testing file"),
    ("official_novel", "official_novel", (), False,
     "official split, test restricted to flows never seen in training"),
    ("official_novel_no_ttl", "official_novel", TTL_FEATURES, False,
     "same protocol without the TTL testbed-artifact features"),
    ("official_novel_hardened", "official_novel", TTL_FEATURES, True,
     "the served model: no TTL + adversarial training (headline)"),
]
HEADLINE = "official_novel_hardened"
BASELINE = "official_novel"


def rf_params() -> dict:
    path = REPORTS_DIR / "random_forest_results.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8")).get("best_params") or {}
        return {**DEFAULT_RF, **{k: v for k, v in saved.items() if k in DEFAULT_RF}}
    return DEFAULT_RF


def run_protocol(df: pd.DataFrame, protocol: str, drop: tuple[str, ...],
                 adversarial: bool, params: dict, seed: int, n_boot: int,
                 fast: bool) -> dict:
    parts, stats = split_positions(df, protocol, seed)
    assert_disjoint(parts)
    if fast:  # smoke run: subsample the training partition
        parts["train"] = parts["train"][:: max(1, len(parts["train"]) // 20_000)]
    data = build_splits(df, parts, seed=seed, drop_features=drop)
    X_fit, y_fit = data["X_train"], data["y_train"]
    if adversarial:  # same rows/seed as random_forest.py's hardened model
        names = X_fit.columns.tolist()
        pipeline = {"imputers": data["imputers"], "encoder": data["encoder_state"],
                    "scaler": data["scaler"]}
        aug = adversarial_rows(df.loc[parts["train"]].query("Label == 1"),
                               pipeline, names, seed)
        X_fit = pd.concat([X_fit, aug[names]], ignore_index=True)
        y_fit = pd.concat([pd.Series(np.asarray(y_fit)),
                           pd.Series(np.ones(len(aug), dtype=int))], ignore_index=True)
    model = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    start = time.time()
    model.fit(X_fit, y_fit)
    fit_s = time.time() - start

    y_val, y_test = data["y_val"].to_numpy(), data["y_test"].to_numpy()
    p_val = model.predict_proba(data["X_val"])[:, 1]
    p_test = model.predict_proba(data["X_test"])[:, 1]
    thr_1pct = threshold_for_fpr(y_val, p_val, 0.01)

    at_05 = binary_report(y_test, p_test, 0.5)
    at_1pct = binary_report(y_test, p_test, thr_1pct)
    ci = bootstrap_ci(y_test, p_test, n_boot=n_boot, seed=seed)
    families = data["attack_cat"]["test"].to_numpy()
    return {
        "split_stats": stats,
        "dropped_features": list(drop),
        "adversarial_training": adversarial,
        "fit_seconds": round(fit_s, 1),
        "test_at_0.5": at_05,
        "ci95_at_0.5": ci,
        "val_threshold_fpr_1pct": thr_1pct,
        "test_at_val_fpr_1pct": at_1pct,
        "family_recall_at_0.5": per_family_recall(p_test >= 0.5, families),
        "family_recall_at_val_fpr_1pct": per_family_recall(p_test >= thr_1pct, families),
    }


def fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def render_markdown(results: dict, params: dict) -> str:
    lines = [
        "# Rigorous evaluation: one model, five protocols (+ the served model)",
        "",
        "> Generated by `src/evaluation/rigorous_eval.py`. Random Forest "
        f"`{params}`, SMOTE on train, seed 42. CIs: percentile bootstrap "
        "over test rows.",
        "",
        "| run | test rows | attack share | accuracy | F1-macro (95% CI) | ROC-AUC | PR-AUC | FPR@0.5 |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name, r in results["runs"].items():
        t = r["test_at_0.5"]
        lines.append(
            f"| `{name}` | {t['n']:,} | {t['attack_share']:.2f} | {t['accuracy']:.4f} | "
            f"{t['f1_macro']:.4f} {fmt_ci(r['ci95_at_0.5']['f1_macro'])} | "
            f"{t['roc_auc']:.4f} | {t['pr_auc']:.4f} | {t['fpr']:.4f} |")
    lines += ["", "Run descriptions:", ""]
    lines += [f"- `{n}`: {d}" for n, _, _, _, d in RUNS]

    lines += ["", "## Split hygiene", "",
              "| run | rows removed as duplicates / seen-in-train |", "|---|---|"]
    for name, r in results["runs"].items():
        s = {k: v for k, v in r["split_stats"].items() if k.startswith("dropped")}
        lines.append(f"| `{name}` | {s or '-'} |")

    h = results["runs"][HEADLINE]
    lines += [
        "", f"## Operating point (headline `{HEADLINE}`)", "",
        "Threshold chosen on the **validation** split for FPR <= 1%, "
        "then applied to test:", "",
        "| threshold | TPR (attack recall) | FPR | precision on test mix |",
        "|---:|---:|---:|---:|",
    ]
    for key, label in (("test_at_0.5", "0.5"), ("test_at_val_fpr_1pct",
                                                  f"{h['val_threshold_fpr_1pct']:.3f}")):
        t = h[key]
        lines.append(f"| {label} | {t['tpr']:.4f} | {t['fpr']:.4f} | {t['precision']:.4f} |")

    lines += ["", "### What that means for a SOC (1M flows/day)", "",
              "| attack prevalence | threshold | alert precision | true alerts/day | false alerts/day |",
              "|---:|---:|---:|---:|---:|"]
    for row in results["base_rate"]:
        lines.append(
            f"| {row['attack_prevalence']:.1%} | {row['threshold']} | "
            f"{row['alert_precision']:.3f} | {row['true_alerts_per_day']:,} | "
            f"{row['false_alerts_per_day']:,} |")

    b = results["runs"][BASELINE]
    lines += ["", f"## Per-family recall (served `{HEADLINE}` vs plain `{BASELINE}`)", "",
              "| family | served @0.5 | served @val-FPR-1% | plain @0.5 | plain @val-FPR-1% |",
              "|---|---:|---:|---:|---:|"]
    for fam, rec in h["family_recall_at_0.5"].items():
        lines.append(f"| {fam} | {rec:.3f} | "
                     f"{h['family_recall_at_val_fpr_1pct'][fam]:.3f} | "
                     f"{b['family_recall_at_0.5'][fam]:.3f} | "
                     f"{b['family_recall_at_val_fpr_1pct'][fam]:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--fast", action="store_true",
                        help="subsample training to ~20k rows (smoke test)")
    args = parser.parse_args()

    df = load_raw(args.input)
    params = rf_params()
    results: dict = {"rf_params": params, "runs": {}}
    for name, protocol, drop, adversarial, _ in RUNS:
        print(f"\n=== {name} ===")
        results["runs"][name] = run_protocol(
            df, protocol, drop, adversarial, params, args.seed, args.n_boot, args.fast)
        t = results["runs"][name]["test_at_0.5"]
        print(f"  acc={t['accuracy']:.4f} f1={t['f1_macro']:.4f} "
              f"auc={t['roc_auc']:.4f} fpr={t['fpr']:.4f}")

    h = results["runs"][HEADLINE]
    results["base_rate"] = []
    for key, thr in (("test_at_0.5", 0.5),
                     ("test_at_val_fpr_1pct", round(h["val_threshold_fpr_1pct"], 3))):
        for row in base_rate_table(h[key]["tpr"], h[key]["fpr"], (0.01, 0.001)):
            results["base_rate"].append({**row, "threshold": thr})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "rigorous_eval.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    (REPORTS_DIR / "rigorous_eval.md").write_text(
        render_markdown(results, params), encoding="utf-8")
    print(f"\n[ok] -> {REPORTS_DIR / 'rigorous_eval.md'}")


if __name__ == "__main__":
    main()
