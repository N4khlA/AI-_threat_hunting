"""Poisoning PoC: the v1 label path let any client blind the model to one family.

The v1 API accepted ``Label`` in the unauthenticated /predict payload,
logged it as ground truth, and ``retrain.py`` trained on it. An attacker
who knows that only needs to replay their own tool's flows with
``"Label": 0`` (MITRE ATLAS AML.T0020 Poison Training Data).

This script measures what that buys them, offline and reproducibly:

1. incumbent  = RF trained on 60% of the training split (the deployed
                model before a retrain),
2. candidate  = RF trained on 100% of the training split - a *genuine*
                improvement from new data,
3. poisoned   = candidate + n copies of ``--family`` training flows
                labelled benign (the attacker's submissions), for several
                budgets n (as % of the training set).

For every poisoned model it reports aggregate accuracy/F1, the target
family's recall, and the verdict of two guards against the incumbent:

    v1 guard  accuracy / F1 must improve           (what shipped)
    v2 guard  ... and no family's recall may drop  (src/models/retrain.py)

The point: poison rides along with a real improvement, aggregate metrics
barely move, so an aggregate-only guard waves it through. The fixes are
(a) labels only via the authenticated analyst ``/feedback`` endpoint and
(b) the per-family recall tripwire. Both are covered by tests.

Usage:
    python src/attacks/poisoning.py --family Backdoor
Outputs: reports/poisoning.json, reports/poisoning.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.retrain import evaluate, should_replace  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
BUDGETS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)


def poison_rows(X: pd.DataFrame, families: pd.Series, family: str, n: int,
                seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """``n`` attacker submissions: copies of ``family`` flows labelled benign."""
    pool = X.loc[families.to_numpy() == family]
    if pool.empty or n <= 0:
        return X.iloc[:0], pd.Series([], dtype=int)
    rows = pool.sample(n, replace=True, random_state=seed)
    return rows, pd.Series(np.zeros(n, dtype=int), index=rows.index)


def v1_guard(old: dict, new: dict) -> bool:
    strip = lambda m: {k: m[k] for k in ("accuracy", "f1_macro")}  # noqa: E731
    return should_replace(strip(old), strip(new))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="Backdoor")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_train.csv")["label"]
    cats = pd.read_csv(PROCESSED_DIR / "attack_cat_train.csv")["attack_cat"]
    # SMOTE appends synthetic rows after the originals; they are 'Normal'
    # minority samples here, so pad the family labels accordingly.
    fam = pd.concat([cats, pd.Series(["Normal"] * (len(X) - len(cats)))],
                    ignore_index=True)
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["label"]
    fam_test = pd.read_csv(PROCESSED_DIR / "attack_cat_test.csv")["attack_cat"]

    def fit(Xf, yf):
        return RandomForestClassifier(n_estimators=100, min_samples_split=5,
                                      n_jobs=-1, random_state=args.seed).fit(Xf, yf)

    rng = np.random.default_rng(args.seed)
    subset = rng.random(len(X)) < 0.6
    incumbent = evaluate(fit(X[subset], y[subset]), X_test, y_test, fam_test)
    print(f"incumbent: acc={incumbent['accuracy']:.4f} "
          f"{args.family} recall={incumbent['family_recall'][args.family]:.3f}")

    runs = []
    for budget in BUDGETS:
        n = int(round(budget * len(X)))
        Xp, yp = poison_rows(X, fam, args.family, n, args.seed)
        model = fit(pd.concat([X, Xp], ignore_index=True),
                    pd.concat([y, yp], ignore_index=True))
        m = evaluate(model, X_test, y_test, fam_test)
        run = {
            "budget_share_of_train": budget, "poison_rows": n,
            "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
            "target_recall": m["family_recall"][args.family],
            "other_families_min_recall_delta": min(
                m["family_recall"][f] - incumbent["family_recall"][f]
                for f in m["family_recall"] if f != args.family),
            "v1_guard_accepts": v1_guard(incumbent, m),
            "v2_guard_accepts": should_replace(incumbent, m)[0],
            "v2_reason": should_replace(incumbent, m)[1],
        }
        runs.append(run)
        print(f"budget={budget:.2%} n={n:5d} acc={m['accuracy']:.4f} "
              f"{args.family} recall={run['target_recall']:.3f} "
              f"v1={run['v1_guard_accepts']} v2={run['v2_guard_accepts']}")

    result = {"family": args.family, "train_rows": int(len(X)),
              "incumbent": incumbent, "runs": runs}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "poisoning.json").write_text(json.dumps(result, indent=2),
                                                encoding="utf-8")
    lines = [
        f"# Label-poisoning PoC: blinding the model to `{args.family}`", "",
        "> Generated by `src/attacks/poisoning.py`. Incumbent = RF on 60% of "
        "train; each candidate = RF on 100% of train + `n` attacker-submitted "
        f"copies of {args.family} flows labelled benign (the v1 `/predict` "
        "`Label` field).", "",
        f"Incumbent: accuracy {incumbent['accuracy']:.4f}, "
        f"{args.family} recall {incumbent['family_recall'][args.family]:.3f}.", "",
        "| poison budget | rows | accuracy | F1-macro | "
        f"{args.family} recall | v1 guard | v2 guard |",
        "|---:|---:|---:|---:|---:|---|---|"]
    for r in runs:
        lines.append(
            f"| {r['budget_share_of_train']:.2%} | {r['poison_rows']:,} | "
            f"{r['accuracy']:.4f} | {r['f1_macro']:.4f} | {r['target_recall']:.3f} | "
            f"{'ACCEPT' if r['v1_guard_accepts'] else 'refuse'} | "
            f"{'ACCEPT' if r['v2_guard_accepts'] else 'refuse'} |")
    (REPORTS_DIR / "poisoning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] -> {REPORTS_DIR / 'poisoning.md'}")


if __name__ == "__main__":
    main()
