"""Train and tune a Random Forest classifier on the processed splits.

Loads the splits produced by ``src/features/preprocess.py``, performs a
grid search over key hyperparameters, and reports:

    * the best hyperparameters (printed, plus saved via joblib),
    * feature importances ranked descending (printed + bar chart in
      ``reports/feature_importances.png``),
    * validation accuracy/F1 as a sanity check.

The ``--fast`` flag subsamples the training set and shrinks the grid so
the pipeline can be smoke-tested quickly; omit it for the full run.

Usage:
    python src/models/random_forest.py
    python src/models/random_forest.py --fast
    python src/models/random_forest.py --n-jobs 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import threshold_for_fpr  # noqa: E402
from src.models.manifest import write_manifest  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "random_forest_best.joblib"

FULL_GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [10, 25, None],
    "min_samples_split": [2, 5, 10],
}
FAST_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [10, None],
    "min_samples_split": [2, 5],
}
FAST_SAMPLE = 25_000

# UNSW-NB15 "additional features": flow-derived window counts / content
# indicators with borderline provenance. `--drop-borderline` removes them
# for an ablation so you can quantify how much of the score relies on them.
BORDERLINE_FEATURES = [
    "ct_state_ttl", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "is_sm_ips_ports",
]


def load_splits(fast: bool, seed: int, drop_borderline: bool = False
                ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load processed CSVs; optionally subsample the training split."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv", usecols=["label"]).iloc[:, 0]
    X_val = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    y_val = pd.read_csv(PROCESSED_DIR / "y_val.csv", usecols=["label"]).iloc[:, 0]
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv", usecols=["label"]).iloc[:, 0]
    if drop_borderline:
        kept = [c for c in X_train.columns if c not in BORDERLINE_FEATURES]
        removed = [c for c in X_train.columns if c in BORDERLINE_FEATURES]
        X_train, X_val, X_test = X_train[kept], X_val[kept], X_test[kept]
        print(f"[ablation] dropped {len(removed)} borderline features: {removed}")
    if fast and len(X_train) > FAST_SAMPLE:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_train), FAST_SAMPLE, replace=False)
        X_train, y_train = X_train.iloc[idx], y_train.iloc[idx]
        print(f"[fast] subsampled training set to {FAST_SAMPLE:,} rows")
    print(f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}  "
          f"features={X_train.shape[1]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def grid_search(X_train: pd.DataFrame, y_train: pd.Series, grid: dict,
                use_class_weight: bool, n_jobs: int, cv: int, seed: int
                ) -> tuple[RandomForestClassifier, GridSearchCV]:
    """GridSearchCV over the Random Forest hyperparameter grid."""
    base = RandomForestClassifier(
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced" if use_class_weight else None,
    )
    search = GridSearchCV(
        base, grid, cv=cv, scoring="f1_macro",
        n_jobs=1, verbose=1, refit=True,
    )
    print(f"\nGrid search over {np.prod([len(v) for v in grid.values()])} "
          f"combinations x {cv}-fold CV ...")
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV f1_macro: {search.best_score_:.4f}  "
          f"(took {elapsed / 60:.1f} min)")
    return search.best_estimator_, search


def report_importances(model: RandomForestClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Print feature importances ranked descending; return the table."""
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCES (ranked descending)")
    print("=" * 60)
    print(importances.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return importances


def save_importance_chart(importances: pd.DataFrame, top_n: int = 25) -> Path:
    """Bar chart of the top-N feature importances -> reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    top = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(top))))
    ax.barh(top["feature"], top["importance"], color="tab:orange")
    ax.set_xlabel("importance (Gini decrease)")
    ax.set_title(f"Random Forest - top {len(top)} feature importances")
    fig.tight_layout()
    path = REPORTS_DIR / "feature_importances.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] importance chart -> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="subsample training data and use a small grid")
    parser.add_argument("--drop-borderline", action="store_true",
                        help="ablation: remove ct_*/content features from the splits")
    parser.add_argument("--no-search", action="store_true",
                        help="skip the grid search; fit the best params recorded "
                             "in reports/random_forest_results.json")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(
        args.fast, args.seed, args.drop_borderline)

    metadata_path = MODELS_DIR / "preprocess_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    use_class_weight = metadata.get("use_class_weight", False)

    results_path = REPORTS_DIR / "random_forest_results.json"
    previous = (json.loads(results_path.read_text(encoding="utf-8"))
                if results_path.exists() else {})
    if args.no_search and previous.get("best_params"):
        params = previous["best_params"]
        print(f"[no-search] fitting recorded best params {params}")
        model = RandomForestClassifier(
            random_state=args.seed, n_jobs=args.n_jobs,
            class_weight="balanced" if use_class_weight else None, **params)
        model.fit(X_train, y_train)
        # the recorded CV score belongs to the earlier search, not this fit
        best_cv = None
    else:
        grid = FAST_GRID if args.fast else FULL_GRID
        model, search = grid_search(X_train, y_train, grid, use_class_weight,
                                    args.n_jobs, args.cv, args.seed)
        best_cv = float(search.best_score_)

    joblib.dump(model, MODEL_PATH)
    print(f"[ok] best model -> {MODEL_PATH}")

    val_acc = accuracy_score(y_val, model.predict(X_val))
    val_f1 = f1_score(y_val, model.predict(X_val), average="macro")
    test_acc = accuracy_score(y_test, model.predict(X_test))
    test_f1 = f1_score(y_test, model.predict(X_test), average="macro")
    # Alert threshold for the API: picked on validation for FPR <= 1%, so
    # the operating point is never tuned on the test split.
    threshold = threshold_for_fpr(y_val, model.predict_proba(X_val)[:, 1], 0.01)
    print(f"\nValidation: accuracy={val_acc:.4f}  f1_macro={val_f1:.4f}  "
          f"threshold@FPR<=1%={threshold:.3f}")
    print(f"Test:       accuracy={test_acc:.4f}  f1_macro={test_f1:.4f}")

    importances = report_importances(model, X_train.columns.tolist())
    save_importance_chart(importances)
    importances.to_csv(REPORTS_DIR / "feature_importances.csv", index=False)

    summary = {
        "best_params": {k: (None if v is None else int(v)) for k, v in model.get_params().items()
                        if k in {"n_estimators", "max_depth", "min_samples_split"}},
        "best_cv_f1_macro": best_cv,
        "params_source": ("recorded grid-search params (--no-search)"
                          if args.no_search and previous.get("best_params")
                          else "grid search on this split"),
        "operating_threshold": threshold,
        "operating_threshold_rule": "lowest threshold with validation FPR <= 1%",
        "split_protocol": metadata.get("split_protocol", "random"),
        "val_accuracy": float(val_acc), "val_f1_macro": float(val_f1),
        "test_accuracy": float(test_acc), "test_f1_macro": float(test_f1),
        "class_weight": model.class_weight,
        "ablation_drop_borderline": bool(args.drop_borderline),
    }
    (REPORTS_DIR / "random_forest_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] results -> {REPORTS_DIR / 'random_forest_results.json'}")
    write_manifest(MODELS_DIR)


if __name__ == "__main__":
    main()
