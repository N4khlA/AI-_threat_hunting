"""Evasion benchmark: can an attacker reshape their flows to look benign?

Threat model (docs/threat_model.md, MITRE ATLAS AML.T0015 / AML.T0043):

* **Black-box, query access** to a verdict (e.g. by watching whether a
  connection gets blocked), no model internals.
* **Attacker controls only their own side of the flow** and cannot
  undo the attack's purpose, so every perturbation only *adds* cost:
    - pad:   append junk to payloads           sbytes  x (1 + pad)
    - pkts:  split into / add more packets     Spkts   x (1 + pkts)
                                               (+40 B header each)
    - slow:  stretch the flow in time          dur, inter-packet gaps
                                               x (1 + slow)
    - ttl:   choose the initial IP TTL         sttl -> a benign-typical
                                               value (31 or 62)
  Derived fields (Sload, Dload, smeansz, Sintpkt) are recomputed so the
  record stays self-consistent. Destination-side fields are untouched.
* **Budget eps**: each continuous knob is drawn from [0, eps]; random
  search with ``--queries`` tries per flow. Success = P(attack) drops
  below the model's operating threshold.

Models compared (same official_novel split, thresholds picked on
validation for FPR <= 1%):
    baseline    the deployed Random Forest
    no_ttl      retrained without sttl/dttl/ct_state_ttl
    adv_trained no_ttl + adversarial training: perturbed copies of the
                training attacks (eps ~ U(0, 2)) added as attacks

Usage:
    python src/attacks/evasion.py                    # ~5 min
    python src/attacks/evasion.py --samples 300 --queries 10   # smoke

Outputs: reports/evasion.json, reports/evasion.md
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import binary_report, threshold_for_fpr  # noqa: E402
from src.features.preprocess import DEFAULT_INPUT, load_raw  # noqa: E402
from src.models.lstm_model import transform_flows  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TTL_FEATURES = ["sttl", "dttl", "ct_state_ttl"]
# the deployed RF was fitted on a DataFrame; we score column-sliced arrays
warnings.filterwarnings("ignore", message="X does not have valid feature names")
BENIGN_TTLS = np.array([31, 62])
EPSILONS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _ratio(new: pd.Series, old: pd.Series) -> pd.Series:
    old = old.astype(float)
    return (new / old.where(old > 0)).fillna(1.0)


def perturb(df: pd.DataFrame, pad, pkts, slow, ttl=None) -> pd.DataFrame:
    """Attacker-side, self-consistent perturbation of raw flow records.

    ``pad``/``pkts``/``slow`` are scalars or per-row arrays >= 0; ``ttl``
    is None (untouched), a scalar or a per-row array.
    """
    out = df.copy()
    pad, pkts, slow = (np.broadcast_to(np.asarray(v, float), len(df)) for v in (pad, pkts, slow))
    spkts = df["Spkts"].astype(float).clip(lower=1)
    new_spkts = np.ceil(spkts * (1 + pkts))
    sbytes = df["sbytes"].astype(float)
    new_sbytes = sbytes * (1 + pad) + 40.0 * (new_spkts - spkts)
    stretch = 1 + slow

    out["Spkts"] = new_spkts
    out["sbytes"] = new_sbytes
    out["dur"] = df["dur"].astype(float) * stretch
    out["Sload"] = df["Sload"].astype(float) * _ratio(new_sbytes, sbytes) / stretch
    out["Dload"] = df["Dload"].astype(float) / stretch
    out["smeansz"] = new_sbytes / new_spkts
    out["Sintpkt"] = df["Sintpkt"].astype(float) * stretch * spkts / new_spkts
    out["Dintpkt"] = df["Dintpkt"].astype(float) * stretch
    if ttl is not None:
        out["sttl"] = np.broadcast_to(np.asarray(ttl), len(df))
    return out


class Scored:
    """A fitted RF that sees only ``columns`` of the full feature vector."""

    def __init__(self, model, names: list[str], columns: list[str], threshold: float):
        self.model, self.threshold = model, threshold
        self.idx = [names.index(c) for c in columns]

    def p_attack(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X[:, self.idx])[:, 1]


def random_search(raw: pd.DataFrame, scorer: Scored, pipeline: dict, names: list[str],
                  eps: float, queries: int, allow_ttl: bool, seed: int) -> np.ndarray:
    """Boolean array: which flows were pushed below the threshold."""
    rng = np.random.default_rng(seed)
    evaded = np.zeros(len(raw), dtype=bool)
    for q in range(queries):
        todo = np.where(~evaded)[0]
        if not len(todo):
            break
        n = len(todo)
        # first query: the corner of the budget box (max of every knob)
        knobs = (np.full((3, n), eps) if q == 0 else rng.uniform(0, eps, (3, n)))
        ttl = rng.choice(BENIGN_TTLS, n) if allow_ttl else None
        cand = perturb(raw.iloc[todo], *knobs, ttl=ttl)
        p = scorer.p_attack(transform_flows(cand, pipeline, names))
        evaded[todo[p < scorer.threshold]] = True
    return evaded


def fit_rf(X: np.ndarray, y: np.ndarray, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, min_samples_split=5,
                                  n_jobs=-1, random_state=seed).fit(X, y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    raw = load_raw(args.input)
    pipeline = joblib.load(MODELS_DIR / "preprocess_pipeline.joblib")  # nosec B301
    meta = json.loads((MODELS_DIR / "preprocess_metadata.json").read_text(encoding="utf-8"))
    names = meta["feature_names"]
    pos = {k: pd.read_csv(PROCESSED_DIR / f"{k}_positions.csv")["position"]
           for k in ("train", "val", "test")}
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")[names].to_numpy()
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["label"].to_numpy()
    X_val = transform_flows(raw.loc[pos["val"]], pipeline, names)
    y_val = raw.loc[pos["val"], "Label"].to_numpy()
    test_raw = raw.loc[pos["test"]]
    X_test = transform_flows(test_raw, pipeline, names)
    y_test = test_raw["Label"].to_numpy()

    no_ttl_cols = [c for c in names if c not in TTL_FEATURES]
    no_ttl_idx = [names.index(c) for c in no_ttl_cols]

    # adversarial training set: perturbed copies of the training attacks
    train_attacks = raw.loc[pos["train"]].query("Label == 1")
    k = len(train_attacks)
    aug = perturb(train_attacks, rng.uniform(0, 2, k), rng.uniform(0, 2, k),
                  rng.uniform(0, 2, k), ttl=rng.choice(BENIGN_TTLS, k))
    X_aug = np.vstack([X_train, transform_flows(aug, pipeline, names)])
    y_aug = np.concatenate([y_train, np.ones(k, dtype=int)])

    models = {}
    rf = joblib.load(MODELS_DIR / "random_forest_best.joblib")  # nosec B301
    for name, model, cols in (
        ("baseline", rf, names),
        ("no_ttl", fit_rf(X_train[:, no_ttl_idx], y_train, args.seed), no_ttl_cols),
        ("adv_trained", fit_rf(X_aug[:, no_ttl_idx], y_aug, args.seed), no_ttl_cols),
    ):
        tmp = Scored(model, names, cols, 0.5)
        thr = threshold_for_fpr(y_val, tmp.p_attack(X_val), 0.01)
        models[name] = Scored(model, names, cols, thr)
        print(f"[model] {name}: threshold@valFPR1% = {thr:.3f}")

    results: dict = {"samples": args.samples, "queries": args.queries,
                     "epsilons": list(EPSILONS), "models": {}}
    for name, scorer in models.items():
        p_test = scorer.p_attack(X_test)
        clean = binary_report(y_test, p_test, scorer.threshold)
        detected = np.where((y_test == 1) & (p_test >= scorer.threshold))[0]
        pick = rng.choice(detected, min(args.samples, len(detected)), replace=False)
        sample = test_raw.iloc[pick]
        fams = sample["attack_cat"].to_numpy()
        entry = {"clean": clean, "threshold": scorer.threshold, "evasion": {}}
        for allow_ttl in (False, True):
            scenario = "volume_timing_ttl" if allow_ttl else "volume_timing"
            entry["evasion"][scenario] = {}
            for eps in EPSILONS:
                ok = random_search(sample, scorer, pipeline, names, eps,
                                   args.queries, allow_ttl, args.seed)
                per_family = {f: float(ok[fams == f].mean())
                              for f in sorted(set(fams)) if (fams == f).sum() >= 10}
                entry["evasion"][scenario][str(eps)] = {
                    "evasion_rate": float(ok.mean()), "per_family": per_family}
                print(f"  {name:12s} {scenario:18s} eps={eps:<4} "
                      f"evasion={ok.mean():.3f}")
        results["models"][name] = entry

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evasion.json").write_text(json.dumps(results, indent=2),
                                              encoding="utf-8")
    (REPORTS_DIR / "evasion.md").write_text(render(results), encoding="utf-8")
    print(f"[ok] -> {REPORTS_DIR / 'evasion.md'}")


def render(r: dict) -> str:
    eps = r["epsilons"]
    lines = ["# Evasion benchmark", "",
             "> Generated by `src/attacks/evasion.py`. Black-box random search, "
             f"{r['queries']} queries/flow, {r['samples']} correctly-detected test "
             "attacks per model. Evasion rate = share pushed below the model's "
             "operating threshold (validation FPR <= 1%).", "",
             "## Clean performance (official_novel test)", "",
             "| model | threshold | F1-macro | TPR | FPR |", "|---|---:|---:|---:|---:|"]
    for name, m in r["models"].items():
        c = m["clean"]
        lines.append(f"| `{name}` | {m['threshold']:.3f} | {c['f1_macro']:.4f} | "
                     f"{c['tpr']:.4f} | {c['fpr']:.4f} |")
    for scenario, title in (("volume_timing", "Attacker pads bytes, adds packets, slows down"),
                            ("volume_timing_ttl", "... and also picks its initial TTL")):
        lines += ["", f"## {title}", "",
                  "| model | " + " | ".join(f"eps={e}" for e in eps) + " |",
                  "|---|" + "---:|" * len(eps)]
        for name, m in r["models"].items():
            cells = [f"{m['evasion'][scenario][str(e)]['evasion_rate']:.1%}" for e in eps]
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines += ["", "## Per-family evasion at eps=1.0, TTL allowed", "",
              "| family | " + " | ".join(f"`{n}`" for n in r["models"]) + " |",
              "|---|" + "---:|" * len(r["models"])]
    fams = sorted({f for m in r["models"].values()
                   for f in m["evasion"]["volume_timing_ttl"]["1.0"]["per_family"]})
    for f in fams:
        cells = [m["evasion"]["volume_timing_ttl"]["1.0"]["per_family"].get(f)
                 for m in r["models"].values()]
        lines.append(f"| {f} | " + " | ".join("-" if c is None else f"{c:.1%}"
                                               for c in cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
