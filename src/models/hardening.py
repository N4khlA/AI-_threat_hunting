"""Hardening recipe for the served Random Forest.

The evasion benchmark (``reports/evasion.md``) showed two weaknesses of
a plainly trained RF:

1. **TTL is a skeleton key.** Observed ``sttl`` = 31 is benign in 100%
   of UNSW-NB15 rows and the attacker chooses their initial TTL, so a TTL
   rewrite evaded the model on 100% of detected attacks. Dropping the TTL
   features costs ~0.002 macro-F1.
2. **Padding / slowing down evades 21-64%** of detections even without
   TTL. Adversarial training - adding perturbed copies of the training
   attacks, still labelled as attacks - cuts that to a few percent.

The served model applies both. ``random_forest.py``, ``retrain.py`` and
``rigorous_eval.py`` all call this module so the recipe cannot drift
between training, retraining and evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.attacks.perturb import BENIGN_TTLS, perturb

TTL_FEATURES = ("sttl", "dttl", "ct_state_ttl")
ADV_EPS = 2.0  # augmentation knobs ~ U(0, ADV_EPS); TTL from BENIGN_TTLS


def model_features(all_features: list[str], hardened: bool = True) -> list[str]:
    """Features the classifier sees (the pipeline still computes all)."""
    if not hardened:
        return list(all_features)
    return [f for f in all_features if f not in TTL_FEATURES]


def adversarial_rows(raw_attacks: pd.DataFrame, pipeline: dict, names: list[str],
                     seed: int, eps: float = ADV_EPS) -> pd.DataFrame:
    """Perturbed, pipeline-transformed copies of ``raw_attacks`` (label: attack)."""
    from src.models.lstm_model import transform_flows  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    k = len(raw_attacks)
    aug = perturb(raw_attacks, rng.uniform(0, eps, k), rng.uniform(0, eps, k),
                  rng.uniform(0, eps, k), ttl=rng.choice(BENIGN_TTLS, k))
    return pd.DataFrame(transform_flows(aug, pipeline, names), columns=names)


def harden_training_set(X_train: pd.DataFrame, y_train: pd.Series,
                        raw_train: pd.DataFrame, pipeline: dict, names: list[str],
                        seed: int) -> tuple[pd.DataFrame, pd.Series, dict]:
    """(X, y) restricted to model features, plus adversarial attack copies."""
    feats = model_features(names)
    attacks = raw_train[raw_train["Label"] == 1]
    aug = adversarial_rows(attacks, pipeline, names, seed)
    X = pd.concat([X_train[feats], aug[feats]], ignore_index=True)
    y = pd.concat([pd.Series(np.asarray(y_train)), pd.Series(np.ones(len(aug), dtype=int))],
                  ignore_index=True)
    info = {"dropped_features": list(TTL_FEATURES),
            "adversarial_training": {"rows": int(len(aug)), "eps": ADV_EPS,
                                     "ttl_values": BENIGN_TTLS.tolist(),
                                     "seed": seed}}
    return X, y, info


def served_features(models_dir, all_features: list[str]) -> list[str]:
    """Feature subset the served RF was trained on (pinned in operating_point.json)."""
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    path = Path(models_dir) / "operating_point.json"
    if path.exists():
        feats = json.loads(path.read_text(encoding="utf-8")).get("features")
        if feats:
            return feats
    return list(all_features)
