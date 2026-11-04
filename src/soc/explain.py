"""Per-alert explanations for the Random Forest (exact, no SHAP dependency).

Decision-path contributions (Saabas): walking a tree from root to leaf,
every split moves P(attack) from the parent node's value to the child's.
That delta is credited to the split feature. Averaged over the forest:

    P(attack | x) = bias + sum_f contribution_f(x)

holds *exactly* (bias = mean root value = training attack share after
SMOTE). This is the identity ``tests/test_soc.py`` checks, so the
"why was this flagged" text shown to an analyst is never a guess.
"""

from __future__ import annotations

import numpy as np


def _node_attack_prob(tree) -> np.ndarray:
    """P(class 1) at every node, robust to sklearn's count/fraction storage."""
    value = tree.tree_.value[:, 0, :]
    return value[:, 1] / value.sum(axis=1)


def path_contributions(forest, X: np.ndarray, class_index: int = 1
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return (bias[n], contributions[n, n_features]) for P(class_index)."""
    X = np.asarray(X, dtype=np.float32)
    n, n_features = X.shape
    bias = np.zeros(n)
    contrib = np.zeros((n, n_features))
    for est in forest.estimators_:
        tree = est.tree_
        prob = _node_attack_prob(est)
        if class_index == 0:
            prob = 1.0 - prob
        paths = est.decision_path(X)
        for row in range(n):
            nodes = paths.indices[paths.indptr[row]:paths.indptr[row + 1]]
            bias[row] += prob[nodes[0]]
            deltas = prob[nodes[1:]] - prob[nodes[:-1]]
            np.add.at(contrib[row], tree.feature[nodes[:-1]], deltas)
    k = len(forest.estimators_)
    return bias / k, contrib / k


def top_drivers(contrib_row: np.ndarray, feature_names: list[str],
                raw_values: dict | None = None, k: int = 5) -> list[dict]:
    """The k features with the largest |contribution|, signed."""
    order = np.argsort(-np.abs(contrib_row))[:k]
    raw_values = raw_values or {}
    return [{
        "feature": feature_names[i],
        "contribution": round(float(contrib_row[i]), 4),
        "direction": "towards attack" if contrib_row[i] > 0 else "towards benign",
        "value": raw_values.get(feature_names[i]),
    } for i in order if contrib_row[i] != 0]
