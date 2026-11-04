"""Attacker-side flow perturbations shared by the evasion benchmark and
adversarial training (``src/models/hardening.py``).

The threat model is documented in ``src/attacks/evasion.py``: the
attacker controls only their own side of the flow and can only add cost
(pad bytes, add packets, slow down, choose the initial TTL).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BENIGN_TTLS = np.array([31, 62])


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
    # relative update: UNSW stores smeansz rounded, so recomputing
    # sbytes/Spkts would perturb flows even at zero budget
    out["smeansz"] = (df["smeansz"].astype(float) * _ratio(new_sbytes, sbytes)
                      / (new_spkts / spkts))
    out["Sintpkt"] = df["Sintpkt"].astype(float) * stretch * spkts / new_spkts
    out["Dintpkt"] = df["Dintpkt"].astype(float) * stretch
    if ttl is not None:
        out["sttl"] = np.broadcast_to(np.asarray(ttl), len(df))
    return out
