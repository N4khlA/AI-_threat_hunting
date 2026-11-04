"""Turn a model verdict into an alert a SOC analyst can act on.

An alert carries: severity (from P(attack), not from max(p, 1-p)), the
family classifier's top guess and its probability, the ATT&CK mapping
with its fidelity, the features that drove the decision (exact path
contributions, see ``explain.py``), and a short plain-text triage note.

Field names loosely follow the Elastic Common Schema (``event.*``,
``threat.*``, ``network.*``) so the JSON can be shipped to a SIEM as is.
"""

from __future__ import annotations

import datetime as dt

from src.soc.attack_mapping import techniques_for

# UNSW-NB15 testbed artifacts: hop distances were fixed per host role, so
# TTL features separate classes in this dataset for reasons that do not
# transfer. An alert driven mainly by them deserves extra scepticism.
ARTIFACT_FEATURES = {"sttl", "dttl", "ct_state_ttl"}

SEVERITY_BANDS = ((0.95, "high"), (0.80, "medium"))


def severity(p_attack: float, threshold: float = 0.5) -> str | None:
    """Map P(attack) to a severity; None below the alert threshold."""
    if p_attack < threshold:
        return None
    for floor, name in SEVERITY_BANDS:
        if p_attack >= floor:
            return name
    return "low"


def artifact_share(drivers: list[dict]) -> float:
    """Share of the |contribution| mass carried by testbed-artifact features."""
    total = sum(abs(d["contribution"]) for d in drivers)
    if not total:
        return 0.0
    return sum(abs(d["contribution"]) for d in drivers
               if d["feature"] in ARTIFACT_FEATURES) / total


def triage_note(p_attack: float, sev: str, family: str | None,
                family_p: float | None, mapping: dict, drivers: list[dict],
                record: dict) -> str:
    flow = (f"{record.get('proto', '?')}/{record.get('service', '?')} "
            f"state={record.get('state', '?')}")
    lines = [f"[{sev.upper()}] Random Forest flagged flow {flow} as malicious "
             f"(P(attack)={p_attack:.2f})."]
    if family:
        techs = ", ".join(f"{t['id']} {t['name']}" for t in mapping["techniques"]) \
            or "no faithful ATT&CK mapping"
        lines.append(f"Most likely family: {family} (p={family_p:.2f}) -> "
                     f"{techs} [mapping fidelity: {mapping['fidelity']}].")
    if drivers:
        why = "; ".join(
            f"{d['feature']}={d['value']} ({d['contribution']:+.3f})"
            for d in drivers[:3])
        lines.append(f"Main drivers: {why}.")
    if artifact_share(drivers) >= 0.5:
        lines.append("Caution: most of this score comes from TTL features, "
                     "a known UNSW-NB15 testbed artifact. Verify before escalating.")
    lines += [f"Next: {step}" for step in mapping["next_steps"]]
    return "\n".join(lines)


def build_alert(prediction_id: int, record: dict, p_attack: float,
                family_probs: dict[str, float], drivers: list[dict],
                threshold: float = 0.5, model_version: str | None = None
                ) -> dict | None:
    """Alert dict for one flow, or None when P(attack) < threshold."""
    sev = severity(p_attack, threshold)
    if sev is None:
        return None
    family = max(family_probs, key=family_probs.get) if family_probs else None
    family_p = family_probs.get(family) if family else None
    mapping = techniques_for(family)
    return {
        "@timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event": {"kind": "alert", "module": "ml-ids", "severity": sev,
                  "risk_score": round(100 * p_attack, 1),
                  "reference": f"prediction:{prediction_id}"},
        "network": {"transport": record.get("proto"),
                    "protocol": record.get("service")},
        "threat": {"framework": "MITRE ATT&CK", "tactic": mapping["tactic"],
                   "technique": mapping["techniques"],
                   "mapping_fidelity": mapping["fidelity"]},
        "ml": {"model": "random_forest", "model_version": model_version,
               "p_attack": round(p_attack, 4), "threshold": threshold,
               "family": family,
               "family_probability": round(family_p, 4) if family_p is not None else None,
               "drivers": drivers,
               "artifact_driven": artifact_share(drivers) >= 0.5},
        "triage_note": triage_note(p_attack, sev, family, family_p, mapping,
                                   drivers, record),
    }
