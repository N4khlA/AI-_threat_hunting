"""UNSW-NB15 attack families -> MITRE ATT&CK (Enterprise) techniques.

UNSW-NB15 labels come from the IXIA PerfectStorm tool, not from ATT&CK,
so every mapping here is an analyst's best fit, with a fidelity rating:

    high   the family *is* the technique (e.g. DoS -> T1498)
    medium the family usually manifests as the technique
    low    a loose fit; treat as a starting point for triage only

``Generic`` (IXIA's block-cipher attacks against crypto implementations)
has no faithful ATT&CK technique and is deliberately left unmapped
rather than forced into one.
"""

from __future__ import annotations

ATTACK_MAP: dict[str, dict] = {
    "Reconnaissance": {
        "tactic": "Reconnaissance / Discovery",
        "techniques": [("T1595", "Active Scanning"),
                       ("T1046", "Network Service Discovery")],
        "fidelity": "high",
        "next_steps": [
            "Pivot on the source: count distinct dst ports/hosts in the last 15 min.",
            "Check whether the scan is followed by Exploits from the same source.",
        ],
    },
    "Analysis": {
        "tactic": "Reconnaissance",
        "techniques": [("T1595", "Active Scanning"),
                       ("T1046", "Network Service Discovery")],
        "fidelity": "low",
        "next_steps": [
            "UNSW 'Analysis' mixes port scans, spam and HTML probes: inspect payload/service.",
            "Correlate with other low-volume probes from the same source (per-flow model is weak here).",
        ],
    },
    "Fuzzers": {
        "tactic": "Reconnaissance",
        "techniques": [("T1595.002", "Active Scanning: Vulnerability Scanning")],
        "fidelity": "medium",
        "next_steps": [
            "Look for malformed requests / protocol errors in the service logs.",
            "Fuzzers are the family this model misses most: check neighbouring benign-labelled flows too.",
        ],
    },
    "Exploits": {
        "tactic": "Initial Access / Lateral Movement",
        "techniques": [("T1190", "Exploit Public-Facing Application"),
                       ("T1210", "Exploitation of Remote Services")],
        "fidelity": "medium",
        "next_steps": [
            "Identify the targeted service and version; check it against known CVEs.",
            "Hunt for follow-on activity from the destination host (new outbound connections).",
        ],
    },
    "DoS": {
        "tactic": "Impact",
        "techniques": [("T1498", "Network Denial of Service"),
                       ("T1499", "Endpoint Denial of Service")],
        "fidelity": "high",
        "next_steps": [
            "Check service availability and flow volume to the destination.",
            "Rate-limit or upstream-filter the source if confirmed; do not auto-block on one flow.",
        ],
    },
    "Backdoor": {
        "tactic": "Command and Control / Persistence",
        "techniques": [("T1571", "Non-Standard Port"),
                       ("T1095", "Non-Application Layer Protocol")],
        "fidelity": "low",
        "next_steps": [
            "Treat the destination as potentially compromised: pull process/network telemetry from it.",
            "Look for periodic (beacon-like) connections between the same pair.",
        ],
    },
    "Shellcode": {
        "tactic": "Execution",
        "techniques": [("T1203", "Exploitation for Client Execution"),
                       ("T1190", "Exploit Public-Facing Application")],
        "fidelity": "low",
        "next_steps": [
            "Retrieve the payload (pcap) if available and analyse in a sandbox.",
            "Check the destination for crashes or new child processes of the targeted service.",
        ],
    },
    "Worms": {
        "tactic": "Lateral Movement",
        "techniques": [("T1210", "Exploitation of Remote Services"),
                       ("T1570", "Lateral Tool Transfer")],
        "fidelity": "medium",
        "next_steps": [
            "Check whether the destination starts scanning/exploiting others (propagation).",
            "Prioritise containment: worms spread faster than ticket queues move.",
        ],
    },
    "Generic": {
        "tactic": None,
        "techniques": [],
        "fidelity": "none",
        "next_steps": [
            "No faithful ATT&CK technique (IXIA block-cipher attack class); triage on the flow evidence.",
        ],
    },
}


def techniques_for(family: str | None) -> dict:
    """Mapping entry for ``family`` (unknown/None -> an explicit 'unmapped')."""
    entry = ATTACK_MAP.get(family or "")
    if entry is None:
        return {"tactic": None, "techniques": [], "fidelity": "none",
                "next_steps": ["Unknown family: triage on the flow evidence."]}
    return {**entry, "techniques": [{"id": t, "name": n} for t, n in entry["techniques"]]}
