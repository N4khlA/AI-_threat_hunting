"""Feature engineering: raw network flow records -> ML-ready features.

The functions here convert raw (NetFlow-style / UNSW-NB15-style) flow
records into a compact set of engineered features. Every function includes
a docstring explaining the *security relevance* of what it produces -
i.e. which attack behaviours the feature helps expose.

Attacks tend to perturb traffic statistics, so most engineered features
capture distributional shape (mean/min/max/std), asymmetry between the
forward and backward direction, and volume per unit time. Botnets, DoS,
and scan behaviour typically show up as abnormal packet-size, timing, or
asymmetry patterns.

Typical input columns (subset required, missing ones are skipped):
    spkts, dpkts, sbytes, dbytes, smeansz, dmeansz, Sintpkt, Dintpkt,
    Sjit, Djit, dur, proto, sport, dsport, Sload, Dload, payload

Usage (library):
    from src.features.extract import extract_features
    df_feat, names = extract_features(df_raw)

Usage (CLI):
    python src/features/extract.py --input data/raw/UNSW_NB15_combined.csv
                                   --output data/processed/features.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SAFE_ZERO = 1e-9  # guard against division by zero


def packet_size_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Packet-size statistics: mean, min, max, and std of packet sizes.

    Security relevance: normal flows have stable, well-bounded packet
    sizes; DoS, amplification, and scanning traffic often deviate sharply
    (e.g. floods of tiny packets or oversized requests). The standard
    deviation flags mixed-size bursts typical of malware C2 beacons.

    Derived per direction from byte/packet counts (``smeansz``/``dmeansz``
    are used directly when present).
    """
    out = pd.DataFrame(index=df.index)
    if {"smeansz", "dmeansz"}.issubset(df.columns):
        fwd_size = pd.to_numeric(df["smeansz"], errors="coerce")
        bwd_size = pd.to_numeric(df["dmeansz"], errors="coerce")
        all_sizes = pd.concat([fwd_size, bwd_size], axis=1)
        out["pkt_size_mean"] = all_sizes.mean(axis=1)
        out["pkt_size_min"] = all_sizes.min(axis=1)
        out["pkt_size_max"] = all_sizes.max(axis=1)
        out["pkt_size_std"] = all_sizes.std(axis=1)
    elif {"spkts", "dpkts", "sbytes", "dbytes"}.issubset(df.columns):
        spkts = pd.to_numeric(df["spkts"], errors="coerce").replace(0, np.nan)
        dpkts = pd.to_numeric(df["dpkts"], errors="coerce").replace(0, np.nan)
        sbytes = pd.to_numeric(df["sbytes"], errors="coerce")
        dbytes = pd.to_numeric(df["dbytes"], errors="coerce")
        fwd_size = sbytes / spkts
        bwd_size = dbytes / dpkts
        all_sizes = pd.concat([fwd_size, bwd_size], axis=1)
        out["pkt_size_mean"] = all_sizes.mean(axis=1)
        out["pkt_size_min"] = all_sizes.min(axis=1)
        out["pkt_size_max"] = all_sizes.max(axis=1)
        out["pkt_size_std"] = all_sizes.std(axis=1)
    return out


def inter_arrival_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Inter-arrival time statistics: mean, std, min, max; plus jitter.

    Security relevance: inter-arrival time (IAT) is one of the strongest
    discriminators between human/web-like traffic and machine-generated
    activity. Port scanners and botnets emit packets on near-constant
    clocks (low IAT variance), while DDoS floods produce near-zero IAT.
    Jitter (``Sjit``/``Djit``) captures clock regularity of each peer.
    """
    out = pd.DataFrame(index=df.index)
    if {"Sintpkt", "Dintpkt"}.issubset(df.columns):
        sint = pd.to_numeric(df["Sintpkt"], errors="coerce")
        dint = pd.to_numeric(df["Dintpkt"], errors="coerce")
        iat = pd.concat([sint, dint], axis=1)
        out["iat_mean"] = iat.mean(axis=1)
        out["iat_std"] = iat.std(axis=1)
        out["iat_min"] = iat.min(axis=1)
        out["iat_max"] = iat.max(axis=1)
    if {"Sjit", "Djit"}.issubset(df.columns):
        out["jit_mean"] = pd.concat(
            [pd.to_numeric(df["Sjit"], errors="coerce"),
             pd.to_numeric(df["Djit"], errors="coerce")], axis=1).mean(axis=1)
    return out


def byte_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Total byte/packet counts and bytes-per-packet aggregate.

    Security relevance: total transferred volume flags data exfiltration
    (large outbound byte counts) and network saturation attacks. The
    bytes-per-packet ratio complements ``packet_size_stats`` as a coarse
    volume-normalized measure that is robust to flow duration.
    """
    out = pd.DataFrame(index=df.index)
    if {"sbytes", "dbytes"}.issubset(df.columns):
        sbytes = pd.to_numeric(df["sbytes"], errors="coerce")
        dbytes = pd.to_numeric(df["dbytes"], errors="coerce")
        out["total_bytes"] = sbytes + dbytes
    if {"spkts", "dpkts"}.issubset(df.columns):
        spkts = pd.to_numeric(df["spkts"], errors="coerce")
        dpkts = pd.to_numeric(df["dpkts"], errors="coerce")
        out["total_pkts"] = spkts + dpkts
    if "total_bytes" in out and "total_pkts" in out:
        out["bytes_per_pkt"] = out["total_bytes"] / (out["total_pkts"] + SAFE_ZERO)
    return out


def duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flow duration and throughput (bytes per second).

    Security relevance: very short flows dominate port scans and
    connection probes; very long flows may indicate persistent tunnels
    (C2 keep-alives) or data exfiltration. Throughput (bytes/second)
    normalizes volume by time, exposing high-intensity flood behaviour.
    """
    out = pd.DataFrame(index=df.index)
    if "dur" in df.columns:
        dur = pd.to_numeric(df["dur"], errors="coerce")
        out["duration"] = dur
        out["flow_rate"] = 1.0 / (dur + SAFE_ZERO)
    if {"total_bytes", "duration"}.issubset(out.columns):
        out["throughput_bps"] = out["total_bytes"] / (out["duration"] + SAFE_ZERO)
    return out


def port_features(df: pd.DataFrame) -> pd.DataFrame:
    """Port numbers plus well-known/registered port flags.

    Security relevance: ports anchor service identification - scans
    spread requests across many destinations on the *same* port, while
    unusual source ports can betray custom malware clients. The flags
    compress high-cardinality port values into behaviourally meaningful
    indicators (well-known < 1024, registered < 49152, else ephemeral).
    """
    out = pd.DataFrame(index=df.index)
    for col, feat in (("sport", "sport"), ("dsport", "dsport")):
        if col not in df.columns:
            continue
        ports = pd.to_numeric(df[col], errors="coerce")
        out[feat] = ports
        out[f"{feat}_well_known"] = (ports < 1024).astype(int)
        out[f"{feat}_registered"] = ((ports >= 1024) & (ports < 49152)).astype(int)
    if {"sport", "dsport"}.issubset(out.columns):
        out["port_diff"] = (out["dsport"] - out["sport"]).abs()
    return out


def protocol_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding of the transport protocol (tcp/udp/icmp/other).

    Security relevance: many attacks are protocol-specific - SYN floods
    abuse TCP, amplification abuses UDP (DNS/NTP/SSDP), and ICMP is
    abused for ping floods and tunneling. Encoding protocol lets a model
    branch on the transport layer directly while keeping the feature
    space small.
    """
    if "proto" not in df.columns:
        return pd.DataFrame(index=df.index)
    proto = df["proto"].astype(str).str.strip().str.lower()
    normalized = proto.where(proto.isin(["tcp", "udp", "icmp"]), "other")
    encoded = pd.get_dummies(normalized, prefix="proto")
    for col in ["proto_tcp", "proto_udp", "proto_icmp", "proto_other"]:
        if col not in encoded.columns:
            encoded[col] = 0
    return encoded.reindex(columns=["proto_tcp", "proto_udp", "proto_icmp", "proto_other"])


def forward_backward_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Forward/backward traffic ratios (bytes, packets, load).

    Security relevance: benign web traffic is dominated by the server
    (backward) direction; exfiltration and botnet C2 produce forward-heavy
    flows; reflection/amplification attacks are sharply backward-heavy.
    Directional ratios summarize this asymmetry in a scale-free form that
    generalizes across network speeds.
    """
    out = pd.DataFrame(index=df.index)
    if {"sbytes", "dbytes"}.issubset(df.columns):
        sbytes = pd.to_numeric(df["sbytes"], errors="coerce")
        dbytes = pd.to_numeric(df["dbytes"], errors="coerce")
        out["fwd_byte_ratio"] = sbytes / (sbytes + dbytes + SAFE_ZERO)
        out["fwd_byte_ratio_alt"] = sbytes / (dbytes + SAFE_ZERO)
    if {"spkts", "dpkts"}.issubset(df.columns):
        spkts = pd.to_numeric(df["spkts"], errors="coerce")
        dpkts = pd.to_numeric(df["dpkts"], errors="coerce")
        out["fwd_pkt_ratio"] = spkts / (spkts + dpkts + SAFE_ZERO)
    if {"Sload", "Dload"}.issubset(df.columns):
        sload = pd.to_numeric(df["Sload"], errors="coerce")
        dload = pd.to_numeric(df["Dload"], errors="coerce")
        out["fwd_load_ratio"] = sload / (sload + dload + SAFE_ZERO)
    return out


def shannon_entropy(data: bytes) -> float:
    """Shannon entropy (bits/byte) of a byte sequence; NaN for empty input."""
    if not data:
        return math.nan
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8))
    probs = counts[counts > 0] / len(data)
    return float(-(probs * np.log2(probs)).sum())


def payload_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """Payload entropy (Shannon, bits/byte) where payload bytes are available.

    Security relevance: encrypted or compressed data (TLS tunnels, packed
    malware, obfuscated C2) has near-maximum entropy (~8 bits/byte),
    whereas plaintext protocols (HTTP, DNS, FTP) are far more redundant.
    Abrupt entropy shifts in a flow signal encryption wrapping or payload
    obfuscation even when no payload is otherwise inspectable.

    Falls back to NaN (no signal) when no ``payload`` column exists -
    e.g. UNSW-NB15 ships aggregated statistics only.
    """
    out = pd.DataFrame(index=df.index)
    if "payload" not in df.columns:
        return out
    out["payload_entropy"] = [
        shannon_entropy(bytes(bytearray(v))) if isinstance(v, (bytes, bytearray)) else math.nan
        for v in df["payload"]
    ]
    return out


# Order matters: duration_features depends on byte_counts output.
EXTRACTORS = [
    ("packet_size_stats", packet_size_stats),
    ("inter_arrival_stats", inter_arrival_stats),
    ("byte_counts", byte_counts),
    ("duration_features", duration_features),
    ("port_features", port_features),
    ("protocol_one_hot", protocol_one_hot),
    ("forward_backward_ratios", forward_backward_ratios),
    ("payload_entropy", payload_entropy),
]


def extract_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply every extractor; return (augmented_df, engineered_feature_names).

    The returned feature list contains only *newly engineered* columns,
    not the raw inputs, so downstream pipelines know exactly what to use.
    """
    augmented = df.copy()
    engineered: list[str] = []
    for name, extractor in EXTRACTORS:
        block = extractor(df)
        if not block.empty:
            augmented = pd.concat([augmented, block], axis=1)
            engineered.extend(block.columns)
    return augmented, engineered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="raw flow records CSV")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "data" / "processed" / "features.csv",
                        help="output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} raw records x {len(df.columns)} columns")
    augmented, names = extract_features(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(args.output, index=False)
    print(f"Engineered {len(names)} features from {len(EXTRACTORS)} extractors")
    for name in names:
        print(f"  + {name}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
