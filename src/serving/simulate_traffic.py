"""Stream held-out test flows to the FastAPI service, one at a time.

Demonstrates real-time inference end-to-end: reads the held-out test
positions saved by ``src/features/preprocess.py``
(data/processed/test_positions.csv), fetches the corresponding RAW flow
records from data/raw/UNSW_NB15_combined.csv, and POSTs them to the
API. (data/processed/X_test.csv holds *processed* vectors; the API
expects raw records, so the simulator sends the raw fields of the
held-out positions.)

The simulator plays two roles, each with its own API key:

    sensor   (IDS_SENSOR_KEY)  submits flows - labels are NOT sent, the
                               API rejects them;
    analyst  (IDS_ANALYST_KEY) optional: "confirms" each verdict with the
                               ground truth via /feedback, standing in for
                               the human-in-the-loop review that feeds the
                               dashboard and retrain.py.

Usage (start the API first, see README "2-minute demo"):
    python src/serving/simulate_traffic.py                     # row mode
    python src/serving/simulate_traffic.py --mode sequence --seq-size 10
    python src/serving/simulate_traffic.py --limit 200 --show-alerts 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_INPUT = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_combined.csv"
TEST_POSITIONS = PROJECT_ROOT / "data" / "processed" / "test_positions.csv"


# never sent by a sensor: labels, and identity columns the cleaned
# dataset leaves empty
NOT_SENT = {"Label", "attack_cat", "id"}


def clean_record(row: pd.Series) -> dict:
    """pandas row -> JSON-safe sensor payload (NaN dropped, ints as ints)."""
    out = {}
    for k, v in row.to_dict().items():
        if k in NOT_SENT or pd.isna(v):
            continue
        out[k] = int(v) if isinstance(v, float) and v.is_integer() else v
    return out


def _headers(key: str | None) -> dict:
    return {"X-API-Key": key} if key else {}


def stream_rows(base_url: str, raw: pd.DataFrame, positions: list[int],
                delay: float, limit: int, sensor_key: str | None,
                analyst_key: str | None, show_alerts: int) -> None:
    correct = total = shown = 0
    for i, pos in enumerate(positions[:limit]):
        record = clean_record(raw.iloc[pos])
        response = requests.post(f"{base_url}/predict", json=record,
                                 headers=_headers(sensor_key), timeout=30)
        response.raise_for_status()
        out = response.json()
        total += 1
        true_label = int(raw.iloc[pos]["Label"])
        hit = "OK  " if out["prediction"] == true_label else "MISS"
        correct += out["prediction"] == true_label
        alert = out.get("alert") or {}
        fam = alert.get("ml", {}).get("family") or "-"
        sev = alert.get("event", {}).get("severity") or "-"
        print(f"[{i + 1:>4}] pos={pos:>6} {hit} pred={out['prediction']} "
              f"p_attack={out['p_attack']:.3f} severity={sev:<6} family={fam} "
              f"(truth: {raw.iloc[pos]['attack_cat']})")
        if alert and shown < show_alerts:
            shown += 1
            print("\n" + alert["triage_note"] + "\n")
        if analyst_key:
            fb = {"prediction_id": out["prediction_id"], "label": true_label}
            if true_label:
                fb["attack_cat"] = raw.iloc[pos]["attack_cat"]
            requests.post(f"{base_url}/feedback", json=fb,
                          headers=_headers(analyst_key), timeout=30).raise_for_status()
        time.sleep(delay)
    if total:
        print(f"\naccuracy on streamed {total:,} flows: {correct / total:.4f}")


def stream_sequences(base_url: str, raw: pd.DataFrame, positions: list[int],
                     delay: float, limit: int, seq_size: int,
                     sensor_key: str | None) -> None:
    n_windows = 0
    for start in range(0, min(limit, len(positions)), seq_size):
        batch_pos = positions[start:start + seq_size]
        flows = [clean_record(raw.iloc[p]) for p in batch_pos]
        response = requests.post(f"{base_url}/predict/sequence",
                                 json={"flows": flows},
                                 headers=_headers(sensor_key), timeout=60)
        response.raise_for_status()
        out = response.json()
        for win in out["windows"]:
            n_windows += 1
            raw_pos = [batch_pos[i] for i in win["flow_positions"]]
            window_true = int(max(raw.iloc[p]["Label"] for p in raw_pos))
            hit = "OK  " if win["prediction"] == window_true else "MISS"
            print(f"  window {win['window_id']} [{win['group']}] "
                  f"{hit} pred={win['prediction']} p_attack={win['p_attack']:.3f} "
                  f"({win['n_flows']} flows)")
        time.sleep(delay)
    print(f"\ntotal windows scored: {n_windows}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--delay", type=float, default=0.05,
                        help="seconds between requests (default 0.05)")
    parser.add_argument("--limit", type=int, default=50,
                        help="max flows to stream (default 50)")
    parser.add_argument("--mode", choices=["row", "sequence"], default="row")
    parser.add_argument("--seq-size", type=int, default=10,
                        help="flows per /predict/sequence batch")
    parser.add_argument("--show-alerts", type=int, default=1,
                        help="print the triage note of the first N alerts")
    parser.add_argument("--seed", type=int, default=0,
                        help="shuffle seed (the test file is ordered by class)")
    args = parser.parse_args()

    if not RAW_INPUT.exists() or not TEST_POSITIONS.exists():
        sys.exit("[!] missing data/raw/UNSW_NB15_combined.csv or "
                 "data/processed/test_positions.csv - run download + preprocess")
    raw = pd.read_csv(RAW_INPUT)
    positions = (pd.read_csv(TEST_POSITIONS)["position"]
                 .sample(frac=1, random_state=args.seed).tolist())
    sensor_key = os.environ.get("IDS_SENSOR_KEY")
    analyst_key = os.environ.get("IDS_ANALYST_KEY")
    if not sensor_key:
        sys.exit("[!] set IDS_SENSOR_KEY (see README: 2-minute demo)")
    print(f"{len(positions):,} held-out flows available; streaming "
          f"{args.limit} in {args.mode} mode to http://{args.host}:{args.port}")

    base_url = f"http://{args.host}:{args.port}"
    health = requests.get(f"{base_url}/health", timeout=15)
    health.raise_for_status()
    print("API /health:", health.json())

    if args.mode == "row":
        stream_rows(base_url, raw, positions, args.delay, args.limit,
                    sensor_key, analyst_key, args.show_alerts)
    else:
        stream_sequences(base_url, raw, sorted(positions), args.delay, args.limit,
                         args.seq_size, sensor_key)


if __name__ == "__main__":
    main()
