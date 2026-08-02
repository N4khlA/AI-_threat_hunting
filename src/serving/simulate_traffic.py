"""Stream held-out test flows to the FastAPI service, one at a time.

Demonstrates real-time inference end-to-end: reads the held-out test
positions saved by ``src/features/preprocess.py``
(data/processed/test_positions.csv), fetches the corresponding RAW flow
records from data/raw/UNSW_NB15_combined.csv, and POSTs them to the
API. (data/processed/X_test.csv holds *processed* vectors; the API
expects raw records, so the simulator sends the raw fields of the
held-out positions.)

Usage (start the API first):
    python -m uvicorn src.serving.api:app --port 8000

    python src/serving/simulate_traffic.py                     # row mode
    python src/serving/simulate_traffic.py --mode sequence --seq-size 10
    python src/serving/simulate_traffic.py --delay 0.2 --limit 100 --no-label
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_INPUT = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_combined.csv"
TEST_POSITIONS = PROJECT_ROOT / "data" / "processed" / "test_positions.csv"


def clean_record(row: pd.Series) -> dict:
    """pandas row -> JSON-safe dict (NaN -> None)."""
    return {
        k: (None if pd.isna(v) else v)
        for k, v in row.to_dict().items()
    }


def stream_rows(base_url: str, raw: pd.DataFrame, positions: list[int],
                delay: float, limit: int, show_label: bool) -> None:
    correct = total = 0
    for i, pos in enumerate(positions[:limit]):
        record = clean_record(raw.iloc[pos])
        response = requests.post(f"{base_url}/predict", json=record, timeout=30)
        response.raise_for_status()
        out = response.json()
        total += 1
        hit = ""
        if show_label:
            true_label = int(raw.iloc[pos]["Label"])
            hit = "OK " if out["prediction"] == true_label else "MISS"
            correct += out["prediction"] == true_label
        top = sorted(out["attack_family_probs"].items(),
                     key=lambda kv: kv[1], reverse=True)[:1]
        top_fam = f"{top[0][0]}={top[0][1]:.3f}" if top else "n/a"
        print(f"[{i + 1:>4}] pos={pos:>6} {hit}pred={out['prediction']} "
              f"conf={out['confidence']:.3f} top_family={top_fam}")
        time.sleep(delay)
    if show_label and total:
        print(f"\naccuracy on streamed {total:,} flows: "
              f"{correct / total:.4f}")


def stream_sequences(base_url: str, raw: pd.DataFrame, positions: list[int],
                     delay: float, limit: int, seq_size: int,
                     show_label: bool) -> None:
    n_windows = 0
    for start in range(0, min(limit, len(positions)), seq_size):
        batch_pos = positions[start:start + seq_size]
        flows = [clean_record(raw.iloc[p]) for p in batch_pos]
        response = requests.post(f"{base_url}/predict/sequence",
                                 json={"flows": flows}, timeout=60)
        response.raise_for_status()
        out = response.json()
        for win in out["windows"]:
            n_windows += 1
            hit = ""
            if show_label:
                raw_pos = [batch_pos[i] for i in win["flow_positions"]]
                window_true = int(max(raw.iloc[p]["Label"] for p in raw_pos))
                hit = "OK " if win["prediction"] == window_true else "MISS"
            print(f"  window {win['window_id']} [{win['group']}] "
                  f"{hit}pred={win['prediction']} conf={win['confidence']:.3f} "
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
    parser.add_argument("--no-label", action="store_true",
                        help="skip ground-truth labels in the output")
    args = parser.parse_args()

    if not RAW_INPUT.exists() or not TEST_POSITIONS.exists():
        sys.exit("[!] missing data/raw/UNSW_NB15_combined.csv or "
                 "data/processed/test_positions.csv - run download + preprocess")
    raw = pd.read_csv(RAW_INPUT)
    positions = pd.read_csv(TEST_POSITIONS)["position"].tolist()
    print(f"{len(positions):,} held-out flows available; streaming "
          f"{args.limit} in {args.mode} mode to http://{args.host}:{args.port}")

    base_url = f"http://{args.host}:{args.port}"
    health = requests.get(f"{base_url}/health", timeout=15)
    health.raise_for_status()
    print("API /health:", health.json())

    if args.mode == "row":
        stream_rows(base_url, raw, positions, args.delay, args.limit,
                    not args.no_label)
    else:
        stream_sequences(base_url, raw, positions, args.delay, args.limit,
                         args.seq_size, not args.no_label)


if __name__ == "__main__":
    main()
