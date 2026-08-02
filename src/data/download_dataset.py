"""Download and prepare the UNSW-NB15 network intrusion detection dataset.

The UNSW-NB15 dataset (Moustafa & Slay, 2015) contains ~2.5M flow records
captured from real network traffic at the UNSW Canberra Cyber Range. It
combines benign ("Normal") traffic with nine attack families: Fuzzers,
Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and
Worms. Each record is a bidirectional flow with 49 features spanning flow
statistics, connection metadata and derived content/timing metrics.

Usage:
    python src/data/download_dataset.py            # all splits
    python src/data/download_dataset.py --split training
    python src/data/download_dataset.py --help

Strategy: UNSW Canberra hosts the official files behind interactive
SharePoint pages with no stable API, so this script tries known mirrors in
order:

    1. the legacy official CloudStor links (strict 49-column validation),
    2. community GitHub mirrors of the canonical CSVs,
    3. a HuggingFace mirror of the widely used *cleaned* 45-column variant,
       which this script normalizes back to the canonical schema (renames
       columns, fills the missing identity columns with NaN) and validates
       on row count + core flow columns.

If every mirror fails, step-by-step manual instructions are printed and the
files can be dropped into ``data/raw/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# name -> expected number of records (used to validate the download)
SPLITS = {
    "training": {"filename": "UNSW_NB15_training-set.csv", "rows": 175_341},
    "testing": {"filename": "UNSW_NB15_testing-set.csv", "rows": 82_332},
}

# Mirrors tried in order: (variant, url).
#   "canonical" = the official 49-column schema (capitalized names, real
#                 srcip/sport/dstip/dsport/Stime/Ltime).
#   "cleaned"   = the 45-column variant (no IPs/ports/timestamps),
#                 normalized to the canonical schema.
# Note: the canonical files are login-gated on UNSW SharePoint, so this
# mirror list may end at "cleaned"; see MANUAL_INSTRUCTIONS for the
# steps to obtain the full canonical CSVs (needed for srcip/dstip-based
# sequence construction in the LSTM stage).
MIRRORS = {
    "training": [
        ("canonical", "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEwRvwNA/download"),
        ("canonical", "https://raw.githubusercontent.com/cyberheartmi9/UNSW-NB15-Dataset/master/UNSW_NB15_training-set.csv"),
        ("cleaned", "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv"),
        ("cleaned", "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_training-set.csv"),
    ],
    "testing": [
        ("canonical", "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEwRvwNA/download"),
        ("canonical", "https://raw.githubusercontent.com/cyberheartmi9/UNSW-NB15-Dataset/master/UNSW_NB15_testing-set.csv"),
        ("cleaned", "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_testing-set.csv"),
    ],
}

EXPECTED_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt",
    "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label",
]

# Columns required (row count aside) for a download to be trusted.
MIN_COLUMNS = [
    "proto", "state", "service", "dur", "sbytes", "dbytes",
    "Sload", "Dload", "Spkts", "Dpkts", "attack_cat", "Label",
]

# Renames applied to the cleaned 45-column variant so it matches the
# canonical schema. Columns not listed keep their names; 'id'/'rate' are
# dropped; missing canonical columns are added as NaN.
CLEANED_RENAMES = {
    "label": "Label", "spkts": "Spkts", "dpkts": "Dpkts",
    "sload": "Sload", "dload": "Dload", "sinpkt": "Sintpkt", "dinpkt": "Dintpkt",
    "sjit": "Sjit", "djit": "Djit", "smean": "smeansz", "dmean": "dmeansz",
    "response_body_len": "res_bdy_len",
}
CLEANED_DROP = {"id", "rate"}
CLEANED_FILL = ["srcip", "sport", "dstip", "dsport", "Stime", "Ltime"]


def _download(url: str, destination: Path, timeout: int = 3600) -> bool:
    """Stream ``url`` to ``destination``; return True on success."""
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(destination, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
        return True
    except (requests.RequestException, OSError) as exc:
        print(f"  [x] {url}\n      -> {exc}", file=sys.stderr)
        return False


def normalize_cleaned(path: Path) -> None:
    """Rewrite a cleaned-variant CSV in place with the canonical schema."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in CLEANED_DROP if c in df.columns])
    df = df.rename(columns=CLEANED_RENAMES)
    for col in CLEANED_FILL:
        if col not in df.columns:
            df[col] = float("nan")
    df.to_csv(path, index=False)
    print(f"  [~] normalized cleaned variant to canonical schema "
          f"({len(df.columns)} columns)")


def validate(path: Path, expected_rows: int, variant: str) -> bool:
    """Check a downloaded CSV parses and matches the expected schema."""
    try:
        df = pd.read_csv(path, nrows=expected_rows + 1)
        if len(df) != expected_rows:
            print(f"  [x] {path.name}: {len(df)} rows, expected {expected_rows}")
            return False
        required = EXPECTED_COLUMNS if variant == "canonical" else MIN_COLUMNS
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [x] {path.name}: missing columns {missing}")
            return False
        print(f"  [ok] {path.name}: {len(df)} rows, {len(df.columns)} columns "
              f"({variant})")
        return True
    except Exception as exc:  # noqa: BLE001 - validation must never crash
        print(f"  [x] {path.name} failed validation: {exc}")
        return False


def download_split(split: str) -> Path | None:
    """Download one split to ``data/raw``; returns its path or None."""
    spec = SPLITS[split]
    destination = RAW_DIR / spec["filename"]
    if destination.exists() and validate(destination, spec["rows"], "canonical"):
        print(f"  [=] {destination.name} already present and valid")
        return destination

    print(f"  [~] downloading {split} split -> {destination}")
    for variant, url in MIRRORS[split]:
        if not _download(url, destination):
            continue
        if variant == "canonical_lower":
            normalize_lower(destination)
        elif variant == "cleaned":
            normalize_cleaned(destination)
        if validate(destination, spec["rows"], variant):
            return destination
        destination.unlink(missing_ok=True)
    return None


def combine(splits: list[Path]) -> Path:
    """Merge the downloaded splits into one labelled dataset for modeling."""
    frames = [pd.read_csv(path) for path in splits]
    combined = pd.concat(frames, ignore_index=True)
    out = RAW_DIR / "UNSW_NB15_combined.csv"
    combined.to_csv(out, index=False)
    print(f"  [ok] combined dataset -> {out} ({len(combined):,} records)")
    return out


MANUAL_INSTRUCTIONS = """
===========================================================================
  MANUAL DOWNLOAD REQUIRED - the automated mirrors were unreachable.
===========================================================================

1. Open the official UNSW-NB15 page:
     https://research.unsw.edu.au/projects/unsw-nb15-dataset

   The "Download" section links the source files (pcap, BRO/Zeek, Argus
   and CSV packages) hosted on UNSW SharePoint:
     https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/
     EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS

   The two files used by this project are:
     - UNSW_NB15_training-set.csv   (175,341 records)
     - UNSW_NB15_testing-set.csv    (82,332 records)

   IMPORTANT: only the official files contain real srcip/sport/dstip/
   dsport/Stime/Ltime columns. They are required for the LSTM sequence
   stage (grouping flows by srcip/dstip, sorting by Stime). The cleaned
   HF mirrors below drop them, so the LSTM falls back to weaker grouping.

2. Alternative mirrors that host the same files:
     - HuggingFace (cleaned 45-column variant):
         https://huggingface.co/datasets/Mouwiya/UNSW-NB15
     - Kaggle:
         https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
     - GitHub:
         https://github.com/cyberheartmi9/UNSW-NB15-Dataset

3. Save the CSVs into the folder:
     {raw_dir}
   keeping the original file names. Then re-run this script (or skip it):
     python src/data/download_dataset.py --validate

===========================================================================
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=list(SPLITS), default=None,
        help="download a single split (default: both)",
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="also merge the splits into UNSW_NB15_combined.csv",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="only validate files already present in data/raw/",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    splits = [s for s in SPLITS if args.split in (None, s)]
    if args.validate:
        results = [validate(RAW_DIR / SPLITS[s]["filename"], SPLITS[s]["rows"], "canonical")
                   for s in splits]
        return None if all(results) else sys.exit(1)

    paths = {s: download_split(s) for s in splits}
    failed = [s for s, p in paths.items() if p is None]
    if failed:
        print(MANUAL_INSTRUCTIONS.format(raw_dir=RAW_DIR))
        sys.exit(1)

    if args.combine or all(paths.values()):
        combine([paths[s] for s in splits if paths[s]])
    print("Done. Next: python src/data/explore.py")


if __name__ == "__main__":
    main()
