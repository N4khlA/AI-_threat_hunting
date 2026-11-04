#!/usr/bin/env bash
# Rebuild every artifact and report from the raw dataset, in order.
# ~15 min on 8 CPU cores (no GPU). Every step is seeded.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

$PY src/data/download_dataset.py --combine        # data/raw/ (needs network)
$PY src/data/explore.py                           # EDA -> reports/
$PY src/features/preprocess.py                    # official_novel split + pipeline
$PY src/models/random_forest.py --no-search       # RF (drop --no-search for the 40-min grid)
$PY src/models/family_classifier.py               # attack-family RF (alert -> ATT&CK)
$PY src/models/lstm_model.py                      # LSTM, leak-free window split
$PY src/models/lstm_model.py --split-mode random --results-name lstm_results_random_split.json
$PY src/evaluation/per_class_metrics.py
$PY src/evaluation/evaluate.py                    # RF vs LSTM, McNemar
$PY src/evaluation/rigorous_eval.py               # 5 protocols, CIs, base rates
$PY src/attacks/poisoning.py --family Backdoor    # label-poisoning PoC
$PY src/attacks/evasion.py                        # evasion benchmark
$PY src/models/manifest.py --verify               # artifacts match their pins
$PY -m pytest tests -q -p no:warnings
