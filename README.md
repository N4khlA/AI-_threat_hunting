# AI Threat Detection - ML-based Network Intrusion Detection

End-to-end machine learning system that detects network attacks (DoS,
scans, exploits, fuzzing, botnet-style traffic, ...) from raw **UNSW-NB15**
flow records, and serves the models through a live API with a monitoring
dashboard and a guarded retraining pipeline.

```
data -> preprocessing -> Random Forest (per-flow)  -> FastAPI /predict
                      -> Bidirectional LSTM (per-window, sequences)
                                                        |
                         prediction_log.sqlite <--------/
                                |
                    Streamlit dashboard (live monitoring)
                                |
                    retrain.py (compare-and-replace guard) -> models/
```

## Model Results (held-out test set)

| Model | Unit | Accuracy | F1 (macro) | AUC |
|---|---|---|---|---|
| Random Forest (grid-searched) | per-flow | **0.9514** | **0.9476** | 0.9922 |
| Bidirectional LSTM (2x64, window 10) | per-window | **0.9805** | **0.9783** | 0.9980 |

- LSTM window labels use `any` mode (attack if any flow in the window is
  an attack), so the units are not directly comparable. The statistically
  valid head-to-head is the **aligned McNemar test** in
  `reports/model_comparison.md`: on 1,199 flows present in both test
  sets, LSTM 0.9716 vs RF 0.9575 accuracy, McNemar p = 0.058 - the
  sequence model is numerically stronger but the gap is not significant
  at the 0.05 level.
- Per-family detection (RF, `reports/model_comparison.md`): recall >= 0.97
  for every family except **Fuzzers (72.0%)** and **Analysis (87.9%)** -
  stable weak spots of UNSW-NB15 classifiers (see `docs/limitations.md`).
- Inference speed: RF scores ~700k flows/s on CPU; LSTM ~6.3k windows/s.

## Setup

Requires **64-bit Python 3.12** (TensorFlow does not ship 32-bit wheels).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Everything below assumes the venv is active (or prefix commands with
`.\.venv\Scripts\python.exe`).

## Pipeline (run in order)

```powershell
# 1. Download the dataset (mirrors: cloudstor -> HF cleaned 45-col variant;
#    prints manual fallback steps if every mirror is down)
python src/data/download_dataset.py --combine

# 2. Exploratory analysis -> reports/eda_summary.txt, *.png
python src/data/explore.py

# 3. Preprocessing -> data/processed/ splits, models/{scaler,pipeline,metadata}
python src/features/preprocess.py --imbalance smote
#    alternatives: --imbalance weights | none, --categorical onehot,
#                  --target multiclass (predict attack_cat instead)

# 4. Random Forest: grid search + feature importances
python src/models/random_forest.py           # full grid (slow)
python src/models/random_forest.py --fast    # smoke run
python src/models/random_forest.py --fast --drop-borderline  # ablation

# 5. Per-class / per-attack-family evaluation -> reports/per_class_metrics.*
python src/evaluation/per_class_metrics.py

# 6. Leakage sanity scan -> reports/leakage_scan.csv
python src/data/leakage_check.py

# 7. LSTM on flow sequences -> models/lstm_best.keras
python src/models/lstm_model.py                # full run (~2 min CPU)
python src/models/lstm_model.py --fast         # smoke test
python src/models/lstm_model.py --label-mode majority

# 8. Head-to-head evaluation -> reports/model_comparison.md,
#    reports/evaluation_summary.json, reports/confusion_matrix_{rf,lstm}.png
python src/evaluation/evaluate.py

# 9. Tests (49 tests: pipeline, leakage check, LSTM windowing, API
#    endpoints, retrain guard)
python -m pytest tests/ -q
```

## Serving: REST API

```powershell
python -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
```

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | - | model status, uptime |
| `POST /predict` | one raw flow record (49 fields, extras allowed) | `{prediction, confidence, attack_family_probs, method}` |
| `POST /predict/sequence` | `{"flows": [...]}` raw records | LSTM window predictions with grouping + covered flow indices |

```powershell
# single flow
$body = Get-Content flow.json -Raw   # any subset of the 49 fields
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body $body

# or use the simulator: streams held-out test flows at the API,
# printing live predictions compared with ground truth
python src/serving/simulate_traffic.py --mode row --limit 100 --delay 0.1
python src/serving/simulate_traffic.py --mode sequence --seq-size 10 --limit 100
```

Every prediction (timestamp, prediction, confidence, true label and
attack family when the request carried them, raw flow record) is appended
to **`reports/prediction_log.sqlite`** - the input for the dashboard and
the retraining script.

## Monitoring Dashboard

```powershell
streamlit run src/dashboard/app.py
```

Reads `reports/prediction_log.sqlite` and renders:

- detection rate over time (rolling window) and false-positive-rate trend,
- bar chart of detected attack types,
- live confusion matrix as ground-truth labels arrive,
- model confidence distribution,
- manual **Refresh now** button and an **auto-refresh** toggle
  (5-60 s interval).

## Retraining (with a "don't replace if worse" guard)

```powershell
# harvest logged predictions: confirmed labels + high-confidence
# pseudo-labels (confidence >= 0.95), deduplicated vs the base dataset
python src/models/retrain.py

# confirmed labels only (no pseudo-labels)
python src/models/retrain.py --labeled-only

# evaluate without writing anything
python src/models/retrain.py --dry-run
```

`retrain.py` re-runs the preprocessing pipeline on base data + harvested
rows, trains a new Random Forest, and evaluates it against the current
model on the **exact same held-out test set**
(`data/processed/test_positions.csv`). It replaces
`models/random_forest_best.joblib` (and the pipeline artifacts) only if
the new model wins - otherwise it prints the comparison and refuses
(exit code 1). The verdict is recorded in `reports/retrain_results.json`.

## Known Issues & Fixes (debugging timeline)

This section records the notable bugs found and fixed during
development - not because they were fun, but because each one is a
lesson about silent failure modes in ML systems.

**1. Label leakage into features (preprocessing).**
`attack_cat` (the attack-family column) is redundant with the binary
`Label` target: if one label column is kept in the feature matrix while
the other is the target, the model learns the leak and reports
impossibly good scores. Fixed by dropping *both* label columns from `X`
in `src/features/preprocess.py`, backed by
`src/data/leakage_check.py` - a scan that flags any raw feature
correlating ~1.0 with the label (runs as pipeline step 6).

**2. NaN/None dtype mismatch in retrain deduplication.**
The retraining script dedupes harvested predictions against the base
dataset before retraining. The base CSV carries all-NaN columns (e.g.
`srcip`) as `float64`, but JSON log records carry `None` (`object`), so
the merge silently failed and test-replayed flows could have been kept
as "new" training rows (test leakage into retraining). Fixed by coercing
the harvested rows to the base column dtypes before joining; regression
test added in `tests/test_retrain.py`.

**3. Pipeline/model desync (deployed RF stale vs its own pipeline).**
The Random Forest was trained on an earlier `X_train.csv`; a later
preprocessing re-run regenerated every artifact except the model, so the
deployed RF was silently under-trained relative to its own pipeline -
evaluation reported 0.9372 test accuracy, the fair value is **0.9514**.
The retrain guard caught it: with zero new rows it still wanted to
replace the model. Same audit found a related guard bug (`max_depth=None`
from the grid search was dropped, substituted with 25). Both fixed;
the guard now refuses bit-for-bit when nothing changed (regression
tested). Lesson and production recommendations:
`docs/limitations.md` section 3 (pipeline-version pinning).

**4. JSON nulls misrouted in the serving transform.**
`transform_flows` in `src/models/lstm_model.py` originally inferred
feature types from the incoming JSON, so numeric fields arriving as
`null` were routed to the categorical branch and the LSTM scored those
requests wrong. Now the transform uses the imputer's fitted
`feature_names_in_` (same 43 features as training) regardless of
incoming dtypes.

## Project Structure

```
├── data/
│   ├── raw/            # UNSW-NB15 CSVs (download_dataset.py or manual)
│   └── processed/      # splits, test positions, attack_cat labels, lstm_test.npz
├── src/
│   ├── data/           # download_dataset.py, explore.py, leakage_check.py
│   ├── features/       # preprocess.py (impute/encode/scale/split/SMOTE)
│   ├── models/         # random_forest.py, lstm_model.py, retrain.py
│   ├── evaluation/     # per_class_metrics.py, evaluate.py (RF vs LSTM)
│   ├── serving/        # api.py (FastAPI), prediction_log.py, simulate_traffic.py
│   └── dashboard/      # app.py (Streamlit monitor)
├── models/             # scaler, pipeline, metadata, best RF/LSTM
├── reports/            # EDA, importances, evaluation, prediction_log.sqlite
├── docs/limitations.md # evasion, weak spots, pipeline pinning, drift, SOC
├── tests/              # pytest suite (49 tests)
└── requirements.txt
```

## Known Limitations

See **`docs/limitations.md`**: adversarial evasion (RF feature-importance
transparency tells attackers what to perturb - e.g. lower Sload/sbytes),
the stable Fuzzers/Analysis weak spots (benign-mimicry attacks defeat
anomaly detection), the RF/pipeline desync incident and why pipeline
versions must be pinned with models, concept drift as legitimate traffic
shifts, why ML must be paired with signature/rule systems, and how a
`/predict` alert realistically feeds a SOC workflow (ticket creation,
confidence-to-severity mapping, human-in-the-loop review threshold that
closes the retraining loop).

## License & Citation

Dataset: N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set
for network intrusion detection systems," 2015 Military Communications
and Information Systems Conference (MilCIS), 2015.
