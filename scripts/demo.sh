#!/usr/bin/env bash
# 2-minute demo: start the hardened API, stream held-out flows as a sensor,
# confirm verdicts as an analyst, print a triage note, show the controls.
# Needs the trained artifacts (scripts/reproduce.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
PORT=${PORT:-8000}
URL="http://127.0.0.1:$PORT"

new_key() { $PY -m src.serving.security new-key "$1" "$2"; }
SENSOR_OUT=$(new_key sensor-1 sensor); ANALYST_OUT=$(new_key alice analyst)
export IDS_SENSOR_KEY=$(sed -n 's/^API key.*: //p' <<<"$SENSOR_OUT")
export IDS_ANALYST_KEY=$(sed -n 's/^API key.*: //p' <<<"$ANALYST_OUT")
export IDS_API_KEYS="$(sed -n 's/^IDS_API_KEYS entry: //p' <<<"$SENSOR_OUT"),$(sed -n 's/^IDS_API_KEYS entry: //p' <<<"$ANALYST_OUT")"
export PREDICTION_LOG_PATH=${PREDICTION_LOG_PATH:-reports/demo_log.sqlite}

$PY -m uvicorn src.serving.api:app --host 127.0.0.1 --port "$PORT" --log-level warning &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
for _ in $(seq 60); do curl -fs "$URL/health" >/dev/null 2>&1 && break; sleep 1; done

echo "== 1. stream 40 held-out flows (sensor) + analyst feedback =="
$PY src/serving/simulate_traffic.py --port "$PORT" --limit 40 --delay 0 --show-alerts 1

echo; echo "== 2. security controls =="
attack='{"proto":"tcp","service":"-","state":"INT","sbytes":1000,"Label":0}'
printf 'no key                      -> '; curl -s -o /dev/null -w '%{http_code}\n' -X POST "$URL/predict" -H 'content-type: application/json' -d '{"proto":"tcp"}'
printf 'sensor sends a Label (v1 poisoning path) -> '; curl -s -o /dev/null -w '%{http_code}\n' -X POST "$URL/predict" -H "X-API-Key: $IDS_SENSOR_KEY" -H 'content-type: application/json' -d "$attack"
printf 'sensor calls /feedback     -> '; curl -s -o /dev/null -w '%{http_code}\n' -X POST "$URL/feedback" -H "X-API-Key: $IDS_SENSOR_KEY" -H 'content-type: application/json' -d '{"prediction_id":1,"label":0}'
printf 'sttl=300                   -> '; curl -s -o /dev/null -w '%{http_code}\n' -X POST "$URL/predict" -H "X-API-Key: $IDS_SENSOR_KEY" -H 'content-type: application/json' -d '{"sttl":300}'
echo; echo "Dashboard: PREDICTION_LOG_PATH=$PREDICTION_LOG_PATH streamlit run src/dashboard/app.py"
