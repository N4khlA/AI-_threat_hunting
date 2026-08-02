"""SQLite prediction log shared by the API, the dashboard and retraining.

Every prediction served by ``src/serving/api.py`` is appended to
``reports/prediction_log.sqlite`` (overridable with the
``PREDICTION_LOG_PATH`` environment variable, used by the test suite).

Schema
------
predictions
    id              autoincrement
    ts              unix epoch seconds (UTC)
    model           'rf' (per-flow) or 'lstm' (per-window)
    prediction      0/1 binary prediction
    confidence      [0, 1] model confidence
    true_label      0/1 ground truth when the request carried a label
    predicted_family  top predicted attack family (rf rows)
    true_family     ground-truth attack family when available
    n_flows         flows covered by this prediction (1 for rf)
    record_json     the raw flow record as JSON (rf rows only)

The dashboard reads this table for live monitoring; ``retrain.py`` uses
``record_json`` + labels as fresh training material (with exact-duplicate
deduplication against the base dataset).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "reports" / "prediction_log.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    model TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    confidence REAL NOT NULL,
    true_label INTEGER,
    predicted_family TEXT,
    true_family TEXT,
    n_flows INTEGER NOT NULL DEFAULT 1,
    record_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);
"""


def log_db_path() -> Path:
    """Path of the prediction log; override with PREDICTION_LOG_PATH."""
    override = os.environ.get("PREDICTION_LOG_PATH")
    return Path(override) if override else DEFAULT_DB


def init_db(path: Path | None = None) -> Path:
    """Create the schema (if needed) and return the resolved path."""
    path = path or log_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
    return path


def log_prediction(
    model: str,
    prediction: int,
    confidence: float,
    true_label: int | None = None,
    predicted_family: str | None = None,
    true_family: str | None = None,
    n_flows: int = 1,
    record_json: str | None = None,
    ts: float | None = None,
    path: Path | None = None,
) -> None:
    """Append one prediction row. Opens its own connection (thread-safe)."""
    if ts is None:
        import time

        ts = time.time()
    path = path or log_db_path()
    try:
        with sqlite3.connect(path) as conn:
            conn.execute(
                """INSERT INTO predictions
                   (ts, model, prediction, confidence, true_label,
                    predicted_family, true_family, n_flows, record_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, model, int(prediction), float(confidence),
                 true_label, predicted_family, true_family,
                 int(n_flows), record_json),
            )
    except sqlite3.Error as exc:  # logging must never break inference
        print(f"[prediction_log] WARNING: could not log prediction: {exc}")


def load_predictions(path: Path | None = None) -> list[tuple]:
    """All prediction rows as (ts, model, prediction, confidence,
    true_label, predicted_family, true_family, n_flows, record_json)."""
    path = path or log_db_path()
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            return [tuple(r) for r in conn.execute(
                "SELECT ts, model, prediction, confidence, true_label,"
                "       predicted_family, true_family, n_flows, record_json"
                "  FROM predictions ORDER BY ts")]
    except sqlite3.Error as exc:
        print(f"[prediction_log] WARNING: could not read log: {exc}")
        return []
