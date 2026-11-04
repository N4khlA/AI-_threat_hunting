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
    confidence      [0, 1] max(p_attack, 1 - p_attack) (dashboard)
    p_attack        [0, 1] the model's attack probability
    true_label      0/1 ground truth - written ONLY by record_feedback()
    predicted_family  family classifier's top guess (rf attack rows)
    true_family     analyst-confirmed attack family
    n_flows         flows covered by this prediction (1 for rf)
    record_json     the validated flow record as JSON (rf rows only)
    sensor          key name of the sensor that submitted the flow
    label_source    'analyst' when a label came through /feedback
    labeled_by      key name of the analyst (audit trail)
    labeled_at      unix epoch seconds of the feedback

Labels never come from the scoring request: the old API took ``Label``
from the sensor payload, which let any caller poison retraining. The
dashboard reads this table for live monitoring; ``retrain.py`` uses
``record_json`` + analyst labels as fresh training material.
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
    record_json TEXT,
    p_attack REAL,
    sensor TEXT,
    label_source TEXT,
    labeled_by TEXT,
    labeled_at REAL
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);
"""

# Columns added after v1; migrated in place on older log files.
ADDED_COLUMNS = {"p_attack": "REAL", "sensor": "TEXT", "label_source": "TEXT",
                 "labeled_by": "TEXT", "labeled_at": "REAL"}


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
        have = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        for col, kind in ADDED_COLUMNS.items():
            if col not in have:  # identifiers come from the constant above
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {kind}")
    return path


def log_prediction(
    model: str,
    prediction: int,
    p_attack: float,
    predicted_family: str | None = None,
    n_flows: int = 1,
    record_json: str | None = None,
    sensor: str | None = None,
    ts: float | None = None,
    path: Path | None = None,
) -> int:
    """Append one prediction row; return its id (0 if logging failed)."""
    if ts is None:
        import time

        ts = time.time()
    path = path or log_db_path()
    confidence = max(p_attack, 1.0 - p_attack)
    try:
        with sqlite3.connect(path) as conn:
            cur = conn.execute(
                """INSERT INTO predictions
                   (ts, model, prediction, confidence, p_attack,
                    predicted_family, n_flows, record_json, sensor)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, model, int(prediction), float(confidence), float(p_attack),
                 predicted_family, int(n_flows), record_json, sensor),
            )
            return int(cur.lastrowid)
    except sqlite3.Error as exc:  # logging must never break inference
        print(f"[prediction_log] WARNING: could not log prediction: {exc}")
        return 0


def record_feedback(prediction_id: int, label: int, family: str | None,
                    analyst: str, path: Path | None = None) -> bool:
    """Attach an analyst verdict to a logged prediction; False if unknown id."""
    import time

    path = path or log_db_path()
    with sqlite3.connect(path) as conn:
        cur = conn.execute(
            """UPDATE predictions
                  SET true_label = ?, true_family = ?, label_source = 'analyst',
                      labeled_by = ?, labeled_at = ?
                WHERE id = ?""",
            (int(label), family, analyst, time.time(), int(prediction_id)))
        return cur.rowcount == 1


SELECT_BASIC = ("SELECT ts, model, prediction, confidence, true_label,"
                " predicted_family, true_family, n_flows, record_json"
                " FROM predictions ORDER BY ts")
SELECT_EXTENDED = ("SELECT ts, model, prediction, confidence, true_label,"
                   " predicted_family, true_family, n_flows, record_json,"
                   " p_attack, label_source, labeled_by"
                   " FROM predictions ORDER BY ts")


def load_predictions(path: Path | None = None, extended: bool = False) -> list[tuple]:
    """All prediction rows as (ts, model, prediction, confidence,
    true_label, predicted_family, true_family, n_flows, record_json)
    [+ (p_attack, label_source, labeled_by) when ``extended``]."""
    path = path or log_db_path()
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            return [tuple(r) for r in conn.execute(
                SELECT_EXTENDED if extended else SELECT_BASIC)]
    except sqlite3.Error as exc:
        print(f"[prediction_log] WARNING: could not read log: {exc}")
        return []
