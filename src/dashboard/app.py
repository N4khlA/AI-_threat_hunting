"""Streamlit monitoring dashboard for the threat-detection API.

Reads the prediction log written by ``src/serving/api.py``
(``reports/prediction_log.sqlite``) and renders live monitoring:

    * detection rate over time (rolling window),
    * false-positive-rate trend,
    * bar chart of detected attack types,
    * live confusion matrix (as ground-truth labels arrive),
    * model confidence distribution.

Run (API not required, but the DB only fills up while it runs):
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.prediction_log import init_db, load_predictions, log_db_path  # noqa: E402

st.set_page_config(page_title="Threat Detection Monitor",
                   page_icon="monitor", layout="wide")

COLUMNS = ["ts", "model", "prediction", "confidence", "true_label",
           "predicted_family", "true_family", "n_flows", "record_json"]

SIDEBAR_HINT = (
    "Stream traffic at the API (see README: uvicorn + simulate_traffic.py); "
    "every prediction lands in reports/prediction_log.sqlite and this "
    "dashboard refreshes from it."
)


def load_df() -> pd.DataFrame:
    rows = load_predictions()
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


def bucket_rates(df: pd.DataFrame, bucket_s: int, roll: int) -> pd.DataFrame:
    """Detection rate + FPR per time bucket, rolling-mean smoothed."""
    lab = df[df["true_label"].notna()].copy()
    if lab.empty:
        return pd.DataFrame(columns=["detection_rate", "fpr", "n", "ts"])
    lab["bucket"] = (lab["ts"].astype("int64") // 1_000_000_000) // bucket_s
    lab["tp"] = ((lab["prediction"] == 1) & (lab["true_label"] == 1)).astype(int)
    lab["fn"] = ((lab["prediction"] == 0) & (lab["true_label"] == 1)).astype(int)
    lab["fp"] = ((lab["prediction"] == 1) & (lab["true_label"] == 0)).astype(int)
    lab["tn"] = ((lab["prediction"] == 0) & (lab["true_label"] == 0)).astype(int)
    agg = lab.groupby("bucket")[["tp", "fn", "fp", "tn"]].sum()
    agg["detection_rate"] = agg["tp"] / (agg["tp"] + agg["fn"])
    agg["fpr"] = agg["fp"] / (agg["fp"] + agg["tn"])
    agg["n"] = agg[["tp", "fn", "fp", "tn"]].sum(axis=1)
    agg[["detection_rate", "fpr"]] = agg[["detection_rate", "fpr"]].rolling(
        roll, min_periods=1).mean()
    agg["ts"] = pd.to_datetime(agg.index * bucket_s, unit="s", utc=True)
    return agg.reset_index(drop=True)


def confusion_matrix_figure(df: pd.DataFrame) -> plt.Figure:
    lab = df[df["true_label"].notna()]
    tn = int(((lab["prediction"] == 0) & (lab["true_label"] == 0)).sum())
    fp = int(((lab["prediction"] == 1) & (lab["true_label"] == 0)).sum())
    fn = int(((lab["prediction"] == 0) & (lab["true_label"] == 1)).sum())
    tp = int(((lab["prediction"] == 1) & (lab["true_label"] == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["pred normal", "pred attack"],
                yticklabels=["true normal", "true attack"], ax=ax)
    ax.set_title(f"Confusion matrix (n={len(lab):,} labeled predictions)")
    fig.tight_layout()
    return fig


def confidence_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not df.empty:
        for label, color in ((0, "tab:green"), (1, "tab:red")):
            subset = df[df["prediction"] == label]["confidence"]
            if len(subset):
                ax.hist(subset, bins=40, alpha=0.55, color=color,
                        label=f"prediction={label} (n={len(subset):,})")
        ax.set_xlabel("confidence")
        ax.set_ylabel("count")
        ax.set_title("Model confidence distribution")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no predictions yet", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def main() -> None:
    st.title("Threat Detection Monitor")
    st.caption(SIDEBAR_HINT)

    with st.sidebar:
        st.header("Controls")
        if st.button("Refresh now"):
            st.rerun()
        auto = st.toggle("Auto-refresh", value=False,
                         help="Re-run the dashboard automatically")
        interval = st.selectbox("Auto-refresh interval (s)",
                                [5, 10, 15, 30, 60], index=1)
        bucket_s = st.selectbox("Time-bucket size (s)",
                                [5, 10, 30, 60], index=1)
        roll = st.slider("Rolling mean window (buckets)", 1, 30, 10)
        st.caption(f"Log: `{log_db_path()}`")

    init_db()  # idempotent: also covers the 'no log yet' first run
    df = load_df()
    if df.empty:
        st.info("No predictions logged yet. Start the API "
                "(`uvicorn src.serving.api:app --port 8000`) and stream "
                "traffic (`python src/serving/simulate_traffic.py`) - this "
                "page will pick them up.")
        return

    labeled = df[df["true_label"].notna()]
    n_attack_true = int((labeled["true_label"] == 1).sum())

    tp = int(((labeled["prediction"] == 1) & (labeled["true_label"] == 1)).sum())
    fn = int(((labeled["prediction"] == 0) & (labeled["true_label"] == 1)).sum())
    fp = int(((labeled["prediction"] == 1) & (labeled["true_label"] == 0)).sum())
    tn = int(((labeled["prediction"] == 0) & (labeled["true_label"] == 0)).sum())
    det_rate = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Predictions", f"{len(df):,}")
    m2.metric("With ground truth", f"{len(labeled):,}")
    m3.metric("Predicted attacks", f"{int((df['prediction'] == 1).sum()):,}")
    m4.metric("Detection rate", f"{det_rate:.1%}" if not np.isnan(det_rate) else "-",
              help="recall on labeled attack flows")
    m5.metric("False positive rate", f"{fpr:.1%}" if not np.isnan(fpr) else "-",
              help="alarms among labeled normal flows")

    rates = bucket_rates(df, bucket_s, roll)
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Detection rate over time")
        if rates.empty:
            st.caption("No labeled predictions yet - ground truth arrives "
                       "when requests carry Label/attack_cat (e.g. via the "
                       "traffic simulator).")
        else:
            chart = rates.set_index("ts")[["detection_rate"]]
            st.line_chart(chart, height=260)

    with col_b:
        st.subheader("False positive rate trend")
        if rates.empty:
            st.caption("Awaiting labeled predictions.")
        else:
            chart = rates.set_index("ts")[["fpr"]]
            st.line_chart(chart, height=260)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Attack types detected")
        attacks = df[df["prediction"] == 1].copy()
        if attacks.empty:
            st.caption("No attacks predicted yet.")
        else:
            attacks["family"] = attacks["predicted_family"].fillna(
                attacks["true_family"])
            counts = attacks["family"].value_counts()
            if counts.empty:
                st.caption("No attack-family info logged.")
            else:
                st.bar_chart(counts, height=260)

    with col_d:
        st.subheader("Confusion matrix (labeled)")
        if labeled.empty:
            st.caption("Awaiting labeled predictions.")
        else:
            st.pyplot(confusion_matrix_figure(df))

    st.subheader("Confidence distribution")
    st.pyplot(confidence_figure(df))
    st.caption(
        f"DB buckets: {n_attack_true:,} labeled attacks · "
        f"models: {', '.join(sorted(df['model'].unique()))}")

    if auto:
        time.sleep(interval)
        st.rerun()


if __name__ == "__main__":
    main()
