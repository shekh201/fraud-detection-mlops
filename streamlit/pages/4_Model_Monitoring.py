
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# Load shared API client
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIENT_PATH = PROJECT_ROOT / "streamlit" / "api_client.py"

spec = importlib.util.spec_from_file_location(
    "fraud_api_client",
    CLIENT_PATH,
)
api_client = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api_client
spec.loader.exec_module(api_client)

# --------------------------------------------------
# Page configuration and styling
# --------------------------------------------------

st.set_page_config(
    page_title="Model Monitoring | Fraud Intelligence",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.20);
        padding: 16px;
        border-radius: 12px;
    }
    .monitor-caption {
        color: #808080;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚙️ Model & System Monitoring")
st.caption(
    "Monitor API health, operational counters and feature drift."
)

header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.markdown(
        '<p class="monitor-caption">'
        'Environment: Local Kubernetes · Metrics: Prometheus'
        '</p>',
        unsafe_allow_html=True,
    )

with header_col2:
    if st.button("↻ Refresh", type="primary", use_container_width=True):
        st.rerun()

# --------------------------------------------------
# API health
# --------------------------------------------------

st.subheader("System Health")

health = None

try:
    health = api_client.get_health()

    status = str(health.get("status", "unknown")).upper()
    model = health.get("model", "Unknown")
    source = health.get("model_source", "Unknown")

    h1, h2, h3 = st.columns(3)

    h1.metric("API Status", status)
    h2.metric("Active Model", model)
    h3.metric("Model Source", source)

    if status == "HEALTHY":
        st.success("API health endpoint responded successfully.")
    else:
        st.warning(f"API returned status: {status}")

except api_client.FraudAPIError as exc:
    st.error(f"API health check failed: {exc}")

st.divider()

# --------------------------------------------------
# Prometheus parser
# --------------------------------------------------

def parse_prometheus_metrics(text):
    """Parse numeric Prometheus samples, preserving labels."""
    samples = []

    for line in text.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        try:
            metric_part, value_part = line.rsplit(None, 1)
            value = float(value_part)
        except (ValueError, TypeError):
            continue

        samples.append(
            {
                "metric": metric_part.split("{", 1)[0],
                "labels": metric_part,
                "value": value,
            }
        )

    return samples


try:
    raw_metrics = api_client.get_prometheus_metrics()
    metrics_df = pd.DataFrame(parse_prometheus_metrics(raw_metrics))

    if metrics_df.empty:
        st.warning("No Prometheus metric samples were returned.")
        st.stop()

    def metric_values(name):
        return metrics_df.loc[
            metrics_df["metric"] == name, "value"
        ].astype(float)

    def metric_sum(name):
        values = metric_values(name)
        return float(values.sum()) if not values.empty else None

    def format_count(value):
        return f"{value:,.0f}" if value is not None else "N/A"

    # --------------------------------------------------
    # Operational KPI cards
    # --------------------------------------------------

    st.subheader("Operational Metrics")

    predictions = metric_sum("fraud_prediction_requests_total")
    auth_failures = metric_sum("fraud_auth_failures_total")
    rate_limits = metric_sum("fraud_rate_limit_exceeded_total")
    cache_hits = metric_sum("fraud_cache_hits_total")
    cache_misses = metric_sum("fraud_cache_misses_total")

    cache_hit_rate = None
    if cache_hits is not None and cache_misses is not None:
        cache_total = cache_hits + cache_misses
        if cache_total > 0:
            cache_hit_rate = cache_hits / cache_total

    k1, k2, k3 = st.columns(3)

    k1.metric("Prediction Requests", format_count(predictions))
    k2.metric("Authentication Failures", format_count(auth_failures))
    k3.metric("Rate Limit Events", format_count(rate_limits))

    k4, k5, k6 = st.columns(3)

    k4.metric("Cache Hits", format_count(cache_hits))
    k5.metric("Cache Misses", format_count(cache_misses))
    k6.metric(
        "Cache Hit Rate",
        f"{cache_hit_rate:.1%}" if cache_hit_rate is not None else "N/A",
    )

    st.caption(
        "These are cumulative counter values exposed by the API metrics "
        "endpoint. With multiple API replicas, these values may represent "
        "only the process serving this request, not guaranteed cluster-wide totals."
    )

    # --------------------------------------------------
    # Feature drift
    # --------------------------------------------------

    st.divider()
    st.subheader("Feature Drift Analysis")

    drift_values = metric_values("fraud_drift_monitor_status")
    status_value = int(drift_values.iloc[-1]) if not drift_values.empty else None

    status_names = {
        0: "SUCCESS",
        1: "INSUFFICIENT DATA",
        2: "ERROR",
    }

    d1, d2 = st.columns([1, 2])

    with d1:
        if status_value is None:
            st.metric("Drift Monitor", "N/A")
            st.info("The drift status metric is not available.")
        elif status_value == 0:
            st.metric("Drift Monitor", "SUCCESS")
            st.success("The latest drift-monitor status indicates success.")
        elif status_value == 1:
            st.metric("Drift Monitor", "INSUFFICIENT DATA")
            st.warning(
                "At least 1,000 stored transactions are required by "
                "the current drift-monitor configuration."
            )
        else:
            st.metric("Drift Monitor", status_names.get(status_value, "UNKNOWN"))
            st.error("The drift monitor reported an error.")

    psi_df = metrics_df[
        metrics_df["metric"] == "fraud_feature_drift_psi"
    ].copy()

    if not psi_df.empty:
        psi_df["feature"] = psi_df["labels"].str.extract(
            r'feature="([^"]+)"'
        )
        psi_df = psi_df.dropna(subset=["feature"])

        # Keep the most recent exported value per feature label.
        psi_df = psi_df.drop_duplicates(
            subset=["feature"],
            keep="last",
        )

    with d2:
        if psi_df.empty:
            st.info(
                "No feature PSI values are available yet. "
                "PSI will appear after a successful drift calculation."
            )
        else:
            chart_df = psi_df[["feature", "value"]].rename(
                columns={"feature": "Feature", "value": "PSI"}
            )

            fig = px.bar(
                chart_df.sort_values("PSI", ascending=False),
                x="Feature",
                y="PSI",
                title="Population Stability Index by Feature",
                color="PSI",
                color_continuous_scale="RdYlGn_r",
            )
            fig.add_hline(
                y=0.10,
                line_dash="dash",
                annotation_text="PSI 0.10",
            )
            fig.add_hline(
                y=0.25,
                line_dash="dash",
                annotation_text="PSI 0.25 — high drift",
            )
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=60, b=10),
                coloraxis_showscale=False,
                xaxis_title="",
                yaxis_title="PSI",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Reference guide: PSI below 0.10 is often considered low, "
                "0.10–0.25 moderate, and 0.25 or above high. "
                "These are monitoring heuristics, not proof of model failure."
            )

            display_df = chart_df.copy()
            display_df["PSI"] = display_df["PSI"].round(4)
            st.dataframe(
                display_df.sort_values("PSI", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------
    # Raw metrics and metric inventory
    # --------------------------------------------------

    st.divider()
    st.subheader("Prometheus Metric Inventory")

    inventory = (
        metrics_df.groupby("metric", as_index=False)
        .agg(Samples=("value", "size"))
        .sort_values("metric")
    )

    st.dataframe(
        inventory,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("View raw Prometheus metrics"):
        st.code(raw_metrics, language="text")

except api_client.FraudAPIError as exc:
    st.error(f"Could not load Prometheus metrics: {exc}")