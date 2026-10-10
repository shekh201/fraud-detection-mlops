
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fraud Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# Load shared API client
# --------------------------------------------------

CLIENT_PATH = STREAMLIT_DIR / "api_client.py"

spec = importlib.util.spec_from_file_location(
    "fraud_api_client",
    CLIENT_PATH,
)

api_client = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api_client
spec.loader.exec_module(api_client)


# --------------------------------------------------
# Professional styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(100, 116, 139, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.22);
        padding: 18px;
        border-radius: 12px;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        margin-top: -8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Executive Overview
# --------------------------------------------------

def executive_overview():

    st.title("🛡️ Fraud Intelligence Platform")
    st.markdown(
        '<p class="hero-subtitle">'
        'Transaction risk analytics • Model operations • API health'
        '</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Refresh button
    header_col, refresh_col = st.columns([5, 1])

    with header_col:
        st.subheader("Executive Overview")

    with refresh_col:
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()

    # --------------------------------------------------
    # API health
    # --------------------------------------------------

    try:
        health = api_client.get_health()
        api_healthy = health.get("status", "").lower() == "healthy"

        if api_healthy:
            st.success("● Fraud Detection API is healthy")
        else:
            st.warning(
                f"API returned status: {health.get('status', 'Unknown')}"
            )

        health_col1, health_col2, health_col3 = st.columns(3)

        health_col1.metric(
            "API Status",
            "ONLINE" if api_healthy else "CHECK STATUS",
        )

        health_col2.metric(
            "Registered Model",
            health.get("model", "Unknown"),
        )

        health_col3.metric(
            "Model Source",
            health.get("model_source", "Unknown"),
        )

    except api_client.FraudAPIError as exc:
        st.error(f"API health check failed: {exc}")
        st.warning(
            "Check that the Kubernetes HTTPS port-forward is running."
        )
        return

    st.divider()

    # --------------------------------------------------
    # Fetch actual transaction history
    # --------------------------------------------------

    try:
        with st.spinner("Loading transaction analytics..."):
            result = api_client.get_transactions(
                limit=100,
                offset=0,
            )

        transactions = result.get("items", [])
        total_transactions = int(result.get("total", 0))

    except api_client.FraudAPIError as exc:
        st.error(f"Could not load transaction history: {exc}")
        return

    df = pd.DataFrame(transactions)

    # --------------------------------------------------
    # KPI cards
    # --------------------------------------------------

    st.subheader("Transaction Intelligence")

    if df.empty:
        allow_count = 0
        review_count = 0
        block_count = 0
        average_probability = None
        total_amount = 0.0
    else:
        decisions = df["risk_decision"].value_counts()

        allow_count = int(decisions.get("ALLOW", 0))
        review_count = int(decisions.get("REVIEW", 0))
        block_count = int(decisions.get("BLOCK", 0))

        average_probability = float(
            df["fraud_probability"].mean() * 100
        )

        total_amount = float(df["amount"].sum())

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric(
        "Stored Predictions",
        f"{total_transactions:,}",
        help="Total saved predictions reported by the API.",
    )

    kpi2.metric(
        "ALLOW",
        f"{allow_count:,}",
        help="ALLOW decisions in the currently loaded 100 records.",
    )

    kpi3.metric(
        "REVIEW",
        f"{review_count:,}",
        help="REVIEW decisions in the currently loaded 100 records.",
    )

    kpi4.metric(
        "BLOCK",
        f"{block_count:,}",
        help="BLOCK decisions in the currently loaded 100 records.",
    )

    if df.empty:
        st.info(
            "No saved predictions are available yet. "
            "Run a transaction through Transaction Analyzer first."
        )
        return

    second1, second2 = st.columns(2)

    second1.metric(
        "Average Fraud Probability",
        f"{average_probability:.4f}%",
        help="Average probability across the currently loaded records.",
    )

    second2.metric(
        "Amount Across Loaded Records",
        f"{total_amount:,.2f}",
        help="Sum of transaction amounts in the currently loaded records; not a balance or loss estimate.",
    )

    st.caption(
        f"Analytics are based on {len(df)} loaded records out of "
        f"{total_transactions:,} saved predictions. "
        "Decision cards and charts below describe this loaded sample."
    )

    st.divider()

    # --------------------------------------------------
    # Charts
    # --------------------------------------------------

    st.subheader("Risk Analytics")

    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("#### Risk Decision Distribution")

        decision_counts = (
            df["risk_decision"]
            .value_counts()
            .reindex(["ALLOW", "REVIEW", "BLOCK"], fill_value=0)
            .rename_axis("Decision")
            .reset_index(name="Transactions")
        )

        fig_decisions = px.bar(
            decision_counts,
            x="Decision",
            y="Transactions",
            color="Decision",
            color_discrete_map={
                "ALLOW": "#16A34A",
                "REVIEW": "#F59E0B",
                "BLOCK": "#DC2626",
            },
            text="Transactions",
        )

        fig_decisions.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Number of transactions",
            xaxis_title="Risk decision",
        )

        st.plotly_chart(
            fig_decisions,
            use_container_width=True,
        )

    with chart2:
        st.markdown("#### Transaction Types")

        type_counts = (
            df["type"]
            .value_counts()
            .rename_axis("Transaction Type")
            .reset_index(name="Transactions")
        )

        fig_types = px.pie(
            type_counts,
            names="Transaction Type",
            values="Transactions",
            hole=0.55,
        )

        fig_types.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title="Transaction type",
        )

        st.plotly_chart(
            fig_types,
            use_container_width=True,
        )

    # --------------------------------------------------
    # Recent transactions
    # --------------------------------------------------

    st.divider()
    st.subheader("Recent Transactions")

    recent = df.head(10).copy()

    recent["fraud_probability"] = recent[
        "fraud_probability"
    ].map(lambda value: f"{float(value) * 100:.4f}%")

    recent = recent.rename(
        columns={
            "id": "Transaction ID",
            "created_at": "Timestamp (UTC)",
            "type": "Transaction Type",
            "amount": "Amount",
            "fraud_probability": "Fraud Probability",
            "risk_decision": "Risk Decision",
            "anomaly_score": "Anomaly Score",
        }
    )

    columns_to_show = [
        "Transaction ID",
        "Timestamp (UTC)",
        "Transaction Type",
        "Amount",
        "Fraud Probability",
        "Risk Decision",
        "Anomaly Score",
    ]

    st.dataframe(
        recent[
            [column for column in columns_to_show if column in recent.columns]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Risk decisions are model outputs and should not be treated "
        "as definitive proof of fraud."
    )


# --------------------------------------------------
# Explicit navigation
# --------------------------------------------------

pages = [
    st.Page(
        executive_overview,
        title="Executive Overview",
        icon=":material/dashboard:",
        default=True,
    ),
    st.Page(
        str(STREAMLIT_DIR / "pages" / "2_Transaction_Analyzer.py"),
        title="Transaction Analyzer",
        icon=":material/search:",
    ),
    st.Page(
        str(STREAMLIT_DIR / "pages" / "3_Transaction_Explorer.py"),
        title="Transaction Explorer",
        icon=":material/table_view:",
    ),
    st.Page(
        str(STREAMLIT_DIR / "pages" / "4_Model_Monitoring.py"),
        title="Model Monitoring",
        icon=":material/monitor_heart:",
    ),
]

navigation = st.navigation(
    pages,
    position="sidebar",
)

with st.sidebar:
    st.divider()
    st.caption("FRAUD INTELLIGENCE")
    st.caption("Environment: Local Kubernetes")

navigation.run()
