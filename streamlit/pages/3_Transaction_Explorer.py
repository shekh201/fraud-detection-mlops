
import importlib.util
import sys
from pathlib import Path

import pandas as pd
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
# Page
# --------------------------------------------------

st.title("📋 Transaction Explorer")
st.caption("Search, filter and export saved prediction history.")

decision_options = ["All", "ALLOW", "REVIEW", "BLOCK"]
type_options = [
    "All",
    "TRANSFER",
    "CASH_OUT",
    "PAYMENT",
    "CASH_IN",
    "DEBIT",
]

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    decision_filter = st.selectbox(
        "Risk decision",
        decision_options,
    )

with filter_col2:
    type_filter = st.selectbox(
        "Transaction type",
        type_options,
    )

with filter_col3:
    page_size = st.selectbox(
        "Rows per page",
        [10, 20, 50, 100],
        index=1,
    )

page_number = st.number_input(
    "Page",
    min_value=1,
    value=1,
    step=1,
)

offset = (page_number - 1) * page_size

try:
    with st.spinner("Loading transaction history..."):
        result = api_client.get_transactions(
            limit=page_size,
            offset=offset,
            risk_decision=(
                None if decision_filter == "All"
                else decision_filter
            ),
            transaction_type=(
                None if type_filter == "All"
                else type_filter
            ),
        )

    items = result.get("items", [])
    total = result.get("total", 0)

    st.divider()

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric("Matching transactions", total)
    metric_col2.metric(
        "Current page",
        f"{page_number} / {max(1, (total + page_size - 1) // page_size)}",
    )

    if items:
        df = pd.DataFrame(items)

        # Search only within the currently loaded page.
        search_id = st.text_input(
            "Search transaction ID on this page",
            placeholder="Enter transaction ID",
        )

        if search_id.strip():
            df = df[
                df["id"].astype(str).str.contains(
                    search_id.strip(),
                    case=False,
                    na=False,
                )
            ]

        preferred_columns = [
            "id",
            "created_at",
            "type",
            "amount",
            "fraud_probability",
            "risk_decision",
            "anomaly_score",
        ]

        display_columns = [
            column for column in preferred_columns
            if column in df.columns
        ]

        display_df = df[display_columns].copy()

        if "fraud_probability" in display_df.columns:
            display_df["fraud_probability"] = (
                display_df["fraud_probability"] * 100
            ).map(lambda value: f"{value:.4f}%")

        display_df = display_df.rename(
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

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="fraud_transaction_history.csv",
            mime="text/csv",
            type="primary",
        )

        st.caption(
            "Fraud probability is displayed as a percentage in the table."
        )

    else:
        st.info(
            "No transactions found for these filters. "
            "Try another filter or submit a prediction first."
        )

except api_client.FraudAPIError as exc:
    st.error(f"Could not load transaction history: {exc}")
