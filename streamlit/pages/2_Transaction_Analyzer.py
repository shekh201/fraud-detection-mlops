
import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# Shared API client
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLIENT_PATH = PROJECT_ROOT / "streamlit" / "api_client.py"

spec = importlib.util.spec_from_file_location(
    "fraud_api_client",
    CLIENT_PATH,
)

api_client = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api_client
spec.loader.exec_module(api_client)

predict_transaction = api_client.predict_transaction
FraudAPIError = api_client.FraudAPIError


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Transaction Analyzer | Fraud Intelligence",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Transaction Analyzer")
st.caption(
    "Analyze transaction risk using the deployed fraud detection model."
)


# --------------------------------------------------
# Transaction input
# --------------------------------------------------

with st.form("transaction_form"):

    st.subheader("Transaction details")

    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Transaction step",
            min_value=1,
            value=1,
        )

        transaction_type = st.selectbox(
            "Transaction type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
        )

        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=10000.0,
            step=100.0,
        )

        oldbalance_org = st.number_input(
            "Sender balance before",
            min_value=0.0,
            value=20000.0,
            step=100.0,
        )

    with col2:
        newbalance_orig = st.number_input(
            "Sender balance after",
            min_value=0.0,
            value=10000.0,
            step=100.0,
        )

        oldbalance_dest = st.number_input(
            "Receiver balance before",
            min_value=0.0,
            value=5000.0,
            step=100.0,
        )

        newbalance_dest = st.number_input(
            "Receiver balance after",
            min_value=0.0,
            value=15000.0,
            step=100.0,
        )

    submitted = st.form_submit_button(
        "Analyze transaction",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    payload = {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
    }

    # These checks are diagnostic only. They do not change the model output.
    sender_change = oldbalance_org - newbalance_orig
    receiver_change = newbalance_dest - oldbalance_dest
    sender_balance_error = oldbalance_org - amount - newbalance_orig

    try:
        with st.spinner("Analyzing transaction..."):
            result = predict_transaction(payload)

        probability = float(result["fraud_probability"])
        decision = result["risk_decision"]
        anomaly_score = result.get("anomaly_score")

        if not 0 <= probability <= 1:
            st.error("The API returned a probability outside the valid 0–1 range.")
            st.stop()

        analyzed_at = datetime.now(timezone.utc).isoformat()

        # Store report data across Streamlit reruns.
        report = {
            "analyzed_at_utc": analyzed_at,
            "transaction": payload,
            "prediction": result,
            "diagnostics": {
                "sender_balance_change": sender_change,
                "receiver_balance_change": receiver_change,
                "sender_balance_error": sender_balance_error,
            },
        }

        st.session_state["latest_fraud_report"] = report

    except FraudAPIError as exc:
        st.error(str(exc))
        st.stop()

    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Unexpected API response: {exc}")
        st.stop()


# --------------------------------------------------
# Display latest result
# --------------------------------------------------

report = st.session_state.get("latest_fraud_report")

if report:

    payload = report["transaction"]
    result = report["prediction"]
    diagnostics = report["diagnostics"]

    probability = float(result["fraud_probability"])
    decision = result["risk_decision"]
    anomaly_score = result.get("anomaly_score")

    st.divider()
    st.subheader("Prediction results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Fraud probability",
        f"{probability:.4%}",
    )

    c2.metric(
        "Risk decision",
        decision,
    )

    c3.metric(
        "Anomaly score",
        f"{float(anomaly_score):.4f}"
        if anomaly_score is not None
        else "N/A",
    )

    # --------------------------------------------------
    # Probability gauge
    # --------------------------------------------------

    st.subheader("Risk score")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".4f"},
            title={"text": "Model-estimated fraud probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#38BDF8"},
                "steps": [
                    {"range": [0, 1], "color": "#DCFCE7"},
                    {"range": [1, 99], "color": "#FEF3C7"},
                    {"range": [99, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#DC2626", "width": 3},
                    "thickness": 0.75,
                    "value": 99,
                },
            },
        )
    )

    gauge.update_layout(
        height=300,
        margin=dict(l=25, r=25, t=55, b=10),
    )

    st.plotly_chart(gauge, use_container_width=True)

    # --------------------------------------------------
    # Decision explanation
    # --------------------------------------------------

    st.subheader("Risk assessment")

    if decision == "BLOCK":
        st.error(
            "Risk engine decision: BLOCK. "
            "The configured risk policy classified this transaction as high risk."
        )

    elif decision == "REVIEW":
        st.warning(
            "Risk engine decision: REVIEW. "
            "This transaction requires additional review under the configured policy."
        )

    elif decision == "ALLOW":
        st.success(
            "Risk engine decision: ALLOW. "
            "The configured risk policy allowed this transaction."
        )

    st.caption(
        "The probability is a model estimate, not proof of fraud. "
        "The risk decision follows the configured risk-engine policy."
    )

    # --------------------------------------------------
    # Balance diagnostics
    # --------------------------------------------------

    st.divider()
    st.subheader("Balance validation")

    sender_change = diagnostics["sender_balance_change"]
    receiver_change = diagnostics["receiver_balance_change"]
    balance_error = diagnostics["sender_balance_error"]

    balance_col1, balance_col2, balance_col3 = st.columns(3)

    balance_col1.metric(
        "Sender balance change",
        f"{sender_change:,.2f}",
    )

    balance_col2.metric(
        "Receiver balance change",
        f"{receiver_change:,.2f}",
    )

    balance_col3.metric(
        "Sender balance error",
        f"{balance_error:,.2f}",
    )

    tolerance = 0.01

    if abs(balance_error) <= tolerance:
        st.success(
            "Sender balance is consistent with the supplied amount."
        )
    else:
        st.warning(
            "Sender balance does not exactly match "
            "old balance − transaction amount. Verify the input."
        )

    if receiver_change < -tolerance:
        st.info(
            "Receiver balance decreased. Verify the transaction details "
            "and the expected behavior for this transaction type."
        )

    # --------------------------------------------------
    # Downloadable report
    # --------------------------------------------------

    st.divider()
    st.subheader("Export analysis")

    report_json = json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )

    csv_row = {
        "analyzed_at_utc": report["analyzed_at_utc"],
        **payload,
        **result,
        **diagnostics,
    }

    report_csv = pd.DataFrame([csv_row]).to_csv(index=False)

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            "Download JSON report",
            data=report_json,
            file_name="fraud_transaction_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with download_col2:
        st.download_button(
            "Download CSV report",
            data=report_csv,
            file_name="fraud_transaction_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("View complete API response"):
        st.json(result)

    st.caption(
        f"Analysis timestamp (UTC): {report['analyzed_at_utc']}"
    )

else:
    st.info(
        "Enter transaction details above and select "
        "'Analyze transaction' to view the risk assessment."
    )
