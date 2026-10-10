
import os
from typing import Any

import requests
import streamlit as st
import urllib3


# --------------------------------------------------
# API configuration
# --------------------------------------------------

API_URL = os.getenv(
    "FRAUD_API_URL",
    "https://localhost:8443"
).rstrip("/")

API_KEY = os.getenv(
    "FRAUD_API_KEY",
    "fraud-dev-secret-2026"
)

# Local development only: ingress uses a self-signed TLS certificate.
# Production must use a trusted certificate with verification enabled.
VERIFY_TLS = os.getenv(
    "FRAUD_API_VERIFY_TLS",
    "false"
).lower() in ("1", "true", "yes")

if not VERIFY_TLS:
    urllib3.disable_warnings(
        urllib3.exceptions.InsecureRequestWarning
    )


# --------------------------------------------------
# Shared HTTP client
# --------------------------------------------------

class FraudAPIError(Exception):
    """Raised when the Fraud Detection API request fails."""


def api_request(
    method: str,
    endpoint: str,
    *,
    params: dict | None = None,
    payload: dict | None = None,
) -> Any:
    """Send an authenticated request to the Fraud Detection API."""

    headers = {
        "X-API-Key": API_KEY,
        "Host": "fraud.localhost",
    }

    try:
        response = requests.request(
            method=method,
            url=f"{API_URL}/{endpoint.lstrip('/')}",
            headers=headers,
            params=params,
            json=payload,
            timeout=20,
            verify=VERIFY_TLS,
        )

        if not response.ok:
            raise FraudAPIError(
                f"API returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        return response.json()

    except requests.exceptions.SSLError as exc:
        raise FraudAPIError(
            "TLS certificate verification failed. "
            "Check FRAUD_API_VERIFY_TLS and the API certificate."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise FraudAPIError(
            "Cannot connect to the API. Check that the "
            "Kubernetes HTTPS port-forward is running on port 8443."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise FraudAPIError(
            "The API request timed out. Please try again."
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise FraudAPIError(
            f"API request failed: {exc}"
        ) from exc


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

def get_health() -> dict:
    """Return API and model health."""
    return api_request("GET", "/health")


def predict_transaction(payload: dict) -> dict:
    """Get a fraud prediction for one transaction."""
    return api_request(
        "POST",
        "/predict",
        payload=payload,
    )


def get_transactions(
    limit: int = 20,
    offset: int = 0,
    risk_decision: str | None = None,
    transaction_type: str | None = None,
) -> dict:
    """Fetch paginated transaction history with optional filters."""

    params = {
        "limit": limit,
        "offset": offset,
    }

    if risk_decision:
        params["risk_decision"] = risk_decision

    if transaction_type:
        params["transaction_type"] = transaction_type

    return api_request(
        "GET",
        "/transactions",
        params=params,
    )


def get_prometheus_metrics() -> str:
    """Fetch raw Prometheus metrics for monitoring."""

    headers = {
        "Host": "fraud.localhost",
    }

    try:
        response = requests.get(
            f"{API_URL}/metrics",
            headers=headers,
            timeout=20,
            verify=VERIFY_TLS,
        )

        response.raise_for_status()
        return response.text

    except requests.exceptions.RequestException as exc:
        raise FraudAPIError(
            f"Could not retrieve Prometheus metrics: {exc}"
        ) from exc
