import os

import requests
from dotenv import load_dotenv


load_dotenv()


BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY")


def get_metrics():
    response = requests.get(
        f"{BASE_URL}/metrics"
    )

    assert response.status_code == 200

    return response.text


def get_metric_value(metrics_text, metric_name):
    for line in metrics_text.splitlines():
        if line.startswith(metric_name + " "):
            return float(line.split()[-1])

    return 0.0


def test_auth_failure_increments_metric():
    assert API_KEY is not None
    assert API_KEY != ""

    before_metrics = get_metrics()

    before_failures = get_metric_value(
        before_metrics,
        "fraud_auth_failures_total"
    )

    response = requests.post(
        f"{BASE_URL}/predict",
        headers={
            "X-API-Key": "definitely-wrong-api-key"
        },
        json={
            "step": 950,
            "type": "TRANSFER",
            "amount": 100000.0,
            "oldbalanceOrg": 100000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 50000.0,
            "newbalanceDest": 150000.0,
        }
    )

    assert response.status_code == 401

    after_metrics = get_metrics()

    after_failures = get_metric_value(
        after_metrics,
        "fraud_auth_failures_total"
    )

    assert after_failures == before_failures + 1