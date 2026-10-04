import os
import uuid

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


def test_cache_hit_and_miss_metrics():
    assert API_KEY is not None
    assert API_KEY != ""

    unique_value = uuid.uuid4().int % 100000

    transaction = {
        "step": 800,
        "type": "TRANSFER",
        "amount": 500000.0 + unique_value,
        "oldbalanceOrg": 500000.0 + unique_value,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 100000.0,
        "newbalanceDest": 600000.0 + unique_value,
    }

    before_metrics = get_metrics()

    before_misses = get_metric_value(
        before_metrics,
        "fraud_cache_misses_total"
    )

    before_hits = get_metric_value(
        before_metrics,
        "fraud_cache_hits_total"
    )

    first_response = requests.post(
        f"{BASE_URL}/predict",
        headers={
            "X-API-Key": API_KEY
        },
        json=transaction
    )

    assert first_response.status_code == 200

    after_first_metrics = get_metrics()

    after_first_misses = get_metric_value(
        after_first_metrics,
        "fraud_cache_misses_total"
    )

    assert after_first_misses == before_misses + 1

    second_response = requests.post(
        f"{BASE_URL}/predict",
        headers={
            "X-API-Key": API_KEY
        },
        json=transaction
    )

    assert second_response.status_code == 200

    after_second_metrics = get_metrics()

    after_second_hits = get_metric_value(
        after_second_metrics,
        "fraud_cache_hits_total"
    )

    assert after_second_hits == before_hits + 1