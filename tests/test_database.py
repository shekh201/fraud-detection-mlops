import os
import uuid

import requests
from dotenv import load_dotenv

from src.database import (
    get_database_connection,
    get_prediction_count
)


load_dotenv()


BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY")


def test_database_connection():
    connection = get_database_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

        assert result[0] == 1

    finally:
        connection.close()


def test_prediction_count_returns_integer():
    count = get_prediction_count()

    assert isinstance(count, int)
    assert count >= 0


def test_api_prediction_is_saved_to_database():
    assert API_KEY is not None
    assert API_KEY != ""

    before_count = get_prediction_count()

    unique_value = uuid.uuid4().int % 100000

    transaction = {
        "step": 700,
        "type": "TRANSFER",
        "amount": 987654.0 + unique_value,
        "oldbalanceOrg": 987654.0 + unique_value,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 100000.0,
        "newbalanceDest": 1087654.0 + unique_value,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        headers={
            "X-API-Key": API_KEY
        },
        json=transaction
    )

    assert response.status_code == 200

    prediction = response.json()

    assert "fraud_probability" in prediction
    assert "risk_decision" in prediction
    assert "anomaly_score" in prediction

    after_count = get_prediction_count()

    assert after_count == before_count + 1


def test_cached_prediction_does_not_create_duplicate_database_record():
    assert API_KEY is not None
    assert API_KEY != ""

    unique_value = uuid.uuid4().int % 100000

    transaction = {
        "step": 701,
        "type": "TRANSFER",
        "amount": 876543.0 + unique_value,
        "oldbalanceOrg": 876543.0 + unique_value,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 200000.0,
        "newbalanceDest": 1076543.0 + unique_value,
    }

    before_count = get_prediction_count()

    first_response = requests.post(
        f"{BASE_URL}/predict",
        headers={
            "X-API-Key": API_KEY
        },
        json=transaction
    )

    assert first_response.status_code == 200

    first_prediction = first_response.json()

    after_first_count = get_prediction_count()

    assert after_first_count == before_count + 1

    second_response = requests.post(
        f"{BASE_URL}/predict",
        headers={
            "X-API-Key": API_KEY
        },
        json=transaction
    )

    assert second_response.status_code == 200

    second_prediction = second_response.json()

    after_second_count = get_prediction_count()

    assert after_second_count == after_first_count

    assert second_prediction == first_prediction