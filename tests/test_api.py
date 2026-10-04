import os

import requests
from dotenv import load_dotenv


load_dotenv()


BASE_URL = "http://localhost:8000"

API_KEY = os.getenv("API_KEY")


def test_health_endpoint():
    response = requests.get(
        f"{BASE_URL}/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "fraud_detection_model"


def test_predict_endpoint():
    assert API_KEY is not None
    assert API_KEY != ""
    assert len(API_KEY) > 10

    transaction = {
        "step": 501,
        "type": "TRANSFER",
        "amount": 123456.0,
        "oldbalanceOrg": 123456.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 200000.0,
        "newbalanceDest": 323456.0,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        headers={
            "X-API-Key": API_KEY
        },
        json=transaction
    )

    print("Status:", response.status_code)
    print("Response:", response.text)

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "risk_decision" in data
    assert "anomaly_score" in data

    assert 0.0 <= data["fraud_probability"] <= 1.0

    assert data["risk_decision"] in {
        "ALLOW",
        "REVIEW",
        "BLOCK"
    }
    
def test_predict_without_api_key():
    transaction = {
        "step": 501,
        "type": "TRANSFER",
        "amount": 123456.0,
        "oldbalanceOrg": 123456.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 200000.0,
        "newbalanceDest": 323456.0,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=transaction
    )

    assert response.status_code == 401

    data = response.json()

    assert data["detail"] == "API key is missing"


def test_predict_with_wrong_api_key():
    transaction = {
        "step": 501,
        "type": "TRANSFER",
        "amount": 123456.0,
        "oldbalanceOrg": 123456.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 200000.0,
        "newbalanceDest": 323456.0,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        headers={
            "X-API-Key": "wrong-api-key"
        },
        json=transaction
    )

    assert response.status_code == 401

    data = response.json()

    assert data["detail"] == "Invalid API key"