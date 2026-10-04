import os

import requests
from dotenv import load_dotenv


load_dotenv()


BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY")


def test_get_transactions():
    assert API_KEY is not None
    assert API_KEY != ""

    response = requests.get(
        f"{BASE_URL}/transactions",
        headers={
            "X-API-Key": API_KEY
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "items" in data
    assert "total" in data
    assert "limit" in data
    assert "offset" in data

    assert isinstance(data["items"], list)
    assert isinstance(data["total"], int)
    assert isinstance(data["limit"], int)
    assert isinstance(data["offset"], int)

    assert data["total"] >= 0
    assert data["limit"] > 0
    assert data["offset"] >= 0
    
def test_get_transactions_pagination():
    assert API_KEY is not None
    assert API_KEY != ""

    first_response = requests.get(
        f"{BASE_URL}/transactions",
        headers={
            "X-API-Key": API_KEY
        },
        params={
            "limit": 5,
            "offset": 0
        }
    )

    assert first_response.status_code == 200

    first_data = first_response.json()

    assert first_data["limit"] == 5
    assert first_data["offset"] == 0
    assert len(first_data["items"]) <= 5

    second_response = requests.get(
        f"{BASE_URL}/transactions",
        headers={
            "X-API-Key": API_KEY
        },
        params={
            "limit": 5,
            "offset": 5
        }
    )

    assert second_response.status_code == 200

    second_data = second_response.json()

    assert second_data["limit"] == 5
    assert second_data["offset"] == 5
    assert len(second_data["items"]) <= 5

    if first_data["items"] and second_data["items"]:
        assert first_data["items"][0]["id"] != second_data["items"][0]["id"]
        
def test_get_transactions_by_risk_decision():
    assert API_KEY is not None
    assert API_KEY != ""

    response = requests.get(
        f"{BASE_URL}/transactions",
        headers={
            "X-API-Key": API_KEY
        },
        params={
            "risk_decision": "BLOCK"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "items" in data
    assert "total" in data

    for item in data["items"]:
        assert item["risk_decision"] == "BLOCK"
        
        
def test_get_transactions_by_transaction_type():
    assert API_KEY is not None
    assert API_KEY != ""

    response = requests.get(
        f"{BASE_URL}/transactions",
        headers={
            "X-API-Key": API_KEY
        },
        params={
            "transaction_type": "TRANSFER"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "items" in data
    assert "total" in data

    for item in data["items"]:
        assert item["type"] == "TRANSFER"