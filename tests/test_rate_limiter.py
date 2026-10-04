import os
import uuid
from unittest.mock import MagicMock

import redis
import requests
from dotenv import load_dotenv

import src.rate_limiter as rate_limiter




load_dotenv()


BASE_URL = "http://localhost:8000"

API_KEY = os.getenv("API_KEY")


def test_rate_limit_allows_requests_within_limit():

    mock_redis = MagicMock()

    mock_pipeline = MagicMock()

    mock_pipeline.execute.return_value = [
        1,
        True
    ]

    mock_redis.pipeline.return_value.__enter__.return_value = (
        mock_pipeline
    )

    original_redis_client = rate_limiter.redis_client

    try:

        rate_limiter.redis_client = mock_redis

        result = rate_limiter.is_rate_limit_exceeded(
            "test-client-allow"
        )

        assert result is False

        mock_redis.pipeline.assert_called_once_with(
            transaction=True
        )

        mock_pipeline.incr.assert_called_once_with(
            "rate_limit:test-client-allow"
        )

        mock_pipeline.expire.assert_called_once_with(
            "rate_limit:test-client-allow",
            rate_limiter.WINDOW_SECONDS
        )

        mock_pipeline.execute.assert_called_once()

    finally:

        rate_limiter.redis_client = original_redis_client


def test_rate_limit_blocks_requests_over_limit():

    mock_redis = MagicMock()

    mock_pipeline = MagicMock()

    mock_pipeline.execute.return_value = [
        rate_limiter.RATE_LIMIT + 1,
        True
    ]

    mock_redis.pipeline.return_value.__enter__.return_value = (
        mock_pipeline
    )

    original_redis_client = rate_limiter.redis_client

    try:

        rate_limiter.redis_client = mock_redis

        result = rate_limiter.is_rate_limit_exceeded(
            "test-client-block"
        )

        assert result is True

        mock_redis.pipeline.assert_called_once_with(
            transaction=True
        )

        mock_pipeline.incr.assert_called_once_with(
            "rate_limit:test-client-block"
        )

        mock_pipeline.expire.assert_called_once_with(
            "rate_limit:test-client-block",
            rate_limiter.WINDOW_SECONDS
        )

        mock_pipeline.execute.assert_called_once()

    finally:

        rate_limiter.redis_client = original_redis_client


def test_api_rate_limit():

    assert API_KEY is not None
    assert API_KEY != ""

    redis_client = redis.Redis(
        host="localhost",
        port=6379,
        decode_responses=True
    )

    client_ip = "172.18.0.1"

    redis_key = f"rate_limit:{client_ip}"

    redis_client.delete(redis_key)

    transaction = {
        "step": 900,
        "type": "TRANSFER",
        "amount": 100000.0,
        "oldbalanceOrg": 100000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 50000.0,
        "newbalanceDest": 150000.0
    }

    headers = {
        "X-API-Key": API_KEY
    }

    try:

        for _ in range(rate_limiter.RATE_LIMIT):

            response = requests.post(
                f"{BASE_URL}/predict",
                headers=headers,
                json=transaction
            )

            assert response.status_code == 200

        response = requests.post(
            f"{BASE_URL}/predict",
            headers=headers,
            json=transaction
        )

        assert response.status_code == 429

    finally:

        redis_client.delete(redis_key)