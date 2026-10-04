import json
import os

import redis


redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True
)


def set_prediction_cache(cache_key, prediction, ttl=300):
    redis_client.setex(
        cache_key,
        ttl,
        json.dumps(prediction)
    )


def get_prediction_cache(cache_key):
    cached_prediction = redis_client.get(cache_key)

    if cached_prediction is None:
        return None

    return json.loads(cached_prediction)