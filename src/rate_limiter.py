from src.cache import redis_client


RATE_LIMIT = 20
WINDOW_SECONDS = 60


def is_rate_limit_exceeded(client_ip):

    key = f"rate_limit:{client_ip}"

    with redis_client.pipeline(transaction=True) as pipe:

        pipe.incr(key)

        pipe.expire(
            key,
            WINDOW_SECONDS
        )

        current_count, _ = pipe.execute()

    return current_count > RATE_LIMIT