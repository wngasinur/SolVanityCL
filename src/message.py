from redis.backoff import ExponentialBackoff
from redis.retry import Retry
import redis
import os

retry = Retry(ExponentialBackoff(), 3)

REDIS_HOST = os.environ.get("REDIS_HOST", "")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
print(f"REDIS {REDIS_HOST} {REDIS_PASSWORD}")

r = redis.Redis(
    host=REDIS_HOST, port=6379,
    ssl=True,
    username="default", 
    password=REDIS_PASSWORD,
    retry=retry, retry_on_timeout=True)
