import redis
import json
import os 

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
print(f"REDIS {REDIS_HOST} {REDIS_PASSWORD}")

r = redis.Redis(
    host=REDIS_HOST, port=6379,
    ssl=True,
    username="default", # use your Redis user. More info https://redis.io/docs/latest/operate/oss_and_stack/management/security/acl/
    password=REDIS_PASSWORD)
