"""
Short-term memory: keeps recent conversation turns for a given
conversation_id so the agent has context across multiple messages in the
same session (e.g. "actually make that refund $50 instead").

Deliberately scoped as SHORT-term (bounded window, TTL) rather than a full
long-term memory/vector-store system — that's a distinct, larger feature
this project doesn't claim to solve.
"""
import json
import os
from datetime import timedelta

import redis

REDIS_HOST = os.environ.get("TASKPILOT_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("TASKPILOT_REDIS_PORT", 6379))
MAX_TURNS = 10
TTL = timedelta(hours=2)

_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def _key(conversation_id: str) -> str:
    return f"taskpilot:memory:{conversation_id}"


def get_history(conversation_id: str) -> list[dict]:
    raw = _client.get(_key(conversation_id))
    return json.loads(raw) if raw else []


def append_turn(conversation_id: str, role: str, content: str) -> None:
    history = get_history(conversation_id)
    history.append({"role": role, "content": content})
    history = history[-MAX_TURNS:]  # bounded window — prevents unbounded context growth
    _client.set(_key(conversation_id), json.dumps(history), ex=TTL)


def clear_history(conversation_id: str) -> None:
    _client.delete(_key(conversation_id))
