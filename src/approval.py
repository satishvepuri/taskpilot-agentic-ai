"""
Human-in-the-loop approval gate.

Any subtask flagged `requires_approval=True` by the planner (real-world
side effects: refunds, budget changes, notifications) pauses graph
execution and writes a pending-approval record instead of running
immediately. A human (or an approval UI/Slack bot in a real deployment)
calls `resolve_approval()` to unblock it.

This is intentionally decoupled from any particular UI — in production
this would be backed by a queue + webhook, not polling, but the interface
(`request_approval` / `resolve_approval` / `poll_approval`) stays the same.
"""
import json
import os
from datetime import timedelta

import redis

REDIS_HOST = os.environ.get("TASKPILOT_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("TASKPILOT_REDIS_PORT", 6379))
APPROVAL_TTL = timedelta(hours=1)

_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def _key(conversation_id: str, step_id: int) -> str:
    return f"taskpilot:approval:{conversation_id}:{step_id}"


def request_approval(conversation_id: str, subtask: dict) -> None:
    record = {"subtask": subtask, "decision": None}
    _client.set(_key(conversation_id, subtask["step_id"]), json.dumps(record, default=str), ex=APPROVAL_TTL)


def resolve_approval(conversation_id: str, step_id: int, approved: bool, reason: str | None = None) -> None:
    key = _key(conversation_id, step_id)
    raw = _client.get(key)
    if raw is None:
        raise ValueError(f"No pending approval found for conversation={conversation_id} step={step_id}")
    record = json.loads(raw)
    if not approved and not reason:
        reason = "No reason provided"
    record["decision"] = {"approved": approved, "reason": reason}
    _client.set(key, json.dumps(record, default=str), ex=APPROVAL_TTL)


def poll_approval(conversation_id: str, step_id: int) -> dict | None:
    """Returns the decision dict once resolved, or None if still pending."""
    raw = _client.get(_key(conversation_id, step_id))
    if raw is None:
        return None
    record = json.loads(raw)
    return record["decision"]


# --- Paused workflow state ---
# When a run pauses for approval, the CLI process exits (Step 10's
# `python main.py ...` command finishes running). To resume correctly
# later — in a brand new process — we need to remember exactly where the
# workflow was: which plan, which steps already ran, which step is
# waiting. This stores that full snapshot in Redis, keyed by conversation.

def _paused_state_key(conversation_id: str) -> str:
    return f"taskpilot:paused_state:{conversation_id}"


def save_paused_state(conversation_id: str, state: dict) -> None:
    # default=str is a safety net: if any value in state isn't natively
    # JSON-safe (e.g. a datetime that slipped through from elsewhere),
    # fall back to converting it to a string rather than crashing the
    # whole paused workflow.
    _client.set(_paused_state_key(conversation_id), json.dumps(state, default=str), ex=APPROVAL_TTL)


def load_paused_state(conversation_id: str) -> dict | None:
    raw = _client.get(_paused_state_key(conversation_id))
    return json.loads(raw) if raw else None


def clear_paused_state(conversation_id: str) -> None:
    _client.delete(_paused_state_key(conversation_id))
