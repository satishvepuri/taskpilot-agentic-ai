"""
External API call tool — simulates calling out to a third-party or
internal service (e.g. issuing a refund, updating a campaign budget).

This is mocked rather than hitting a real service, but the shape is real:
it's exactly these calls (side-effecting, hard to undo) that get flagged
with requires_approval=True in the planner, and routed through the
human-approval gate before execution.
"""
import random
import time

# Simulated action registry — in a real system these would be actual
# HTTP calls to internal/external services.
MOCK_ACTIONS = {
    "issue_refund": lambda params: {"refund_id": f"RF-{random.randint(10000,99999)}", "status": "processed"},
    "update_campaign_budget": lambda params: {"campaign_id": params.get("campaign_id"), "new_budget": params.get("amount"), "status": "updated"},
    "send_notification": lambda params: {"status": "sent", "channel": params.get("channel", "email")},
}


def call_api(tool_input: str) -> dict:
    """tool_input format: 'action_name(param=value, ...)' e.g. 'issue_refund(order_id=ORD-1029, amount=45.00)'"""
    import re
    match = re.match(r"^(\w+)\((.*)\)$", tool_input.strip())
    if not match:
        raise ValueError(f"call_api input must look like 'action_name(param=value, ...)', got: {tool_input!r}")

    action_name, args_str = match.groups()
    params = {}
    if args_str.strip():
        for pair in args_str.split(","):
            key, _, value = pair.strip().partition("=")
            params[key.strip()] = value.strip().strip("'\"")

    if action_name not in MOCK_ACTIONS:
        raise ValueError(f"Unknown action '{action_name}'. Available: {list(MOCK_ACTIONS.keys())}")

    time.sleep(0.2)  # simulate network latency
    result = MOCK_ACTIONS[action_name](params)
    return {"action": action_name, "params": params, "result": result}
