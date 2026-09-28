"""Generate a deterministic 3,000-conversation TaskPilot benchmark.

This creates labeled synthetic requests. It does NOT claim they have been
run. The resume statement "evaluated across 3K+" becomes true only after
`python -m eval.run_eval --file eval/test_conversations_3000.jsonl` finishes
and writes results for all 3,000 cases.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "test_conversations_3000.jsonl"


def add(cases, category, requests, expected_tools, expected_contains, expects_approval, count):
    start = len(cases) + 1
    for i in range(count):
        request = requests[i % len(requests)]
        tokens = expected_contains[i % len(expected_contains)] if expected_contains else []
        cases.append({
            "id": f"bench-{start+i:04d}",
            "request": request,
            "expected_tools": expected_tools,
            "expects_approval": expects_approval,
            "category": category,
            "expected_response_contains": tokens,
        })


def build_cases():
    cases = []
    add(cases, "order_status", [
        "What's the status of order ORD-1029?",
        "Check order ORD-1030 and tell me its current status.",
        "Look up the status for ORD-1031.",
    ], ["db_query"], [["shipped"], ["processing"], ["delivered"]], False, 500)

    add(cases, "refund_policy", [
        "What is our refund policy?",
        "How long do refunds take according to the policy?",
        "Search the policy docs and tell me when refunds are issued.",
    ], ["retrieve_docs"], [["5-7"], ["5-7"], ["5-7"]], False, 400)

    add(cases, "shipping_policy", [
        "What does our shipping policy say about standard shipping?",
        "How long does express shipping take?",
        "Search the docs for shipping delivery times.",
    ], ["retrieve_docs"], [["3-5"], ["1-2"], ["shipping"]], False, 300)

    add(cases, "arithmetic", [
        "Calculate 240 * (1 - 0.15).",
        "What is 59.99 - 50?",
        "Calculate 15400 / 1200000 * 100.",
        "What is 45 + 8.50?",
    ], ["calculate"], [["204"], ["9.99"], ["1.28"], ["53.5"]], False, 500)

    add(cases, "customer_lookup", [
        "Show me the orders for customer CUST-001.",
        "Look up customer CUST-002's recent orders.",
        "Get the customer profile for CUST-001.",
    ], ["db_query"], [["ORD-1029"], ["ORD-1031"], ["Jane"]], False, 300)

    add(cases, "campaign_lookup", [
        "Get the metrics for campaign CAMP-500.",
        "Look up campaign CAMP-501 metrics.",
    ], ["db_query"], [["15400"], ["9600"]], False, 250)

    add(cases, "refund_action", [
        "Issue a $45 refund for order ORD-1029.",
        "Refund $20 for order ORD-1030.",
        "Process a $9.99 refund for ORD-1031.",
    ], ["call_api"], [["processed"], ["processed"], ["processed"]], True, 300)

    add(cases, "notification_action", [
        "Send an email notification to customer CUST-001 saying the order shipped.",
        "Send a notification to CUST-002 that their order was delivered.",
    ], ["call_api"], [["sent"], ["sent"]], True, 200)

    add(cases, "budget_action", [
        "Update campaign CAMP-500 budget to $5000.",
        "Change CAMP-501 campaign budget to $3000.",
    ], ["call_api"], [["updated"], ["updated"]], True, 150)

    add(cases, "lookup_plus_math", [
        "Look up CAMP-500 metrics and calculate click-through rate as clicks divided by impressions times 100.",
        "Get CAMP-501 metrics and calculate its CTR percentage.",
    ], ["db_query", "calculate"], [["1.28"], ["1.2"]], False, 100)

    if len(cases) != 3000:
        raise RuntimeError(f"Expected 3000 cases, got {len(cases)}")
    return cases


if __name__ == "__main__":
    cases = build_cases()
    with open(OUT, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print(f"Wrote {len(cases)} labeled conversations to {OUT}")
