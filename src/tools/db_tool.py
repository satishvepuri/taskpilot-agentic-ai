"""
Database query tool.

Deliberately does NOT let the LLM write raw SQL against the live database —
that's an injection and correctness risk with a non-deterministic caller.
Instead, the LLM chooses from a small set of parameterized, pre-validated
query templates. This mirrors the "constrained text-to-SQL" pattern
(intent -> validated template -> parameters) rather than free-form SQL.
"""
import os
import re
from typing import Any

import psycopg2
import psycopg2.extras

DB_DSN = os.environ.get(
    "TASKPILOT_DB_DSN", "dbname=taskpilot user=postgres password=postgres host=localhost port=5432"
)

# Whitelisted query templates. Adding a new capability means adding a new
# template here — not opening up arbitrary SQL execution to the model.
QUERY_TEMPLATES = {
    "get_order_status": "SELECT order_id, status, updated_at FROM orders WHERE order_id = %(order_id)s",
    "get_customer_orders": "SELECT order_id, status, total_amount FROM orders WHERE customer_id = %(customer_id)s ORDER BY created_at DESC LIMIT 10",
    "get_customer_profile": "SELECT customer_id, name, email, signup_date FROM customers WHERE customer_id = %(customer_id)s",
    "get_campaign_metrics": "SELECT campaign_id, impressions, clicks, spend FROM campaigns WHERE campaign_id = %(campaign_id)s",
}

# Each template takes exactly one identifier param. Used as a fallback so a
# slightly-off LLM phrasing (e.g. "get_order_status: ORD-1029" instead of
# "get_order_status(order_id=ORD-1029)") still resolves correctly instead
# of hard-failing on a formatting technicality.
TEMPLATE_PARAM_NAME = {
    "get_order_status": "order_id",
    "get_customer_orders": "customer_id",
    "get_customer_profile": "customer_id",
    "get_campaign_metrics": "campaign_id",
}

# e.g. "get_order_status(order_id=ORD-1029)"
CALL_PATTERN = re.compile(r"^(\w+)\((.*)\)$")
# fallback e.g. "get_order_status: ORD-1029" or "get_order_status ORD-1029"
FALLBACK_PATTERN = re.compile(r"^(\w+)[\s:]+(.+)$")


def parse_tool_call(tool_input: str) -> tuple[str, dict]:
    cleaned = tool_input.strip()

    match = CALL_PATTERN.match(cleaned)
    if match:
        template_name, args_str = match.groups()
        params = {}
        if args_str.strip():
            for pair in args_str.split(","):
                key, _, value = pair.strip().partition("=")
                params[key.strip()] = value.strip().strip("'\"")
        return template_name, params

    # Fallback: model gave "template_name: value" or "template_name value"
    # instead of the exact "template_name(key=value)" syntax.
    fallback_match = FALLBACK_PATTERN.match(cleaned)
    if fallback_match:
        template_name, raw_value = fallback_match.groups()
        if template_name in TEMPLATE_PARAM_NAME:
            param_name = TEMPLATE_PARAM_NAME[template_name]
            value = raw_value.strip().strip("'\"")
            return template_name, {param_name: value}

    raise ValueError(
        f"db_query input must look like 'template_name(key=value, ...)', got: {tool_input!r}"
    )


def _json_safe(value):
    """Postgres returns rich Python types (datetime, Decimal, etc.) that
    json.dumps can't serialize by default. Converting to plain strings here
    keeps every downstream consumer (LLM prompt text, Redis storage, the
    eval harness) working with simple, predictable JSON."""
    if hasattr(value, "isoformat"):  # datetime, date, time
        return value.isoformat()
    return value


def db_query(tool_input: str) -> dict[str, Any]:
    """
    tool_input format: "template_name(param=value, ...)"
    e.g. "get_order_status(order_id=ORD-1029)"
    """
    template_name, params = parse_tool_call(tool_input)

    if template_name not in QUERY_TEMPLATES:
        raise ValueError(
            f"Unknown query template '{template_name}'. "
            f"Available: {list(QUERY_TEMPLATES.keys())}"
        )

    sql = QUERY_TEMPLATES[template_name]

    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        safe_rows = [{k: _json_safe(v) for k, v in dict(r).items()} for r in rows]
        return {"template": template_name, "params": params, "rows": safe_rows}
    finally:
        conn.close()
