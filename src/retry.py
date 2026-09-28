"""
Retry handling for tool execution.

Tool calls fail for reasons that often aren't the model's fault — a
transient DB connection blip, a rate-limited embedding API, a flaky mock
service. Retrying blindly forever is also wrong (infinite loops, wasted
cost), so this caps attempts and distinguishes retryable vs. terminal
failures.
"""
import time
from typing import Any, Callable

MAX_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 0.5

# Errors we consider transient and worth retrying. A malformed tool_input
# (e.g. bad SQL template name) is NOT retryable — retrying won't fix a
# planning error, it'll just waste attempts. That distinction matters:
# blind retry-on-any-exception hides real bugs behind eventual timeouts.
RETRYABLE_ERROR_SUBSTRINGS = (
    "connection",
    "timeout",
    "rate limit",
    "temporarily unavailable",
    "network",
)


def is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in RETRYABLE_ERROR_SUBSTRINGS)


def run_with_retry(fn: Callable[[str], Any], tool_input: str) -> tuple[bool, Any, str | None, int]:
    """
    Returns (success, output, error_message, attempts_made).
    Non-retryable errors fail fast on attempt 1 rather than burning the
    full retry budget.
    """
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = fn(tool_input)
            return True, result, None, attempt
        except Exception as e:
            last_error = str(e)
            if not is_retryable(e) or attempt == MAX_ATTEMPTS:
                return False, None, last_error, attempt
            backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            time.sleep(backoff)

    return False, None, last_error, MAX_ATTEMPTS
