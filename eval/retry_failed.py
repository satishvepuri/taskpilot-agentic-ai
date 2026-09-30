"""Retry only failed TaskPilot benchmark cases and replace them in the saved 3K results.

Usage:
    python -m eval.retry_failed --limit 5
    python -m eval.retry_failed

The script:
- reads eval/test_conversations_3000.jsonl
- reads eval/test_conversations_3000_results.jsonl
- reruns only rows that still contain an error or completed=false
- replaces those rows in-place
- saves progress every 10 retries, so it can safely be run again after interruption
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from eval.run_eval import evaluate_case, print_summary

HERE = Path(__file__).parent
CASES_FILE = HERE / "test_conversations_3000.jsonl"
RESULTS_FILE = HERE / "test_conversations_3000_results.jsonl"
BACKUP_FILE = HERE / "test_conversations_3000_results_before_retry.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def atomic_write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Retry only the first N currently failed cases")
    args = parser.parse_args()

    cases = load_jsonl(CASES_FILE)
    results = load_jsonl(RESULTS_FILE)

    if len(cases) != 3000:
        raise RuntimeError(f"Expected 3000 benchmark cases, found {len(cases)}")
    if len(results) != 3000:
        raise RuntimeError(f"Expected 3000 saved results, found {len(results)}")

    if not BACKUP_FILE.exists():
        shutil.copy2(RESULTS_FILE, BACKUP_FILE)
        print(f"Backup created: {BACKUP_FILE}")

    case_by_id = {row["id"]: row for row in cases}
    result_by_id = {row["id"]: row for row in results}

    failed_ids = [
        row["id"]
        for row in cases
        if (
            row["id"] not in result_by_id
            or bool(result_by_id[row["id"]].get("error"))
            or not bool(result_by_id[row["id"]].get("completed"))
        )
    ]

    print(f"Failed cases currently needing retry: {len(failed_ids)}")

    if args.limit is not None:
        failed_ids = failed_ids[: args.limit]
        print(f"Retrying only {len(failed_ids)} case(s) in this test run.")

    if not failed_ids:
        ordered = [result_by_id[row["id"]] for row in cases]
        print_summary(ordered)
        print("\nNo failed cases remain.")
        return

    for i, case_id in enumerate(failed_ids, start=1):
        new_result = evaluate_case(case_by_id[case_id])
        result_by_id[case_id] = new_result

        # Save every 10 cases (and at the end), so an interruption does not
        # throw away most of the retry work.
        if i == 1 or i % 10 == 0 or i == len(failed_ids):
            ordered = [result_by_id[row["id"]] for row in cases]
            atomic_write(RESULTS_FILE, ordered)
            status = "PASS" if new_result.get("completed") and not new_result.get("error") else "ERROR"
            print(f"Retried {i}/{len(failed_ids)} this run — latest: {case_id} [{status}]")

    ordered = [result_by_id[row["id"]] for row in cases]
    print_summary(ordered)

    remaining_errors = sum(
        1 for row in ordered
        if row.get("error") or not row.get("completed")
    )
    print(f"\nRemaining failed/error cases: {remaining_errors}")
    print(f"Updated results saved to: {RESULTS_FILE}")

    if remaining_errors == 0:
        print("\n3000/3000 completed with no execution errors.")
        print("Use the reported metric values above as the final benchmark results.")


if __name__ == "__main__":
    main()
