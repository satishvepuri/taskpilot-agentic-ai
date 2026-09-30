"""Re-score saved TaskPilot results against the updated benchmark, then
rerun only rows that are still genuinely imperfect.

This avoids wasting API money rerunning rows that only failed because the old
benchmark expected an unsafe/less-specific tool sequence or because a number
was formatted with commas.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from eval.run_eval import evaluate_case, exact_tool_sequence, response_quality, print_summary

HERE = Path(__file__).parent
CASES_FILE = HERE / "test_conversations_3000.jsonl"
RESULTS_FILE = HERE / "test_conversations_3000_results.jsonl"
BACKUP_FILE = HERE / "test_conversations_3000_results_before_policy_fix.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    tmp.replace(path)


def rescore_existing(case: dict, result: dict) -> dict:
    updated = dict(result)

    # Errors cannot be rescored; they need a real rerun.
    if result.get("error") or not result.get("completed"):
        return updated

    expected_tools = case.get("expected_tools", [])
    planned_tools = result.get("planned_tools", [])
    final_response = result.get("final_response", "") or ""

    updated["expected_tools"] = expected_tools
    updated["tool_selection_score"] = exact_tool_sequence(planned_tools, expected_tools)
    updated["response_quality"] = response_quality(
        final_response,
        case.get("expected_response_contains", []),
    )
    return updated


def imperfect(row: dict) -> bool:
    return (
        bool(row.get("error"))
        or not bool(row.get("completed"))
        or not bool(row.get("approval_handling_correct"))
        or float(row.get("tool_selection_score", 0)) < 1.0
        or float(row.get("response_quality", 0)) < 1.0
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Rerun only the first N remaining imperfect rows")
    args = parser.parse_args()

    cases = load_jsonl(CASES_FILE)
    results = load_jsonl(RESULTS_FILE)

    if len(cases) != 3000 or len(results) != 3000:
        raise RuntimeError(
            f"Expected 3000 cases and 3000 results, found {len(cases)} and {len(results)}"
        )

    if not BACKUP_FILE.exists():
        shutil.copy2(RESULTS_FILE, BACKUP_FILE)
        print(f"Backup created: {BACKUP_FILE}")

    case_by_id = {c["id"]: c for c in cases}
    result_by_id = {r["id"]: r for r in results}

    # First fix score-only issues without making any API calls.
    for case in cases:
        rid = case["id"]
        result_by_id[rid] = rescore_existing(case, result_by_id[rid])

    ordered = [result_by_id[c["id"]] for c in cases]
    save_jsonl(RESULTS_FILE, ordered)

    remaining = [c["id"] for c in cases if imperfect(result_by_id[c["id"]])]

    print("\nAfter free re-scoring (no API calls):")
    print_summary(ordered)
    print(f"Rows still needing a real rerun: {len(remaining)}")

    if args.limit is not None:
        remaining = remaining[: args.limit]
        print(f"Rerunning only {len(remaining)} row(s) in this test run.")

    if not remaining:
        print("\nNo imperfect rows remain.")
        return

    for i, case_id in enumerate(remaining, start=1):
        result_by_id[case_id] = evaluate_case(case_by_id[case_id])

        if i == 1 or i % 10 == 0 or i == len(remaining):
            ordered = [result_by_id[c["id"]] for c in cases]
            save_jsonl(RESULTS_FILE, ordered)
            row = result_by_id[case_id]
            label = "PASS" if not imperfect(row) else "NEEDS-RETRY"
            print(f"Reran {i}/{len(remaining)} — latest: {case_id} [{label}]")

    ordered = [result_by_id[c["id"]] for c in cases]
    print_summary(ordered)

    still_bad = sum(1 for r in ordered if imperfect(r))
    print(f"\nRows still imperfect: {still_bad}")
    print(f"Updated results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
