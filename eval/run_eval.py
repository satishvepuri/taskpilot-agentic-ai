"""TaskPilot evaluation harness.

Measures:
  1. Exact tool-sequence accuracy.
  2. Task completion rate.
  3. Human-approval routing accuracy.
  4. Rule-based response quality using expected answer tokens.

Approval-required cases are automatically approved during evaluation so the
whole workflow can finish and response quality can be scored. The API tool is
mocked, so this does not issue a real refund or modify a real external system.

Examples:
  python -m eval.run_eval
  python -m eval.run_eval --file eval/test_conversations_3000.jsonl --limit 10
  python -m eval.run_eval --file eval/test_conversations_3000.jsonl --resume
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.graph import run_agent, resume_agent
from src.state import Plan
from src import memory, approval

DEFAULT_FILE = Path(__file__).parent / "test_conversations.jsonl"


def load_test_cases(path: Path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def exact_tool_sequence(planned: list[str], expected: list[str]) -> float:
    return 1.0 if planned == expected else 0.0


def _normalize_text(value) -> str:
    text = str(value or "").lower()
    text = text.replace("$", "").replace(",", "")
    text = " ".join(text.split())
    return text


def response_quality(response: str, expected_tokens: list[str]) -> float:
    if not expected_tokens:
        return 1.0 if response and response.strip() else 0.0
    text = _normalize_text(response)
    hits = sum(1 for token in expected_tokens if _normalize_text(token) in text)
    return hits / len(expected_tokens)


def finish_approvals(conv_id: str, state: dict) -> dict:
    """Auto-approve mock side effects so eval cases reach a final answer."""
    safety = 0
    while state.get("status") == "awaiting_approval":
        safety += 1
        if safety > 10:
            raise RuntimeError("Too many approval pauses in one eval case")
        subtask = state["pending_approval"]
        approval.resolve_approval(
            conv_id,
            int(subtask["step_id"]),
            approved=True,
            reason="Automated benchmark approval",
        )
        state = resume_agent(conv_id)
    return state


def evaluate_case(case: dict) -> dict:
    conv_id = f"eval-{case['id']}"
    memory.clear_history(conv_id)

    try:
        state = run_agent(conv_id, case["request"])
        initially_awaited = state.get("status") == "awaiting_approval"
        state = finish_approvals(conv_id, state)
    except Exception as exc:
        return {
            "id": case["id"],
            "category": case["category"],
            "error": str(exc),
            "completed": False,
            "tool_selection_score": 0.0,
            "approval_handling_correct": False,
            "response_quality": 0.0,
        }

    plan = Plan.model_validate(state["plan"]) if state.get("plan") else None
    planned_tools = [st.tool.value for st in plan.subtasks if st.tool.value != "respond"] if plan else []
    expected_tools = case.get("expected_tools", [])
    final_response = state.get("final_response", "") or ""

    return {
        "id": case["id"],
        "category": case["category"],
        "planned_tools": planned_tools,
        "expected_tools": expected_tools,
        "tool_selection_score": exact_tool_sequence(planned_tools, expected_tools),
        "completed": state.get("status") == "done",
        "approval_handling_correct": initially_awaited == bool(case.get("expects_approval")),
        "response_quality": response_quality(final_response, case.get("expected_response_contains", [])),
        "final_status": state.get("status"),
        "final_response": final_response,
    }


def read_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    return ids


def load_results(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    out = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def print_summary(results: list[dict]):
    n = len(results)
    if not n:
        print("No results yet.")
        return
    avg_tool = sum(float(r.get("tool_selection_score", 0)) for r in results) / n
    completion = sum(1 for r in results if r.get("completed")) / n
    approval_acc = sum(1 for r in results if r.get("approval_handling_correct")) / n
    quality = sum(float(r.get("response_quality", 0)) for r in results) / n
    errors = sum(1 for r in results if r.get("error"))

    print("\n=== TaskPilot Evaluation Summary ===")
    print(f"Conversations evaluated: {n}")
    print(f"Exact tool-sequence accuracy: {avg_tool:.2%}")
    print(f"Task completion rate: {completion:.2%}")
    print(f"Approval routing accuracy: {approval_acc:.2%}")
    print(f"Response quality score: {quality:.2%}")
    print(f"Errors: {errors}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(DEFAULT_FILE), help="JSONL benchmark file")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N remaining cases")
    parser.add_argument("--output", default=None, help="Output JSONL file")
    parser.add_argument("--resume", action="store_true", help="Skip case IDs already in output")
    args = parser.parse_args()

    test_file = Path(args.file)
    cases = load_test_cases(test_file)
    output = Path(args.output) if args.output else Path(__file__).parent / (test_file.stem + "_results.jsonl")

    done_ids = read_completed_ids(output) if args.resume else set()
    if output.exists() and not args.resume:
        output.unlink()

    pending = [c for c in cases if c["id"] not in done_ids]
    if args.limit is not None:
        pending = pending[:args.limit]

    mode = "a" if args.resume else "w"
    with open(output, mode, encoding="utf-8") as f:
        for i, case in enumerate(pending, start=1):
            result = evaluate_case(case)
            f.write(json.dumps(result, default=str) + "\n")
            f.flush()
            if i == 1 or i % 10 == 0 or i == len(pending):
                print(f"Completed {i}/{len(pending)} this run — latest: {case['id']}")

    results = load_results(output)
    print_summary(results)
    print(f"\nResults saved to: {output}")
    if len(results) >= 3000:
        print("\n3K+ EVALUATION MILESTONE REACHED: the resume claim is now supported by saved results.")
    else:
        print(f"\n3K milestone not reached yet: {len(results)}/3000 saved results.")


if __name__ == "__main__":
    main()
