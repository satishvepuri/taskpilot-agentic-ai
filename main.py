"""
CLI entrypoint for TaskPilot.

Usage:
    python main.py "What's the status of order ORD-1029?"
    python main.py --conversation demo-1 "Issue a $45 refund for order ORD-1029"

If the agent pauses for approval, approve it with:
    python main.py --approve demo-1:2
"""
import argparse
import json
import sys

from src.graph import run_agent, resume_agent
from src import approval


def print_final_state(final_state: dict, conversation_id: str):
    if final_state["status"] == "awaiting_approval":
        subtask = final_state["pending_approval"]
        print("⏸  PAUSED — human approval required for:")
        print(json.dumps(subtask, indent=2))
        print(f"\nApprove with: python main.py --approve {conversation_id}:{subtask['step_id']}")
        print(f"Reject with:  python main.py --reject {conversation_id}:{subtask['step_id']}")
    else:
        print("Plan:", json.dumps(final_state.get("plan", {}), indent=2))
        print("\nTool results:")
        for r in final_state.get("tool_results", []):
            print(" ", r)
        print("\n=== Final response ===")
        print(final_state.get("final_response", "(no response generated)"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", nargs="?", help="The user request to send to the agent")
    parser.add_argument("--conversation", default="cli-session", help="Conversation id for memory continuity")
    parser.add_argument("--approve", help="Resolve a pending approval, format 'conversation_id:step_id'")
    parser.add_argument("--reject", help="Reject a pending approval, format 'conversation_id:step_id'")
    args = parser.parse_args()

    if args.approve or args.reject:
        target = args.approve or args.reject
        conv_id, step_id = target.split(":")
        approval.resolve_approval(conv_id, int(step_id), approved=bool(args.approve))
        print(f"Approval {'granted' if args.approve else 'rejected'} for {target}")
        print("Resuming workflow...\n")

        final_state = resume_agent(conv_id)
        print_final_state(final_state, conv_id)
        return

    if not args.request:
        parser.print_help()
        sys.exit(1)

    print(f"[TaskPilot] Processing: {args.request}\n")
    final_state = run_agent(args.conversation, args.request)
    print_final_state(final_state, args.conversation)


if __name__ == "__main__":
    main()
