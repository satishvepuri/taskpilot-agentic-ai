"""
The TaskPilot agent graph, built with LangGraph.

Flow:
  planner -> tool_executor -> (loop until plan complete or approval needed)
          -> if a step needs approval: save state and stop cleanly (no
             infinite polling loop) — a later `python main.py --approve`
             call resumes it via resume_agent()
          -> responder (synthesizes final answer once all steps are done)

Each node is a plain function over AgentState -> partial AgentState update,
which is the LangGraph pattern. Structured outputs (Plan, ToolResult) keep
node boundaries typed and testable in isolation from the LLM.
"""
from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import AgentState, Plan, ToolResult, ToolName
from src.retry import run_with_retry
from src import memory, approval
from src.tools.db_tool import db_query
from src.tools.retrieval_tool import retrieve_docs
from src.tools.calculator_tool import calculate
from src.tools.api_tool import call_api
from src.llm import build_chat_model, build_structured_model, to_langchain_messages, message_text

TOOL_DISPATCH = {
    ToolName.DB_QUERY: db_query,
    ToolName.RETRIEVE_DOCS: retrieve_docs,
    ToolName.CALCULATE: calculate,
    ToolName.CALL_API: call_api,
}

PLANNER_SYSTEM_PROMPT = """You are a planning module for an agentic task assistant.
Given a user request and conversation history, decompose it into an ordered
list of subtasks. Each subtask must use exactly one tool:

- db_query: structured lookups against orders/customers/campaigns
  (templates: get_order_status, get_customer_orders, get_customer_profile, get_campaign_metrics)
- retrieve_docs: answer questions about policies/FAQs from the knowledge base
- calculate: any arithmetic (fees, totals, percentages)
- call_api: actions with real-world side effects (issue_refund, update_campaign_budget, send_notification)
- respond: use only as the final step once no more tool calls are needed

Mark requires_approval=true for any call_api subtask, since these have
real side effects and must be human-approved before executing.

IMPORTANT — tool_input format: for db_query and call_api, tool_input MUST
be written EXACTLY as "template_name(param=value)", using parentheses and
an equals sign — for example: "get_order_status(order_id=ORD-1029)".
Do NOT use a colon or a space instead of parentheses (e.g. do NOT write
"get_order_status: ORD-1029").

Respond ONLY with valid JSON matching this schema:
{"reasoning": str, "subtasks": [{"step_id": int, "description": str, "tool": str, "tool_input": str, "requires_approval": bool}]}
"""


def planner_node(state: AgentState) -> dict:
    history = memory.get_history(state["conversation_id"])
    messages = to_langchain_messages(history, PLANNER_SYSTEM_PROMPT, state["user_request"])

    # Structured output keeps planning machine-readable. For OpenAI this
    # explicitly uses function/tool calling; Claude uses its native tool-use
    # structured-output path through LangChain.
    planner = build_structured_model(Plan)
    plan = planner.invoke(messages)
    if not isinstance(plan, Plan):
        plan = Plan.model_validate(plan)

    memory.append_turn(state["conversation_id"], "user", state["user_request"])

    return {
        "plan": plan.model_dump(),
        "current_step_index": 0,
        "tool_results": [],
        "status": "executing",
    }


def tool_executor_node(state: AgentState) -> dict:
    plan = Plan.model_validate(state["plan"])
    idx = state["current_step_index"]
    subtask = plan.subtasks[idx]

    if subtask.tool == ToolName.RESPOND:
        return {"status": "executing"}  # handled by router -> responder

    if subtask.requires_approval:
        approval.request_approval(state["conversation_id"], subtask.model_dump())
        return {
            "pending_approval": subtask.model_dump(),
            "status": "awaiting_approval",
        }

    fn = TOOL_DISPATCH[subtask.tool]
    success, output, error, attempts = run_with_retry(fn, subtask.tool_input)

    result = ToolResult(
        step_id=subtask.step_id, tool=subtask.tool, success=success,
        output=output, error=error, attempts=attempts,
    )

    updated_results = state.get("tool_results", []) + [result.model_dump()]
    return {
        "tool_results": updated_results,
        "current_step_index": idx + 1,
        "status": "executing",
    }


def _execute_approved_subtask(subtask: dict, decision: dict) -> ToolResult:
    """Shared logic for running (or recording rejection of) a subtask once
    a human has made a decision on it. Used both by the initial run (if a
    decision somehow already exists) and by resume_agent (the normal path)."""
    if not decision["approved"]:
        return ToolResult(
            step_id=subtask["step_id"], tool=subtask["tool"], success=False,
            output=None, error=(f"Rejected by human reviewer: {decision.get('reason')}"
                   if decision.get("reason") else "Rejected by human reviewer"), attempts=0,
        )
    fn = TOOL_DISPATCH[ToolName(subtask["tool"])]
    success, output, error, attempts = run_with_retry(fn, subtask["tool_input"])
    return ToolResult(
        step_id=subtask["step_id"], tool=subtask["tool"], success=success,
        output=output, error=error, attempts=attempts,
    )


def responder_node(state: AgentState) -> dict:
    plan = Plan.model_validate(state["plan"])
    results = state.get("tool_results", [])

    summary_lines = []
    for r in results:
        line = f"- Step {r['step_id']} ({r['tool']}): "
        line += "SUCCESS" if r["success"] else f"FAILED ({r['error']})"
        if r["success"]:
            line += f" -> {r['output']}"
        summary_lines.append(line)

    responder = build_chat_model("responder")
    response = responder.invoke([
        SystemMessage(content=(
            "Synthesize a clear, concise answer to the user's original request "
            "using only the tool execution results below. If any step failed, "
            "acknowledge it plainly. Do not invent policy rules or facts that "
            "are not present in the tool results."
        )),
        HumanMessage(content=(
            f"Original request: {state['user_request']}\n\n"
            f"Plan reasoning: {plan.reasoning}\n\n"
            f"Tool results:\n" + "\n".join(summary_lines)
        )),
    ])
    final_text = message_text(response)
    memory.append_turn(state["conversation_id"], "assistant", final_text)

    return {"final_response": final_text, "status": "done"}


def route_after_executor(state: AgentState) -> Literal["tool_executor", "responder", "end"]:
    if state["status"] == "awaiting_approval":
        # Stop the graph run cleanly here rather than looping — a separate
        # process (triggered by `python main.py --approve ...`) will resume
        # from the saved paused state once a human has made a decision.
        return "end"

    plan = Plan.model_validate(state["plan"])
    idx = state["current_step_index"]

    if idx >= len(plan.subtasks):
        return "responder"

    next_tool = plan.subtasks[idx].tool
    if next_tool == ToolName.RESPOND:
        return "responder"

    return "tool_executor"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("responder", responder_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool_executor")

    graph.add_conditional_edges("tool_executor", route_after_executor, {
        "tool_executor": "tool_executor",
        "responder": "responder",
        "end": END,
    })

    graph.add_edge("responder", END)

    return graph.compile()


def run_agent(conversation_id: str, user_request: str) -> AgentState:
    app = build_graph()
    initial_state: AgentState = {
        "conversation_id": conversation_id,
        "user_request": user_request,
        "status": "planning",
    }
    # recursion_limit guards against a pathological plan looping forever
    final_state = app.invoke(initial_state, config={"recursion_limit": 25})

    if final_state["status"] == "awaiting_approval":
        approval.save_paused_state(conversation_id, final_state)

    return final_state


def resume_agent(conversation_id: str) -> AgentState:
    """
    Continues a paused workflow after a human has approved or rejected the
    pending step. This runs the remaining subtasks directly (not through
    graph.invoke) since LangGraph's entry point is fixed to `planner` —
    resuming mid-plan means replaying the loop ourselves using the exact
    same node functions the graph itself uses, so behavior stays identical.
    """
    state = approval.load_paused_state(conversation_id)
    if state is None:
        raise ValueError(
            f"No paused workflow found for conversation '{conversation_id}'. "
            "It may have already been resumed, or expired."
        )

    subtask = state["pending_approval"]
    decision = approval.poll_approval(conversation_id, subtask["step_id"])
    if decision is None:
        # Not approved/rejected yet — nothing to do. Caller (main.py) should
        # tell the user to run --approve or --reject first.
        return state

    result = _execute_approved_subtask(subtask, decision)
    state = {
        **state,
        "tool_results": state.get("tool_results", []) + [result.model_dump()],
        "current_step_index": state["current_step_index"] + 1,
        "pending_approval": None,
        "status": "executing",
    }
    approval.clear_paused_state(conversation_id)

    # Continue running any remaining steps, which may include hitting
    # another approval-required step (handled the same way).
    while True:
        plan = Plan.model_validate(state["plan"])
        idx = state["current_step_index"]

        if idx >= len(plan.subtasks) or plan.subtasks[idx].tool == ToolName.RESPOND:
            update = responder_node(state)
            return {**state, **update}

        update = tool_executor_node(state)
        state = {**state, **update}

        if state["status"] == "awaiting_approval":
            approval.save_paused_state(conversation_id, state)
            return state
