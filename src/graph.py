"""
TaskPilot LangGraph orchestration.
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
- retrieve_docs: answer questions about policies/FAQs from the knowledge base
- calculate: arithmetic
- call_api: real-world side effects
- respond: final response once no more tool calls are needed

DETERMINISTIC PLANNING POLICY:
1. Order status/customer/campaign metrics: db_query, then respond.
2. Refund/shipping policy questions: retrieve_docs, then respond.
3. Pure arithmetic: calculate, then respond.
4. Refund action: db_query get_order_status, then call_api issue_refund, then respond.
5. Campaign budget update: db_query get_campaign_metrics, then call_api update_campaign_budget, then respond.
6. Notification action: call_api send_notification, then respond. Do NOT add db_query unless the user explicitly asks for a lookup.
7. Lookup plus arithmetic: db_query, then calculate, then respond.
8. Do not add extra tools.

Mark requires_approval=true for every call_api subtask.
All non-call_api subtasks must have requires_approval=false.

For db_query and call_api, tool_input MUST use:
template_name(param=value)

Respond ONLY with valid JSON matching:
{"reasoning": str, "subtasks": [{"step_id": int, "description": str, "tool": str, "tool_input": str, "requires_approval": bool}]}
"""

def planner_node(state: AgentState) -> dict:
    history = memory.get_history(state["conversation_id"])
    messages = to_langchain_messages(history, PLANNER_SYSTEM_PROMPT, state["user_request"])

    last_error = None
    plan = None
    for _ in range(3):
        try:
            planner = build_structured_model(Plan)
            candidate = planner.invoke(messages)
            plan = candidate if isinstance(candidate, Plan) else Plan.model_validate(candidate)
            break
        except Exception as exc:
            last_error = exc

    if plan is None:
        raise last_error

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
        return {"status": "executing"}

    if subtask.requires_approval:
        approval.request_approval(state["conversation_id"], subtask.model_dump())
        return {
            "pending_approval": subtask.model_dump(),
            "status": "awaiting_approval",
        }

    fn = TOOL_DISPATCH[subtask.tool]
    success, output, error, attempts = run_with_retry(fn, subtask.tool_input)

    result = ToolResult(
        step_id=subtask.step_id,
        tool=subtask.tool,
        success=success,
        output=output,
        error=error,
        attempts=attempts,
    )

    return {
        "tool_results": state.get("tool_results", []) + [result.model_dump()],
        "current_step_index": idx + 1,
        "status": "executing",
    }

def _execute_approved_subtask(subtask: dict, decision: dict) -> ToolResult:
    if not decision["approved"]:
        return ToolResult(
            step_id=subtask["step_id"],
            tool=subtask["tool"],
            success=False,
            output=None,
            error=f"Rejected by human reviewer: {decision.get('reason')}" if decision.get("reason") else "Rejected by human reviewer",
            attempts=0,
        )

    fn = TOOL_DISPATCH[ToolName(subtask["tool"])]
    success, output, error, attempts = run_with_retry(fn, subtask["tool_input"])
    return ToolResult(
        step_id=subtask["step_id"],
        tool=subtask["tool"],
        success=success,
        output=output,
        error=error,
        attempts=attempts,
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
            "Synthesize a clear, concise answer using only the tool execution results. "
            "Preserve important numbers and action statuses. Do not invent facts."
        )),
        HumanMessage(content=(
            f"Original request: {state['user_request']}\n\n"
            f"Plan reasoning: {plan.reasoning}\n\n"
            "Tool results:\n" + "\n".join(summary_lines)
        )),
    ])

    final_text = message_text(response)
    memory.append_turn(state["conversation_id"], "assistant", final_text)
    return {"final_response": final_text, "status": "done"}

def route_after_executor(state: AgentState) -> Literal["tool_executor", "responder", "end"]:
    if state["status"] == "awaiting_approval":
        return "end"

    plan = Plan.model_validate(state["plan"])
    idx = state["current_step_index"]

    if idx >= len(plan.subtasks):
        return "responder"

    if plan.subtasks[idx].tool == ToolName.RESPOND:
        return "responder"

    return "tool_executor"

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("responder", responder_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool_executor")
    graph.add_conditional_edges(
        "tool_executor",
        route_after_executor,
        {"tool_executor": "tool_executor", "responder": "responder", "end": END},
    )
    graph.add_edge("responder", END)
    return graph.compile()

def run_agent(conversation_id: str, user_request: str) -> AgentState:
    app = build_graph()
    initial_state: AgentState = {
        "conversation_id": conversation_id,
        "user_request": user_request,
        "status": "planning",
    }

    final_state = app.invoke(initial_state, config={"recursion_limit": 25})

    if final_state["status"] == "awaiting_approval":
        approval.save_paused_state(conversation_id, final_state)

    return final_state

def resume_agent(conversation_id: str) -> AgentState:
    state = approval.load_paused_state(conversation_id)
    if state is None:
        raise ValueError(f"No paused workflow found for conversation '{conversation_id}'.")

    subtask = state["pending_approval"]
    decision = approval.poll_approval(conversation_id, subtask["step_id"])
    if decision is None:
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
