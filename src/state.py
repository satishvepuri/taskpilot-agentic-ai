"""
Shared state definitions for the TaskPilot LangGraph agent.

Structured outputs matter here for two reasons:
  1. Reliability — free-text LLM output is hard to route/parse downstream;
     Pydantic schemas force the model into a shape the graph can act on.
  2. Evaluability — the eval harness (eval/run_eval.py) can only measure
     "did it pick the right tool" if tool selection is a structured field,
     not buried in prose.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    DB_QUERY = "db_query"
    RETRIEVE_DOCS = "retrieve_docs"
    CALCULATE = "calculate"
    CALL_API = "call_api"
    RESPOND = "respond"  # terminal "tool" meaning: enough info, answer now


class SubTask(BaseModel):
    """One step in the decomposed plan."""
    step_id: int
    description: str = Field(..., description="What this step accomplishes, in plain language")
    tool: ToolName
    tool_input: str = Field(..., description="The concrete input to pass to the tool")
    requires_approval: bool = Field(
        default=False,
        description="True for actions with real-world side effects or spend (e.g. writes, "
                    "refunds, external API calls that mutate state) — routed to human approval.",
    )


class Plan(BaseModel):
    """Structured output of the planner node — the decomposition step."""
    reasoning: str = Field(..., description="Brief reasoning behind the decomposition")
    subtasks: list[SubTask]


class ToolResult(BaseModel):
    step_id: int
    tool: ToolName
    success: bool
    output: Any
    error: Optional[str] = None
    attempts: int = 1


class ApprovalDecision(BaseModel):
    step_id: int
    approved: bool
    reason: Optional[str] = None


class AgentState(TypedDict, total=False):
    """
    The full graph state, threaded through every node. LangGraph merges
    partial updates into this dict as nodes return.
    """
    conversation_id: str
    user_request: str
    chat_history: list[dict]        # short-term memory: prior turns this session
    plan: Optional[dict]            # serialized Plan
    current_step_index: int
    tool_results: list[dict]        # serialized ToolResult objects, accumulated
    pending_approval: Optional[dict]  # a SubTask awaiting human sign-off
    final_response: Optional[str]
    status: Literal["planning", "executing", "awaiting_approval", "done", "failed"]
