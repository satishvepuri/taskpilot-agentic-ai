"""LLM provider layer for TaskPilot.

TaskPilot can use either OpenAI or Anthropic Claude for planning and final
response generation. LangChain is used as the common model interface.

Environment variables:
  TASKPILOT_LLM_PROVIDER=openai|claude  (default: openai)
  TASKPILOT_OPENAI_MODEL                (default: gpt-4o)
  TASKPILOT_CLAUDE_MODEL                (default: claude-sonnet-5)
  OPENAI_API_KEY                        required for OpenAI
  ANTHROPIC_API_KEY                     required for Claude
"""
from __future__ import annotations

import os
from typing import Any, Type

from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


def provider_name() -> str:
    raw = os.environ.get("TASKPILOT_LLM_PROVIDER", "openai").strip().lower()
    if raw in {"anthropic", "claude"}:
        return "claude"
    if raw == "openai":
        return "openai"
    raise ValueError(
        "TASKPILOT_LLM_PROVIDER must be 'openai' or 'claude', "
        f"got {raw!r}"
    )


def build_chat_model(purpose: str = "planner"):
    """Return a LangChain chat model for the selected provider."""
    provider = provider_name()
    if provider == "openai":
        model = os.environ.get("TASKPILOT_OPENAI_MODEL", "gpt-4o")
        temperature = 0 if purpose == "planner" else 0.2
        return ChatOpenAI(model=model, temperature=temperature)

    model = os.environ.get("TASKPILOT_CLAUDE_MODEL", "claude-sonnet-5")
    # Do not force sampling parameters for Claude; newer Claude models can
    # reject non-default temperature/top-p/top-k settings.
    return ChatAnthropic(model=model, max_tokens=2048)


def build_structured_model(schema: Type[BaseModel]):
    """Planner model that returns a validated Pydantic object.

    For OpenAI we explicitly use function/tool calling so the resume claim
    is literal rather than just describing free-form JSON parsing. Claude's
    LangChain structured-output adapter uses Claude's native tool-use path.
    """
    model = build_chat_model("planner")
    if provider_name() == "openai":
        return model.with_structured_output(schema, method="function_calling")
    return model.with_structured_output(schema)


def to_langchain_messages(history: list[dict], system_prompt: str, user_text: str):
    messages: list[Any] = [SystemMessage(content=system_prompt)]
    for item in history:
        role = item.get("role")
        content = str(item.get("content", ""))
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=user_text))
    return messages


def message_text(message: Any) -> str:
    """Extract text from LangChain messages across providers."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content)
