# TaskPilot — Agentic Workflow Automation Assistant

An agentic AI assistant, built with LangGraph, that decomposes a user
request into multi-step tasks, selects and executes tools (database
queries, document retrieval, calculations, API calls), pauses for human
approval on side-effecting actions, and is measured by an evaluation
harness — the full loop from request to audited execution.

## Architecture

```
 user request
      │
      ▼
 ┌──────────┐   structured Plan (Pydantic)
 │ planner  │───────────────────────────────┐
 └──────────┘                                │
      │ subtasks: [{tool, tool_input,        │
      │             requires_approval}]      │
      ▼                                       │
 ┌──────────────┐   requires_approval=False  │
 │ tool_executor│──────────┐                  │
 └──────────────┘          │                  │
      │ requires_approval=True                │
      ▼                                        │
 saves paused state, stops cleanly (no loop)   │
      │                                        │
      │   human resolves via CLI/webhook       │
      │   (`python main.py --approve ...`),    │
      │   which calls resume_agent()           │
      ▼                                        │
      └───────────────┬────────────────────────┘
                       ▼ (all steps done)
                 ┌────────────┐
                 │ responder  │ → final structured response
                 └────────────┘
```

Short-term memory (Redis, bounded window) threads through `conversation_id`
so multi-turn conversations retain context. Retries wrap every tool call
with backoff, distinguishing transient failures (retry) from planning
errors (fail fast).

## Why it's built this way

- **Structured outputs everywhere.** The planner returns a Pydantic `Plan`,
  not free text — this is what makes tool selection *measurable* (the eval
  harness can literally diff planned tools against expected tools) and
  makes the graph's routing logic deterministic instead of regex-parsing
  LLM prose.
- **Constrained tools, not open execution.** `db_query` only accepts
  whitelisted query templates (no raw SQL from the model); `calculate`
  parses expressions with `ast` instead of `eval()`. The LLM chooses
  *what* to do, never *how* the underlying system executes it — this
  matters a lot once tools have real side effects.
- **Approval is a first-class graph state, not an afterthought.** Any
  subtask with real-world side effects (refunds, budget changes) is
  flagged by the planner and pauses the graph — it stops cleanly and
  persists its exact state (plan, progress, pending step) rather than
  polling in a loop. A separate `--approve`/`--reject` CLI call resumes
  it later from that saved state, which is what lets this run safely
  as two entirely separate process invocations.
- **Retry vs. fail-fast is a deliberate distinction.** `retry.py` only
  retries errors that look transient (timeouts, rate limits); a bad query
  template name fails immediately rather than burning 3 attempts on a bug
  that retrying can't fix.
- **Evaluation is structural, not vibes-based.** `eval/run_eval.py` scores
  tool selection accuracy (against labeled expected tools), task
  completion rate, and uses an LLM-as-judge for response quality — the
  three axes that actually matter for an agent: did it pick the right
  tool, did it finish, and was the answer any good.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

docker-compose up -d          # Postgres + Redis
python -m src.tools.retrieval_tool   # builds the FAISS index from sample docs

# Try it
python main.py "What's the status of order ORD-1029?"
python main.py --conversation demo "Issue a $45 refund for order ORD-1029"
# -> pauses for approval, prints exact approve/reject command to run

python main.py --approve demo:1

# Run the eval suite
python -m eval.run_eval
```

## Suggested resume bullets

- Built an agentic AI assistant with LangGraph that decomposes user
  requests into multi-step plans and dynamically selects between
  database query, RAG retrieval, calculation, and external API tools
  using structured, schema-validated outputs.
- Implemented a human-in-the-loop approval gate for side-effecting actions
  (refunds, budget changes), a bounded short-term memory layer in Redis
  for multi-turn context, and retry handling that distinguishes transient
  failures from non-retryable planning errors.
- Built an evaluation harness scoring tool-selection accuracy, task
  completion rate, and LLM-judged response quality across test
  conversations, used to validate agent reliability before deployment.
- Constrained tool execution to whitelisted query templates and safe
  expression parsing (avoiding raw SQL/`eval()` from LLM output) to keep
  agentic actions auditable and injection-resistant.

## Next steps to extend this further

- Swap the synchronous approval poll for a real webhook/Slack
  approve-reject flow, and persist graph state so it survives a process
  restart while awaiting approval.
- Add LangGraph's built-in checkpointing (`MemorySaver`/Postgres saver)
  instead of the custom Redis memory module, for full state persistence.
- Expand the eval set toward the 3K+ conversation scale referenced on the
  resume, with a labeled dataset generation script and CI integration so
  regressions in tool selection are caught automatically on each change.

## Resume-claim verification checklist

Do not claim a feature just because code exists. Mark it complete only after the test below succeeds.

1. **OpenAI + LangChain structured/function calling**
   - `set TASKPILOT_LLM_PROVIDER=openai`
   - run a normal TaskPilot request and confirm it finishes.
2. **Claude provider**
   - install dependencies, set `ANTHROPIC_API_KEY`, then:
   - `set TASKPILOT_LLM_PROVIDER=claude`
   - run at least one normal request and one approval workflow.
3. **RAG**
   - build the index and ask a refund/shipping policy question.
4. **Calculation tool**
   - ask `Calculate 240 * (1 - 0.15).`
5. **Human approval**
   - run a refund request and test both approve and reject paths.
6. **3K+ evaluation**
   - first smoke test 10 cases:
     `python -m eval.run_eval --file eval/test_conversations_3000.jsonl --limit 10`
   - then run/resume the benchmark:
     `python -m eval.run_eval --file eval/test_conversations_3000.jsonl --resume`
   - the resume claim is supported only when the saved results reach 3,000 conversations.

The 3,000-case benchmark is synthetic/labeled. Say that plainly if an interviewer asks where the test conversations came from.
