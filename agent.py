"""CareerLens agent: the MCP Host/Client, built with the OpenAI Agents SDK.

Connects to the CareerLens FastMCP server over Streamable HTTP and lets an LLM
agent route user requests to the Data, Action, and Orchestration tools.

Human approval: any call that would write to the career plan (save_skill_gap, or
investigate_job_fit with save_gap=true) pauses the run until the user approves or
rejects it. Read-only calls never ask. Without an interactive terminal, writes are
rejected (fail closed).

Usage (start the server first in another terminal: python -m server.mcp_server)
    python agent.py                 interactive chat
    python agent.py "question"      answer a single question and exit
    python agent.py --tools         list the server's tools (no LLM call)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

from agents import Agent, ItemHelpers, RunContextWrapper, Runner
from agents.items import ToolApprovalItem, ToolCallItem, ToolCallOutputItem
from agents.mcp import MCPServerStreamableHttp

load_dotenv()

MCP_URL = os.getenv("CAREERLENS_MCP_URL", "http://127.0.0.1:8000/mcp")
MODEL = os.getenv("CAREERLENS_MODEL") or None  # None = Agents SDK default model

INSTRUCTIONS = """You are CareerLens, an assistant that compares a candidate profile with a job
posting using ONLY the CareerLens MCP tools. Never invent jobs, skills, evidence, or results.

Routing:
- The user only wants to know what a job requires -> get_job_requirements.
- The user wants a fit analysis, a match report, or skill gaps for a candidate and a job
  -> investigate_job_fit. Set save_gap=true ONLY if the user explicitly asks to save gaps.
- The user asks to save one specific entry (skill, gap type, reason) -> save_skill_gap.
  Only use a skill and gap_type the user asked for or that a fit report justifies, and
  write a rationale that cites the profile evidence.

Reporting:
- Keep the four labels distinct: supported_match, possible_match, skill_gap, missing_evidence.
  Missing evidence means the profile does not show the skill; never call it a gap.
- Cite the evidence behind each label. The coverage score is evidence coverage, not a
  probability of being hired; say so when you give it.
- Report every save exactly as the tool returned it (saved, already_exists, or failed, with
  the record_id). Never say something was saved unless a tool returned status "saved".
- If a tool returns an error, tell the user the error message plainly and stop that task.
  Do not retry with guessed IDs; you may list the valid IDs the error provides.
- Saves require the user's approval. If the user declines, say clearly that nothing was
  saved and do not try to save it another way.
"""

WRITE_TOOLS = ("save_skill_gap", "investigate_job_fit")
REJECTION_MESSAGE = "The user declined this save. Nothing was written to the career plan."


def show_tool_error(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """Pass the real failure reason to the model instead of the SDK's generic message."""
    return f"Tool call failed: {error}"


async def _approval_for_orchestration(ctx: RunContextWrapper[Any], args: dict[str, Any], call_id: str) -> bool:
    """investigate_job_fit only writes when save_gap is true, so only then does it need approval."""
    return args.get("save_gap") is True


def needs_approval_for(tool_name: str) -> Any:
    """Argument-aware approval policy for one CareerLens tool."""
    if tool_name == "save_skill_gap":
        return True
    if tool_name == "investigate_job_fit":
        return _approval_for_orchestration
    return False


class CareerLensMCPServer(MCPServerStreamableHttp):
    """MCP client for CareerLens with an argument-aware human-approval policy.

    The SDK's public require_approval setting decides per tool, without seeing the
    call's arguments. This override lets a read-only analysis run freely while a
    saving analysis still pauses for approval. If the SDK ever stops calling this
    hook, the require_approval fallback below still asks before every write-capable
    call (fail closed).
    """

    def _get_needs_approval_for_tool(self, tool: Any, agent: Any) -> Any:  # SDK hook (openai-agents 0.24)
        return needs_approval_for(tool.name)


def build_server() -> MCPServerStreamableHttp:
    return CareerLensMCPServer(
        name="CareerLens",
        params={"url": MCP_URL, "timeout": 10},
        cache_tools_list=True,
        client_session_timeout_seconds=30,
        failure_error_function=show_tool_error,
        require_approval={name: "always" for name in WRITE_TOOLS},
    )


def build_agent(server: MCPServerStreamableHttp) -> Agent:
    kwargs: dict[str, Any] = {"name": "CareerLens", "instructions": INSTRUCTIONS, "mcp_servers": [server]}
    if MODEL:
        kwargs["model"] = MODEL
    return Agent(**kwargs)


def _unwrap(value: Any) -> Any:
    """Turn the SDK's tool output wrapper ({"type": "text", "text": ...}) back into the server's payload."""
    if isinstance(value, list):
        parts = [_unwrap(v) for v in value]
        return parts[0] if len(parts) == 1 else parts
    if isinstance(value, dict) and value.get("type") == "text" and "text" in value:
        value = value["text"]
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _summarize_report(r: dict) -> str:
    """One-line view of an investigate_job_fit report, so the save outcome is visible."""
    saves = ", ".join(f"{o['skill']}={o['status']}({o.get('record_id') or '-'})" for o in r.get("saved_entries", []))
    return (f"fit report {r['candidate_id']} vs {r['job_id']}: coverage={r['required_coverage_score']}% "
            f"labels={r.get('label_counts')} | saves: {saves or 'none'} | {r.get('save_note', '')}")


def _short(value: Any, limit: int = 300) -> str:
    value = _unwrap(value)
    if isinstance(value, dict) and "required_coverage_score" in value and "candidate_id" in value:
        return _summarize_report(value)
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ..."


def print_trace(items: list, seen: set | None = None) -> None:
    """Show each step of the round trip: agent -> MCP client -> server -> back."""
    seen = set() if seen is None else seen
    for item in items:
        key = (type(item).__name__, id(item.raw_item))
        if key in seen:
            continue
        seen.add(key)
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            name = getattr(raw, "name", None) or (raw.get("name") if isinstance(raw, dict) else "?")
            args = getattr(raw, "arguments", None) or (raw.get("arguments") if isinstance(raw, dict) else "")
            print(f"  [agent -> MCP] call {name}({args})")
        elif isinstance(item, ToolCallOutputItem):
            if _unwrap(item.output) == REJECTION_MESSAGE:  # blocked by the host; never reached the server
                print(f"  [host -> agent] {REJECTION_MESSAGE}")
            else:
                print(f"  [MCP -> agent] {_short(item.output)}")


def describe_pending_write(item: ToolApprovalItem) -> str:
    """Plain-language description of what an approval would allow."""
    try:
        args = json.loads(item.arguments or "{}")
    except ValueError:
        args = {}
    if item.name == "save_skill_gap":
        return (f"save '{args.get('skill')}' as {args.get('gap_type')} for {args.get('candidate_id')} / "
                f"{args.get('job_id')}\n      rationale: {args.get('rationale')}")
    if item.name == "investigate_job_fit":
        return (f"analyze {args.get('candidate_id')} / {args.get('job_id')} and save every justified "
                f"skill gap to the career plan")
    return f"call {item.name}({item.arguments})"


def ask_user_to_approve(item: ToolApprovalItem) -> bool:
    print(f"\n  [approval needed] CareerLens wants to {describe_pending_write(item)}")
    try:
        answer = input("  Approve this write? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""  # no one to ask: fail closed
    approved = answer in {"y", "yes"}
    print(f"  [approval] {'approved' if approved else 'rejected'}\n")
    return approved


async def ask(agent: Agent, history: list, question: str) -> list:
    seen: set = set()
    result = await Runner.run(agent, history + [{"role": "user", "content": question}])
    while result.interruptions:  # the run paused before a write: ask the human
        print_trace(result.new_items, seen)
        state = result.to_state()
        for item in result.interruptions:
            if ask_user_to_approve(item):
                state.approve(item)
            else:
                state.reject(item, rejection_message=REJECTION_MESSAGE)
        result = await Runner.run(agent, state)
    print_trace(result.new_items, seen)
    print(f"\nCareerLens: {result.final_output}\n")
    return result.to_input_list()


def _hint(annotations: Any, name: str) -> Any:
    if annotations is None:
        return None
    camel = name.split("_")[0] + "".join(p.title() for p in name.split("_")[1:])
    return getattr(annotations, name, getattr(annotations, camel, None))


def describe_annotations(annotations: Any) -> str:
    """Turn MCP tool annotations into a short behavior label."""
    if annotations is None:
        return "no annotations"
    if _hint(annotations, "read_only_hint"):
        label = "read-only"
    else:
        destructive = _hint(annotations, "destructive_hint")
        label = "writes: " + ("additive" if destructive is False else "may overwrite/delete")
        if _hint(annotations, "idempotent_hint"):
            label += ", idempotent"
    if _hint(annotations, "open_world_hint") is False:
        label += ", local data only"
    return label


async def list_tools(server: MCPServerStreamableHttp) -> None:
    tools = await server.list_tools()
    print(f"Connected to {MCP_URL}. {len(tools)} tools:")
    for t in tools:
        first_line = (t.description or "").strip().splitlines()[0] if t.description else ""
        print(f"  - {t.name}: {first_line}")
        print(f"      behavior: {describe_annotations(getattr(t, 'annotations', None))}")


async def main(argv: list[str]) -> int:
    if "--tools" not in argv and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    server = build_server()
    try:
        await server.connect()
    except Exception as e:
        print(f"Could not connect to the CareerLens MCP server at {MCP_URL}: {e}")
        print("Start it first in another terminal:  python -m server.mcp_server")
        return 1

    try:
        if "--tools" in argv:
            await list_tools(server)
            return 0

        agent = build_agent(server)
        question = " ".join(a for a in argv if not a.startswith("--"))
        if question:
            await ask(agent, [], question)
            return 0

        print(f"CareerLens agent connected to {MCP_URL}. Type 'exit' to quit.\n")
        history: list = []
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            try:
                history = await ask(agent, history, question)
            except Exception as e:  # e.g. API key or network problems: report, keep the chat alive
                print(f"\n[error] {type(e).__name__}: {e}\n")
        return 0
    finally:
        await server.cleanup()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(sys.argv[1:])))
    except KeyboardInterrupt:
        sys.exit(0)
