"""CareerLens agent: the MCP Host/Client, built with the OpenAI Agents SDK.

Connects to the CareerLens FastMCP server over Streamable HTTP and lets an LLM
agent route user requests to the Data, Action, and Orchestration tools.

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
from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem
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
"""


def show_tool_error(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """Pass the real failure reason to the model instead of the SDK's generic message."""
    return f"Tool call failed: {error}"


def build_server() -> MCPServerStreamableHttp:
    return MCPServerStreamableHttp(
        name="CareerLens",
        params={"url": MCP_URL, "timeout": 10},
        cache_tools_list=True,
        client_session_timeout_seconds=30,
        failure_error_function=show_tool_error,
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


def print_trace(items: list) -> None:
    """Show each step of the round trip: agent -> MCP client -> server -> back."""
    for item in items:
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            name = getattr(raw, "name", None) or (raw.get("name") if isinstance(raw, dict) else "?")
            args = getattr(raw, "arguments", None) or (raw.get("arguments") if isinstance(raw, dict) else "")
            print(f"  [agent -> MCP] call {name}({args})")
        elif isinstance(item, ToolCallOutputItem):
            print(f"  [MCP -> agent] {_short(item.output)}")


async def ask(agent: Agent, history: list, question: str) -> list:
    result = await Runner.run(agent, history + [{"role": "user", "content": question}])
    print_trace(result.new_items)
    print(f"\nCareerLens: {result.final_output}\n")
    return result.to_input_list()


async def list_tools(server: MCPServerStreamableHttp) -> None:
    tools = await server.list_tools()
    print(f"Connected to {MCP_URL}. {len(tools)} tools:")
    for t in tools:
        first_line = (t.description or "").strip().splitlines()[0] if t.description else ""
        print(f"  - {t.name}: {first_line}")


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
