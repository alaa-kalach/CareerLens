"""Smoke check for the Data tool. Run from the project root:
    python -m scripts.smoke_data_tool
"""
import asyncio

from fastmcp import Client
from fastmcp.exceptions import ToolError

from server import careerlens as core
from server.mcp_server import mcp


def check_logic() -> None:
    print("== 1. Logic layer ==")
    job = core.get_job_requirements("j003")
    print("OK:", job.job_id, "-", job.title, "| required:", [s.skill for s in job.required_skills])
    try:
        core.get_job_requirements("J999")
        print("FAIL: unknown job did not raise")
    except core.JobNotFoundError as e:
        print("OK (expected error):", e)


async def check_mcp() -> None:
    print("\n== 2. Through MCP (in-memory client) ==")
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Registered tools:", [t.name for t in tools])
        result = await client.call_tool("get_job_requirements", {"job_id": "J003"})
        print("OK:", getattr(result, "data", result))
        try:
            await client.call_tool("get_job_requirements", {"job_id": "J999"})
            print("FAIL: unknown job did not raise")
        except ToolError as e:
            print("OK (expected error over MCP):", e)


if __name__ == "__main__":
    check_logic()
    asyncio.run(check_mcp())