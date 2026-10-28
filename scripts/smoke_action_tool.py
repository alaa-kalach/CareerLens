"""Smoke check for the Action tool. Uses a temporary copy of the data,
so data/career_plan.json is never touched. Run from the project root:
    python -m scripts.smoke_action_tool
"""
import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from fastmcp import Client
from fastmcp.exceptions import ToolError

from server import careerlens as core
from server.mcp_server import mcp

REAL_DATA = Path("data")
TMP = Path(tempfile.mkdtemp(prefix="careerlens_smoke_"))
for name in ("candidates.json", "jobs.json", "career_plan.json"):
    shutil.copy(REAL_DATA / name, TMP / name)
os.environ["CAREERLENS_DATA_DIR"] = str(TMP)

RATIONALE = "Candidate explicitly lists Docker under known gaps."


def plan_count() -> int:
    return len(json.loads((TMP / "career_plan.json").read_text(encoding="utf-8"))["entries"])


def expect_error(label: str, fn, *args) -> None:
    before = plan_count()
    try:
        fn(*args)
        print(f"FAIL  {label}: no error raised")
    except core.CareerLensError as e:
        unchanged = plan_count() == before
        print(f"{'OK  ' if unchanged else 'FAIL'}  {label}: {type(e).__name__}: {e}"
              f"{'' if unchanged else '  <-- FILE CHANGED!'}")


def check_logic() -> None:
    print("== 1. Logic layer ==")
    r = core.save_skill_gap("C001", "J003", "docker", "missing_skill", RATIONALE)
    print(f"OK    valid save: {r.status} {r.record_id} skill={r.entry.skill!r} | entries={plan_count()}")
    r2 = core.save_skill_gap("C001", "J003", "Docker", "missing_skill", RATIONALE)
    print(f"OK    duplicate: {r2.status} {r2.record_id} | entries={plan_count()}")

    expect_error("invalid gap_type", core.save_skill_gap, "C001", "J003", "Docker", "weakness", RATIONALE)
    expect_error("empty skill", core.save_skill_gap, "C001", "J003", "  ", "missing_skill", RATIONALE)
    expect_error("short rationale", core.save_skill_gap, "C001", "J003", "Docker", "missing_skill", "no")
    expect_error("skill not in job", core.save_skill_gap, "C001", "J003", "Excel", "missing_skill", RATIONALE)
    expect_error("unknown job", core.save_skill_gap, "C001", "J999", "Docker", "missing_skill", RATIONALE)
    expect_error("unknown candidate", core.save_skill_gap, "C999", "J003", "Docker", "missing_skill", RATIONALE)

    plan_file = TMP / "career_plan.json"
    good = plan_file.read_text(encoding="utf-8")
    plan_file.write_text("{ this is not json", encoding="utf-8")
    try:
        core.save_skill_gap("C001", "J003", "Git", "insufficient_evidence", RATIONALE)
        print("FAIL  corrupt plan file: no error raised")
    except core.StorageError as e:
        intact = plan_file.read_text(encoding="utf-8") == "{ this is not json"
        print(f"{'OK  ' if intact else 'FAIL'}  corrupt plan file refused, not overwritten: {e}")
    plan_file.write_text(good, encoding="utf-8")


async def check_mcp() -> None:
    print("\n== 2. Through MCP (in-memory client) ==")
    async with Client(mcp) as client:
        print("Registered tools:", [t.name for t in await client.list_tools()])
        result = await client.call_tool("save_skill_gap", {
            "candidate_id": "C001", "job_id": "J003", "skill": "Git",
            "gap_type": "insufficient_evidence",
            "rationale": "Git is listed as a skill but no project or role shows it being used.",
        })
        data = getattr(result, "data", result)
        print("OK    saved over MCP:", getattr(data, "status", data), getattr(data, "record_id", ""),
              f"| entries={plan_count()}")
        try:
            await client.call_tool("save_skill_gap", {
                "candidate_id": "C001", "job_id": "J003", "skill": "Git",
                "gap_type": "weakness", "rationale": RATIONALE,
            })
            print("FAIL  invalid gap_type accepted over MCP")
        except ToolError as e:
            print("OK    invalid gap_type rejected over MCP:", str(e)[:120])


if __name__ == "__main__":
    try:
        check_logic()
        asyncio.run(check_mcp())
        real = json.loads((REAL_DATA / "career_plan.json").read_text(encoding="utf-8"))["entries"]
        print(f"\nReal data/career_plan.json still has {len(real)} entries (should be 0).")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
