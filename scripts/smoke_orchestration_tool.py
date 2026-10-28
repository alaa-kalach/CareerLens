"""Smoke check for the Orchestration tool. Uses a temporary copy of the data,
so data/career_plan.json is never touched. Run from the project root:
    python -m scripts.smoke_orchestration_tool
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


def plan_count() -> int:
    return len(json.loads((TMP / "career_plan.json").read_text(encoding="utf-8"))["entries"])


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'OK  ' if ok else 'FAIL'}  {label}{': ' + detail if detail else ''}")


def check_logic() -> None:
    print("== 1. Logic layer ==")
    r = core.investigate_job_fit("C001", "J003")
    print(f"      {r.candidate_name} vs {r.job_title}: coverage {r.required_coverage_score}%")
    for a in r.skills:
        print(f"      - {a.skill:<10} {a.requirement:<9} {a.label:<17} {'; '.join(a.evidence)}")
    print(f"      experience: {r.experience.status} ({r.experience.explanation})")
    check("all four labels present", all(r.label_counts[l] > 0 for l in core.LABEL_ORDER), str(r.label_counts))
    check("save_gap=False writes nothing", plan_count() == 0, r.save_note)

    r = core.investigate_job_fit("C001", "J003", save_gap=True)
    check("save_gap=True saves the Docker gap", plan_count() == 1 and r.saved_entries[0].status == "saved",
          f"{[(o.skill, o.status, o.record_id) for o in r.saved_entries]}")
    r = core.investigate_job_fit("C001", "J003", save_gap=True)
    check("repeat save does not duplicate", plan_count() == 1 and r.saved_entries[0].status == "already_exists",
          r.save_note)

    r = core.investigate_job_fit("C003", "J001", save_gap=True)
    check("incomplete profile: limitations reported", len(r.profile_limitations) > 0, "; ".join(r.profile_limitations))
    check("incomplete profile: no invented matches", r.label_counts["supported_match"] == 0, str(r.label_counts))
    check("missing evidence is never saved", plan_count() == 1, r.save_note)

    for label, args in [("unknown job", ("C001", "J999", True)), ("unknown candidate", ("C999", "J003", True))]:
        try:
            core.investigate_job_fit(*args)
            check(label, False, "no error raised")
        except core.CareerLensError as e:
            check(f"{label} stops cleanly, nothing written", plan_count() == 1, f"{type(e).__name__}: {e}")


async def check_mcp() -> None:
    print("\n== 2. Through MCP (in-memory client) ==")
    async with Client(mcp) as client:
        tools = [t.name for t in await client.list_tools()]
        check("exactly three tools registered", len(tools) == 3, str(tools))
        result = await client.call_tool("investigate_job_fit", {"candidate_id": "C002", "job_id": "J002"})
        data = getattr(result, "data", result)
        check("report over MCP", hasattr(data, "required_coverage_score"),
              f"coverage={getattr(data, 'required_coverage_score', '?')}% labels={getattr(data, 'label_counts', '?')}")
        try:
            await client.call_tool("investigate_job_fit", {"candidate_id": "C001", "job_id": "J999", "save_gap": True})
            check("unknown job over MCP", False, "no error raised")
        except ToolError as e:
            check("unknown job rejected over MCP", plan_count() == 1, str(e))


if __name__ == "__main__":
    try:
        check_logic()
        asyncio.run(check_mcp())
        real = json.loads((REAL_DATA / "career_plan.json").read_text(encoding="utf-8"))["entries"]
        print(f"\nReal data/career_plan.json still has {len(real)} entries (should be 0).")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
