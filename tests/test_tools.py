"""CareerLens test suite.

Layers:
  1. Logic tests for the Data, Action, and Orchestration tools.
  2. MCP-layer tests through FastMCP's in-memory client.
  3. Integration: a real HTTP server called through the OpenAI Agents SDK MCP client.
     An optional live LLM round trip runs only when RUN_LLM_TESTS=1.
"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from server import careerlens as core
from server.mcp_server import mcp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REASON = "Candidate explicitly lists Docker under known gaps."


def plan_entries(data_dir: Path) -> list:
    return json.loads((data_dir / "career_plan.json").read_text(encoding="utf-8"))["entries"]


# ===========================================================================
# 1a. DATA tool: get_job_requirements
# ===========================================================================
def test_data_valid_job_returns_structured_requirements(data_dir):
    job = core.get_job_requirements("J003")
    assert job.title == "AI Engineer Intern"
    assert [s.skill for s in job.required_skills] == ["Python", "PyTorch", "Docker", "Git", "REST APIs"]
    assert job.min_experience_months == 3


def test_data_job_id_is_case_insensitive(data_dir):
    assert core.get_job_requirements(" j001 ").job_id == "J001"


def test_data_unknown_job_raises_clear_error(data_dir):
    with pytest.raises(core.JobNotFoundError, match="Available job IDs: J001, J002, J003, J004, J005"):
        core.get_job_requirements("J999")


def test_data_empty_job_id_rejected(data_dir):
    with pytest.raises(core.InvalidInputError):
        core.get_job_requirements("   ")


def test_data_missing_dataset_raises_storage_error(data_dir):
    (data_dir / "jobs.json").unlink()
    with pytest.raises(core.StorageError, match="not found"):
        core.get_job_requirements("J001")


def test_data_corrupt_dataset_raises_storage_error(data_dir):
    (data_dir / "jobs.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(core.StorageError, match="not valid JSON"):
        core.get_job_requirements("J001")


# ===========================================================================
# 1b. ACTION tool: save_skill_gap
# ===========================================================================
def test_action_valid_save_writes_one_entry(data_dir):
    r = core.save_skill_gap("C001", "J003", "docker", "missing_skill", REASON)
    assert r.status == "saved" and r.record_id.startswith("GAP-")
    entries = plan_entries(data_dir)
    assert len(entries) == 1
    assert entries[0]["record_id"] == r.record_id
    assert entries[0]["skill"] == "Docker"  # stored with the job's spelling


def test_action_duplicate_is_not_written_twice(data_dir):
    first = core.save_skill_gap("C001", "J003", "Docker", "missing_skill", REASON)
    second = core.save_skill_gap("C001", "J003", "Docker", "missing_skill", REASON)
    assert second.status == "already_exists" and second.record_id == first.record_id
    assert len(plan_entries(data_dir)) == 1


@pytest.mark.parametrize(
    "args, error",
    [
        (("C001", "J003", "Docker", "weakness", REASON), core.InvalidInputError),       # invalid gap type
        (("C001", "J003", "  ", "missing_skill", REASON), core.InvalidInputError),      # empty skill
        (("C001", "J003", "Docker", "missing_skill", "no"), core.InvalidInputError),    # rationale too short
        (("C001", "J003", "Excel", "missing_skill", REASON), core.InvalidInputError),   # skill not in job
        (("C001", "J999", "Docker", "missing_skill", REASON), core.JobNotFoundError),   # unknown job
        (("C999", "J003", "Docker", "missing_skill", REASON), core.CandidateNotFoundError),
    ],
    ids=["invalid_gap_type", "empty_skill", "short_rationale", "skill_not_in_job", "unknown_job", "unknown_candidate"],
)
def test_action_invalid_input_rejected_and_nothing_saved(data_dir, args, error):
    with pytest.raises(error):
        core.save_skill_gap(*args)
    assert plan_entries(data_dir) == []


def test_action_missing_plan_file_is_created(data_dir):
    (data_dir / "career_plan.json").unlink()
    core.save_skill_gap("C001", "J003", "Docker", "missing_skill", REASON)
    assert len(plan_entries(data_dir)) == 1


def test_action_corrupt_plan_file_is_never_overwritten(data_dir):
    plan = data_dir / "career_plan.json"
    plan.write_text("{ broken", encoding="utf-8")
    with pytest.raises(core.StorageError):
        core.save_skill_gap("C001", "J003", "Docker", "missing_skill", REASON)
    assert plan.read_text(encoding="utf-8") == "{ broken"


def test_action_concurrent_saves_lose_nothing_and_never_duplicate(data_dir):
    same = ("C001", "J003", "Docker", "missing_skill", REASON)
    different = [("C001", "J003", skill, "development_goal", f"Candidate wants to develop {skill}.")
                 for skill in ("Python", "PyTorch", "Git", "REST APIs", "FastAPI", "Linux")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda a: core.save_skill_gap(*a), [same] * 8 + different))
    assert sum(r.status == "saved" for r in results[:8]) == 1
    assert len({r.record_id for r in results[:8]}) == 1
    assert len(plan_entries(data_dir)) == 1 + len(different)


# ===========================================================================
# 1c. ORCHESTRATION tool: investigate_job_fit
# ===========================================================================
def labels(report) -> dict:
    return {a.skill: a.label for a in report.skills}


def test_orchestration_report_has_evidence_and_all_labels(data_dir):
    r = core.investigate_job_fit("C001", "J003")
    assert labels(r) == {
        "Python": "supported_match", "PyTorch": "possible_match", "Docker": "skill_gap",
        "Git": "possible_match", "REST APIs": "missing_evidence",
        "FastAPI": "missing_evidence", "Linux": "missing_evidence",
    }
    python = next(a for a in r.skills if a.skill == "Python")
    assert any("Student Performance Prediction" in e for e in python.evidence)
    assert r.required_coverage_score == 40.0
    assert "not the probability of being hired" in r.score_explanation
    assert r.experience.status == "meets"
    assert r.next_steps


def test_orchestration_save_gap_false_writes_nothing(data_dir):
    r = core.investigate_job_fit("C001", "J003", save_gap=False)
    assert r.saved_entries == [] and plan_entries(data_dir) == []


def test_orchestration_save_gap_true_saves_only_justified_gaps(data_dir):
    r = core.investigate_job_fit("C001", "J003", save_gap=True)
    assert [(o.skill, o.status) for o in r.saved_entries] == [("Docker", "saved")]
    entries = plan_entries(data_dir)
    assert [(e["skill"], e["gap_type"]) for e in entries] == [("Docker", "missing_skill")]
    # Missing-evidence skills (REST APIs, FastAPI, Linux) must never be saved as gaps.


def test_orchestration_repeat_save_reports_already_exists(data_dir):
    core.investigate_job_fit("C001", "J003", save_gap=True)
    r = core.investigate_job_fit("C001", "J003", save_gap=True)
    assert r.saved_entries[0].status == "already_exists"
    assert len(plan_entries(data_dir)) == 1


@pytest.mark.parametrize("cid, jid, error", [
    ("C001", "J999", core.JobNotFoundError),
    ("C999", "J003", core.CandidateNotFoundError),
])
def test_orchestration_failed_retrieval_stops_and_writes_nothing(data_dir, cid, jid, error):
    with pytest.raises(error):
        core.investigate_job_fit(cid, jid, save_gap=True)
    assert plan_entries(data_dir) == []


def test_orchestration_incomplete_profile_reports_limitations(data_dir):
    r = core.investigate_job_fit("C003", "J001", save_gap=True)
    assert r.profile_limitations
    assert r.label_counts["supported_match"] == 0  # no invented qualifications
    assert r.saved_entries == [] and plan_entries(data_dir) == []
    assert "nothing was saved" in r.save_note


def test_orchestration_explicit_gap_outranks_related_evidence(data_dir):
    # Omar lists Tableau as a known gap even though he uses the related Power BI.
    r = core.investigate_job_fit("C002", "J002")
    assert labels(r)["Tableau"] == "skill_gap"


def test_orchestration_malformed_candidate_fails_clearly(data_dir):
    path = data_dir / "candidates.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["candidates"][0]["experience"][0]["months"] = "four"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(core.StorageError, match="malformed"):
        core.investigate_job_fit("C001", "J003", save_gap=True)
    assert plan_entries(data_dir) == []


def test_orchestration_storage_failure_is_reported_not_hidden(data_dir):
    (data_dir / "career_plan.json").write_text("{ broken", encoding="utf-8")
    r = core.investigate_job_fit("C001", "J003", save_gap=True)
    assert r.saved_entries[0].status == "failed"
    assert "failed 1" in r.save_note


# ===========================================================================
# 2. MCP layer (FastMCP in-memory client)
# ===========================================================================
def test_mcp_exposes_exactly_three_tools():
    async def run():
        async with Client(mcp) as client:
            return sorted(t.name for t in await client.list_tools())
    assert asyncio.run(run()) == ["get_job_requirements", "investigate_job_fit", "save_skill_gap"]


def test_mcp_tools_have_descriptions_and_typed_inputs():
    async def run():
        async with Client(mcp) as client:
            return await client.list_tools()
    for tool in asyncio.run(run()):
        assert tool.description and tool.description.startswith("[")
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema")
        assert schema["properties"]


def test_mcp_unexpected_errors_are_masked(monkeypatch):
    def boom(job_id):
        raise RuntimeError("secret internal path C:/hidden/db")
    monkeypatch.setattr(core, "get_job_requirements", boom)

    async def run():
        async with Client(mcp) as client:
            with pytest.raises(ToolError) as info:
                await client.call_tool("get_job_requirements", {"job_id": "J001"})
            return str(info.value)
    assert "secret internal path" not in asyncio.run(run())


def test_mcp_round_trip_and_error_reach_client(data_dir):
    async def run():
        async with Client(mcp) as client:
            ok = await client.call_tool("investigate_job_fit", {"candidate_id": "C001", "job_id": "J003"})
            with pytest.raises(ToolError, match="No job with job_id 'J999'"):
                await client.call_tool("get_job_requirements", {"job_id": "J999"})
            with pytest.raises(ToolError):
                await client.call_tool("save_skill_gap", {
                    "candidate_id": "C001", "job_id": "J003", "skill": "Docker",
                    "gap_type": "weakness", "rationale": REASON})
            return ok
    result = asyncio.run(run())
    assert result.structured_content["required_coverage_score"] == 40.0
    assert plan_entries(data_dir) == []


# ===========================================================================
# 3. Integration: real HTTP server + OpenAI Agents SDK MCP client
# ===========================================================================
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def http_server(data_dir):
    port = _free_port()
    env = {**os.environ, "CAREERLENS_DATA_DIR": str(data_dir), "CAREERLENS_PORT": str(port),
           "FASTMCP_CHECK_FOR_UPDATES": "off"}
    proc = subprocess.Popen([sys.executable, "-m", "server.mcp_server"], cwd=PROJECT_ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("MCP server did not start")
    yield f"http://127.0.0.1:{port}/mcp"
    proc.terminate()
    proc.wait(timeout=10)


def _is_error(result) -> bool:
    value = getattr(result, "is_error", None)
    return bool(result.isError if value is None else value)


def test_integration_agents_sdk_client_round_trip(http_server, data_dir):
    from agents.mcp import MCPServerStreamableHttp

    async def run():
        server = MCPServerStreamableHttp(params={"url": http_server, "timeout": 10}, name="CareerLens")
        await server.connect()
        try:
            names = sorted(t.name for t in await server.list_tools())
            report = await server.call_tool("investigate_job_fit",
                                            {"candidate_id": "C001", "job_id": "J003", "save_gap": True})
            failure = await server.call_tool("get_job_requirements", {"job_id": "J999"})
            return names, report, failure
        finally:
            await server.cleanup()

    names, report, failure = asyncio.run(run())
    assert names == ["get_job_requirements", "investigate_job_fit", "save_skill_gap"]
    assert not _is_error(report)
    assert [e["skill"] for e in plan_entries(data_dir)] == ["Docker"]
    assert _is_error(failure) and "Available job IDs" in failure.content[0].text


@pytest.mark.skipif(os.getenv("RUN_LLM_TESTS") != "1" or not os.getenv("OPENAI_API_KEY"),
                    reason="Live LLM test: set RUN_LLM_TESTS=1 and OPENAI_API_KEY to run.")
def test_integration_live_agent_uses_toolkit(http_server, data_dir):
    from agents import Runner
    from agents.items import ToolCallItem
    from agents.mcp import MCPServerStreamableHttp

    import agent as host

    async def run():
        server = MCPServerStreamableHttp(params={"url": http_server, "timeout": 10}, name="CareerLens",
                                         failure_error_function=host.show_tool_error)
        await server.connect()
        try:
            return await Runner.run(host.build_agent(server), "Analyze how candidate C001 fits job J003.")
        finally:
            await server.cleanup()

    result = asyncio.run(run())
    called = [getattr(i.raw_item, "name", None) for i in result.new_items if isinstance(i, ToolCallItem)]
    assert "investigate_job_fit" in called
    assert "Docker" in str(result.final_output)
    assert plan_entries(data_dir) == []  # no save was requested
