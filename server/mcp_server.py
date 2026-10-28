"""CareerLens MCP server.

Exposes exactly three MCP tools:
  - DATA:          get_job_requirements
  - ACTION:        save_skill_gap
  - ORCHESTRATION: investigate_job_fit

Run (from the project root):  python -m server.mcp_server
"""
import logging
import os
from typing import Annotated

from fastmcp import FastMCP
from mcp_types import ToolAnnotations
from pydantic import Field

from server import careerlens as core

logger = logging.getLogger("careerlens.server")

# MCP tool annotations: protocol-level hints that tell clients how each tool behaves.
# All three tools only touch local CareerLens data, so none is open-world.
DATA_ANNOTATIONS = ToolAnnotations(
    title="Get job requirements",
    read_only_hint=True,       # never modifies anything
    open_world_hint=False,
)
ACTION_ANNOTATIONS = ToolAnnotations(
    title="Save skill gap",
    read_only_hint=False,      # writes to career_plan.json
    destructive_hint=False,    # additive only: never overwrites or deletes entries
    idempotent_hint=True,      # repeating the same call returns 'already_exists'
    open_world_hint=False,
)
ORCHESTRATION_ANNOTATIONS = ToolAnnotations(
    title="Investigate job fit",
    read_only_hint=False,      # may write, but only when save_gap=true
    destructive_hint=False,    # saves go through the additive Action logic
    idempotent_hint=True,      # a repeated save reports 'already_exists'
    open_world_hint=False,
)

mcp = FastMCP(
    name="CareerLens",
    instructions=(
        "CareerLens compares a candidate profile with a job posting and reports "
        "evidence-based matches, missing evidence, and skill gaps."
    ),
    # CareerLens errors are ToolErrors and always reach the agent with their message.
    # Any unexpected exception (a bug) is masked so internal details never leak to clients.
    mask_error_details=True,
)


@mcp.tool(annotations=DATA_ANNOTATIONS)
def get_job_requirements(
    job_id: Annotated[str, Field(description="Job identifier, e.g. 'J003'. Case-insensitive.")],
) -> core.JobRequirements:
    """[DATA] Retrieve the requirements of one job posting.

    Returns the job title, description, required skills, preferred skills
    (each with related skills that count as partial evidence), minimum months
    of experience, and experience requirements.

    Read-only: does not assess any candidate and does not write anything.
    Fails with a clear error if job_id is empty or unknown; the error lists the
    valid job IDs.
    """
    logger.info("DATA tool called: get_job_requirements(job_id=%r)", job_id)
    try:
        result = core.get_job_requirements(job_id)
    except core.CareerLensError as e:
        logger.warning("DATA tool failed: %s", e)
        raise
    logger.info("DATA tool returned job %s (%s)", result.job_id, result.title)
    return result


@mcp.tool(annotations=ACTION_ANNOTATIONS)
def save_skill_gap(
    candidate_id: Annotated[str, Field(description="Candidate identifier, e.g. 'C001'.")],
    job_id: Annotated[str, Field(description="Job the entry relates to, e.g. 'J003'.")],
    skill: Annotated[str, Field(description="A required or preferred skill of that job, e.g. 'Docker'.")],
    gap_type: Annotated[
        core.GapType,
        Field(
            description=(
                "missing_skill = available evidence shows the candidate lacks the skill; "
                "insufficient_evidence = the profile does not show whether they have it; "
                "development_goal = a skill the candidate has chosen to develop."
            )
        ),
    ],
    rationale: Annotated[
        str, Field(description="Why this entry is justified, citing profile evidence (at least 10 characters).")
    ],
) -> core.SaveSkillGapResult:
    """[ACTION] Save ONE skill-gap or development entry to the local career plan.

    Validates every argument, checks that the candidate and job exist and that
    the skill belongs to that job, then writes a single entry and returns its
    record_id. Returns status 'already_exists' (and writes nothing) if an
    identical entry is already saved.

    Does not run any matching. Only call when the user asks to save an entry.
    Fails with a clear error on invalid input or storage problems; nothing is
    written on failure.
    """
    logger.info(
        "ACTION tool called: save_skill_gap(candidate_id=%r, job_id=%r, skill=%r, gap_type=%r)",
        candidate_id, job_id, skill, gap_type,
    )
    try:
        result = core.save_skill_gap(candidate_id, job_id, skill, gap_type, rationale)
    except core.CareerLensError as e:
        logger.warning("ACTION tool failed: %s", e)
        raise
    logger.info("ACTION tool result: %s (%s)", result.status, result.record_id)
    return result


@mcp.tool(annotations=ORCHESTRATION_ANNOTATIONS)
def investigate_job_fit(
    candidate_id: Annotated[str, Field(description="Candidate identifier, e.g. 'C001'.")],
    job_id: Annotated[str, Field(description="Job identifier, e.g. 'J003'.")],
    save_gap: Annotated[
        bool,
        Field(description="If true, save each justified skill gap to the career plan. Default false (read-only)."),
    ] = False,
) -> core.JobFitReport:
    """[ORCHESTRATION] Produce an evidence-based fit report for one candidate and one job.

    Retrieves the job requirements (Data logic), compares them with the
    candidate profile, and labels every required and preferred skill as
    supported_match, possible_match, skill_gap, or missing_evidence, with the
    evidence behind each label. Also assesses experience, lists profile
    limitations, and suggests next steps. The coverage score is transparent and
    is NOT a hiring probability.

    Only when save_gap is true, saves each explicit skill gap through the Action
    logic and reports every save outcome truthfully. Missing evidence is never
    saved as a gap. If the job or candidate cannot be found, it stops with a
    clear error and writes nothing.
    """
    logger.info(
        "ORCHESTRATION tool called: investigate_job_fit(candidate_id=%r, job_id=%r, save_gap=%r)",
        candidate_id, job_id, save_gap,
    )
    try:
        report = core.investigate_job_fit(candidate_id, job_id, save_gap)
    except core.CareerLensError as e:
        logger.warning("ORCHESTRATION tool failed (nothing written): %s", e)
        raise
    logger.info(
        "ORCHESTRATION tool result: %s vs %s | coverage=%s%% | labels=%s | %s",
        report.candidate_id, report.job_id, report.required_coverage_score, report.label_counts, report.save_note,
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    try:
        mcp.run(transport="http", host="127.0.0.1", port=int(os.getenv("CAREERLENS_PORT", "8000")))
    except KeyboardInterrupt:
        logger.info("Server stopped.")
