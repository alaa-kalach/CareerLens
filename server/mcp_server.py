"""CareerLens MCP server.

Exposes exactly three MCP tools:
  - DATA:          get_job_requirements
  - ACTION:        save_skill_gap        (added in Step 4)
  - ORCHESTRATION: investigate_job_fit   (added in Step 5)

Run (from the project root):  python -m server.mcp_server
"""
import logging
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from server import careerlens as core

logger = logging.getLogger("careerlens.server")

mcp = FastMCP(
    name="CareerLens",
    instructions=(
        "CareerLens compares a candidate profile with a job posting and reports "
        "evidence-based matches, missing evidence, and skill gaps."
    ),
)


@mcp.tool
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


@mcp.tool
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    try:
        mcp.run(transport="http", host="127.0.0.1", port=8000)
    except KeyboardInterrupt:
        logger.info("Server stopped.")
