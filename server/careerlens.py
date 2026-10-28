"""Core CareerLens logic, independent of MCP.

server/mcp_server.py wraps these functions as MCP tools. Keeping the logic here
lets the Orchestration tool reuse the Data and Action logic directly, and lets
tests exercise everything without running a server.
"""
from __future__ import annotations

import contextlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, get_args

from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def data_dir() -> Path:
    """Folder holding the JSON datasets. Override with CAREERLENS_DATA_DIR (used by tests)."""
    return Path(os.environ.get("CAREERLENS_DATA_DIR", DEFAULT_DATA_DIR))


# ---------------------------------------------------------------------------
# Errors. All subclass ToolError so the message always reaches the agent.
# ---------------------------------------------------------------------------
class CareerLensError(ToolError):
    """Base class for all predictable CareerLens failures."""


class InvalidInputError(CareerLensError):
    """An argument is empty, malformed, or not an allowed value."""


class JobNotFoundError(CareerLensError):
    """The requested job_id does not exist in the job dataset."""


class CandidateNotFoundError(CareerLensError):
    """The requested candidate_id does not exist in the candidate dataset."""


class StorageError(CareerLensError):
    """A data file is missing, unreadable, malformed, or could not be written."""


# ---------------------------------------------------------------------------
# Typed models
# ---------------------------------------------------------------------------
class SkillRequirement(BaseModel):
    skill: str = Field(description="Name of the required or preferred skill.")
    related: list[str] = Field(
        default_factory=list,
        description="Related skills that count only as partial evidence for this requirement.",
    )


class JobRequirements(BaseModel):
    job_id: str
    title: str
    description: str
    required_skills: list[SkillRequirement]
    preferred_skills: list[SkillRequirement] = Field(default_factory=list)
    min_experience_months: int = Field(0, ge=0)
    experience_requirements: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------
def _load_json(filename: str) -> Any:
    path = data_dir() / filename
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise StorageError(f"Data file '{filename}' not found in {path.parent}.") from None
    except json.JSONDecodeError as e:
        raise StorageError(f"Data file '{filename}' is not valid JSON (line {e.lineno}): {e.msg}.") from None
    except OSError as e:
        raise StorageError(f"Could not read data file '{filename}': {e.strerror}.") from None


def _load_records(filename: str, key: str) -> list[dict]:
    data = _load_json(filename)
    if not isinstance(data, dict) or not isinstance(data.get(key), list):
        raise StorageError(f"Data file '{filename}' must be an object with a '{key}' list.")
    return data[key]


def _normalize_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"{field_name} must be a non-empty string.")
    return value.strip().upper()


# ---------------------------------------------------------------------------
# DATA tool logic
# ---------------------------------------------------------------------------
def get_job_requirements(job_id: str) -> JobRequirements:
    """Return the requirements of one job. Read-only: never assesses or writes anything."""
    jid = _normalize_id(job_id, "job_id")
    jobs = _load_records("jobs.json", "jobs")

    for job in jobs:
        if str(job.get("job_id", "")).strip().upper() == jid:
            try:
                return JobRequirements.model_validate(job)
            except ValidationError as e:
                raise StorageError(
                    f"Job record '{jid}' is malformed ({e.error_count()} validation error(s))."
                ) from None

    available = ", ".join(sorted(str(j.get("job_id", "?")) for j in jobs)) or "none"
    raise JobNotFoundError(f"No job with job_id '{jid}'. Available job IDs: {available}.")


# ---------------------------------------------------------------------------
# Candidate lookup (internal helper, deliberately NOT an exposed MCP tool)
# ---------------------------------------------------------------------------
def find_candidate(candidate_id: str) -> dict:
    """Return the raw candidate record. Used by the Action and Orchestration logic."""
    cid = _normalize_id(candidate_id, "candidate_id")
    candidates = _load_records("candidates.json", "candidates")
    for c in candidates:
        if str(c.get("candidate_id", "")).strip().upper() == cid:
            return c
    available = ", ".join(sorted(str(c.get("candidate_id", "?")) for c in candidates)) or "none"
    raise CandidateNotFoundError(f"No candidate with candidate_id '{cid}'. Available candidate IDs: {available}.")


# ---------------------------------------------------------------------------
# ACTION tool logic
# ---------------------------------------------------------------------------
GapType = Literal["missing_skill", "insufficient_evidence", "development_goal"]
ALLOWED_GAP_TYPES: tuple[str, ...] = get_args(GapType)
MIN_RATIONALE_CHARS = 10
CAREER_PLAN_FILE = "career_plan.json"


class CareerPlanEntry(BaseModel):
    record_id: str
    candidate_id: str
    job_id: str
    skill: str
    gap_type: GapType
    rationale: str
    created_at: str = Field(description="UTC timestamp in ISO 8601 format.")


class SaveSkillGapResult(BaseModel):
    status: Literal["saved", "already_exists"] = Field(
        description="'saved' = a new entry was written; 'already_exists' = an identical entry was found and nothing was written."
    )
    record_id: str
    message: str
    entry: CareerPlanEntry


def _read_career_plan() -> dict:
    path = data_dir() / CAREER_PLAN_FILE
    if not path.exists():
        return {"entries": []}  # a missing plan simply starts empty; it is created on first save
    data = _load_json(CAREER_PLAN_FILE)
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise StorageError(
            f"'{CAREER_PLAN_FILE}' is malformed (expected an object with an 'entries' list); "
            "refusing to overwrite it."
        )
    return data


def _write_career_plan(plan: dict) -> None:
    path = data_dir() / CAREER_PLAN_FILE
    tmp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)  # atomic: the old file stays intact if anything above fails
    except OSError as e:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise StorageError(f"Could not write '{CAREER_PLAN_FILE}': {e.strerror}. Nothing was saved.") from None


def save_skill_gap(
    candidate_id: str, job_id: str, skill: str, gap_type: str, rationale: str
) -> SaveSkillGapResult:
    """Validate and save ONE career-plan entry. Never runs any matching."""
    # 1. Cheap input checks (no I/O)
    if not isinstance(skill, str) or not skill.strip():
        raise InvalidInputError("skill must be a non-empty string.")
    if gap_type not in ALLOWED_GAP_TYPES:
        raise InvalidInputError(
            f"Invalid gap_type '{gap_type}'. Allowed values: {', '.join(ALLOWED_GAP_TYPES)}."
        )
    rationale = rationale.strip() if isinstance(rationale, str) else ""
    if len(rationale) < MIN_RATIONALE_CHARS:
        raise InvalidInputError(
            f"rationale must explain why this entry is justified (at least {MIN_RATIONALE_CHARS} characters)."
        )

    # 2. Reference checks
    candidate = find_candidate(candidate_id)
    job = get_job_requirements(job_id)
    job_skills = {s.skill.lower(): s.skill for s in job.required_skills + job.preferred_skills}
    canonical_skill = job_skills.get(skill.strip().lower())
    if canonical_skill is None:
        raise InvalidInputError(
            f"'{skill.strip()}' is not a required or preferred skill of {job.job_id} ({job.title}). "
            f"Valid skills: {', '.join(job_skills.values())}."
        )
    cid = str(candidate["candidate_id"]).upper()

    # 3. Duplicate check: never write the same entry twice
    plan = _read_career_plan()
    for raw in plan["entries"]:
        if (
            str(raw.get("candidate_id", "")).upper() == cid
            and str(raw.get("job_id", "")).upper() == job.job_id
            and str(raw.get("skill", "")).lower() == canonical_skill.lower()
            and raw.get("gap_type") == gap_type
        ):
            try:
                existing = CareerPlanEntry.model_validate(raw)
            except ValidationError:
                raise StorageError(f"Existing entry in '{CAREER_PLAN_FILE}' is malformed.") from None
            return SaveSkillGapResult(
                status="already_exists",
                record_id=existing.record_id,
                message=f"An identical entry already exists ({existing.record_id}); nothing was written.",
                entry=existing,
            )

    # 4. Write
    entry = CareerPlanEntry(
        record_id=f"GAP-{uuid.uuid4().hex[:8].upper()}",
        candidate_id=cid,
        job_id=job.job_id,
        skill=canonical_skill,
        gap_type=gap_type,
        rationale=rationale,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    plan["entries"].append(entry.model_dump())
    _write_career_plan(plan)
    return SaveSkillGapResult(
        status="saved",
        record_id=entry.record_id,
        message=f"Saved {gap_type} entry for '{canonical_skill}' ({cid} / {job.job_id}).",
        entry=entry,
    )
