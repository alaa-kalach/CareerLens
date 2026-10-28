"""Core CareerLens logic, independent of MCP.

server/mcp_server.py wraps these functions as MCP tools. Keeping the logic here
lets the Orchestration tool reuse the Data and Action logic directly, and lets
tests exercise everything without running a server.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
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


# Typed models
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


# Storage helpers
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
    if not all(isinstance(record, dict) for record in data[key]):
        raise StorageError(f"Data file '{filename}' contains a '{key}' entry that is not an object.")
    return data[key]


def _normalize_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"{field_name} must be a non-empty string.")
    return value.strip().upper()


# DATA tool logic
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


# Candidate lookup (internal helper, deliberately NOT an exposed MCP tool)
def find_candidate(candidate_id: str) -> dict:
    """Return the raw candidate record. Used by the Action and Orchestration logic."""
    cid = _normalize_id(candidate_id, "candidate_id")
    candidates = _load_records("candidates.json", "candidates")
    for c in candidates:
        if str(c.get("candidate_id", "")).strip().upper() == cid:
            return c
    available = ", ".join(sorted(str(c.get("candidate_id", "?")) for c in candidates)) or "none"
    raise CandidateNotFoundError(f"No candidate with candidate_id '{cid}'. Available candidate IDs: {available}.")


# ACTION tool logic
GapType = Literal["missing_skill", "insufficient_evidence", "development_goal"]
ALLOWED_GAP_TYPES: tuple[str, ...] = get_args(GapType)
MIN_RATIONALE_CHARS = 10
CAREER_PLAN_FILE = "career_plan.json"
# FastMCP runs sync tools in a threadpool, so concurrent saves are possible. The lock makes the
# read -> duplicate check -> write sequence atomic, so no entry is lost or duplicated.
_PLAN_LOCK = threading.Lock()


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
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("entries"), list)
        or not all(isinstance(entry, dict) for entry in data["entries"])
    ):
        raise StorageError(
            f"'{CAREER_PLAN_FILE}' is malformed (expected an object with an 'entries' list of objects); "
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
        os.replace(tmp, path)  
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

    with _PLAN_LOCK:
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


# ORCHESTRATION tool logic
MatchLabel = Literal["supported_match", "possible_match", "skill_gap", "missing_evidence"]
LABEL_ORDER: tuple[str, ...] = get_args(MatchLabel)


class SkillAssessment(BaseModel):
    skill: str
    requirement: Literal["required", "preferred"]
    label: MatchLabel
    evidence: list[str] = Field(default_factory=list, description="Where in the profile the label comes from.")
    explanation: str


class ExperienceAssessment(BaseModel):
    required_months: int
    evidenced_months: int
    status: Literal["meets", "below_minimum", "not_established", "not_required"]
    explanation: str


class SaveOutcome(BaseModel):
    skill: str
    gap_type: GapType
    status: Literal["saved", "already_exists", "failed"]
    record_id: str | None = None
    message: str


class JobFitReport(BaseModel):
    candidate_id: str
    candidate_name: str
    job_id: str
    job_title: str
    required_coverage_score: float = Field(
        description="Percent of required skills covered (supported = 1, possible = 0.5). Not a hiring probability."
    )
    score_explanation: str
    label_counts: dict[str, int]
    skills: list[SkillAssessment]
    experience: ExperienceAssessment
    profile_limitations: list[str]
    next_steps: list[str]
    save_requested: bool
    saved_entries: list[SaveOutcome]
    save_note: str


def _evidence_index(candidate: dict) -> dict[str, list[str]]:
    """Map lowercase skill -> descriptions of the projects/roles where it was used."""
    index: dict[str, list[str]] = {}
    for p in candidate.get("projects") or []:
        where = f"project '{p.get('name', '?')}' ({p.get('project_id', '?')})"
        for s in p.get("skills_used") or []:
            index.setdefault(str(s).lower(), []).append(where)
    for e in candidate.get("experience") or []:
        where = f"role '{e.get('role', '?')}' at {e.get('organization', '?')} ({e.get('experience_id', '?')})"
        for s in e.get("skills_used") or []:
            index.setdefault(str(s).lower(), []).append(where)
    return index


def _assess_skill(req: SkillRequirement, requirement: str, candidate: dict, evidence: dict[str, list[str]]) -> SkillAssessment:
    key = req.skill.lower()
    listed = {str(s).lower() for s in candidate.get("skills") or []}
    known_gaps = {str(s).lower() for s in candidate.get("known_gaps") or []}

    if key in evidence:
        return SkillAssessment(
            skill=req.skill, requirement=requirement, label="supported_match", evidence=evidence[key],
            explanation=f"{req.skill} is used in {len(evidence[key])} project(s)/role(s).",
        )
    if key in known_gaps:
        return SkillAssessment(
            skill=req.skill, requirement=requirement, label="skill_gap",
            evidence=["listed under known_gaps in the profile"],
            explanation=f"The candidate has stated they do not currently have {req.skill}.",
        )
    related_hits = [r for r in req.related if r.lower() in evidence or r.lower() in listed]
    if related_hits:
        ev = []
        for r in related_hits:
            ev += [f"related skill {r} used in {w}" for w in evidence.get(r.lower(), [])] or [f"related skill {r} listed in skills"]
        return SkillAssessment(
            skill=req.skill, requirement=requirement, label="possible_match", evidence=ev,
            explanation=f"No direct evidence of {req.skill}, but related experience exists ({', '.join(related_hits)}).",
        )
    if key in listed:
        return SkillAssessment(
            skill=req.skill, requirement=requirement, label="possible_match",
            evidence=["listed in skills section only"],
            explanation=f"{req.skill} is listed, but no project or role shows it being used.",
        )
    return SkillAssessment(
        skill=req.skill, requirement=requirement, label="missing_evidence",
        explanation=f"The profile does not show whether the candidate has {req.skill}. This is not treated as a gap.",
    )


def _assess_experience(job: JobRequirements, candidate: dict) -> ExperienceAssessment:
    roles = candidate.get("experience") or []
    months = sum(int(e.get("months") or 0) for e in roles)
    need = job.min_experience_months
    if need == 0:
        return ExperienceAssessment(required_months=0, evidenced_months=months, status="not_required",
                                    explanation="The job has no minimum experience requirement.")
    if not roles:
        return ExperienceAssessment(required_months=need, evidenced_months=0, status="not_established",
                                    explanation=f"The job asks for {need} months; the profile lists no experience, so this cannot be assessed.")
    if months >= need:
        return ExperienceAssessment(required_months=need, evidenced_months=months, status="meets",
                                    explanation=f"{months} months of listed experience meets the {need}-month minimum.")
    return ExperienceAssessment(required_months=need, evidenced_months=months, status="below_minimum",
                                explanation=f"{months} months of listed experience is below the {need}-month minimum.")


def _profile_limitations(candidate: dict) -> list[str]:
    notes = []
    if not candidate.get("skills"):
        notes.append("No skills are listed in the profile.")
    if not candidate.get("projects") and not candidate.get("experience"):
        notes.append("No projects or work experience are listed, so no skill can be marked as a supported match.")
    if not str(candidate.get("education") or "").strip():
        notes.append("Education is not specified.")
    return notes


def _next_steps(skills: list[SkillAssessment], experience: ExperienceAssessment) -> list[str]:
    steps = []
    for a in sorted(skills, key=lambda a: a.requirement != "required"):
        tag = "required" if a.requirement == "required" else "preferred"
        if a.label == "skill_gap":
            steps.append(f"Close the {tag} gap in {a.skill} with a small hands-on project.")
        elif a.label == "missing_evidence":
            steps.append(f"{a.skill} ({tag}) is not shown: if you have it, add a project or role that demonstrates it; otherwise consider learning it.")
        elif a.label == "possible_match" and a.evidence and a.evidence[0].startswith("related skill"):
            steps.append(f"Your related experience could transfer to {a.skill} ({tag}); a short project using {a.skill} would make it supported evidence.")
        elif a.label == "possible_match":
            steps.append(f"Describe where you used {a.skill} ({tag}) so the claim is backed by evidence.")
    if experience.status == "below_minimum":
        steps.append(f"Gain about {experience.required_months - experience.evidenced_months} more months of relevant experience.")
    elif experience.status == "not_established":
        steps.append("Add your work or project experience (with durations) to the profile.")
    return steps


def investigate_job_fit(candidate_id: str, job_id: str, save_gap: bool = False) -> JobFitReport:
    """Coordinate the workflow: Data logic -> candidate comparison -> (optional) Action logic."""
    # 1. Retrieve the job first. If this fails, the error propagates and nothing is written.
    job = get_job_requirements(job_id)
    # 2. Load the candidate (internal helper, not an exposed tool).
    candidate = find_candidate(candidate_id)
    cid = str(candidate["candidate_id"]).upper()

    # 3. Compare (a malformed profile is reported clearly instead of crashing)
    try:
        evidence = _evidence_index(candidate)
        skills = [_assess_skill(r, "required", candidate, evidence) for r in job.required_skills]
        skills += [_assess_skill(r, "preferred", candidate, evidence) for r in job.preferred_skills]
        experience = _assess_experience(job, candidate)
        limitations = _profile_limitations(candidate)
    except (TypeError, ValueError, AttributeError) as e:
        raise StorageError(f"Candidate record '{cid}' is malformed ({type(e).__name__}); nothing was saved.") from None

    required = [a for a in skills if a.requirement == "required"]
    points = sum(1.0 if a.label == "supported_match" else 0.5 if a.label == "possible_match" else 0.0 for a in required)
    score = round(100 * points / len(required), 1) if required else 0.0
    counts = {label: sum(a.label == label for a in skills) for label in LABEL_ORDER}

    # 4. Optionally save: only explicit skill gaps are justified entries
    outcomes: list[SaveOutcome] = []
    if not save_gap:
        note = "Saving was not requested; the career plan was not modified."
    else:
        gaps = [a for a in skills if a.label == "skill_gap"]
        for a in gaps:
            rationale = (f"{cid} lists {a.skill} under known gaps; it is a {a.requirement} skill for "
                         f"{job.job_id} ({job.title}).")
            try:
                r = save_skill_gap(cid, job.job_id, a.skill, "missing_skill", rationale)
                outcomes.append(SaveOutcome(skill=a.skill, gap_type="missing_skill", status=r.status,
                                            record_id=r.record_id, message=r.message))
            except CareerLensError as e:
                outcomes.append(SaveOutcome(skill=a.skill, gap_type="missing_skill", status="failed", message=str(e)))
        if not gaps:
            note = ("No skill gap is justified by the profile (missing evidence is not treated as a gap); "
                    "nothing was saved.")
        else:
            n = {s: sum(o.status == s for o in outcomes) for s in ("saved", "already_exists", "failed")}
            note = f"Saved {n['saved']}, already existed {n['already_exists']}, failed {n['failed']}."

    return JobFitReport(
        candidate_id=cid,
        candidate_name=str(candidate.get("name") or cid),
        job_id=job.job_id,
        job_title=job.title,
        required_coverage_score=score,
        score_explanation=(f"{points:g} of {len(required)} required skills covered (supported = 1, possible = 0.5). "
                           "This measures evidence coverage, not the probability of being hired."),
        label_counts=counts,
        skills=skills,
        experience=experience,
        profile_limitations=limitations,
        next_steps=_next_steps(skills, experience),
        save_requested=save_gap,
        saved_entries=outcomes,
        save_note=note,
    )
