# CareerLens

**An MCP toolkit for evidence-based CV-to-job matching and skill-gap analysis.**
Built with [FastMCP](https://gofastmcp.com) and the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/).

CareerLens compares a candidate profile with a job posting and explains, skill by skill, what the profile actually shows. It keeps two things apart that CV tools often mix up:

- a **skill gap**, where the profile gives a positive signal that the candidate lacks the skill, and
- **missing evidence**, where the CV simply does not show the skill.

A skill that is absent from a CV is never treated as proof that the candidate lacks it. With the user's approval, a justified gap can be saved to a local career plan.

---

## The three tools

The server exposes **exactly three MCP tools**, one from each category.

| Category | Tool | Arguments | Returns | Predictable failures |
|---|---|---|---|---|
| **Data** | `get_job_requirements` | `job_id: str` | `JobRequirements`: title, description, required and preferred skills (each with related skills), minimum months, experience requirements | Empty ID → `InvalidInputError`; unknown ID → `JobNotFoundError` listing the valid IDs; missing, corrupt or malformed dataset → `StorageError` |
| **Action** | `save_skill_gap` | `candidate_id: str`, `job_id: str`, `skill: str`, `gap_type: "missing_skill" \| "insufficient_evidence" \| "development_goal"`, `rationale: str` | `SaveSkillGapResult`: `status` (`saved` / `already_exists`), `record_id`, `message`, the stored `entry` | Empty skill, invalid gap type, rationale under 10 characters, skill not part of that job, unknown candidate or job → `InvalidInputError` / `*NotFoundError`; corrupt plan file or write failure → `StorageError`. **Nothing is written on any failure.** |
| **Orchestration** | `investigate_job_fit` | `candidate_id: str`, `job_id: str`, `save_gap: bool = False` | `JobFitReport`: a label and evidence for every skill, coverage score, experience check, profile limitations, next steps, and a truthful outcome for every save | Unknown job or candidate → stops before any comparison and writes nothing; malformed profile → `StorageError`; a failed save is reported as `failed` in the report, never hidden |

**Single-purpose boundaries:**
- The **Data** tool only *retrieves* job requirements. It never assesses a candidate and never writes.
- The **Action** tool only *validates and saves one entry*. It never runs any matching.
- The **Orchestration** tool *coordinates*. It retrieves the job through the Data logic, compares it with the profile, and saves through the Action logic only when `save_gap=true`. The candidate profile is loaded internally rather than through a fourth tool.

### MCP tool annotations

Each tool also declares standard MCP behavior hints, which clients receive over the protocol (`python agent.py --tools` shows them):

| Tool | read-only | destructive | idempotent | open-world |
|---|---|---|---|---|
| `get_job_requirements` | yes | n/a | n/a | no |
| `save_skill_gap` | no | no (additive only) | yes (duplicates return `already_exists`) | no |
| `investigate_job_fit` | no (writes only if `save_gap=true`) | no | yes | no |

### Matching labels

Each required and preferred skill gets exactly one label, checked in this order:

| Label | Rule |
|---|---|
| `supported_match` | The skill is used in a listed project or role. The evidence names that project or role. |
| `skill_gap` | The candidate lists the skill under `known_gaps`. This is the only negative signal. |
| `possible_match` | The skill is listed without demonstrated use, **or** a related skill is evidenced (e.g. TensorFlow for PyTorch). |
| `missing_evidence` | None of the above. This is **not** treated as a gap and is never saved automatically. |

Demonstrated use outranks self-reported doubt: if a skill appears both in a project and in `known_gaps`, it is labeled supported. An explicit known gap does outrank related evidence. For example, Omar's Tableau is a gap even though he uses Power BI.

The **coverage score** is the share of *required* skills with evidence (supported = 1, possible = 0.5). The report states that it measures evidence coverage and **is not a probability of being hired**.

---

## Architecture

```
 User
   |
   v
+------------------------------------------------------------------+
| agent.py                                                         |
| OpenAI Agents SDK agent  (MCP Host / Client)                     |
+------------------------------------------------------------------+
   |  MCP over Streamable HTTP  (http://127.0.0.1:8000/mcp)
   v
+------------------------------------------------------------------+
| server/mcp_server.py  -  FastMCP server, exactly three tools     |
|                                                                  |
| get_job_requirements    save_skill_gap     investigate_job_fit   |
|       (Data)              (Action)          (Orchestration)      |
+------------------------------------------------------------------+
        |                      |                     |
        |                      |       reuses the Data logic, then
        |                      |       the Action logic only if save_gap=true
        v                      v                     v
  data/jobs.json      data/career_plan.json    data/candidates.json
```

- **`agent.py`** is the MCP Host. An OpenAI Agents SDK `Agent` connects to the server with `MCPServerStreamableHttp` and routes each request to the right tool.
- **`server/mcp_server.py`** contains thin `@mcp.tool` wrappers with logging, annotations and error masking.
- **`server/careerlens.py`** holds all the logic as plain Python, independent of MCP. The Orchestration tool reuses the same Data and Action functions that the Data and Action tools wrap, so it never calls a decorated tool object, and the logic can be tested without a server.

### Human approval before writes

Any call that could write to the career plan pauses the agent run until the user approves it in the terminal:

- `save_skill_gap` → always asks
- `investigate_job_fit` → asks only when `save_gap=true`
- `get_job_requirements` → never asks

Answering anything other than `y` rejects the write. The agent is told "Nothing was written to the career plan" and reports that honestly. When there is no one to answer (no interactive terminal), writes are rejected (fail closed).

The SDK's public `require_approval` setting decides per tool, without seeing the call's arguments. `CareerLensMCPServer` overrides an SDK hook so that the policy can look at `save_gap`. If that hook ever changes, the tool-level `require_approval` fallback still asks before every call that can write. This is one reason `openai-agents` is pinned in `requirements.txt`.

### Reliability details

- **Atomic writes:** each save goes to a temporary file, then `os.replace`, so a crash cannot leave a half-written plan.
- **No silent overwrites:** a missing plan file is created; a corrupt or malformed one is refused and left untouched.
- **Thread-safe saves:** FastMCP runs sync tools in a threadpool, so a lock makes "read, check for duplicates, write" a single step. A test runs 14 parallel saves.
- **Clear errors, no leaks:** all CareerLens errors subclass FastMCP's `ToolError`, so their message always reaches the agent. Any unexpected exception is masked (`mask_error_details=True`). On the client side, a custom `failure_error_function` passes the real reason to the model instead of the SDK's generic message.

---

## Setup

Requirements: **Python 3.10+** and an **OpenAI API key**. Tested with Python 3.13 on Windows.

```powershell
git clone https://github.com/alaa-kalach/CareerLens.git
cd CareerLens
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                # macOS/Linux: cp .env.example .env
```

Then open `.env` and set your key. `.env` is git-ignored and must never be committed.

```
OPENAI_API_KEY=sk-...
```

### Configuration

All settings are optional except the API key.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | Used by the agent for model calls |
| `CAREERLENS_MODEL` | Agents SDK default | Model the agent uses, e.g. `gpt-4.1-mini` |
| `CAREERLENS_MCP_URL` | `http://127.0.0.1:8000/mcp` | Server URL the agent connects to |
| `CAREERLENS_PORT` | `8000` | Port the server listens on |
| `CAREERLENS_DATA_DIR` | `./data` | Folder holding the JSON datasets (tests point this to a temporary copy) |

---

## Running

Use two terminals, both with the virtual environment active.

**Terminal 1: start the MCP server**
```powershell
python -m server.mcp_server
```
The server listens on `http://127.0.0.1:8000/mcp` and logs every tool call (`DATA tool called`, `ACTION tool result`, ...).

**Terminal 2: run the agent**
```powershell
python agent.py --tools      # list the tools and their behavior hints (no LLM call)
python agent.py              # interactive chat
python agent.py "What does job J003 require?"   # single question
```

Every tool call is traced in the agent terminal, so the full round trip is visible:

```
You: Analyze how C001 fits J003
  [agent -> MCP] call investigate_job_fit({"candidate_id":"C001","job_id":"J003","save_gap":false})
  [MCP -> agent] fit report C001 vs J003: coverage=40.0% labels={'supported_match': 1, ...} | saves: none | ...
```

### Example prompts

| Prompt | Tool used | What it shows |
|---|---|---|
| `What does job J003 require?` | Data | Structured requirements |
| `Analyze how C001 fits J003` | Orchestration | All four labels, with evidence |
| `Save the justified skill gap` | Action (with approval) | Only Docker is saved; missing evidence is not |
| `Save an insufficient_evidence entry for Git for C001 and J003: it's listed but no project shows it` | Action (with approval) | A direct, validated save |
| `Analyze C001 for J999` | Orchestration | A clear error with the valid IDs; nothing is written |
| `Analyze C003 for J001` | Orchestration | An incomplete profile: limitations are reported and nothing is invented |

To reset the demo career plan:
```powershell
Set-Content data\career_plan.json '{ "entries": [] }'
```

---

## Tests

```powershell
pytest
```

The tests run against a temporary copy of the data, so `data/` is never modified. Expected result: **38 passed, 1 skipped**.

| Layer | Covers |
|---|---|
| Data tool | Valid job, case-insensitive IDs, unknown and empty IDs, missing, corrupt or malformed dataset |
| Action tool | Valid save, no duplicates, six invalid inputs (each checking nothing was written), missing plan created, corrupt or malformed plan never overwritten, 14 concurrent saves |
| Orchestration tool | Labels, evidence and score wording; `save_gap=false` writes nothing; `save_gap=true` saves only justified gaps; repeat saves; failed retrieval stops with no write; incomplete and malformed profiles; failed saves are reported |
| MCP layer | Exactly three tools, descriptions and typed schemas, annotations, error masking, errors reaching the client |
| Agent host | Approval policy, fail-closed fallback, reject when no one can answer |
| Integration | A real HTTP server in a subprocess, called through the Agents SDK MCP client |
| Live (optional) | A real LLM agent run: `$env:RUN_LLM_TESTS="1"; pytest -k live` |

Quick, readable checks per tool are also available as example scripts:
```powershell
python -m scripts.smoke_data_tool
python -m scripts.smoke_action_tool
python -m scripts.smoke_orchestration_tool
```

---

## Project structure

```
CareerLens/
├── agent.py                     # OpenAI Agents SDK host / MCP client (approval, trace)
├── server/
│   ├── mcp_server.py            # FastMCP server: the three @mcp.tool wrappers
│   └── careerlens.py            # Core logic, models, errors (no MCP code)
├── data/
│   ├── candidates.json          # 3 fictional candidates
│   ├── jobs.json                # 5 internship / junior roles
│   └── career_plan.json         # Saved entries (starts empty)
├── tests/
│   ├── conftest.py              # Temporary-data fixture
│   └── test_tools.py            # Unit, MCP-layer and integration tests
├── scripts/                     # Smoke-check example scripts, one per tool
├── written_summary.md           # Part 1: MCP article and server critique
├── reflection.md                # Reflection on tool boundaries
├── requirements.txt
├── pytest.ini
└── .env.example
```

### Sample data

All people in the dataset are fictional.

| ID | Candidate | Purpose in the demo |
|---|---|---|
| C001 | Lina Haddad (ML profile) | Against J003 she produces all four labels and one justified gap (Docker) |
| C002 | Omar Khalil (data analyst) | Skills listed without evidence; an explicit gap with related evidence (Tableau) |
| C003 | Rami Saab | A deliberately incomplete profile |

The jobs are J001 Machine Learning Intern, J002 Data Analyst Intern, J003 AI Engineer Intern, J004 Junior Python Developer and J005 ML Research Assistant.

---