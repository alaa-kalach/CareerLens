"""Shared fixtures. Every test runs against a fresh temporary copy of the data,
so the demo files in data/ are never modified."""
import shutil
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATA = PROJECT_ROOT / "data"
load_dotenv(PROJECT_ROOT / ".env")  # makes OPENAI_API_KEY available to the optional live LLM test


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    for name in ("candidates.json", "jobs.json", "career_plan.json"):
        shutil.copy(REAL_DATA / name, tmp_path / name)
    monkeypatch.setenv("CAREERLENS_DATA_DIR", str(tmp_path))
    return tmp_path
