"""Relocatable paths rooted at the repository directory."""
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WORK_DIR.parent
PROCESSED_DIR = WORK_DIR / "processed_data"
OUTPUT_DIR = WORK_DIR / "output"
ENV_FILE = WORK_DIR / ".env"
