# Base project paths and environment config

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
ENV_PATH = BASE_DIR / ".env"

# Load environment variables from .env if present.
load_dotenv(ENV_PATH, override=False)


def ensure_runtime_dirs() -> None:
    """Create runtime folders required by the bot."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_env(name: str, default: str | None = None) -> str | None:
    """Small wrapper to read environment values consistently."""
    return os.getenv(name, default)
