# Base project paths and environment config
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH, override=False)

def ensure_runtime_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)

# === FIX: Bütün əskik sabitlər ===
BACKTEST_DAYS = 180
BACKTEST_INITIAL_CASH = 10000.0
MODEL_TTL_HOURS = 0.75
TICKERS = ["KO"]

# Backtesting və signal üçün lazım olanlar
SIGNAL_CONFIDENCE_MIN = 50.0
SIGNAL_CONFIDENCE_AL = 55.0
SIGNAL_CONFIDENCE_SAT = 45.0
CONFIDENCE_THRESHOLD = 55.0

# Risk və trading
STOP_LOSS_PCT = 5.0
TAKE_PROFIT_PCT = 8.0
POSITION_SIZE = 0.2
