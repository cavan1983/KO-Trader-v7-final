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

# === FIX: Bütün əskik sabitlər - backtesting, risk_management, model üçün ===
BACKTEST_DAYS = 180
BACKTEST_INITIAL_CASH = 10000.0
MODEL_TTL_HOURS = 0.75
TICKERS = ["NVDA"]

# Signal confidence
SIGNAL_CONFIDENCE_MIN = 50.0
SIGNAL_CONFIDENCE_AL = 55.0
SIGNAL_CONFIDENCE_SAT = 45.0
CONFIDENCE_THRESHOLD = 55.0

# Risk / Position - sənin xətan burda idi
MAX_POSITION_RATIO = 0.2
POSITION_SIZE = 0.2
POSITION_RATIO = 0.2
MAX_POSITION_SIZE = 0.2
STOP_LOSS_PCT = 5.0
TAKE_PROFIT_PCT = 8.0
STOP_LOSS = 5.0
TAKE_PROFIT = 8.0
RISK_PER_TRADE = 0.02

# Trading
INITIAL_CASH = 10000.0
COMMISSION = 0.001
SLIPPAGE = 0.001
