"""
Logging system for KO Trader
Ensures all events are tracked for audit and debugging
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

from config import LOG_DIR

# Ensure logs directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Define log file paths
MAIN_LOG = LOG_DIR / "robot.log"
ERROR_LOG = LOG_DIR / "errors.log"
TRADING_LOG = LOG_DIR / "trading.log"

# Shared formatter for consistency
LOG_FORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_logger(name: str, log_file: Path = None) -> logging.Logger:
    """
    Create or retrieve a logger with file + console handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Optional custom log file path
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(console_handler)
    
    # File handler (DEBUG and above, goes to MAIN_LOG or custom path)
    file_log = log_file or MAIN_LOG
    try:
        file_handler = logging.FileHandler(file_log, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not attach file handler to {file_log}: {e}")
    
    return logger


def log_trade_event(ticker: str, signal: str, confidence: float, price: float, 
                   reason: str = "", status: str = "signal") -> None:
    """
    Log a trading event to the dedicated trading log.
    
    Args:
        ticker: Stock ticker (e.g., "KO")
        signal: AL/GÖZLƏ/SAT
        confidence: Model confidence 0-100%
        price: Current price
        reason: Additional context (e.g., "stop_loss", "take_profit", "model_signal")
        status: "signal", "executed", "failed"
    """
    trading_logger = get_logger("trading", TRADING_LOG)
    msg = f"{ticker} | {signal} | {confidence:.1f}% | ${price:.2f} | {reason} | [{status}]"
    trading_logger.info(msg)


def log_error(error_context: str, exception: Exception = None, 
             module: str = "robot") -> None:
    """
    Log an error with optional exception traceback.
    
    Args:
        error_context: Description of what failed
        exception: Optional exception object for traceback
        module: Which module this error came from
    """
    error_logger = get_logger(f"{module}.error", ERROR_LOG)
    if exception:
        error_logger.error(f"{error_context}", exc_info=True)
    else:
        error_logger.error(error_context)


# Initialize main logger for use in robot.py and other modules
main_logger = get_logger("robot")
