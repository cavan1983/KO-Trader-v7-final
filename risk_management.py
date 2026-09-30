"""Risk management helpers for the trading strategy.

These functions keep the project logic safer and more traceable without
rewriting the whole strategy.
"""

from __future__ import annotations
from config import SIGNAL_CONFIDENCE_MIN, STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_POSITION_RATIO


def validate_signal(signal: str, confidence: float) -> bool:
    """Return True if signal is strong enough to be considered actionable."""
    if signal not in {"AL", "SAT", "GÖZLƏ"}:
        return False
    if signal == "GÖZLƏ":
        return False
    return confidence >= SIGNAL_CONFIDENCE_MIN


def get_trade_risk_levels() -> dict:
    """Return core risk thresholds used by the strategy."""
    return {
        "min_confidence": SIGNAL_CONFIDENCE_MIN,
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "max_position_ratio": MAX_POSITION_RATIO,
    }


def evaluate_position_change(entry_price: float, current_price: float) -> float:
    """Return percentage move from entry price to current price."""
    if entry_price <= 0:
        return 0.0
    return ((current_price - entry_price) / entry_price) * 100.0


def should_exit_position(entry_price: float, current_price: float, signal: str, signal_confidence: float) -> str | None:
    """Return the reason for exit when risk or signal conditions are triggered."""
    pct_change = evaluate_position_change(entry_price, current_price)

    if pct_change <= -STOP_LOSS_PCT:
        return "STOP_LOSS"
    if pct_change >= TAKE_PROFIT_PCT:
        return "TAKE_PROFIT"
    if signal == "SAT" and signal_confidence >= SIGNAL_CONFIDENCE_MIN:
        return "SAT_SIGNAL"
    return None


def max_position_size(balance: float, current_price: float) -> float:
    """Maximum shares allowed for a single position using a simple risk cap."""
    if current_price <= 0 or balance <= 0:
        return 0.0
    max_value = balance * MAX_POSITION_RATIO
    return max_value / current_price
