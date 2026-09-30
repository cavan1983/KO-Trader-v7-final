"""Test suite for backtesting and risk management."""

import pytest
from backtesting import run_backtest
from model_audit import audit_model_accuracy
from risk_management import validate_signal, get_trade_risk_levels


def test_backtest_runs():
    """Backtesting framework işləyir."""
    result = run_backtest("KO", days=30, initial_cash=10000)
    assert isinstance(result, dict)
    assert "final_value" in result
    assert result["final_value"] > 0


def test_signal_validation():
    """Signal validation doğru işləyir."""
    assert validate_signal("AL", 65) == True
    assert validate_signal("AL", 55) == False
    assert validate_signal("GÖZLƏ", 100) == False


def test_risk_levels():
    """Risk config faylı yüklənir."""
    cfg = get_trade_risk_levels()
    assert cfg["min_confidence"] >= 50
    assert cfg["stop_loss_pct"] > 0
    assert cfg["take_profit_pct"] > 0


def test_model_audit_runs():
    """Model audit framework işləyir."""
    result = audit_model_accuracy("KO", lookback_days=30)
    if "signals_generated" in result:
        assert result["signals_generated"] >= 0
