"""Lightweight backtesting framework for model validation.

Goal: Sınaqda modelin keçmişdə necə işlədiyi görülsün.
Virtual 10k portfolyo kimi, hər al/sat-ı simulyasiya et.
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf
from config import BACKTEST_DAYS
from risk_management import get_trade_risk_levels


def run_backtest(symbol: str = "KO", days: int = BACKTEST_DAYS, initial_cash: float = 10000.0) -> dict:
    """Keçmiş veriler üzərində strategiyanı sınaqla.
    
    Returns: {trades, win_rate, total_return_pct, max_drawdown_pct, final_value}
    """
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False, auto_adjust=True, threads=False)
    if df is None or df.empty:
        return {"symbol": symbol, "trades": 0, "win_rate": 0.0, "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "final_value": initial_cash}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().sort_index().copy()

    # Sadə SMA/RSI əsaslı siqnal
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-6)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Siqnal: AL=qiymət SMA20 və SMA50-dən yuxarı + RSI>55, SAT=əksi
    def get_signal(row):
        close, sma20, sma50, rsi = row["Close"], row["SMA20"], row["SMA50"], row["RSI"]
        if close > sma20 and close > sma50 and rsi > 55:
            return "AL"
        if close < sma20 and close < sma50 and rsi < 45:
            return "SAT"
        return "GÖZLƏ"
    df["signal"] = df.apply(get_signal, axis=1)

    # Virtual portfolio: başlanğıc 10k
    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    trades = []
    equity_curve = [initial_cash]
    peak_value = initial_cash

    risk_cfg = get_trade_risk_levels()
    sl = risk_cfg["stop_loss_pct"]
    tp = risk_cfg["take_profit_pct"]

    for _, row in df.iterrows():
        signal, close = row["signal"], float(row["Close"])

        # AL: satın al (25% risk ilə)
        if position == 0 and signal == "AL":
            max_shares = min((cash * 0.25) / close, 1000)
            if max_shares > 0:
                position = max_shares
                entry_price = close
                cash = 0.0
                trades.append({"entry": close, "signal": signal})
        # SAT: sat (SL/TP/signal)
        elif position > 0:
            change = ((close - entry_price) / entry_price) * 100.0
            should_exit = change <= -sl or change >= tp or signal == "SAT"
            if should_exit:
                cash = position * close
                if trades:
                    trades[-1]["exit"] = close
                    trades[-1]["pnl_pct"] = change
                position = 0.0

        # Portfolio dəyəri
        current_value = (position * close) if position > 0 else cash
        equity_curve.append(current_value)
        peak_value = max(peak_value, current_value)

    final_value = equity_curve[-1]
    closed_trades = [t for t in trades if "exit" in t]
    wins = sum(1 for t in closed_trades if t.get("pnl_pct", 0) > 0)
    win_rate = (wins / len(closed_trades)) * 100 if closed_trades else 0.0
    total_return = ((final_value - initial_cash) / initial_cash) * 100

    max_dd = 0.0
    peak = initial_cash
    for v in equity_curve:
        peak = max(peak, v)
        dd = ((peak - v) / peak) * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        "symbol": symbol,
        "period_days": days,
        "trades_closed": len(closed_trades),
        "win_rate_pct": round(win_rate, 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_value": round(final_value, 2),
        "initial_value": initial_cash,
        "status": "✅ Model işləyir" if total_return > 5 else "⚠️ Model xətli"
    }
