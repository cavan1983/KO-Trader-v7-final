"""
backtesting.py - FIXED
KO Trader V8 üçün real backtest
config_fixed.py sabitləri ilə uyğun
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# config_fixed-dən al, yoxdursa default
try:
    from config_fixed import (
        BACKTEST_DAYS, BACKTEST_INITIAL_CASH,
        MAX_POSITION_RATIO, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
        COMMISSION, SIGNAL_CONFIDENCE_AL, SIGNAL_CONFIDENCE_SAT
    )
except ImportError:
    BACKTEST_DAYS = 180
    BACKTEST_INITIAL_CASH = 10000.0
    MAX_POSITION_RATIO = 0.2
    STOP_LOSS_PCT = 5.0
    TAKE_PROFIT_PCT = 8.0
    COMMISSION = 0.001
    SIGNAL_CONFIDENCE_AL = 55.0
    SIGNAL_CONFIDENCE_SAT = 45.0

def run_backtest(ticker="KO", days=180, initial_cash=10000.0):
    period = f"{days+80}d"
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True, threads=False)
    if df is None or df.empty:
        return {"error": "No data from yfinance"}
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().sort_index()

    # Indicators
    df["SMA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["SMA50"] = df["Close"].rolling(50, min_periods=50).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta>0,0).rolling(14, min_periods=14).mean()
    loss = -delta.where(delta<0,0).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-6)
    df["RSI"] = 100 - (100/(1+rs))
    
    df = df.dropna(subset=["SMA20","SMA50","RSI"]).copy()
    if len(df) > days:
        df = df.iloc[-days:].copy()

    # Signal
    def get_signal(r):
        if r["Close"] > r["SMA20"] and r["Close"] > r["SMA50"] and r["RSI"] > SIGNAL_CONFIDENCE_AL:
            return "AL"
        if r["Close"] < r["SMA20"] and r["Close"] < r["SMA50"] and r["RSI"] < SIGNAL_CONFIDENCE_SAT:
            return "SAT"
        return "GÖZLƏ"
    df["signal"] = df.apply(get_signal, axis=1)

    # Backtest loop
    cash = initial_cash
    position = 0
    entry_price = 0
    trades = []
    equity_curve = []

    for i, row in df.iterrows():
        price = float(row["Close"])
        signal = row["signal"]
        
        # Equity
        equity = cash + position * price
        equity_curve.append(equity)

        # SAT -> close long
        if signal == "SAT" and position > 0:
            cash += position * price * (1 - COMMISSION)
            pnl = (price - entry_price) * position
            trades.append({"type":"SELL", "price":price, "pnl":pnl, "date": i})
            position = 0
            entry_price = 0

        # AL -> open long
        if signal == "AL" and position == 0:
            invest = cash * MAX_POSITION_RATIO
            qty = invest / price
            cost = qty * price * (1 + COMMISSION)
            if cost <= cash:
                position = qty
                cash -= cost
                entry_price = price
                trades.append({"type":"BUY", "price":price, "date": i})

        # Stop Loss / Take Profit
        if position > 0:
            change_pct = (price - entry_price) / entry_price * 100
            if change_pct <= -STOP_LOSS_PCT or change_pct >= TAKE_PROFIT_PCT:
                cash += position * price * (1 - COMMISSION)
                pnl = (price - entry_price) * position
                trades.append({"type":"CLOSE_SLTP", "price":price, "pnl":pnl, "date": i})
                position = 0
                entry_price = 0

    # Final close
    if position > 0:
        final_price = float(df["Close"].iloc[-1])
        cash += position * final_price * (1 - COMMISSION)
        position = 0

    final_equity = cash
    total_return = (final_equity - initial_cash) / initial_cash * 100
    wins = len([t for t in trades if t.get("pnl",0) > 0])
    total_sells = len([t for t in trades if t["type"] != "BUY"])
    win_rate = wins / total_sells * 100 if total_sells>0 else 0

    return {
        "ticker": ticker,
        "days": days,
        "initial_cash": initial_cash,
        "final_equity": round(final_equity,2),
        "total_return_pct": round(total_return,2),
        "trades_count": len(trades),
        "win_rate_pct": round(win_rate,2),
        "trades": trades[-10:],  # son 10 trade
        "status": "✅ Qazanclı" if total_return>0 else "❌ Zərər"
    }

if __name__ == "__main__":
    result = run_backtest("KO", BACKTEST_DAYS, BACKTEST_INITIAL_CASH)
    print(result)
