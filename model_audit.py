"""Model accuracy audit: Real siqnallar dəqiq olub-olmadığını yoxla.

Keçmiş siqnalların real nəticəsi ilə müqayisə et.
"""

import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf


def audit_model_accuracy(ticker: str = "KO", lookback_days: int = 30) -> dict:
    """Keçmiş siqnalların dəqiqliyi.
    
    Returns: {accuracy_pct, precision, recall, false_positives}
    """
    df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", progress=False, auto_adjust=True, threads=False)
    if df is None or df.empty:
        return {"error": "No data"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().sort_index().copy()

    # Technical indicators
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-6)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Model siqnalı
    def get_signal(row):
        close, sma20, sma50, rsi = row["Close"], row["SMA20"], row["SMA50"], row["RSI"]
        if close > sma20 and close > sma50 and rsi > 55:
            return "AL"
        if close < sma20 and close < sma50 and rsi < 45:
            return "SAT"
        return "GÖZLƏ"
    df["signal"] = df.apply(get_signal, axis=1)

    # Gələcək qiymət hərəkətini yoxla (AL+3 gün sonra up? SAT+3 gün sonra down?)
    df["future_change"] = df["Close"].shift(-3).pct_change() * 100
    df["correct"] = False

    for i in range(len(df) - 3):
        signal = df.iloc[i]["signal"]
        future = df.iloc[i + 3]["Close"]
        current = df.iloc[i]["Close"]
        change = ((future - current) / current) * 100

        if signal == "AL" and change > 0.5:
            df.iloc[i, df.columns.get_loc("correct")] = True
        elif signal == "SAT" and change < -0.5:
            df.iloc[i, df.columns.get_loc("correct")] = True

    signals = df[df["signal"] != "GÖZLƏ"]
    if len(signals) == 0:
        return {"signals_generated": 0, "accuracy_pct": 0}

    correct = signals[signals["correct"]].shape[0]
    accuracy = (correct / len(signals)) * 100 if len(signals) > 0 else 0

    al_signals = signals[signals["signal"] == "AL"]
    al_correct = al_signals[al_signals["correct"]].shape[0] if len(al_signals) > 0 else 0
    precision_al = (al_correct / len(al_signals)) * 100 if len(al_signals) > 0 else 0

    return {
        "ticker": ticker,
        "lookback_days": lookback_days,
        "signals_generated": len(signals),
        "accuracy_pct": round(accuracy, 2),
        "precision_al_pct": round(precision_al, 2),
        "total_correct": correct,
        "status": "✅ Good" if accuracy > 55 else "⚠️ Needs improvement"
    }
