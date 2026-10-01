import pandas as pd
import yfinance as yf

def audit_model_accuracy(ticker: str = "KO", lookback_days: int = 90) -> dict:
    # FIX: SMA50 üçün kifayət qədər data yüklə
    df = yf.download(ticker, period=f"{lookback_days+80}d", interval="1d", progress=False, auto_adjust=True, threads=False)
    if df is None or df.empty:
        return {"error": "No data"}
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().sort_index().copy()

    df["SMA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["SMA50"] = df["Close"].rolling(50, min_periods=50).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-6)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    df = df.dropna(subset=["SMA20","SMA50","RSI"]).copy()
    if len(df) > lookback_days:
        df = df.iloc[-lookback_days:].copy()

    def get_signal(r):
        if r["Close"] > r["SMA20"] and r["Close"] > r["SMA50"] and r["RSI"] > 55:
            return "AL"
        if r["Close"] < r["SMA20"] and r["Close"] < r["SMA50"] and r["RSI"] < 45:
            return "SAT"
        return "GÖZLƏ"
    df["signal"] = df.apply(get_signal, axis=1)

    # FIX: gələcək return düz hesablanır
    df["future_close"] = df["Close"].shift(-3)
    df["future_ret"] = (df["future_close"] - df["Close"]) / df["Close"] * 100
    eval_df = df.dropna(subset=["future_close"]).copy()

    eval_df["correct"] = False
    eval_df.loc[(eval_df["signal"]=="AL") & (eval_df["future_ret"]>0.5), "correct"] = True
    eval_df.loc[(eval_df["signal"]=="SAT") & (eval_df["future_ret"]<-0.5), "correct"] = True

    signals = eval_df[eval_df["signal"]!="GÖZLƏ"]
    if len(signals)==0:
        return {"signals_generated":0, "accuracy_pct":0, "status":"Siqnal yoxdur"}

    accuracy = signals["correct"].mean()*100
    al = signals[signals["signal"]=="AL"]
    sat = signals[signals["signal"]=="SAT"]
    
    return {
        "ticker": ticker,
        "signals_generated": len(signals),
        "accuracy_pct": round(accuracy,2),
        "precision_al_pct": round(al["correct"].mean()*100,2) if len(al)>0 else 0,
        "precision_sat_pct": round(sat["correct"].mean()*100,2) if len(sat)>0 else 0,
        "total_correct": int(signals["correct"].sum()),
        "false_positives": int(len(signals)-signals["correct"].sum()),
        "status": "✅ Good" if accuracy>55 else "⚠ Needs improvement"
    }
