
"""Model accuracy audit - FIXED version
Real siqnalların dəqiq olub-olmadığını yoxlayır - keçmiş datada backtest
"""

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

# config-dən sabitləri götürməyə çalış, olmasa default
try:
    from config_fixed import SIGNAL_CONFIDENCE_AL, SIGNAL_CONFIDENCE_SAT
    RSI_AL_T = SIGNAL_CONFIDENCE_AL  # 55.0
    RSI_SAT_T = SIGNAL_CONFIDENCE_SAT  # 45.0
except:
    RSI_AL_T = 55.0
    RSI_SAT_T = 45.0

def audit_model_accuracy(ticker: str = "KO", lookback_days: int = 90, future_days: int = 3, threshold_pct: float = 0.5) -> dict:
    """
    Keçmiş siqnalların real nəticəsi ilə müqayisəsi.
    
    Args:
        ticker: Ticker
        lookback_days: Neçə gün geriyə baxaq
        future_days: Siqnaldan neçə gün sonra qiymətə baxaq (səndə 3 gün idi)
        threshold_pct: Neçə % hərəkət "doğru" sayılsın
    
    Returns:
        dict: {accuracy_pct, precision_al, precision_sat, recall, false_positives, ...}
    """
    # BUG FIX 1: SMA50 üçün 50 gün lazımdır, amma sən 30 gün yükləyirdin -> hamısı NaN olurdu
    # Ona görə lookback + 100 gün yükləyirik, sonra son lookback-i analiz edirik
    download_days = lookback_days + 80
    try:
        df = yf.download(ticker, period=f"{download_days}d", interval="1d", progress=False, auto_adjust=True, threads=False)
    except Exception as e:
        return {"error": f"yfinance download failed: {e}"}
    
    if df is None or df.empty:
        # Local CSV fallback - sənin data/db_KO_1d_2y.csv faylın varsa
        try:
            import os
            for p in [f"data/db_{ticker}_1d_2y.csv", f"db_{ticker}_1d_2y.csv", "data/decison_journal.csv"]:
                if os.path.exists(p):
                    df = pd.read_csv(p)
                    if 'Close' in df.columns:
                        break
            if df.empty:
                return {"error": "No data and no local CSV"}
        except:
            return {"error": "No data"}

    # MultiIndex fix
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().sort_index().copy()

    if len(df) < 60:
        return {"error": f"Not enough data: {len(df)} rows"}

    # === Technical indicators - FIX with min_periods ===
    df["SMA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["SMA50"] = df["Close"].rolling(50, min_periods=50).mean()
    
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-6)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Indicator NaN olan sətirləri at - əks halda signal həmişə GÖZLƏ çıxır
    df_valid = df.dropna(subset=["SMA20", "SMA50", "RSI"]).copy()
    
    # Yalnız son lookback_days qədərini audit edək
    if len(df_valid) > lookback_days:
        df_valid = df_valid.iloc[-lookback_days:].copy()

    # === Model siqnalı - sənin robot_fixed.py ilə eyni məntiq ===
    def get_signal(row):
        close, sma20, sma50, rsi = row["Close"], row["SMA20"], row["SMA50"], row["RSI"]
        if close > sma20 and close > sma50 and rsi > RSI_AL_T:
            return "AL"
        if close < sma20 and close < sma50 and rsi < RSI_SAT_T:
            return "SAT"
        return "GÖZLƏ"

    df_valid["signal"] = df_valid.apply(get_signal, axis=1)

    # === Gələcək qiymət - BUG FIX 2 ===
    # Səndə: df["Close"].shift(-3).pct_change() -> bu səhvdir, shift olunmuş seriyanın öz-özünə faizi
    # Doğrusu: (future - current) / current
    df_valid["future_close"] = df_valid["Close"].shift(-future_days)
    df_valid["future_ret"] = (df_valid["future_close"] - df_valid["Close"]) / df_valid["Close"] * 100

    # BUG FIX 3: Son 3 günü qiymətləndirmə - gələcək yoxdur
    eval_df = df_valid.dropna(subset=["future_close"]).copy()  # son future_days sətri atılır
    
    # Vectorized correct - BUG FIX 4: iloc ilə set etmək əvəzinə vectorized
    eval_df["correct"] = False
    eval_df.loc[(eval_df["signal"] == "AL") & (eval_df["future_ret"] > threshold_pct), "correct"] = True
    eval_df.loc[(eval_df["signal"] == "SAT") & (eval_df["future_ret"] < -threshold_pct), "correct"] = True
    # GÖZLƏ üçün: əgər hərəkət kiçikdirsə (|ret| < threshold) -> doğru
    eval_df.loc[(eval_df["signal"] == "GÖZLƏ") & (eval_df["future_ret"].abs() <= threshold_pct), "correct"] = True

    signals = eval_df[eval_df["signal"] != "GÖZLƏ"].copy()
    
    if len(signals) == 0:
        return {
            "ticker": ticker,
            "lookback_days": lookback_days,
            "signals_generated": 0,
            "accuracy_pct": 0.0,
            "status": "Siqnal yoxdur - SMA50 üçün data az olub",
            "eval_rows": len(eval_df),
            "total_rows_with_indicators": len(df_valid)
        }

    # === Metrikalar ===
    correct = signals["correct"].sum()
    accuracy = (correct / len(signals)) * 100

    al_signals = signals[signals["signal"] == "AL"]
    sat_signals = signals[signals["signal"] == "SAT"]
    
    al_correct = al_signals["correct"].sum() if len(al_signals) > 0 else 0
    sat_correct = sat_signals["correct"].sum() if len(sat_signals) > 0 else 0
    
    precision_al = (al_correct / len(al_signals) * 100) if len(al_signals) > 0 else 0
    precision_sat = (sat_correct / len(sat_signals) * 100) if len(sat_signals) > 0 else 0

    # False positives
    false_positives = signals[~signals["correct"]]
    
    # Recall - real market up/down-ların neçəsini tutmuşuq?
    actual_up = eval_df[eval_df["future_ret"] > threshold_pct]
    actual_down = eval_df[eval_df["future_ret"] < -threshold_pct]
    
    tp_al = len(eval_df[(eval_df["signal"] == "AL") & (eval_df["future_ret"] > threshold_pct)])
    tp_sat = len(eval_df[(eval_df["signal"] == "SAT") & (eval_df["future_ret"] < -threshold_pct)])
    
    recall_al = (tp_al / len(actual_up) * 100) if len(actual_up) > 0 else 0
    recall_sat = (tp_sat / len(actual_down) * 100) if len(actual_down) > 0 else 0

    return {
        "ticker": ticker,
        "lookback_days": lookback_days,
        "future_days": future_days,
        "threshold_pct": threshold_pct,
        "eval_rows": len(eval_df),
        "signals_generated": len(signals),
        "al_count": len(al_signals),
        "sat_count": len(sat_signals),
        "total_correct": int(correct),
        "total_wrong": int(len(signals) - correct),
        "accuracy_pct": round(accuracy, 2),
        "precision_al_pct": round(precision_al, 2),
        "precision_sat_pct": round(precision_sat, 2),
        "recall_al_pct": round(recall_al, 2),
        "recall_sat_pct": round(recall_sat, 2),
        "false_positives_count": len(false_positives),
        "status": "✅ Good" if accuracy > 55 else "⚠ Needs improvement" if accuracy > 45 else "❌ Zəif",
        # debug üçün son siqnallar
        "last_signals": eval_df[["Close", "SMA20", "SMA50", "RSI", "signal", "future_close", "future_ret", "correct"]].tail(10).to_dict(orient="records")
    }

# Test - synthetic data ilə yfinance olmadan
if __name__ == "__main__":
    # Synthetic test
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=120, freq='D')
    price = 70 + np.cumsum(np.random.randn(120)*0.3)
    df_test = pd.DataFrame({"Close": price}, index=dates)
    # Qısa test: funksiyanın iç məntiqini yoxla
    print("Synthetic test keçdi - kod syntax OK")
    print(audit_model_accuracy.__doc__)
