"""
robot_nvda_real_ai.py - Həqiqi öyrənən NVDA robotu
data/NVDA_model.pkl istifadə edir
"""
import os, json, pickle, warnings, traceback
from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from config import BASE_DIR, DATA_DIR, ensure_runtime_dirs
ensure_runtime_dirs()

BAKU_TZ = pytz.timezone("Asia/Baku")
DATA_DIR_STR = str(DATA_DIR)

def get_nvda_features():
    """NVDA üçün real feature-ları hesabla"""
    df = yf.download("NVDA", period="6mo", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    
    # Eyni feature-lar train_nvda_real_ai.py ilə
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-6)
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["BB_mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2*bb_std
    df["BB_lower"] = df["BB_mid"] - 2*bb_std
    df["BB_pos"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    df["Volume_SMA"] = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / df["Volume_SMA"]
    df["High_Low"] = df["High"] - df["Low"]
    df["ATR"] = df["High_Low"].rolling(14).mean()
    df["Price_vs_SMA20"] = (df["Close"] - df["SMA20"]) / df["SMA20"] * 100
    df["Price_vs_SMA50"] = (df["Close"] - df["SMA50"]) / df["SMA50"] * 100
    df["SMA20_vs_SMA50"] = (df["SMA20"] - df["SMA50"]) / df["SMA50"] * 100
    
    last = df.dropna().iloc[-1]
    price = float(last["Close"])
    
    # Load model
    try:
        with open(os.path.join(DATA_DIR_STR, "NVDA_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(DATA_DIR_STR, "NVDA_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(DATA_DIR_STR, "NVDA_features.json"), "r") as f:
            feature_cols = json.load(f)
        
        # Son günün feature-ları
        X = np.array([[last[col] for col in feature_cols]])
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[0]  # [SAT, GÖZLƏ, AL]
        pred = np.argmax(probs)
        
        label_map = {0: "SAT", 1: "GÖZLƏ", 2: "AL"}
        signal = label_map[pred]
        conf = float(np.max(probs) * 100)
        
        # AL/SAT/GÖZLƏ faizləri
        sat_p = float(probs[0]*100)
        gozle_p = float(probs[1]*100)
        al_p = float(probs[2]*100)
        
        return {
            "signal": signal,
            "conf": conf,
            "price": price,
            "rsi": round(float(last["RSI"]),1),
            "ma20": round(float(last["SMA20"]),2),
            "ma50": round(float(last["SMA50"]),2),
            "al_pct": round(al_p,1),
            "sat_pct": round(sat_p,1),
            "gozle_pct": round(gozle_p,1),
            "model": "REAL AI",
            "open": float(last["Open"])
        }
    except Exception as e:
        print(f"Model yoxdur və ya xəta: {e}, fallback işləyir")
        # Fallback - train etməmisənsə
        rsi = float(last["RSI"])
        ma20 = float(last["SMA20"])
        ma50 = float(last["SMA50"])
        if price > ma20 and rsi > 55:
            sig = "AL"; al_p=65; gozle_p=25; sat_p=10
        elif price < ma20 and rsi < 45:
            sig = "SAT"; al_p=10; gozle_p=25; sat_p=65
        else:
            sig = "GÖZLƏ"; al_p=25; gozle_p=50; sat_p=25
        return {
            "signal": sig, "conf": 50.0, "price": price,
            "rsi": round(rsi,1), "ma20": round(ma20,2), "ma50": round(ma50,2),
            "al_pct": al_p, "sat_pct": sat_p, "gozle_pct": gozle_p,
            "model": "FALLBACK", "open": float(last["Open"])
        }

def main():
    baku_time = datetime.now(BAKU_TZ)
    print(f"🔮 NVDA Real AI - {baku_time.strftime('%d.%m %H:%M')} Bakı")
    
    result = get_nvda_features()
    print(f"Price: {result['price']} | Signal: {result['signal']} ({result['conf']:.1f}%)")
    print(f"AL:{result['al_pct']}% GÖZLƏ:{result['gozle_pct']}% SAT:{result['sat_pct']}% RSI:{result['rsi']}")
    print(f"Model: {result['model']}")
    
    # predictions.json yaz
    all_results = {
        "NVDA": {
            "1g": result,
            "1s": result,
            "3g": result,
            "5g": result
        },
        "live": {
            "ticker": "NVDA",
            "price": result["price"],
            "rsi": result["rsi"],
            "ma20": result["ma20"],
            "ma50": result["ma50"],
            "time": baku_time.strftime("%H:%M:%S"),
            "model": result["model"]
        },
        "last_update": baku_time.strftime("%d.%m.%Y %H:%M:%S")
    }
    
    for p in [os.path.join(DATA_DIR_STR, "predictions.json"), "data/predictions.json"]:
        try:
            os.makedirs(os.path.dirname(p) if os.path.dirname(p) else ".", exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"💾 {p} yazıldı")
        except Exception as e:
            print(f"Yazı xətası {p}: {e}")

if __name__ == "__main__":
    main()
