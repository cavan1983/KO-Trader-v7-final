"""
robot_nvda_real_ai.py - FIXED for root files + data/ folder both
"""
import os, json, pickle, warnings, traceback, random
from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

try:
    from config import BASE_DIR, DATA_DIR, ensure_runtime_dirs
except ImportError:
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    def ensure_runtime_dirs():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

ensure_runtime_dirs()

BAKU_TZ = pytz.timezone("Asia/Baku")
DATA_DIR_STR = str(DATA_DIR)
BASE_DIR_STR = str(BASE_DIR)

def find_file(names):
    """Try multiple locations: data/, root, ./"""
    candidates = []
    for name in names:
        candidates.extend([
            os.path.join(DATA_DIR_STR, name),
            os.path.join(BASE_DIR_STR, name),
            os.path.join(BASE_DIR_STR, "data", name),
            f"data/{name}",
            name,
            f"./{name}"
        ])
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def get_nvda_features():
    df = None
    try:
        df = yf.download("NVDA", period="6mo", interval="1d", progress=False, auto_adjust=True, threads=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) < 20:
                df = None
    except Exception as e:
        print(f"yfinance failed: {e}")
        df = None

    if df is None or (hasattr(df, 'empty') and df.empty):
        # try local csv from root or data/
        csv_path = find_file(["db_NVDA_1d_2y.csv", "db_NVDA_1d_2y.csv"])
        if not csv_path:
            # also try the names you uploaded
            csv_path = find_file(["db_NVDA_1d_2y.csv"])
        if csv_path:
            try:
                cdf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if not cdf.empty and len(cdf) >= 20:
                    df = cdf
                    print(f"Local cache used: {csv_path} ({len(df)} rows)")
            except Exception as e:
                print(f"Cache fallback failed: {e}")

    finnhub_price = None
    try:
        api_key = os.getenv("FINNHUB_API_KEY")
        if api_key:
            url = f"https://finnhub.io/api/v1/quote?symbol=NVDA&token={api_key}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                finnhub_price = float(data.get('c', 0))
                if finnhub_price > 0:
                    print(f"Finnhub price: {finnhub_price}")
    except Exception as e:
        print(f"Finnhub failed: {e}")

    if df is None or df.empty:
        print("Heç bir data mənbəyi işləmədi, synthetic fallback")
        base_price = finnhub_price if finnhub_price else 175.0
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
        close = base_price + np.cumsum(np.random.randn(100)*0.8)
        df = pd.DataFrame({
            "Close": close,
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Volume": np.random.randint(40_000_000, 60_000_000, 100)
        }, index=dates)

    try:
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
        df["BB_pos"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"]).replace(0,1)
        df["Volume_SMA"] = df["Volume"].rolling(20).mean()
        df["Vol_ratio"] = df["Volume"] / df["Volume_SMA"].replace(0,1)
        df["High_Low"] = df["High"] - df["Low"]
        df["ATR"] = df["High_Low"].rolling(14).mean()
        df["Price_vs_SMA20"] = (df["Close"] - df["SMA20"]) / df["SMA20"].replace(0,1) * 100
        df["Price_vs_SMA50"] = (df["Close"] - df["SMA50"]) / df["SMA50"].replace(0,1) * 100
        df["SMA20_vs_SMA50"] = (df["SMA20"] - df["SMA50"]) / df["SMA50"].replace(0,1) * 100
        
        clean_df = df.dropna()
        if clean_df.empty:
            raise ValueError("clean_df empty")
        last = clean_df.iloc[-1]
        price = float(last["Close"])
    except Exception as e:
        print(f"Feature calc failed: {e}")
        price = finnhub_price if finnhub_price else 175.0
        return {
            "signal": "GÖZLƏ", "conf": 50.0, "price": price,
            "rsi": 55.0, "ma20": price*0.98, "ma50": price*0.95,
            "al_pct": 33.3, "sat_pct": 33.3, "gozle_pct": 33.4,
            "model": "EMERGENCY FALLBACK", "open": price*0.998
        }
    
    try:
        model_path = find_file(["NVDA_model.pkl"])
        scaler_path = find_file(["NVDA_scaler.pkl"])
        features_path = find_file(["NVDA_features.json"])
        
        if not model_path or not scaler_path or not features_path:
            raise FileNotFoundError(f"Model files not found. model={model_path} scaler={scaler_path} features={features_path} cwd={os.listdir('.')} data_exists={os.path.exists('data')}")

        print(f"Model files found: {model_path}, {scaler_path}, {features_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(features_path, "r") as f:
            feature_cols = json.load(f)
        
        X = np.array([[last[col] for col in feature_cols]])
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[0]
        pred = np.argmax(probs)
        
        label_map = {0: "SAT", 1: "GÖZLƏ", 2: "AL"}
        signal = label_map[pred]
        conf = float(np.max(probs) * 100)
        
        return {
            "signal": signal,
            "conf": conf,
            "price": price,
            "rsi": round(float(last["RSI"]),1),
            "ma20": round(float(last["SMA20"]),2),
            "ma50": round(float(last["SMA50"]),2),
            "al_pct": round(float(probs[2]*100),1),
            "sat_pct": round(float(probs[0]*100),1),
            "gozle_pct": round(float(probs[1]*100),1),
            "model": "REAL AI",
            "open": float(last["Open"])
        }
    except Exception as e:
        print(f"Model xətası fallback: {e}")
        rsi = float(last["RSI"]) if "RSI" in last else 55.0
        ma20 = float(last["SMA20"]) if "SMA20" in last else price*0.98
        ma50 = float(last["SMA50"]) if "SMA50" in last else price*0.95
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
            "model": "FALLBACK", "open": float(last["Open"]) if "Open" in last else price*0.998
        }

def main():
    baku_time = datetime.now(BAKU_TZ)
    print(f"NVDA Real AI - {baku_time.strftime('%d.%m %H:%M')} Bakı")
    result = get_nvda_features()
    print(f"Price: {result['price']} | Signal: {result['signal']} ({result['conf']:.1f}%)")
    print(f"AL:{result['al_pct']}% GOZLE:{result['gozle_pct']}% SAT:{result['sat_pct']}% RSI:{result['rsi']}")
    print(f"Model: {result['model']}")
    all_results = {
        "NVDA": {"1g": result, "1s": result, "3g": result, "5g": result},
        "live": {"ticker": "NVDA", "price": result["price"], "rsi": result["rsi"], "ma20": result["ma20"], "ma50": result["ma50"], "time": baku_time.strftime("%H:%M:%S"), "model": result["model"]},
        "last_update": baku_time.strftime("%d.%m.%Y %H:%M:%S")
    }
    for p in [os.path.join(DATA_DIR_STR, "predictions.json"), "data/predictions.json", "predictions.json"]:
        try:
            os.makedirs(os.path.dirname(p) if os.path.dirname(p) else ".", exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"{p} yazıldı")
        except Exception as e:
            print(f"Yazı xətası {p}: {e}")

if __name__ == "__main__":
    main()
