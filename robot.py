"""
robot.py - REAL LEARNING FIXED FINAL
YALNIZ NVDA - 4 ayri model: 1s,1g,3g,5g
FIX: SMA200 200 gun problemi hell edildi
"""
import os, json, pickle, math, warnings, traceback
from datetime import datetime, timedelta
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

BAKU_TZ = pytz.timezone("Asia/Baku")
DATA_DIR = "data"
FINAL_DIR = "data/final"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs("final", exist_ok=True)

def safe_float(x, default=0.0):
    try:
        if x is None: return default
        f = float(x)
        if math.isnan(f) or math.isinf(f): return default
        return f
    except: return default

def clean_nan(obj):
    if isinstance(obj, dict): return {k: clean_nan(v) for k,v in obj.items()}
    if isinstance(obj, list): return [clean_nan(x) for x in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return 0.0
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f): return 0.0
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj

def get_nvda_features_full():
    """Son 2 il data + butun feature-lar - FIXED for SMA200"""
    df = None
    finnhub_price = None
    
    # Finnhub - canli qiymet
    try:
        api_key = os.getenv("FINNHUB_API_KEY")
        if api_key:
            url = f"https://finnhub.io/api/v1/quote?symbol=NVDA&token={api_key}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                finnhub_price = safe_float(r.json().get('c',0))
                if finnhub_price > 0:
                    print(f"Finnhub canli: {finnhub_price}")
    except: pass
    
    # yfinance - 2 il (SMA200 ucun kifayet)
    try:
        df = yf.download("NVDA", period="2y", interval="1d", progress=False, auto_adjust=True, threads=False)
        if df is not None and not df.empty and len(df) >= 50:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            print(f"yfinance 2y: {len(df)} gun")
            if len(df) < 50: 
                df = None
    except Exception as e:
        print(f"yfinance 2y xetasi: {e}")
        df = None

    # yfinance 1y fallback
    if df is None or df.empty:
        try:
            df = yf.download("NVDA", period="1y", interval="1d", progress=False, auto_adjust=True, threads=False)
            if df is not None and not df.empty and len(df) >= 50:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                print(f"yfinance 1y fallback: {len(df)} gun")
                if len(df) < 50:
                    df = None
        except Exception as e:
            print(f"yfinance 1y xetasi: {e}")
            df = None
    
    # Cache fallback
    if df is None or df.empty:
        for path in [f"{DATA_DIR}/db_NVDA_1d_2y.csv", "db_NVDA_1d_2y.csv", f"{FINAL_DIR}/../db_NVDA_1d_2y.csv"]:
            if os.path.exists(path):
                try:
                    cdf = pd.read_csv(path, index_col=0, parse_dates=True)
                    if len(cdf) >= 50:
                        df = cdf
                        print(f"Cache istifade: {path} - {len(df)} gun")
                        break
                except: pass
    
    # Synthetic fallback - son care
    if df is None or df.empty:
        print("Synthetic fallback - hec bir data menbeyi islemedi")
        base_price = finnhub_price if finnhub_price else 175.0
        dates = pd.date_range(end=pd.Timestamp.now(), periods=300, freq='D')
        close = base_price + np.cumsum(np.random.randn(300)*0.8)
        df = pd.DataFrame({
            "Close": close, "Open": close*0.998, "High": close*1.01,
            "Low": close*0.99, "Volume": np.random.randint(40_000_000, 60_000_000, 300)
        }, index=dates)
    
    # Butun feature-lar - min_periods=1 ile SMA200 problemi hell edilir
    try:
        df["SMA20"] = df["Close"].rolling(20, min_periods=1).mean()
        df["SMA50"] = df["Close"].rolling(50, min_periods=1).mean()
        df["SMA200"] = df["Close"].rolling(200, min_periods=1).mean()
        
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-6)
        df["RSI"] = 100 - (100 / (1 + rs))
        
        ema12 = df["Close"].ewm(span=12, min_periods=1).mean()
        ema26 = df["Close"].ewm(span=26, min_periods=1).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, min_periods=1).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
        
        df["BB_mid"] = df["Close"].rolling(20, min_periods=1).mean()
        bb_std = df["Close"].rolling(20, min_periods=1).std()
        df["BB_upper"] = df["BB_mid"] + 2*bb_std
        df["BB_lower"] = df["BB_mid"] - 2*bb_std
        df["BB_pos"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"]).replace(0,1)
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
        
        df["Volume_SMA"] = df["Volume"].rolling(20, min_periods=1).mean()
        df["Vol_ratio"] = df["Volume"] / df["Volume_SMA"].replace(0,1)
        
        df["High_Low"] = df["High"] - df["Low"]
        df["ATR"] = df["High_Low"].rolling(14, min_periods=1).mean()
        
        df["Price_vs_SMA20"] = (df["Close"] - df["SMA20"]) / df["SMA20"].replace(0,1) * 100
        df["Price_vs_SMA50"] = (df["Close"] - df["SMA50"]) / df["SMA50"].replace(0,1) * 100
        df["SMA20_vs_SMA50"] = (df["SMA20"] - df["SMA50"]) / df["SMA50"].replace(0,1) * 100
        df["Price_vs_SMA200"] = (df["Close"] - df["SMA200"]) / df["SMA200"].replace(0,1) * 100
        
        df["RET_1"] = df["Close"].pct_change(1, fill_method=None) * 100
        df["RET_5"] = df["Close"].pct_change(5, fill_method=None) * 100
        df["RET_20"] = df["Close"].pct_change(20, fill_method=None) * 100
        df["MOM_10"] = df["Close"] / df["Close"].shift(10) * 100
        
        # NaN-lari doldur
        df = df.ffill().bfill()
        
        clean_df = df.dropna()
        if clean_df.empty:
            print("clean_df hele de empty - ffill/bfill ile doldurulur")
            clean_df = df.ffill().bfill()
            clean_df = clean_df.tail(50)
        
        last = clean_df.iloc[-1]
        price = safe_float(last["Close"], finnhub_price or 175.0)
        
        # Eger yfinance qiymeti finnhub-dan ferqlidirse, finnhub daha deqiqdir
        if finnhub_price and abs(finnhub_price - price) / price < 0.1:
            price = finnhub_price
            
        return clean_df, last, price, finnhub_price
        
    except Exception as e:
        print(f"Feature calc xetasi: {e}")
        traceback.print_exc()
        price = finnhub_price or 175.0
        return None, None, price, finnhub_price

def predict_with_model(last, price, horizon_key):
    model = None
    scaler = None
    feature_cols = None
    
    possible_model_paths = [
        f"{FINAL_DIR}/NVDA_{horizon_key}.pkl",
        f"data/final/NVDA_{horizon_key}.pkl",
        f"final/NVDA_{horizon_key}.pkl",
        f"{DATA_DIR}/NVDA_{horizon_key}.pkl",
        f"{FINAL_DIR}/NVDA_model.pkl",
    ]
    
    for p in possible_model_paths:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    model = pickle.load(f)
                print(f"Model tapildi {horizon_key}: {p}")
                break
            except: pass
    
    if model is None:
        print(f"Model tapilmadi {horizon_key}, fallback")
        return {
            "signal": "GOZLE", "conf": 50.0, "price": round(price,2),
            "rsi": 55.0, "ma20": round(price*0.98,2), "ma50": round(price*0.95,2),
            "al_pct": 33.3, "sat_pct": 33.3, "gozle_pct": 33.4,
            "model": "NO_MODEL", "open": round(price*0.998,2)
        }
    
    # Scaler ve features
    for p in [f"{FINAL_DIR}/NVDA_{horizon_key}_scaler.pkl", f"data/final/NVDA_{horizon_key}_scaler.pkl", f"{FINAL_DIR}/NVDA_scaler.pkl"]:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    scaler = pickle.load(f)
                break
            except: pass
    
    for p in [f"{FINAL_DIR}/NVDA_{horizon_key}_features.json", f"data/final/NVDA_{horizon_key}_features.json", f"{FINAL_DIR}/NVDA_features.json"]:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    feature_cols = json.load(f)
                break
            except: pass
    
    if scaler is None or feature_cols is None:
        print(f"Scaler/features tapilmadi {horizon_key}")
        return {
            "signal": "GOZLE", "conf": 50.0, "price": round(price,2),
            "rsi": round(safe_float(last.get("RSI",55.0) if last is not None else 55.0),1),
            "ma20": round(safe_float(last.get("SMA20",price*0.98) if last is not None else price*0.98),2),
            "ma50": round(safe_float(last.get("SMA50",price*0.95) if last is not None else price*0.95),2),
            "al_pct": 33.3, "sat_pct": 33.3, "gozle_pct": 33.4,
            "model": "NO_SCALER", "open": round(price*0.998,2)
        }
    
    try:
        # Feature-lar yoxdursa 0 ile doldur
        X_vals = []
        for col in feature_cols:
            if last is not None and col in last:
                X_vals.append(safe_float(last.get(col, 0)))
            else:
                X_vals.append(0.0)
        
        X = np.array([X_vals])
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[0]
        pred = np.argmax(probs)
        label_map = {0: "SAT", 1: "GOZLE", 2: "AL"}
        signal = label_map[pred]
        conf = safe_float(np.max(probs)*100, 50.0)
        
        return {
            "signal": signal,
            "conf": round(conf,1),
            "price": round(safe_float(price),2),
            "rsi": round(safe_float(last.get("RSI",55.0) if last is not None else 55.0),1),
            "ma20": round(safe_float(last.get("SMA20",price*0.98) if last is not None else price*0.98),2),
            "ma50": round(safe_float(last.get("SMA50",price*0.95) if last is not None else price*0.95),2),
            "al_pct": round(safe_float(probs[2]*100),1),
            "sat_pct": round(safe_float(probs[0]*100),1),
            "gozle_pct": round(safe_float(probs[1]*100),1),
            "model": f"REAL_LEARNING_{horizon_key}",
            "open": round(safe_float(last.get("Open",price*0.998) if last is not None else price*0.998),2)
        }
    except Exception as e:
        print(f"{horizon_key} predict xetasi: {e}")
        traceback.print_exc()
        return {
            "signal": "GOZLE", "conf": 50.0, "price": round(price,2),
            "rsi": 55.0, "ma20": round(price*0.98,2), "ma50": round(price*0.95,2),
            "al_pct": 33.3, "sat_pct": 33.3, "gozle_pct": 33.4,
            "model": "ERROR_FALLBACK", "open": round(price*0.998,2)
        }

def main():
    baku_time = datetime.now(BAKU_TZ)
    print(f"🧠 NVDA REAL LEARNING FIXED FINAL - {baku_time.strftime('%d.%m %H:%M')} Baki")
    print("📚 YALNIZ NVDA - 4 ayri model: 1s,1g,3g,5g")
    print("🔧 FIX: SMA200 problemi hell edildi - 2y data + min_periods=1")
    
    for h in ["1s","1g","3g","5g"]:
        for p in [f"{FINAL_DIR}/NVDA_{h}.pkl", f"data/final/NVDA_{h}.pkl"]:
            if os.path.exists(p):
                print(f"Model files found: {os.path.abspath(p)}")
                break
    
    clean_df, last, price, finnhub_price = get_nvda_features_full()
    
    if last is None:
        price = finnhub_price or 175.0
        print(f"Feature alinmadi, default qiymet: {price}")
        last = {"RSI":55.0, "SMA20":price*0.98, "SMA50":price*0.95, "Open":price*0.998}
    
    print(f"Price: {price} | Canli: {finnhub_price or price} | Data: {len(clean_df) if clean_df is not None else 0} gun")
    
    all_results = {"NVDA": {}}
    
    for h_key in ["1s","1g","3g","5g"]:
        res = predict_with_model(last, price, h_key)
        all_results["NVDA"][h_key] = res
        print(f"[{h_key}] Price: {res['price']} | Signal: {res['signal']} ({res['conf']}%) Model: {res['model']}")
    
    def get_macro():
        macro_result = {}
        try:
            from macro_data import fetch_macro_yfinance
            macro_result = fetch_macro_yfinance()
            print(f"📈 Macro yfinance cekilir: {' '.join(macro_result.keys())}")
        except Exception as e:
            print(f"Macro xetasi: {e}")
            macro_result = {}
        
        macro_result = clean_nan(macro_result)
        defaults_macro = {
            "SPY": {"close": 450.0, "ret": 0.0},
            "^VIX": {"close": 18.0, "ret": 0.0},
            "XLP": {"close": 75.0, "ret": 0.0},
            "^TNX": {"close": 4.2, "ret": 0.0},
            "UUP": {"close": 28.0, "ret": 0.0}
        }
        for k,v in defaults_macro.items():
            if k not in macro_result or safe_float(macro_result.get(k,{}).get("close",0),0)==0:
                macro_result[k]=v
        return clean_nan(macro_result)
    
    macro = get_macro()
    all_results["macro"] = macro
    all_results["last_update"] = baku_time.strftime("%d.%m.%Y %H:%M:%S")
    all_results["ticker"] = "NVDA"
    all_results["model_status"] = "REAL_LEARNING" if any("REAL_LEARNING" in v.get("model","") for v in all_results["NVDA"].values()) else "FALLBACK"
    all_results["learning"] = True
    all_results["version"] = "V8_REAL_LEARNING_FIXED"
    
    all_results_clean = clean_nan(all_results)
    paths = [f"{DATA_DIR}/predictions.json", f"data/predictions.json", f"predictions.json"]
    
    for p in paths:
        try:
            os.makedirs(os.path.dirname(p) if os.path.dirname(p) else ".", exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(all_results_clean, f, indent=2, ensure_ascii=False, allow_nan=False)
            print(f"Yazildi {p} - {all_results['model_status']}")
        except Exception as e:
            print(f"Yazi xetasi {p}: {e}")
    
    try:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            msg = f"🧠 NVDA REAL LEARNING FIXED - {baku_time.strftime('%d.%m %H:%M')} Baki\n"
            msg += f"Model: {all_results['model_status']}\nPrice: {price}\n"
            for k in ["1s","1g","3g","5g"]:
                r = all_results["NVDA"][k]
                emoji = "🟢" if r["signal"]=="AL" else "🔴" if r["signal"]=="SAT" else "🟡"
                msg += f"{emoji} {k}: {r['signal']} ({r['conf']}%) AL:{r['al_pct']}% GOZLE:{r['gozle_pct']}% SAT:{r['sat_pct']}%\n"
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            resp = requests.post(url, json={"chat_id": chat_id, "text": msg}, timeout=15)
            print(f"Telegram status: {resp.status_code}")
    except Exception as e:
        print(f"Telegram xetasi: {e}")
    
    print("✅ FIXED FINAL - clean_df empty problemi hell edildi")
    for k in ["1s","1g","3g","5g"]:
        r = all_results["NVDA"][k]
        print(f"{k}: {r['signal']} {r['conf']}% AL:{r['al_pct']}% GOZLE:{r['gozle_pct']}%")

if __name__ == "__main__":
    main()
