"""
train_nvda_REAL_LEARNING.py
HƏQİQƏTƏN ÖYRƏNƏN AI - Hər səhvindən dərs alır, hər həftə yenilənir
YALNIZ NVDA - KO yoxdur!
"""
import os, json, pickle, math, warnings
from datetime import datetime, timedelta
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BAKU_TZ = pytz.timezone("Asia/Baku")
DATA_DIR = "data"
FINAL_DIR = "data/final"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

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

def fetch_nvda_history():
    """3 il NVDA tarixçəsi - yfinance + cache + Finnhub fallback"""
    print("📥 NVDA tarixçəsi çəkilir (3 il)...")
    df = None
    
    # 1. yfinance
    try:
        df = yf.download("NVDA", period="3y", interval="1d", progress=False, auto_adjust=True, threads=False)
        if df is not None and not df.empty and len(df) > 200:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            print(f"✅ yfinance: {len(df)} gün")
    except Exception as e:
        print(f"yfinance xətası: {e}")
        df = None
    
    # 2. Cache
    if df is None or len(df) < 200:
        for path in [f"{DATA_DIR}/db_NVDA_1d_2y.csv", "data/db_NVDA_1d_2y.csv"]:
            if os.path.exists(path):
                try:
                    cdf = pd.read_csv(path, index_col=0, parse_dates=True)
                    if len(cdf) > 200:
                        df = cdf
                        print(f"✅ Cache istifadə: {path} ({len(df)} gün)")
                        break
                except: pass
    
    # 3. Finnhub ilə son qiyməti al
    finnhub_price = None
    try:
        api_key = os.getenv("FINNHUB_API_KEY")
        if api_key:
            url = f"https://finnhub.io/api/v1/quote?symbol=NVDA&token={api_key}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                finnhub_price = safe_float(data.get('c',0))
                print(f"✅ Finnhub canlı: {finnhub_price}")
    except: pass
    
    if df is None or len(df) < 50:
        raise ValueError("Heç bir data mənbəyi işləmədi!")
    
    return df, finnhub_price

def add_features(df):
    """16+ feature hesabla - REAL AI üçün"""
    print("🔧 Feature-lar hesablanır...")
    df = df.copy()
    
    # Əsas
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    
    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-6)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    
    # Bollinger
    df["BB_mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2*bb_std
    df["BB_lower"] = df["BB_mid"] - 2*bb_std
    df["BB_pos"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"]).replace(0,1)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    
    # Volume
    df["Volume_SMA"] = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / df["Volume_SMA"].replace(0,1)
    
    # Volatility
    df["High_Low"] = df["High"] - df["Low"]
    df["ATR"] = df["High_Low"].rolling(14).mean()
    
    # Price vs MA
    df["Price_vs_SMA20"] = (df["Close"] - df["SMA20"]) / df["SMA20"] * 100
    df["Price_vs_SMA50"] = (df["Close"] - df["SMA50"]) / df["SMA50"] * 100
    df["SMA20_vs_SMA50"] = (df["SMA20"] - df["SMA50"]) / df["SMA50"] * 100
    df["Price_vs_SMA200"] = (df["Close"] - df["SMA200"]) / df["SMA200"] * 100
    
    # Returns
    df["RET_1"] = df["Close"].pct_change(1) * 100
    df["RET_5"] = df["Close"].pct_change(5) * 100
    df["RET_20"] = df["Close"].pct_change(20) * 100
    
    # Momentum
    df["MOM_10"] = df["Close"] / df["Close"].shift(10) * 100
    
    return df

def create_labels(df, horizon_days):
    """
    Label yarat: Gələcəkdə nə olacaq?
    horizon_days: 1s=1 gün, 1g=5 gün, 3g=15 gün, 5g=25 gün
    """
    df = df.copy()
    future_close = df["Close"].shift(-horizon_days)
    future_ret = (future_close - df["Close"]) / df["Close"] * 100
    
    # Dinamik threshold - volatilliyə görə
    vol = df["Close"].pct_change().rolling(20).std() * 100
    vol = vol.fillna(1.5)
    
    # Label: 0=SAT, 1=GÖZLƏ, 2=AL
    def label_func(ret, v):
        thresh = max(1.5, v*1.2)  # min 1.5%
        if ret > thresh: return 2  # AL
        elif ret < -thresh: return 0  # SAT
        else: return 1  # GÖZLƏ
    
    df["Label"] = [label_func(r, v) for r, v in zip(future_ret, vol)]
    df["Future_ret"] = future_ret
    
    return df

def load_journal_weights(df):
    """decision_journal.csv-dən səhv proqnozları ağırlıqla öyrən"""
    journal_path = f"{DATA_DIR}/decision_journal.csv"
    weights = np.ones(len(df))
    
    if not os.path.exists(journal_path):
        print("📓 Journal yoxdur, hamısı eyni ağırlıq")
        return weights
    
    try:
        j = pd.read_csv(journal_path)
        if j.empty or 'correct_5g' not in j.columns:
            return weights
        
        # Səhv proqnozların tarixlərini tap
        wrong = j[j['correct_5g'].str.contains("SƏHV", na=False)]
        print(f"📓 Journal: {len(j)} qərar, {len(wrong)} səhv")
        
        # Səhv günlərə daha çox ağırlıq ver
        # (sadə versiya - tarixə görə)
        for idx, row in wrong.iterrows():
            try:
                tarix = pd.to_datetime(row['tarix'], utc=True)
                # df-də həmin tarixə yaxın günlərin ağırlığını artır
                mask = (df.index >= tarix - pd.Timedelta(days=2)) & (df.index <= tarix + pd.Timedelta(days=2))
                weights[mask] *= 1.5  # səhv günlər 1.5x daha vacib
            except: pass
        
        print(f"📓 Ağırlıqlandırma tətbiq edildi")
    except Exception as e:
        print(f"Journal oxu xətası: {e}")
    
    return weights

def train_for_horizon(df_base, horizon_key, horizon_days):
    """Bir horizon üçün model öyrət"""
    print(f"\n{'='*60}")
    print(f"🎯 HORIZON: {horizon_key} ({horizon_days} gün) öyrədilir...")
    print(f"{'='*60}")
    
    df = create_labels(df_base, horizon_days)
    df = df.dropna(subset=["Label", "Future_ret"])
    
    if len(df) < 100:
        print(f"❌ {horizon_key}: kifayət qədər data yoxdur ({len(df)})")
        return None
    
    # Feature sütunları
    feature_cols = [
        "SMA20", "SMA50", "RSI", "MACD", "MACD_signal", "MACD_hist",
        "BB_pos", "BB_width", "Vol_ratio", "ATR",
        "Price_vs_SMA20", "Price_vs_SMA50", "SMA20_vs_SMA50",
        "RET_1", "RET_5", "RET_20", "MOM_10"
    ]
    
    # Mövcud olanları filtrelə
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"📊 Feature sayı: {len(feature_cols)}")
    
    df_clean = df.dropna(subset=feature_cols)
    print(f"📊 Təmiz data: {len(df_clean)} gün (əvvəl {len(df)})")
    
    if len(df_clean) < 100:
        print(f"❌ {horizon_key}: təmiz data azdır")
        return None
    
    X = df_clean[feature_cols].values
    y = df_clean["Label"].values
    
    # Journal ağırlıqları
    weights = load_journal_weights(df_clean)
    
    # Train/Test split - son 20% test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    w_train = weights[:split_idx]
    
    print(f"📚 Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"📊 Label paylanması Train: {np.bincount(y_train)} (0=SAT,1=GÖZLƏ,2=AL)")
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model - RandomForest + GradientBoosting ensemble
    print(f"🧠 Model öyrədilir (RandomForest)...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # GÖZLƏ çoxdursa balansla
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train, sample_weight=w_train)
    
    # Test
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {horizon_key} Test Accuracy: {acc*100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["SAT","GÖZLƏ","AL"]))
    
    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print(f"🔝 Top 5 feature {horizon_key}:")
    for f, imp in feat_imp[:5]:
        print(f"   {f}: {imp*100:.1f}%")
    
    # Save
    model_path = f"{FINAL_DIR}/NVDA_{horizon_key}.pkl"
    scaler_path = f"{FINAL_DIR}/NVDA_{horizon_key}_scaler.pkl"
    features_path = f"{FINAL_DIR}/NVDA_{horizon_key}_features.json"
    meta_path = f"{FINAL_DIR}/NVDA_{horizon_key}_meta.json"
    
    # Həm də köhnə adla saxla - uyğunluq üçün
    model_path_old = f"{FINAL_DIR}/NVDA_model.pkl"
    scaler_path_old = f"{FINAL_DIR}/NVDA_scaler.pkl"
    features_path_old = f"{FINAL_DIR}/NVDA_features.json"
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f)
    
    # Köhnə adla da saxla (robot_nvda_real_ai.py üçün)
    with open(model_path_old, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path_old, "wb") as f:
        pickle.dump(scaler, f)
    with open(features_path_old, "w") as f:
        json.dump(feature_cols, f)
    
    meta = {
        "horizon": horizon_key,
        "horizon_days": horizon_days,
        "train_date": datetime.now(BAKU_TZ).strftime("%Y-%m-%d %H:%M"),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": float(acc),
        "feature_count": len(feature_cols),
        "top_features": [{"name": f, "importance": float(imp)} for f, imp in feat_imp[:5]],
        "model_type": "RandomForest_REAL_LEARNING",
        "version": "V8_REAL_LEARNING"
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(clean_nan(meta), f, indent=2, ensure_ascii=False)
    
    # Həm də data/ qovluğuna kopyala
    os.makedirs(f"{DATA_DIR}/final", exist_ok=True)
    for src in [model_path, scaler_path, features_path, meta_path]:
        if os.path.exists(src):
            dst = src.replace("data/final", "data/final").replace(FINAL_DIR, f"{DATA_DIR}/final")
            # Artıq eyni yerdədir
    
    print(f"💾 Saxlandı: {model_path} ({os.path.getsize(model_path)/1024:.1f} KB)")
    
    return {
        "horizon": horizon_key,
        "accuracy": acc,
        "model_path": model_path
    }

def main():
    print("="*60)
    print("🧠 NVDA REAL LEARNING - HƏQİQƏTƏN ÖYRƏNƏN AI")
    print("="*60)
    print(f"🕐 Başlama: {datetime.now(BAKU_TZ).strftime('%d.%m.%Y %H:%M')} Bakı")
    print("📚 YALNIZ NVDA - KO yoxdur!")
    
    # Data çək
    df_raw, finnhub_price = fetch_nvda_history()
    
    # Feature əlavə et
    df_features = add_features(df_raw)
    print(f"✅ Feature-lar hazır: {len(df_features)} gün, {len(df_features.columns)} sütun")
    
    # Hər horizon üçün öyrət
    horizons = {
        "1s": 1,   # 1 gün sonra
        "1g": 5,   # 1 həftə (5 iş günü)
        "3g": 15,  # 3 həftə
        "5g": 25   # 5 həftə
    }
    
    results = []
    for h_key, h_days in horizons.items():
        try:
            res = train_for_horizon(df_features, h_key, h_days)
            if res:
                results.append(res)
        except Exception as e:
            print(f"❌ {h_key} öyrənmə xətası: {e}")
            import traceback
            traceback.print_exc()
    
    # Xülasə
    print("\n" + "="*60)
    print("🏆 ÖYRƏNMƏ TAMAMLANDI!")
    print("="*60)
    for r in results:
        print(f"✅ {r['horizon']}: {r['accuracy']*100:.1f}% dəqiqlik - {r['model_path']}")
    
    if len(results) == 4:
        print("\n✅ HƏM 4 HORIZON ÖYRƏNDİ - REAL LEARNING!")
        print("📁 Modellər: data/final/NVDA_1s.pkl, NVDA_1g.pkl, NVDA_3g.pkl, NVDA_5g.pkl")
    else:
        print(f"\n⚠️ Yalnız {len(results)}/4 horizon öyrəndi")
    
    # Ümumi meta
    summary = {
        "train_date": datetime.now(BAKU_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "baku_time": datetime.now(BAKU_TZ).strftime("%d.%m.%Y %H:%M"),
        "horizons_trained": len(results),
        "results": clean_nan(results),
        "model_type": "REAL_LEARNING_RandomForest",
        "ticker": "NVDA_ONLY",
        "version": "V8_REAL_LEARNING"
    }
    
    with open(f"{FINAL_DIR}/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 training_summary.json yazıldı")
    print("🚀 İndi robot.py REAL LEARNING modellərini istifadə edəcək!")

if __name__ == "__main__":
    main()
