"""
NVDA Real AI Trainer - Həqiqi öyrənən süni intellekt
KO-dan NVDA-ya keçid üçün
"""
import pandas as pd
import numpy as np
import yfinance as yf
import os, pickle, json
from datetime import datetime

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False
    print("sklearn yoxdur, pip install scikit-learn")

def download_nvda_data():
    print("📥 NVDA datası yüklənir...")
    # 2 il günlük data - sənin db_NVDA_1d_2y.csv yerinə
    df = yf.download("NVDA", period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/db_NVDA_1d_2y.csv")
    print(f"✅ {len(df)} gün yükləndi: {df.index[0]} -> {df.index[-1]}")
    return df

def create_features(df):
    print("🔧 Feature-lar yaradılır...")
    df = df.copy()
    # Texniki indikatorlar
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
    
    # Bollinger
    df["BB_mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2*bb_std
    df["BB_lower"] = df["BB_mid"] - 2*bb_std
    df["BB_pos"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    
    # Volume & ATR
    df["Volume_SMA"] = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / df["Volume_SMA"]
    df["High_Low"] = df["High"] - df["Low"]
    df["ATR"] = df["High_Low"].rolling(14).mean()
    
    # Price position
    df["Price_vs_SMA20"] = (df["Close"] - df["SMA20"]) / df["SMA20"] * 100
    df["Price_vs_SMA50"] = (df["Close"] - df["SMA50"]) / df["SMA50"] * 100
    df["SMA20_vs_SMA50"] = (df["SMA20"] - df["SMA50"]) / df["SMA50"] * 100
    
    # Gələcək return - LABEL üçün (NVDA üçün 1.5% threshold, KO üçün 0.5% idi)
    df["future_close_3d"] = df["Close"].shift(-3)
    df["future_ret_3d"] = (df["future_close_3d"] - df["Close"]) / df["Close"] * 100
    
    def label_return(r):
        if r > 1.5:
            return 2  # AL
        elif r < -1.5:
            return 0  # SAT
        else:
            return 1  # GÖZLƏ
    
    df["label"] = df["future_ret_3d"].apply(label_return)
    
    return df.dropna()

def train_model(df):
    print("🧠 Model öyrənir...")
    feature_cols = [
        "RSI", "MACD", "MACD_signal", "BB_pos", "Vol_ratio", 
        "Price_vs_SMA20", "Price_vs_SMA50", "SMA20_vs_SMA50",
        "ATR", "High_Low"
    ]
    
    X = df[feature_cols]
    y = df["label"]
    
    # Train / Test split - son 20% test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model - RandomForest (XGBoost əvəzi, daha stabil)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Test
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["SAT","GÖZLƏ","AL"]))
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\n🔝 Ən vacib feature-lar:")
    print(importance.to_string())
    
    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/NVDA_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("data/NVDA_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("data/NVDA_features.json", "w") as f:
        json.dump(feature_cols, f)
    
    print("\n✅ Model saxlandı: data/NVDA_model.pkl")
    return model, scaler, feature_cols

if __name__ == "__main__":
    if not SKLEARN_AVAILABLE:
        print("sklearn lazımdır: pip install scikit-learn pandas yfinance")
        exit()
    
    df = download_nvda_data()
    df_feat = create_features(df)
    print(f"Feature-lı data: {len(df_feat)} sətir")
    model, scaler, features = train_model(df_feat)
    print("\n🎉 NVDA AI hazırdır! İndi robot_fixed.py bunu istifadə edəcək.")
