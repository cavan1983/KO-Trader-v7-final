"""
TRADE PRO V8.0 - KO ONLY + MACRO + FRED + SMART
"""
import os, json, pickle, warnings, traceback
from datetime import datetime
from news_sentiment import get_news_sentiment
from macro_data import fetch_macro_yfinance, fetch_finnhub_extras, fetch_yahoo_earnings, fetch_fred_data
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
    print("✅ TF + sklearn OK")
except Exception as e:
    TF_AVAILABLE = False
    print(f"⚠ TF yoxdur: {e}")

TICKERS = ["KO"]
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

print("🧠 KO V8 - MACRO + SENTIMENT + EARNINGS + FRED")

def get_times():
    baku = pytz.timezone("Asia/Baku")
    ny = pytz.timezone("America/New_York")
    return datetime.now(baku), datetime.now(ny)

def is_us_market_open():
    try:
        _, now_ny = get_times()
        if now_ny.weekday() >= 5: return False
        open_t = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_t <= now_ny <= close_t
    except: return True

def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        print(f"Telegram: {r.status_code}")
    except Exception as e:
        print(f"Telegram xətası: {e}")

def fetch_data(ticker, period, interval):
    db_path = f"{DATA_DIR}/db_{ticker}_{interval}_{period}.csv"
    df_db = None
    if os.path.exists(db_path):
        try:
            df_db = pd.read_csv(db_path, parse_dates=True, index_col=0)
            df_db.index = pd.to_datetime(df_db.index, errors='coerce')
            df_db = df_db[~df_db.index.isna()].sort_index()
            if df_db.empty:
                df_db = None
        except Exception as e:
            print(f"db read fail {e}, silirem")
            try: os.remove(db_path)
            except: pass
            df_db = None

    fetch_period = "5d" if df_db is not None and interval == "1d" else ("7d" if df_db is not None else period)
    print(f"[1] fetch {ticker} {interval} {period} -> {fetch_period}")

    df_new = None
    for _ in range(2):
        try:
            df = yf.download(ticker, period=fetch_period, interval=interval, progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.dropna().sort_index()
            if len(df) > 5:
                df_new = df
                break
        except Exception as e:
            print(f" -> FAIL {e}")

    if df_new is None:
        return df_db

    if df_db is not None:
        try:
            combined = pd.concat([df_db, df_new])
            combined.index = pd.to_datetime(combined.index, errors='coerce')
            combined = combined[~combined.index.isna()]
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            combined.to_csv(db_path)
            return combined
        except Exception as e:
            print(f" -> concat fail {e}, db yeniden yaradilir")
            try: os.remove(db_path)
            except: pass
            df_new.to_csv(db_path)
            return df_new
    else:
        df_new.to_csv(db_path)
        return df_new

def build_model(input_shape):
    K.clear_session()
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_xy(df, horizon_key, macro, extras, news_sent, fred=None):
    try:
        df = df.copy()
        close = df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:,0]
        vol = df['Volume']
        if isinstance(vol, pd.DataFrame): vol = vol.iloc[:,0]
        df['Close'] = close
        df['Volume'] = vol
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        df['RET'] = df['Close'].pct_change()
        df['VOL20'] = df['Volume'].rolling(20).mean()
        df['VOL_CH'] = (df['Volume'] - df['VOL20']) / df['VOL20'] * 100
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))

        vix = macro.get("^VIX", {}).get("close", 20) if macro else 20
        spy_ret = macro.get("SPY", {}).get("ret", 0) if macro else 0
        xlp_ret = macro.get("XLP", {}).get("ret", 0) if macro else 0
        tnx = macro.get("^TNX", {}).get("close", 4.0) if macro else 4.0
        uup_ret = macro.get("UUP", {}).get("ret", 0) if macro else 0

        df['VIX'] = vix
        df['SPY_RET'] = spy_ret
        df['XLP_RET'] = xlp_ret
        df['TNX'] = tnx
        df['UUP_RET'] = uup_ret
        df['NEWS_SENT'] = news_sent if news_sent is not None else 0
        df['EARN_DAYS'] = extras.get("earnings_days", 30) if extras else 30
        df['INSIDER'] = extras.get("insider_score", 0) if extras else 0

        # FRED features
        if fred:
            df['FED_FUNDS'] = fred.get('fed_funds', 4.5)
            df['CPI_YOY'] = fred.get('cpi_yoy', 3.0)
        else:
            df['FED_FUNDS'] = 4.5
            df['CPI_YOY'] = 3.0

        shift_n = 1
        if horizon_key == "3g": shift_n = 3
        elif horizon_key == "5g": shift_n = 5
        df['FUTURE'] = df['Close'].shift(-shift_n)
        df['CHANGE'] = (df['FUTURE'] - df['Close']) / df['Close'] * 100
        def label_change(ch):
            if ch > 1.2: return 0
            elif ch < -1.2: return 2
            else: return 1
        df['LABEL'] = df['CHANGE'].apply(label_change)
        df = df.dropna()
        if len(df) < 80: return None, None, None, None
        features = ['MA20','MA50','MA200','RET','RSI','VOL_CH','VIX','SPY_RET','XLP_RET','TNX','UUP_RET','NEWS_SENT','EARN_DAYS','INSIDER','FED_FUNDS','CPI_YOY']
        df[features] = df[features].bfill().fillna(0)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features].values)
        seq_len = 60
        if len(scaled) <= seq_len: return None, None, None, None
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i])
            y.append(df['LABEL'].iloc[i])
        from tensorflow.keras.utils import to_categorical
        X = np.array(X)
        y = to_categorical(y, num_classes=3)
        last_price = float(df['Close'].iloc[-1])
        last_open = float(df['Open'].iloc[-1]) if 'Open' in df else last_price
        meta = {"price": last_price, "open": last_open, "ma20": float(df['MA20'].iloc[-1]), "ma50": float(df['MA50'].iloc[-1]), "ma200": float(df['MA200'].iloc[-1]), "rsi": float(df['RSI'].iloc[-1]), "vol_change": float(df['VOL_CH'].iloc[-1]), "vix": float(vix), "spy_ret": float(spy_ret), "xlp_ret": float(xlp_ret), "tnx": float(tnx), "news_sent": int(news_sent if news_sent else 0), "earn_days": int(extras.get("earnings_days",30) if extras else 30), "insider": int(extras.get("insider_score",0) if extras else 0), "fed_funds": fred.get('fed_funds',0) if fred else 0, "cpi": fred.get('cpi_yoy',0) if fred else 0}
        return X, y, scaler, meta
    except Exception as e:
        print(f"prepare error {e}")
        traceback.print_exc()
        return None, None, None, None

def train_for_ticker(ticker, macro, extras, news_sent, fred=None):
    configs = {"1s": {"period": "2y", "interval": "1h"}, "1g": {"period": "2y", "interval": "1d"}, "3g": {"period": "2y", "interval": "1d"}, "5g": {"period": "2y", "interval": "1d"}}
    results = {}
    for hk, cfg in configs.items():
        try:
            if hk == "1s" and not is_us_market_open(): print(f"skip 1s - bazar bağlı")
            df = fetch_data(ticker, cfg["period"], cfg["interval"])
            if df is None:
                results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": 0.0, "open": 0.0, "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}}
                continue
            X, y, scaler, meta = prepare_xy(df, hk, macro, extras, news_sent, fred)
            if X is None:
                results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": float(df['Close'].iloc[-1]), "open": float(df['Open'].iloc[-1]), "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}}
                continue
            input_shape = (X.shape[1], X.shape[2])
            brain_path = f"{DATA_DIR}/brain_{ticker}_{hk}.keras"
            scaler_path = f"{DATA_DIR}/scaler_{ticker}_{hk}.pkl"
            model = None
            if os.path.exists(brain_path):
                try:
                    from tensorflow.keras.models import load_model
                    model = load_model(brain_path)
                    print(f"🧠 Köhnə beyin: {brain_path}")
                except: model = build_model(input_shape)
            else: model = build_model(input_shape)
            print(f"[4] train {ticker} {hk} {X.shape}")
            if TF_AVAILABLE:
                early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)
                model.fit(X, y, epochs=30, batch_size=16, verbose=0, callbacks=[early_stop])
                model.save(brain_path)
                with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
            last_seq = X[-1:]
            pred = model.predict(last_seq, verbose=0)[0]
            idx = int(np.argmax(pred))
            signals = ["AL", "GÖZLƏ", "SAT"]
            signal = signals[idx]
            conf = float(pred[idx] * 100)
            results[hk] = {"signal": signal, "conf": conf, "price": meta['price'], "open": meta['open'], "probs": {"AL": float(pred[0]*100), "GÖZLƏ": float(pred[1]*100), "SAT": float(pred[2]*100)}, "meta": meta}
            print(f"[5] PRED {ticker} {hk}: {signal} {conf:.0f}% @ {meta['price']:.2f}")
        except Exception as e:
            print(f"[FAIL] {ticker} {hk}: {e}")
            traceback.print_exc()
            results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": 0.0, "open": 0.0, "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}}
    return results

def main():
    baku, ny = get_times()
    print(f"V8.0 KO MACRO+FRED - {baku} | NY {ny.strftime('%A %H:%M')} | Market: {is_us_market_open()}")
    if ny.weekday() >= 5:
        print("🔴 WEEKEND - exit")
        # YENƏ DƏ fayl yarat ki GitHub boş deməsin
        os.makedirs(DATA_DIR, exist_ok=True)
        dummy = {"KO": {"1s": {"signal":"GÖZLƏ","conf":50},"1g": {"signal":"GÖZLƏ","conf":50},"3g": {"signal":"GÖZLƏ","conf":50},"5g": {"signal":"GÖZLƏ","conf":50}}, "updated": str(baku), "market_open": False}
        with open(f"{DATA_DIR}/predictions.json","w") as f: json.dump(dummy,f,indent=2)
        return
    macro = fetch_macro_yfinance()
    fred = fetch_fred_data()
    extras = fetch_finnhub_extras("KO")
    yahoo_days = fetch_yahoo_earnings("KO")
    if extras.get("earnings_days", 30) == 30 and yahoo_days!= 30: extras["earnings_days"] = yahoo_days
    news_data = {}
    for ticker in TICKERS:
        try:
            sent, head, is_new = get_news_sentiment(ticker)
            news_data[ticker] = (sent, head, is_new)
            print(f"📰 News {ticker}: {sent} | {head[:80]}")
        except Exception as e:
            print(f"News err {ticker}: {e}")
            news_data[ticker] = (0, "", False)
    all_results = {}
    for ticker in TICKERS:
        try:
            ns = news_data.get(ticker, (0,"",False))[0]
            res = train_for_ticker(ticker, macro, extras, ns, fred)
            all_results[ticker] = res
        except Exception as e:
            print(f"❌ {ticker} fail: {e}")
            all_results[ticker] = {}

    # YAZ
    os.makedirs(DATA_DIR, exist_ok=True)
    final_out = {"predictions": all_results, "updated": str(baku), "macro": macro, "fred": fred, "extras": extras}
    with open(f"{DATA_DIR}/predictions.json","w") as f: json.dump(final_out,f,indent=2)
    print(f"💾 Saved {DATA_DIR}/predictions.json")
    # Telegram
    try:
        msg = f"🤖 KO V8 {baku.strftime('%d.%m %H:%M')} Baku\n"
        for t in TICKERS:
            for k in ["1s","1g","3g","5g"]:
                if t in all_results and k in all_results[t]:
                    r = all_results[t][k]
                    msg += f"{t} {k}: {r['signal']} {r['conf']:.0f}% @ ${r['price']:.2f}\n"
        send_telegram(msg)
    except Exception as e: print(f"TG err {e}")

if __name__ == "__main__":
    main()
