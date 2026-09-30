"""
TRADE PRO V8.1 - KO ONLY + MACRO + FRED + SL/TP
DISCLAIMER: Bu bot və predictions.json yalnız təhsil/test üçündür.
AL/SAT/GÖZLƏ siqnalları maliyyə tövsiyəsi deyil.
Real pulla ticarət etməzdən əvvəl peşəkar məsləhətçi ilə məsləhətləşin.
"""
import os, json, pickle, warnings, traceback, csv
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
JOURNAL_PATH = f"{DATA_DIR}/decision_journal.csv"
BAKU_TZ = pytz.timezone("Asia/Baku")
NY_TZ = pytz.timezone("America/New_York")

def get_times():
    return datetime.now(BAKU_TZ), datetime.now(NY_TZ)

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
            df_db.index = pd.to_datetime(df_db.index, utc=True, errors='coerce')
            df_db = df_db[~df_db.index.isna()].sort_index()
            if df_db.empty: df_db = None
        except:
            try: os.remove(db_path)
            except: pass
            df_db = None
    fetch_period = "5d" if df_db is not None and interval == "1d" else ("7d" if df_db is not None else period)
    df_new = None
    for _ in range(2):
        try:
            df = yf.download(ticker, period=fetch_period, interval=interval, progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
            df = df.dropna().sort_index()
            if len(df) > 5:
                df_new = df
                break
        except Exception as e:
            print(f" -> FAIL {e}")
    if df_new is None: return df_db
    if df_db is not None:
        try:
            combined = pd.concat([df_db, df_new])
            combined.index = pd.to_datetime(combined.index, utc=True, errors='coerce')
            combined = combined[~combined.index.isna()]
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            combined.to_csv(db_path)
            return combined
        except:
            df_new.to_csv(db_path)
            return df_new
    else:
        df_new.to_csv(db_path)
        return df_new

def build_model(input_shape):
    K.clear_session()
    model = Sequential([LSTM(64, return_sequences=True, input_shape=input_shape), Dropout(0.2), LSTM(32), Dropout(0.2), Dense(16, activation='relu'), Dense(3, activation='softmax')])
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
        ns_val = news_sent if news_sent is not None else 0
        if abs(ns_val) > 1: ns_val = ns_val / 100.0
        df['NEWS_SENT'] = ns_val
        df['EARN_DAYS'] = extras.get("earnings_days", 30) if extras else 30
        df['INSIDER'] = extras.get("insider_score", 0) if extras else 0
        df['SPY_KO_DIFF'] = spy_ret - df['RET']*100
        df['MARKET_STRESS'] = vix * (1 if spy_ret < 0 else -1)
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
            if ch > 0.8: return 0
            elif ch < -0.8: return 2
            else: return 1
        df['LABEL'] = df['CHANGE'].apply(label_change)
        df = df.dropna()
        if len(df) < 80: return None, None, None, None
        features = ['MA20','MA50','MA200','RET','RSI','VOL_CH','VIX','SPY_RET','XLP_RET','TNX','UUP_RET','NEWS_SENT','EARN_DAYS','INSIDER','FED_FUNDS','CPI_YOY','SPY_KO_DIFF','MARKET_STRESS']
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
        meta = {"price": last_price, "open": last_open, "ma20": float(df['MA20'].iloc[-1]), "ma50": float(df['MA50'].iloc[-1]), "ma200": float(df['MA200'].iloc[-1]), "rsi": float(df['RSI'].iloc[-1]), "vol_change": float(df['VOL_CH'].iloc[-1]), "vix": float(vix), "spy_ret": float(spy_ret), "xlp_ret": float(xlp_ret), "tnx": float(tnx), "news_sent": float(ns_val), "earn_days": int(extras.get("earnings_days",30) if extras else 30), "insider": int(extras.get("insider_score",0) if extras else 0), "fed_funds": fred.get('fed_funds',0) if fred else 0, "cpi": fred.get('cpi_yoy',0) if fred else 0}
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
            if hk == "1s" and not is_us_market_open():
                results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": 0.0, "open": 0.0, "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}, "meta": {}, "last": 0}
                continue
            df = fetch_data(ticker, cfg["period"], cfg["interval"])
            if df is None:
                results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": 0.0, "open": 0.0, "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}, "meta": {}, "last": 0}
                continue
            X, y, scaler, meta = prepare_xy(df, hk, macro, extras, news_sent, fred)
            if X is None:
                results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": float(df['Close'].iloc[-1]), "open": float(df['Open'].iloc[-1]), "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}, "meta": {}, "last": float(df['Close'].iloc[-1])}
                continue
            input_shape = (X.shape[1], X.shape[2])
            brain_path = f"{DATA_DIR}/brain_{ticker}_{hk}.keras"
            scaler_path = f"{DATA_DIR}/scaler_{ticker}_{hk}.pkl"
            model = None
            if os.path.exists(brain_path):
                try:
                    from tensorflow.keras.models import load_model
                    model = load_model(brain_path)
                except: model = build_model(input_shape)
            else: model = build_model(input_shape)
            if TF_AVAILABLE:
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
                model.fit(X, y, epochs=30, batch_size=16, verbose=0, validation_split=0.2, callbacks=[early_stop])
                model.save(brain_path)
                with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
            last_seq = X[-1:]
            pred = model.predict(last_seq, verbose=0)[0]
            idx = int(np.argmax(pred))
            signals = ["AL", "GÖZLƏ", "SAT"]
            signal = signals[idx]
            conf = float(pred[idx] * 100)
            results[hk] = {"signal": signal, "conf": conf, "price": meta['price'], "open": meta['open'], "last": meta['price'], "probs": {"AL": float(pred[0]*100), "GÖZLƏ": float(pred[1]*100), "SAT": float(pred[2]*100)}, "meta": meta}
            print(f"[5] PRED {ticker} {hk}: {signal} {conf:.0f}% @ {meta['price']:.2f}")
        except Exception as e:
            print(f"[FAIL] {ticker} {hk}: {e}")
            traceback.print_exc()
            results[hk] = {"signal": "GÖZLƏ", "conf": 50.0, "price": 0.0, "open": 0.0, "last": 0, "probs": {"AL": 33.0, "GÖZLƏ": 50.0, "SAT": 17.0}, "meta": {}}
    return results

def journal_yaz(ticker, results, price_val, macro=None, fred=None, extras=None, news_sent=0):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        jp = f"{DATA_DIR}/decision_journal.csv"
        new = not os.path.exists(jp)
        with open(jp,"a",newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tarix","ticker","s1s","c1s","p1s","rsi1s","ma201s","s1g","c1g","p1g","rsi1g","ma201g","s3g","c3g","p3g","rsi3g","ma203g","s5g","c5g","p5g","rsi5g","ma205g","news_sent","earnings_days","insider","spy","xlp","tnx","uup","vix","fed","cpi","cpi_yoy","unemp","t10y2y"])
            def get_r(h, key, meta_key=None):
                d = results.get(h, {})
                if meta_key: return d.get("meta", {}).get(meta_key, "")
                return d.get(key, "")
            w.writerow([datetime.now(BAKU_TZ).strftime("%Y-%m-%d %H:%M:%S%z"), ticker, get_r("1s","signal"), round(float(get_r("1s","conf") or 0),1), get_r("1s","price"), get_r("1s",None,"rsi"), get_r("1s",None,"ma20"), get_r("1g","signal"), round(float(get_r("1g","conf") or 0),1), get_r("1g","price"), get_r("1g",None,"rsi"), get_r("1g",None,"ma20"), get_r("3g","signal"), round(float(get_r("3g","conf") or 0),1), get_r("3g","price"), get_r("3g",None,"rsi"), get_r("3g",None,"ma20"), get_r("5g","signal"), round(float(get_r("5g","conf") or 0),1), round(float(price_val or 0),2), get_r("5g",None,"rsi"), get_r("5g",None,"ma20"), round(float(news_sent or 0),3), extras.get("earnings_days","") if extras else "", extras.get("insider_score","") if extras else "", macro.get("SPY",{}).get("ret","") if macro else "", macro.get("XLP",{}).get("ret","") if macro else "", macro.get("^TNX",{}).get("close","") if macro else "", macro.get("UUP",{}).get("ret","") if macro else "", macro.get("^VIX",{}).get("close","") if macro else "", fred.get("fed_funds","") if fred else "", fred.get("cpi","") if fred else "", fred.get("cpi_yoy","") if fred else "", fred.get("unemp","") if fred else "", fred.get("t10y2y","") if fred else ""])
    except Exception as e:
        print(f"Journal yazı xətası: {e}")

def journal_oxu(ticker="KO", son_n=5):
    if not os.path.exists(JOURNAL_PATH): return None
    try:
        df = pd.read_csv(JOURNAL_PATH)
        if 'tarix' in df.columns:
            df['tarix'] = pd.to_datetime(df['tarix'], utc=True, errors='coerce')
            df['tarix'] = df['tarix'].dt.tz_convert(BAKU_TZ)
        df = df[df['ticker']==ticker].tail(son_n)
        if df.empty: return None
        print(f"📓 Son {len(df)} qərar")
        return df
    except: return None

def journal_qiymetlendir():
    try:
        if not os.path.exists(JOURNAL_PATH): return
        df_j = pd.read_csv(JOURNAL_PATH)
        if df_j.empty: return
        if 'real_p_5g' not in df_j.columns:
            df_j['real_p_5g'] = ""
            df_j['real_change_5g'] = ""
            df_j['correct_5g'] = ""
        hist = yf.download("KO", period="15d", interval="1d", progress=False, auto_adjust=True)
        if hist.empty: return
        if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
        hist.index = pd.to_datetime(hist.index, utc=True)
        updated = False
        for idx, row in df_j.iterrows():
            if pd.notna(row.get('correct_5g')) and str(row.get('correct_5g'))!= "": continue
            try:
                tarix = pd.to_datetime(row['tarix'], utc=True, errors='coerce')
                if pd.isna(tarix): continue
                if (datetime.now(pytz.UTC) - tarix).days < 5: continue
                p5g = float(row.get('p5g') or 0)
                if p5g == 0: continue
                future = hist[hist.index > tarix]
                if len(future) < 3: continue
                real_price = float(future.iloc[2]['Close'])
                real_change = (real_price - p5g) / p5g * 100
                s5g = str(row.get('s5g', ''))
                if s5g == "AL" and real_change > 0.8: correct = "✅ DÜZ"
                elif s5g == "SAT" and real_change < -0.8: correct = "✅ DÜZ"
                elif s5g == "GÖZLƏ" and abs(real_change) <= 0.8: correct = "✅ DÜZ"
                else: correct = "❌ SƏHV"
                df_j.at[idx, 'real_p_5g'] = round(real_price, 2)
                df_j.at[idx, 'real_change_5g'] = round(real_change, 2)
                df_j.at[idx, 'correct_5g'] = correct
                updated = True
            except: continue
        if updated:
            df_j.to_csv(JOURNAL_PATH, index=False)
            print(f"📊 Qiymətləndirildi")
    except Exception as e:
        print(f"qiymetlendir xətası: {e}")

def main():
    baku, ny = get_times()
    is_open = is_us_market_open()
    print(f"V8.1 SL/TP - {baku} | Market: {is_open}")
    os.makedirs(DATA_DIR, exist_ok=True)

    if not is_open:
        dummy = {"KO": {"1s": {"signal":"GÖZLƏ","conf":50,"price":0,"last":0},"1g": {"signal":"GÖZLƏ","conf":50,"price":0,"last":0},"3g": {"signal":"GÖZLƏ","conf":50,"price":0,"last":0},"5g": {"signal":"GÖZLƏ","conf":50,"price":0,"last":0}}, "updated": str(baku), "market_open": False}
        with open(f"{DATA_DIR}/predictions.json","w") as f: json.dump(dummy,f,indent=2)
        return

    journal_oxu("KO")
    journal_qiymetlendir()
    macro = fetch_macro_yfinance()
    fred = fetch_fred_data()
    extras = fetch_finnhub_extras("KO")
    yahoo_days = fetch_yahoo_earnings("KO")
    if extras.get("earnings_days", 30) == 30 and yahoo_days!= 30:
        extras["earnings_days"] = yahoo_days

    news_data = {}
    for ticker in TICKERS:
        try:
            sent, head, is_new = get_news_sentiment(ticker)
            news_data[ticker] = (sent, head, is_new)
        except:
            news_data[ticker] = (0, "", False)

    all_results = {}
    for ticker in TICKERS:
        res = {}
        try:
            ns = news_data.get(ticker, (0,"",False))[0]
            res = train_for_ticker(ticker, macro, extras, ns, fred)
            all_results[ticker] = res
        except Exception as e:
            print(f"❌ {ticker} fail: {e}")
            traceback.print_exc()
            all_results[ticker] = res if res else {}

    if "KO" in all_results:
        for hk in ["1s", "1g", "3g", "5g"]:
            if hk in all_results["KO"]:
                r = all_results["KO"][hk]
                if r["signal"] in ["AL", "SAT"] and r["conf"] < 55:
                    print(f"⚠ FIX4: {hk} {r['signal']} {r['conf']:.1f}% -> GOZLE")
                    r["signal"] = "GÖZLƏ"
                    r["probs"] = {"AL": 33.0, "GÖZLƏ": 60.0, "SAT": 7.0}

    # ===== FIX 5 + FIX 6: PAPER WALLET + SL/TP + AUTO RESET =====
    wallet_file = f"{DATA_DIR}/paper_wallet.json"
    try:
        if os.path.exists(wallet_file):
            with open(wallet_file, "r") as f:
                wallet = json.load(f)
        else:
            wallet = {"balance": 10000.0, "shares": 0, "buy_price": 0, "trades": [], "resets": 0}

        if "buy_price" not in wallet: wallet["buy_price"] = 0
        if "resets" not in wallet: wallet["resets"] = 0

        if "KO" in all_results and "1g" in all_results["KO"]:
            last_price = all_results["KO"]["1g"]["last"]
            signal = all_results["KO"]["1g"]["signal"]
            conf = all_results["KO"]["1g"]["conf"]

            total_value = wallet["balance"] + (wallet["shares"] * last_price if wallet["shares"] else 0)

            # RESET 9000 alti
            if total_value < 9000 and total_value > 0:
                print(f"💥 Wallet {total_value:.2f}$ < 9000 -> RESET 10k")
                wallet = {"balance": 10000.0, "shares": 0, "buy_price": 0, "trades": [f"RESET {baku}: {total_value:.2f}$ -> 10k"], "resets": wallet.get("resets",0)+1}
                total_value = 10000

            # AL
            if signal == "AL" and conf >= 60 and wallet["shares"] == 0 and wallet["balance"] > 0:
                wallet["shares"] = wallet["balance"] / last_price
                wallet["buy_price"] = last_price
                wallet["balance"] = 0
                wallet["trades"].append(f"AL {baku.strftime('%d.%m %H:%M')}: {last_price:.2f}")
                print(f"💰 PAPER BUY: {last_price:.2f}")

            # SAT - SL / TP / SIQNAL
            elif wallet["shares"] > 0 and wallet["buy_price"] > 0:
                change_pct = (last_price - wallet["buy_price"]) / wallet["buy_price"] * 100
                is_sl = change_pct <= -3.0
                is_tp = change_pct >= 4.0
                is_sig = signal == "SAT" and conf >= 60

                if is_sl or is_tp or is_sig:
                    wallet["balance"] = wallet["shares"] * last_price
                    wallet["shares"] = 0
                    reason = "STOP LOSS" if is_sl else "TAKE PROFIT" if is_tp else "SAT SIQNAL"
                    wallet["trades"].append(f"{reason} {baku.strftime('%d.%m %H:%M')}: {last_price:.2f} ({change_pct:+.1f}%)")
                    print(f"💸 {reason}: {change_pct:+.1f}% @ {last_price:.2f}")
                    wallet["buy_price"] = 0

        cur_price = all_results["KO"]["1g"]["last"] if "KO" in all_results and "1g" in all_results["KO"] else 0
        tot = wallet["balance"] + (wallet["shares"] * cur_price if wallet["shares"] and cur_price else 0)
        if wallet["shares"] == 0: tot = wallet["balance"]
        wallet["total_value"] = round(tot, 2)
        wallet["pnl"] = round(tot - 10000, 2)
        wallet["pnl_percent"] = round((tot - 10000) / 100, 2)

        with open(wallet_file, "w") as f:
            json.dump(wallet, f, indent=2)

        all_results["PAPER_WALLET"] = wallet
        print(f"💼 Wallet: {tot:.2f}$ | PnL: {wallet['pnl']}$ ({wallet['pnl_percent']}%) | Resets: {wallet.get('resets',0)}")

    except Exception as e:
        print(f"Wallet err: {e}")
        traceback.print_exc()

    final_out = {"predictions": all_results, "updated": str(baku), "macro": macro, "fred": fred, "extras": extras}
    with open(f"{DATA_DIR}/predictions.json","w") as f:
        json.dump(final_out,f,indent=2)
    print(f"💾 Saved {DATA_DIR}/predictions.json")

    try:
        msg = f"🤖 KO V8.1 {baku.strftime('%d.%m %H:%M')} Baku\n"
        for t in TICKERS:
            for k in ["1s","1g","3g","5g"]:
                if t in all_results and k in all_results[t]:
                    r = all_results[t][k]
                    msg += f"{t} {k}: {r['signal']} {r['conf']:.0f}% @ ${r['price']:.2f}\n"
        if "PAPER_WALLET" in all_results:
            w = all_results["PAPER_WALLET"]
            msg += f"\n💼 {w['total_value']}$ PnL:{w['pnl']}$ ({w['pnl_percent']}%) Resets:{w.get('resets',0)}"
            if w.get('buy_price',0) > 0:
                ch = (w['total_value']-10000)/100
                msg += f"\nPoz: {w['shares']:.1f} @ {w['buy_price']:.2f}"
                msg += "\n\n⚠️ Təhsil üçündür, maliyyə tövsiyəsi deyil."
        if "KO" in all_results:
            pv = all_results["KO"].get("5g", {}).get("price", 0)
            ns_val = news_data.get("KO", (0,"",False))[0]
            journal_yaz("KO", all_results["KO"], pv, macro=macro, fred=fred, extras=extras, news_sent=ns_val)
        send_telegram(msg)
    except Exception as e:
        print(f"TG err {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
