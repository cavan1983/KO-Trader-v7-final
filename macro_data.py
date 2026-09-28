"""
KO V8 - MACRO DATA FETCHER
A) yfinance 0 limit: ^VIX, SPY, XLP, ^TNX, UUP
C) Finnhub extras: earnings, insider sentiment (24h cache)
D) Yahoo Earnings date
E) FRED - CPI, UNRATE, FEDFUNDS (7 gün cache)
Hamısı ağıllı cache ilə
"""
import os, json, requests
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
MACRO_CACHE = os.path.join(DATA_DIR, "macro_cache.json")
EXTRAS_CACHE = os.path.join(DATA_DIR, "extras_cache.json")
FRED_CACHE = os.path.join(DATA_DIR, "fred_cache.json")

def _load_json(path):
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_json(path, data):
    try:
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"cache save error {path}: {e}")

def fetch_macro_yfinance():
    """A) 5 ticker 1 sorğu - 0 limit"""
    cache = _load_json(MACRO_CACHE)
    now = datetime.utcnow()
    if cache and "time" in cache:
        try:
            ct = datetime.fromisoformat(cache["time"])
            if (now - ct).total_seconds() / 3600 < 6:
                print(f"📈 Macro cache təzədir { (now-ct).total_seconds()/3600:.1f}h - yfinance işləmir")
                return cache
        except:
            pass

    print("📈 Macro yfinance çəkilir: ^VIX SPY XLP ^TNX UUP")
    try:
        tickers = ["^VIX", "SPY", "XLP", "^TNX", "UUP"]
        result = {"time": now.isoformat()}
        for t in tickers:
            try:
                d = yf.download(t, period="10d", interval="1d", progress=False, auto_adjust=True, threads=False)
                if d is None or d.empty:
                    continue
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                close = float(d['Close'].iloc[-1])
                prev = float(d['Close'].iloc[-2]) if len(d) > 1 else close
                ret = (close - prev) / prev * 100 if prev!= 0 else 0
                result[t] = {"close": close, "ret": ret}
            except Exception as e:
                print(f" macro {t} fail {e}")
        _save_json(MACRO_CACHE, result)
        print(f"📈 Macro OK: {result}")
        return result
    except Exception as e:
        print(f"📈 Macro error {e}")
        return cache if cache else {"time": now.isoformat()}

def fetch_finnhub_extras(ticker="KO"):
    """C) Finnhub earnings + insider - 24h cache"""
    cache = _load_json(EXTRAS_CACHE)
    now = datetime.utcnow()
    if ticker in cache:
        try:
            ct = datetime.fromisoformat(cache[ticker].get("time", "2000-01-01"))
            if (now - ct).total_seconds() / 3600 < 24:
                print(f"🔍 Extras cache təzədir {(now-ct).total_seconds()/3600:.1f}h")
                return cache[ticker]
        except:
            pass
    key = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")
    if not key:
        print("⚠️ FINNHUB key yoxdur - extras skip")
        return cache.get(ticker, {"earnings_days": 30, "insider_score": 0, "time": now.isoformat()})
    out = {"time": now.isoformat(), "earnings_days": 30, "insider_score": 0, "next_earnings": "unknown"}
    try:
        from_d = now.strftime("%Y-%m-%d")
        to_d = (now + timedelta(days=90)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/calendar/earnings?from={from_d}&to={to_d}&symbol={ticker}&token={key}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            cals = data.get("earningsCalendar", []) if isinstance(data, dict) else data
            for ev in cals:
                if ev.get("symbol") == ticker:
                    ed = ev.get("date")
                    try:
                        ed_dt = datetime.strptime(ed, "%Y-%m-%d")
                        days = (ed_dt - now).days
                        out["earnings_days"] = max(0, days)
                        out["next_earnings"] = ed
                        print(f"📅 Earnings {ticker} {ed} -> {days} gün")
                        break
                    except:
                        pass
    except Exception as e:
        print(f"📅 Earnings error {e}")
    try:
        url = f"https://finnhub.io/api/v1/stock/insider-sentiment?symbol={ticker}&from={(now - timedelta(days=90)).strftime('%Y-%m-%d')}&to={now.strftime('%Y-%m-%d')}&token={key}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            arr = data.get("data", [])
            if arr:
                last = arr[-1]
                mspr = last.get("mspr", 0)
                out["insider_score"] = int(mspr) if mspr else 0
                print(f"👔 Insider {ticker} mspr={out['insider_score']}")
    except Exception as e:
        print(f"👔 Insider error {e}")
    cache[ticker] = out
    _save_json(EXTRAS_CACHE, cache)
    return out

def fetch_yahoo_earnings(ticker="KO"):
    """D) Yahoo earnings date - yfinance pulsuz"""
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is not None and not cal.empty if hasattr(cal, 'empty') else cal:
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date", [None])[0] if "Earnings Date" in cal else None
                if ed:
                    days = (ed - datetime.now()).days if hasattr(ed, 'date') else 30
                    return max(0, days)
        try:
            edates = tk.earnings_dates
            if edates is not None and not edates.empty:
                future = edates[edates.index > pd.Timestamp.now()]
                if not future.empty:
                    nxt = future.index[0]
                    days = (nxt - pd.Timestamp.now()).days
                    return max(0, days)
        except:
            pass
    except Exception as e:
        print(f"📅 Yahoo earnings error {e}")
    return 30

# ===== YENİ ƏLAVƏ - FRED (SƏNİN API-ın) =====
def fetch_data(ticker, period, interval):
    db_path = f"{DATA_DIR}/db_{ticker}_{interval}_{period}.csv"
    df_db = None
    if os.path.exists(db_path):
        try:
            df_db = pd.read_csv(db_path, parse_dates=True, index_col=0)
            df_db.index = pd.to_datetime(df_db.index, errors='coerce')
            df_db = df_db.dropna().sort_index()
            if df_db.empty:
                df_db = None
        except:
            df_db = None

    fetch_period = "5d" if df_db is not None and interval == "1d" else ("7d" if df_db is not None else period)
    print(f"[1] fetch {ticker} {interval} {period} -> {fetch_period}")

    df_new = None
    for attempt in range(2):
        try:
            df = yf.download(ticker, period=fetch_period, interval=interval, progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty:
                continue
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
        if df_db is not None:
            return df_db
        return None

    if df_db is not None:
        try:
            combined = pd.concat([df_db, df_new])
            combined.index = pd.to_datetime(combined.index, errors='coerce')
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            combined.to_csv(db_path)
            return combined
        except Exception as e:
            print(f" -> concat fail {e}, new only used")
            df_new.to_csv(db_path)
            return df_new
    else:
        df_new.to_csv(db_path)
        return df_new
