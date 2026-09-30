"""
KO V8 - MACRO DATA FETCHER - MARKET AWARE FINAL
A) yfinance ^VIX SPY XLP ^TNX UUP - 45 dəq açıqda / 6 saat bağlıda
C) Finnhub extras - 2 saat açıqda / 24 saat bağlıda
D) Yahoo Earnings
E) FRED - 7 gün
"""
import os, json, requests
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import pytz

from config import DATA_DIR

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


def is_ny_market_open():
    now_ny = datetime.now(pytz.timezone("America/New_York"))
    return now_ny.weekday() < 5 and 9 <= now_ny.hour < 16


def fetch_macro_yfinance():
    cache = _load_json(MACRO_CACHE)
    now = datetime.utcnow()
    TTL = 0.75 if is_ny_market_open() else 6.0

    if cache and "time" in cache:
        try:
            ct = datetime.fromisoformat(cache["time"])
            age_h = (now - ct).total_seconds() / 3600
            if age_h < TTL:
                print(f"📈 Macro cache təzədir {age_h:.1f}h (TTL {TTL}h) - yfinance işləmir")
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
    cache = _load_json(EXTRAS_CACHE)
    now = datetime.utcnow()
    TTL = 2.0 if is_ny_market_open() else 24.0

    if ticker in cache:
        try:
            ct = datetime.fromisoformat(cache[ticker].get("time", "2000-01-01"))
            age_h = (now - ct).total_seconds() / 3600
            if age_h < TTL:
                print(f"🔍 Extras cache təzədir {age_h:.1f}h (TTL {TTL}h)")
                return cache[ticker]
        except:
            pass

    key = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")
    if not key:
        print("⚠ FINNHUB key yoxdur - extras skip")
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
    try:
        tk = yf.Ticker(ticker)
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


def fetch_fred_data():
    cache = _load_json(FRED_CACHE)
    now = datetime.utcnow()
    if cache and "time" in cache:
        try:
            ct = datetime.fromisoformat(cache["time"])
            if (now - ct).days < 7 and "fed_funds" in cache:
                print(f"🏦 FRED cache təzədir {(now-ct).days} gün - API işləmir")
                return cache
        except:
            pass

    key = os.getenv("FRED_API_KEY")
    if not key:
        print("⚠ FRED_API_KEY yoxdur")
        return cache if cache and "fed_funds" in cache else {"time": now.isoformat(), "fed_funds": 4.5, "cpi": 334, "cpi_yoy": 1.3, "unrate": 4.1, "t10y2y": 0.36}

    series_map = {"CPI": "CPIAUCSL", "UNRATE": "UNRATE", "FEDFUNDS": "DFF", "T10Y2Y": "T10Y2Y"}
    out = {"time": now.isoformat()}
    for name, sid in series_map.items():
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}&file_type=json&sort_order=desc&limit=2"
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                obs = r.json().get("observations", [])
                if obs and obs[0]['value']!= '.':
                    curr = float(obs[0]['value'])
                    prev = float(obs[1]['value']) if len(obs) > 1 and obs[1]['value']!= '.' else curr
                    out[name] = {"value": curr, "change": curr-prev}
        except Exception as e:
            print(f"FRED {name} err {e}")

    flat = {
        "time": out.get("time"),
        "fed_funds": out.get("FEDFUNDS", {}).get("value", 4.5),
        "cpi": out.get("CPI", {}).get("value", 334),
        "cpi_yoy": out.get("CPI", {}).get("change", 1.3),
        "unrate": out.get("UNRATE", {}).get("value", 4.1),
        "t10y2y": out.get("T10Y2Y", {}).get("value", 0.36),
    }
    _save_json(FRED_CACHE, flat)
    return flat
