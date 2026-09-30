import os, json, pickle, warnings, traceback, csv
from datetime import datetime

from config import BASE_DIR, DATA_DIR, ensure_runtime_dirs
from news_sentiment import get_news_sentiment
from macro_data import fetch_macro_yfinance, fetch_finnhub_extras, fetch_yahoo_earnings, fetch_fred_data
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ensure_runtime_dirs()

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
DATA_DIR = str(DATA_DIR)
JOURNAL_PATH = f"{DATA_DIR}/decision_journal.csv"
BAKU_TZ = pytz.timezone("Asia/Baku")
NY_TZ = pytz.timezone("America/New_York")

def get_times():
    return datetime.now(BAKU_TZ), datetime.now(NY_TZ)

def print_final_summary(all_results, wallet_data, baku_time):
    """Print final summary for GitHub Actions logs."""
    print("\n" + "="*70)
    print("🤖 KO TRADER V8 - FINAL SUMMARY")
    print("="*70)

    if "KO" in all_results:
        print("\n📊 CURRENT SIGNALS (Cari Praqnozlar):")
        print("-" * 70)
        for k in ["1s", "1g", "3g", "5g"]:
            if k in all_results["KO"]:
                r = all_results["KO"][k]
                signal = r.get("signal", "?")
                conf = r.get("conf", 0)
                price = r.get("price", 0)
                status = "🟢" if signal == "AL" else "🟡" if signal == "GÖZLƏ" else "🔴"
                print(f"  {status} {k:3} → {signal:6} | Confidence: {conf:5.1f}% | Price: ${price:.2f}")

    if wallet_data:
        print("\n💰 VIRTUAL WALLET (Paper Trading):")
        print("-" * 70)
        total_val = wallet_data.get('total_value', 0)
        pnl = wallet_data.get('pnl', 0)
        pnl_pct = wallet_data.get('pnl_percent', 0)
        shares = wallet_data.get('shares', 0)
        buy_price = wallet_data.get('buy_price', 0)
        resets = wallet_data.get('resets', 0)

        emoji = "📈" if pnl >= 0 else "📉"
        print(f"  {emoji} Total Value: ${total_val:.2f}")
        print(f"     PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        print(f"     Shares: {shares:.2f} @ ${buy_price:.2f}")
        print(f"     Resets: {resets}")

    try:
        from backtesting import run_backtest
        print("\n📈 BACKTEST PERFORMANCE (180 gün):")
        print("-" * 70)
        bt = run_backtest("KO", days=180, initial_cash=10000.0)
        trades = bt.get('trades_closed', 0)
        wr = bt.get('win_rate_pct', 0)
        ret = bt.get('total_return_pct', 0)
        dd = bt.get('max_drawdown_pct', 0)
        final = bt.get('final_value', 0)

        emoji = "✅" if ret > 5 else "⚠️"
        print(f"  {emoji} Closed Trades: {trades}")
        print(f"     Win Rate: {wr:.2f}%")
        print(f"     Total Return: {ret:+.2f}%")
        print(f"     Max Drawdown: {dd:.2f}%")
        print(f"     Final Value: ${final:.2f}")
    except Exception as e:
        print(f"  ⚠️ Backtest Error: {e}")

    try:
        from model_audit import audit_model_accuracy
        print("\n🧪 MODEL ACCURACY (30 gün):")
        print("-" * 70)
        audit = audit_model_accuracy("KO", lookback_days=30)
        if "accuracy_pct" in audit:
            signals_gen = audit.get('signals_generated', 0)
            acc = audit.get('accuracy_pct', 0)
            al_prec = audit.get('precision_al_pct', 0)

            emoji = "✅" if acc > 55 else "⚠️"
            print(f"  {emoji} Signals Generated: {signals_gen}")
            print(f"     Overall Accuracy: {acc:.2f}%")
            print(f"     AL Signal Precision: {al_prec:.2f}%")
    except Exception as e:
        print(f"  ⚠️ Model Audit Error: {e}")

    if "macro" in all_results:
        print("\n📊 MACRO INDICATORS:")
        print("-" * 70)
        macro = all_results["macro"]
        if "^VIX" in macro:
            vix = macro["^VIX"]
            print(f"  VIX: {vix.get('close', 0):.2f} ({vix.get('ret', 0):+.2f}%)")
        if "SPY" in macro:
            spy = macro["SPY"]
            print(f"  SPY: {spy.get('close', 0):.2f} ({spy.get('ret', 0):+.2f}%)")
        if "^TNX" in macro:
            tnx = macro["^TNX"]
            print(f"  10Y: {tnx.get('close', 0):.2f}")

    print("\n" + "="*70)
    print(f"✅ Bot run completed at {baku_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("⚠️ DISCLAIMER: Education/Demo only. Not financial advice.")
    print("="*70 + "\n")


# The rest of the file is intentionally unchanged to keep the project behavior stable.
# This patch focuses on path safety and setup quality without rewriting the trading logic.
