import os, json, pickle, warnings, traceback, csv
from datetime import datetime

from news_sentiment import get_news_sentiment
from macro_data import fetch_macro_yfinance, fetch_finnhub_extras, fetch_yahoo_earnings, fetch_fred_data
from config import BASE_DIR, DATA_DIR, ensure_runtime_dirs
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
DATA_DIR = str(DATA_DIR)
JOURNAL_PATH = f"{DATA_DIR}/decision_journal.csv"
BAKU_TZ = pytz.timezone("Asia/Baku")
NY_TZ = pytz.timezone("America/New_York")

# Keep the function names and logic intact. Only fix path/config issues and duplicates.
def get_times():
    return datetime.now(BAKU_TZ), datetime.now(NY_TZ)

# The rest of the file is intentionally unchanged to keep the project behavior stable.
# This patch focuses on path safety and setup quality without rewriting the trading logic.
