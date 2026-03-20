# D5_LSTM_fore.py
# Generic LSTM forecaster – 5 business days ahead for any single ticker
import pickle, logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential, load_model

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

FORECAST_H = 5
TIME_STEP  = 60
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def _next_workdays(start, n):
    days, cur = [], start
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur.date())
        cur += timedelta(days=1)
    return days

def _safe(ticker):
    return ticker.replace("=","_").replace("^","").replace("-","_").replace(".","_")

def _windows(data, ts, h):
    X, y = [], []
    for i in range(len(data) - ts - h):
        X.append(data[i:i+ts])
        y.append(data[i+ts:i+ts+h, 0])
    return np.array(X), np.array(y)

def forecast_ticker(ticker: str, past: int = 1000, retrain: bool = False) -> pd.DataFrame:
    """Prognoza Close na 5 dni roboczych dla dowolnego tickera Yahoo Finance."""
    safe = _safe(ticker)
    model_path  = MODELS_DIR / f"{safe}_model.keras"
    scaler_path = MODELS_DIR / f"{safe}_scaler.pkl"

    # 1. Pobierz dane
    raw = yf.download(ticker, period="8y", interval="1d", progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw[["Close"]].dropna().tail(past)
    if len(raw) < TIME_STEP + FORECAST_H + 30:
        raise ValueError(f"Za mało danych dla {ticker}: {len(raw)} sesji")

    last_price = float(raw["Close"].iloc[-1])
    rr = raw["Close"].pct_change().fillna(0).values.reshape(-1, 1)
    tr_size = int(len(rr) * 0.8)

    if retrain and model_path.exists():
        model_path.unlink()
        if scaler_path.exists():
            scaler_path.unlink()

    if not model_path.exists():
        # Scaler tylko na train – brak data leakage
        scaler = MinMaxScaler(feature_range=(0, 1))
        tr_sc = scaler.fit_transform(rr[:tr_size])
        te_sc = scaler.transform(rr[tr_size:])
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        X_tr, y_tr = _windows(tr_sc, TIME_STEP, FORECAST_H)
        X_te, y_te = _windows(te_sc, TIME_STEP, FORECAST_H)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(TIME_STEP, 1)),
            LSTM(32),
            Dense(FORECAST_H),
        ])
        model.compile(loss="mse", optimizer="adam")
        model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
                  epochs=150, batch_size=32,
                  callbacks=[
                      EarlyStopping(patience=12, restore_best_weights=True, verbose=0),
                      ModelCheckpoint(model_path, save_best_only=True, verbose=0),
                  ], verbose=0)
    else:
        model = load_model(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # 4. Prognoza
    full_sc  = scaler.transform(rr)
    last_win = full_sc[-TIME_STEP:].reshape(1, TIME_STEP, 1)
    pred_sc  = model.predict(last_win, verbose=0)[0]
    pred_rr  = scaler.inverse_transform(pred_sc.reshape(-1, 1)).flatten()

    prices, price = [], last_price
    for rv in pred_rr:
        price = price * (1 + rv)
        prices.append(round(price, 4))

    workdays = _next_workdays(datetime.today(), FORECAST_H)
    return pd.DataFrame({"Date": workdays, "Forecast": prices})
