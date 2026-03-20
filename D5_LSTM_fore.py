# D5_LSTM_fore.py
# Neural network forecaster – 5 business days ahead for any single ticker
# Uses sklearn MLPRegressor (multi-layer perceptron) – no TensorFlow dependency,
# compatible with all Python versions including 3.14+

import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import dump, load
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

FORECAST_H = 5    # dni do przodu
TIME_STEP  = 60   # okno LSTM (lookback)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def _next_workdays(start: datetime, n: int) -> list:
    days, cur = [], start
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur.date())
        cur += timedelta(days=1)
    return days


def _safe(ticker: str) -> str:
    return ticker.replace("=", "_").replace("^", "").replace("-", "_").replace(".", "_")


def _build_windows(data: np.ndarray, time_step: int, horizon: int):
    """Tworzy okna X [time_step features] i etykiety Y [horizon kroków]."""
    X, y = [], []
    for i in range(len(data) - time_step - horizon):
        X.append(data[i : i + time_step].flatten())   # spłaszczamy dla MLP
        y.append(data[i + time_step : i + time_step + horizon, 0])
    return np.array(X), np.array(y)


def forecast_ticker(ticker: str, past: int = 1000, retrain: bool = False) -> pd.DataFrame:
    """
    Pobiera `past` ostatnich sesji dla `ticker`, trenuje (lub wczytuje) MLP,
    zwraca DataFrame z kolumnami Date i Forecast (ceny Close) na 5 dni roboczych.
    """
    safe = _safe(ticker)
    model_path  = MODELS_DIR / f"{safe}_model.joblib"
    scaler_path = MODELS_DIR / f"{safe}_scaler.joblib"

    # 1. Pobierz dane
    raw = yf.download(ticker, period="8y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw[["Close"]].dropna().tail(past)

    if len(raw) < TIME_STEP + FORECAST_H + 30:
        raise ValueError(f"Za mało danych dla {ticker}: {len(raw)} sesji")

    last_price = float(raw["Close"].iloc[-1])

    # 2. Stopy zwrotu (univariate)
    rr = raw["Close"].pct_change().fillna(0).values.reshape(-1, 1)

    tr_size = int(len(rr) * 0.8)

    if retrain:
        for p in (model_path, scaler_path):
            if p.exists():
                p.unlink()

    if not model_path.exists():
        # Scaler fitowany tylko na train – brak data leakage
        scaler = MinMaxScaler(feature_range=(0, 1))
        tr_sc = scaler.fit_transform(rr[:tr_size])
        te_sc = scaler.transform(rr[tr_size:])

        dump(scaler, scaler_path)

        X_tr, y_tr = _build_windows(tr_sc, TIME_STEP, FORECAST_H)
        X_te, y_te = _build_windows(te_sc, TIME_STEP, FORECAST_H)

        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )
        model.fit(X_tr, y_tr)
        dump(model, model_path)
        log.info(f"Model MLP zapisany: {model_path}")
    else:
        model  = load(model_path)
        scaler = load(scaler_path)
        log.info(f"Wczytano model: {model_path}")

    # 4. Prognoza
    full_sc  = scaler.transform(rr)
    last_win = full_sc[-TIME_STEP:].flatten().reshape(1, -1)
    pred_sc  = model.predict(last_win)[0]              # shape (5,)

    # inverse_transform przez dummy array
    dummy = np.zeros((FORECAST_H, 1))
    dummy[:, 0] = pred_sc
    pred_rr = scaler.inverse_transform(dummy).flatten()

    # 5. Stopy zwrotu → poziomy cen
    prices, price = [], last_price
    for rv in pred_rr:
        price = price * (1 + rv)
        prices.append(round(price, 4))

    workdays = _next_workdays(datetime.today(), FORECAST_H)
    return pd.DataFrame({"Date": workdays, "Forecast": prices})
