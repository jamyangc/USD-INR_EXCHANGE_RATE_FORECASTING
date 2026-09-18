import warnings
warnings.filterwarnings("ignore")

import gc
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import lightgbm as lgb

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import threading


RANDOM_STATE = 42
START_DATE = "2020-01-01"
PPP_BASE_YEAR = 2020
MIN_TRAIN_DAYS = 500
HORIZON = 1
DAYS_IN_YEAR = 365
MLP_LOGRET_CLIP = 0.02

# historical_model_selection() retrains LightGBM, Ridge, Decision
# Tree and MLP once per day in this window (a walk-forward
# backtest), only used as a fallback for a pair that doesn't yet
# have enough archived same-day forecasts for the cheap rolling-MAE
# path. At 30 this meant up to ~120 model fits in a single request
# -- 60s+ on Render's free tier. 10 keeps the backtest meaningful
# while cutting that cost roughly 3x. This path should also become
# rare in practice once ENABLE_SCHEDULER keeps every pair's archive
# populated ahead of user requests.
SELECTION_WINDOW = 10

ROLLING_WINDOW_DAYS = 7
MIN_ROLLING_DAYS = 3
DEFAULT_PAIR = "USDINR"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLS = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_5",
    "vol_20",
    "price_ma5_ratio",
    "price_ma20_ratio",
    "ma_ratio",
    "rsi_14"
]

PAIRS = {
    "USDINR": {
        "label": "USD/INR",
        "ticker": "USDINR=X",
        "base": "USD",
        "quote": "INR",
        "wb_base": "USA",
        "wb_quote": "IND"
    },
    "USDJPY": {
        "label": "USD/JPY",
        "ticker": "USDJPY=X",
        "base": "USD",
        "quote": "JPY",
        "wb_base": "USA",
        "wb_quote": "JPN"
    },
    "USDCHF": {
        "label": "USD/CHF",
        "ticker": "USDCHF=X",
        "base": "USD",
        "quote": "CHF",
        "wb_base": "USA",
        "wb_quote": "CHE"
    },
    "USDCAD": {
        "label": "USD/CAD",
        "ticker": "USDCAD=X",
        "base": "USD",
        "quote": "CAD",
        "wb_base": "USA",
        "wb_quote": "CAN"
    },
    "EURUSD": {
        "label": "EUR/USD",
        "ticker": "EURUSD=X",
        "base": "EUR",
        "quote": "USD",
        "wb_base": "EMU",
        "wb_quote": "USA"
    },
    "GBPUSD": {
        "label": "GBP/USD",
        "ticker": "GBPUSD=X",
        "base": "GBP",
        "quote": "USD",
        "wb_base": "GBR",
        "wb_quote": "USA"
    },
    "AUDUSD": {
        "label": "AUD/USD",
        "ticker": "AUDUSD=X",
        "base": "AUD",
        "quote": "USD",
        "wb_base": "AUS",
        "wb_quote": "USA"
    },
    "NZDUSD": {
        "label": "NZD/USD",
        "ticker": "NZDUSD=X",
        "base": "NZD",
        "quote": "USD",
        "wb_base": "NZL",
        "wb_quote": "USA"
    }
}

FORECAST_MODEL_KEYS = [
    "lightgbm",
    "ridge",
    "decision_tree",
    "mlp",
    "ppp",
    "irp"
]

MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "ridge": "Ridge",
    "decision_tree": "Decision Tree",
    "mlp": "MLP",
    "ppp": "PPP",
    "irp": "IRP"
}


def forecast_file_path(pair_key):
    return os.path.join(
        BASE_DIR,
        f"latest_forecast_{pair_key}.json"
    )


def history_file_path(pair_key):
    return os.path.join(
        BASE_DIR,
        f"history_{pair_key}.json"
    )


def forecast_archive_file_path(pair_key):
    return os.path.join(
        BASE_DIR,
        f"forecast_archive_{pair_key}.json"
    )


def next_business_day(date_value):
    ts = pd.Timestamp(date_value).normalize()
    next_day = ts + pd.Timedelta(days=1)

    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)

    return next_day


def stored_forecast_is_stale(pair_key):
    """
    A stored forecast is stale once "today" has reached or passed the
    date it was predicting for -- at that point newer market data is
    available and a fresh forecast should be generated.

    NOTE: this used to recompute next_business_day() from the file's
    own saved `latest_available_date` and compare that against the
    file's own saved `forecast_date`. Since `forecast_date` was
    originally derived from that same `actual_date`, the two values
    were always identical and the check could never return True after
    the first run. Comparing against the real current date fixes it.
    """
    path = forecast_file_path(pair_key)

    if not os.path.exists(path):
        return True

    try:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)

        forecast_date = saved.get("forecast_date")

        if not forecast_date:
            return True

        today = pd.Timestamp.now().normalize()

        return pd.Timestamp(forecast_date) <= today

    except Exception:
        return True


# ============================================================
# PER-PAIR PIPELINE LOCKS
#
# The frontend fires /api/predict and /api/history at the same
# time (Promise.all). Both routes used to independently check
# stored_forecast_is_stale() and independently call
# run_forecast_pipeline() when stale -- so on a cold pair, two
# full pipelines (yfinance download + 2 World Bank calls + 4
# model trainings, each) ran at once, competing for CPU on a
# single-core free instance right when it's also cold-booting.
# That's enough to blow well past a 45s client timeout, which
# is why this showed up worst on mobile.
#
# ensure_fresh_forecast() below serializes refreshes per pair:
# the second caller blocks on the lock instead of duplicating
# the work, then re-checks staleness (the first caller already
# refreshed it) before deciding whether to run again.
# ============================================================

_pipeline_locks = {}
_pipeline_locks_guard = threading.Lock()


def get_pipeline_lock(pair_key):

    with _pipeline_locks_guard:

        if pair_key not in _pipeline_locks:
            _pipeline_locks[pair_key] = threading.Lock()

        return _pipeline_locks[pair_key]


def ensure_fresh_forecast(pair_key):

    if not stored_forecast_is_stale(pair_key):
        return

    lock = get_pipeline_lock(pair_key)

    with lock:

        # Re-check after acquiring the lock -- if another request
        # for this pair got here first and already refreshed it,
        # there's nothing left to do.
        if stored_forecast_is_stale(pair_key):
            run_forecast_pipeline(pair_key)


def get_world_bank_indicator(country, indicator):
    url = (
        f"https://api.worldbank.org/v2/country/"
        f"{country}/indicator/{indicator}"
        f"?format=json&per_page=100"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()

    if len(data) < 2 or data[1] is None:
        raise RuntimeError(
            f"No World Bank data found for {country}/{indicator}"
        )

    records = [
        {
            "year": int(item["date"]),
            "value": float(item["value"])
        }
        for item in data[1]
        if item["value"] is not None
    ]

    if not records:
        raise RuntimeError(
            f"No usable World Bank data found for {country}/{indicator}"
        )

    return pd.DataFrame(records).sort_values("year")


def build_features(df):
    df = df.copy()

    df["ret_1"] = df["price"].pct_change(1)
    df["ret_5"] = df["price"].pct_change(5)
    df["ret_10"] = df["price"].pct_change(10)
    df["ret_20"] = df["price"].pct_change(20)

    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["ma_5"] = df["price"].rolling(5).mean()
    df["ma_20"] = df["price"].rolling(20).mean()

    df["price_ma5_ratio"] = df["price"] / df["ma_5"]
    df["price_ma20_ratio"] = df["price"] / df["ma_20"]
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]

    delta = df["price"].diff()

    gain = (
        delta.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["target_logret"] = np.log(
        df["price"].shift(-HORIZON) / df["price"]
    )

    return df


def create_model():
    # n_jobs was -1 (use every CPU core). On a shared/limited host
    # like Render's free tier, each thread carries its own working
    # memory for tree-building, so more threads means more peak RAM
    # for no real speed win at this data size -- n_jobs=1 trades a
    # bit of wall-clock time for a much smaller memory footprint.
    # n_estimators trimmed from 300 to 150 for the same reason: this
    # model gets refit from scratch on every walk-forward backtest
    # day (see historical_model_selection), so its per-fit cost is
    # what actually matters for staying under the memory ceiling.
    return lgb.LGBMRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        objective="regression",
        verbosity=-1,
        n_jobs=1
    )


def create_ridge_model():
    return Ridge(
        alpha=1.0,
        random_state=RANDOM_STATE
    )


def create_tree_model():
    return DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    )


def create_mlp_model():
    # max_iter dropped from 5000 to 500: with the lbfgs solver on
    # a small (8-unit) hidden layer this converges well within a
    # few hundred iterations in practice, so 5000 was mostly wasted
    # compute -- especially costly since this model gets refit
    # SELECTION_WINDOW times during a walk-forward backtest.
    return MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="lbfgs",
        alpha=0.01,
        max_iter=500,
        random_state=RANDOM_STATE
    )


def clip_logret(value):
    return np.clip(
        value,
        -MLP_LOGRET_CLIP,
        MLP_LOGRET_CLIP
    )


def load_forecast_archive(pair_key):
    path = forecast_archive_file_path(pair_key)

    if not os.path.exists(path):
        return []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, list) else []

    except Exception:
        return []


def save_forecast_archive(pair_key, archive):
    path = forecast_archive_file_path(pair_key)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            archive[-120:],
            f,
            indent=2
        )


def archive_forecast(pair_key, result):
    archive = load_forecast_archive(pair_key)

    forecast_date = result.get("forecast_date")

    archive = [
        item
        for item in archive
        if item.get("forecast_date") != forecast_date
    ]

    archive.append(result)

    save_forecast_archive(
        pair_key,
        archive
    )


def select_model_rolling_window(
    pair_key,
    df,
    window=ROLLING_WINDOW_DAYS,
    min_days=MIN_ROLLING_DAYS
):
    """
    Picks the "best" model by averaging each model's absolute
    forecasting error over the last `window` trading days that have
    both an archived forecast AND a now-known actual price -- rather
    than just comparing on the single most recent day.

    A single day's error is noisy: a model can "win" purely by luck
    on any given day. Averaging over a rolling window means a model
    has to be consistently closer to actual, not just lucky once, to
    get selected -- so the pick changes less often and is more
    trustworthy day to day.

    Returns None if fewer than `min_days` matched days are available
    yet (e.g. the app is newly deployed), so the caller can fall back
    to the walk-forward backtest instead.
    """

    archive = load_forecast_archive(pair_key)

    if not archive:
        return None

    # date string -> actual price, from the full price history
    date_to_price = dict(
        zip(
            df["date"].dt.strftime("%Y-%m-%d"),
            df["price"]
        )
    )

    matched = []

    for item in archive:

        forecast_date = item.get("forecast_date")

        if forecast_date in date_to_price:

            matched.append(
                (
                    forecast_date,
                    item,
                    float(date_to_price[forecast_date])
                )
            )

    if len(matched) < min_days:
        return None

    # Most recent `window` matched trading days
    matched.sort(key=lambda entry: entry[0])
    matched = matched[-window:]

    errors = {
        model_key: []
        for model_key in FORECAST_MODEL_KEYS
    }

    for forecast_date, item, actual_price in matched:

        for model_key in FORECAST_MODEL_KEYS:

            value = item.get(model_key)

            if value is None:
                continue

            try:

                forecast_value = float(value)

                errors[model_key].append(
                    abs(actual_price - forecast_value)
                )

            except (TypeError, ValueError):
                continue

    mae = {}

    for model_key, model_errors in errors.items():

        if model_errors:
            mae[model_key] = float(
                np.mean(model_errors)
            )

    if not mae:
        return None

    selected_model_key = min(
        mae,
        key=mae.get
    )

    return {
        "selected_model_key": selected_model_key,
        "selected_model_label": MODEL_LABELS[selected_model_key],
        "selected_model_error": round(
            mae[selected_model_key],
            6
        ),
        "selection_method": (
            f"{len(matched)}-day rolling MAE"
        ),
        "selection_mae": {
            key: round(value, 6)
            for key, value in mae.items()
        },
        "rolling_window_days": len(matched)
    }


def historical_model_selection(
    df,
    ppp_data,
    rates
):
    usable = (
        df
        .dropna(
            subset=FEATURE_COLS + ["target_logret"]
        )
        .copy()
    )

    if len(usable) < MIN_TRAIN_DAYS + SELECTION_WINDOW:
        return None

    validation_start = (
        len(usable) - SELECTION_WINDOW
    )

    errors = {
        "lightgbm": [],
        "ridge": [],
        "decision_tree": [],
        "mlp": [],
        "ppp": [],
        "irp": []
    }

    for i in range(
        validation_start,
        len(usable)
    ):
        train = usable.iloc[:i].copy()
        test = usable.iloc[[i]].copy()

        if len(train) < MIN_TRAIN_DAYS:
            continue

        X_train = train[FEATURE_COLS]
        y_train = train["target_logret"]
        X_test = test[FEATURE_COLS]

        previous_price = float(
            train["price"].iloc[-1]
        )

        actual_price = float(
            test["price"].iloc[0]
        )

        test_year = int(
            test["year"].iloc[0]
        )

        try:
            model = create_model()

            model.fit(
                X_train,
                y_train
            )

            pred = model.predict(X_test)[0]

            forecast = (
                previous_price *
                np.exp(pred)
            )

            errors["lightgbm"].append(
                abs(actual_price - forecast)
            )

        except Exception as e:
            print(
                f"Historical LightGBM error: {e}"
            )

        try:
            scaler = StandardScaler()

            X_train_scaled = (
                scaler.fit_transform(X_train)
            )

            X_test_scaled = (
                scaler.transform(X_test)
            )

            model = create_ridge_model()

            model.fit(
                X_train_scaled,
                y_train
            )

            pred = model.predict(
                X_test_scaled
            )[0]

            forecast = (
                previous_price *
                np.exp(pred)
            )

            errors["ridge"].append(
                abs(actual_price - forecast)
            )

        except Exception as e:
            print(
                f"Historical Ridge error: {e}"
            )

        try:
            model = create_tree_model()

            model.fit(
                X_train,
                y_train
            )

            pred = model.predict(
                X_test
            )[0]

            forecast = (
                previous_price *
                np.exp(pred)
            )

            errors["decision_tree"].append(
                abs(actual_price - forecast)
            )

        except Exception as e:
            print(
                f"Historical Tree error: {e}"
            )

        try:
            scaler = StandardScaler()

            X_train_scaled = (
                scaler.fit_transform(X_train)
            )

            X_test_scaled = (
                scaler.transform(X_test)
            )

            model = create_mlp_model()

            model.fit(
                X_train_scaled,
                y_train
            )

            pred = model.predict(
                X_test_scaled
            )[0]

            pred = clip_logret(pred)

            forecast = (
                previous_price *
                np.exp(pred)
            )

            errors["mlp"].append(
                abs(actual_price - forecast)
            )

        except Exception as e:
            print(
                f"Historical MLP error: {e}"
            )

        try:
            ppp_rows = (
                ppp_data[
                    ppp_data["year"] <= test_year
                ]
                .dropna(
                    subset=["ppp_rate"]
                )
                .sort_values("year")
            )

            if not ppp_rows.empty:
                ppp_forecast = float(
                    ppp_rows.iloc[-1]["ppp_rate"]
                )

                errors["ppp"].append(
                    abs(
                        actual_price -
                        ppp_forecast
                    )
                )

        except Exception as e:
            print(
                f"Historical PPP error: {e}"
            )

        try:
            rate_rows = (
                rates[
                    rates["year"] <= test_year
                ]
                .dropna(
                    subset=[
                        "quote_rate_decimal",
                        "base_rate_decimal"
                    ]
                )
                .sort_values("year")
            )

            if not rate_rows.empty:
                row = rate_rows.iloc[-1]

                quote_rate = float(
                    row["quote_rate_decimal"]
                )

                base_rate = float(
                    row["base_rate_decimal"]
                )

                irp_forecast = (
                    previous_price *
                    (
                        (1 + quote_rate) /
                        (1 + base_rate)
                    ) **
                    (1 / DAYS_IN_YEAR)
                )

                errors["irp"].append(
                    abs(
                        actual_price -
                        irp_forecast
                    )
                )

        except Exception as e:
            print(
                f"Historical IRP error: {e}"
            )

        # Explicit cleanup between backtest days. This loop creates
        # a brand-new LightGBM/Ridge/Tree/MLP object on every single
        # iteration; Python's garbage collector doesn't always keep
        # up with that churn fast enough to stay under a tight
        # memory ceiling (e.g. Render's free-tier 512MB), so we
        # force a collection pass explicitly rather than letting
        # objects pile up across all SELECTION_WINDOW iterations.
        gc.collect()

    mae = {}

    for model_key, model_errors in errors.items():

        if model_errors:
            mae[model_key] = float(
                np.mean(model_errors)
            )

    if not mae:
        return None

    selected_model_key = min(
        mae,
        key=mae.get
    )

    return {
        "selected_model_key": selected_model_key,
        "selected_model_label": MODEL_LABELS[selected_model_key],
        "selected_model_error": round(
            mae[selected_model_key],
            6
        ),
        "selection_method": (
            f"{SELECTION_WINDOW}-day "
            f"walk-forward MAE"
        ),
        "selection_mae": {
            key: round(value, 6)
            for key, value in mae.items()
        }
    }


def run_forecast_pipeline(
    pair_key=DEFAULT_PAIR
):
    if pair_key not in PAIRS:
        raise ValueError(
            f"Unknown pair: {pair_key}"
        )

    cfg = PAIRS[pair_key]

    fx = yf.download(
        cfg["ticker"],
        period="10y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if fx.empty:
        raise RuntimeError(
            f"Could not download {pair_key} data."
        )

    if isinstance(
        fx.columns,
        pd.MultiIndex
    ):
        fx.columns = (
            fx.columns
            .get_level_values(0)
        )

    fx = fx.reset_index()

    fx = fx.rename(
        columns={
            "Date": "date",
            "Close": "price"
        }
    )

    df = fx[
        ["date", "price"]
    ].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df = df.dropna()

    df = df[
        df["date"] >= pd.Timestamp(
            START_DATE
        )
    ].copy()

    df = (
        df
        .sort_values("date")
        .drop_duplicates(
            subset="date"
        )
        .reset_index(drop=True)
    )

    if len(df) < MIN_TRAIN_DAYS:
        raise RuntimeError(
            f"Not enough data for {pair_key}."
        )

    df["year"] = df["date"].dt.year

    df = build_features(df)

    quote_cpi = (
        get_world_bank_indicator(
            cfg["wb_quote"],
            "FP.CPI.TOTL"
        )
        .rename(
            columns={
                "value": "quote_cpi"
            }
        )
    )

    base_cpi = (
        get_world_bank_indicator(
            cfg["wb_base"],
            "FP.CPI.TOTL"
        )
        .rename(
            columns={
                "value": "base_cpi"
            }
        )
    )

    cpi = pd.merge(
        quote_cpi,
        base_cpi,
        on="year",
        how="inner"
    )

    annual_fx = (
        df
        .groupby("year")["price"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "price": "average_price"
            }
        )
    )

    ppp_data = pd.merge(
        annual_fx,
        cpi,
        on="year",
        how="inner"
    )

    base_row = (
        ppp_data[
            ppp_data["year"] == PPP_BASE_YEAR
        ]
        .iloc[0]
    )

    base_price = float(
        base_row["average_price"]
    )

    base_quote_cpi = float(
        base_row["quote_cpi"]
    )

    base_base_cpi = float(
        base_row["base_cpi"]
    )

    ppp_data["ppp_rate"] = (
        base_price
        *
        (
            ppp_data["quote_cpi"] /
            base_quote_cpi
        )
        /
        (
            ppp_data["base_cpi"] /
            base_base_cpi
        )
    )

    quote_rate = (
        get_world_bank_indicator(
            cfg["wb_quote"],
            "FR.INR.LEND"
        )
        .rename(
            columns={
                "value": "quote_rate"
            }
        )
    )

    base_rate = (
        get_world_bank_indicator(
            cfg["wb_base"],
            "FR.INR.LEND"
        )
        .rename(
            columns={
                "value": "base_rate"
            }
        )
    )

    rates = pd.merge(
        quote_rate,
        base_rate,
        on="year",
        how="inner"
    )

    rates["quote_rate_decimal"] = (
        rates["quote_rate"] / 100
    )

    rates["base_rate_decimal"] = (
        rates["base_rate"] / 100
    )

    train_df = (
        df
        .dropna(
            subset=FEATURE_COLS + ["target_logret"]
        )
        .copy()
    )

    if len(train_df) < MIN_TRAIN_DAYS:
        raise RuntimeError(
            "Not enough usable training data."
        )

    X_train = train_df[
        FEATURE_COLS
    ]

    y_train = train_df[
        "target_logret"
    ]

    lgbm_model = create_model()

    lgbm_model.fit(
        X_train,
        y_train
    )

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    ridge_model = create_ridge_model()

    ridge_model.fit(
        X_train_scaled,
        y_train
    )

    tree_model = create_tree_model()

    tree_model.fit(
        X_train,
        y_train
    )

    mlp_model = create_mlp_model()

    mlp_model.fit(
        X_train_scaled,
        y_train
    )

    last_known_date = (
        df["date"].iloc[-1]
    )

    last_known_price = float(
        df["price"].iloc[-1]
    )

    latest_features = (
        df
        .iloc[[-1]]
        [FEATURE_COLS]
    )

    latest_features_scaled = (
        scaler.transform(
            latest_features
        )
    )

    lgbm_prediction = (
        lgbm_model
        .predict(
            latest_features
        )[0]
    )

    forecast_lgbm = (
        last_known_price *
        np.exp(lgbm_prediction)
    )

    ridge_prediction = (
        ridge_model
        .predict(
            latest_features_scaled
        )[0]
    )

    forecast_ridge = (
        last_known_price *
        np.exp(ridge_prediction)
    )

    tree_prediction = (
        tree_model
        .predict(
            latest_features
        )[0]
    )

    forecast_tree = (
        last_known_price *
        np.exp(tree_prediction)
    )

    mlp_prediction = (
        mlp_model
        .predict(
            latest_features_scaled
        )[0]
    )

    mlp_prediction = clip_logret(
        mlp_prediction
    )

    forecast_mlp = (
        last_known_price *
        np.exp(mlp_prediction)
    )

    latest_ppp_row = (
        ppp_data
        .dropna(
            subset=["ppp_rate"]
        )
        .sort_values("year")
        .iloc[-1]
    )

    forecast_ppp = float(
        latest_ppp_row["ppp_rate"]
    )

    latest_ppp_year = int(
        latest_ppp_row["year"]
    )

    latest_rate_row = (
        rates
        .dropna(
            subset=[
                "quote_rate_decimal",
                "base_rate_decimal"
            ]
        )
        .sort_values("year")
        .iloc[-1]
    )

    quote_rate_value = float(
        latest_rate_row[
            "quote_rate_decimal"
        ]
    )

    base_rate_value = float(
        latest_rate_row[
            "base_rate_decimal"
        ]
    )

    latest_rate_year = int(
        latest_rate_row["year"]
    )

    forecast_irp = (
        last_known_price *
        (
            (1 + quote_rate_value) /
            (1 + base_rate_value)
        ) **
        (1 / DAYS_IN_YEAR)
    )

    forecast_rw = last_known_price

    forecast_date = (
        next_business_day(
            last_known_date
        )
    )

    rolling_selection = (
        select_model_rolling_window(
            pair_key,
            df
        )
    )

    if rolling_selection is None:
        historical_selection = (
            historical_model_selection(
                df,
                ppp_data,
                rates
            )
        )
    else:
        historical_selection = None

    selection = (
        rolling_selection
        if rolling_selection is not None
        else historical_selection
    )

    if selection is not None:

        selected_model_key = (
            selection[
                "selected_model_key"
            ]
        )

        forecasts = {
            "lightgbm": forecast_lgbm,
            "ridge": forecast_ridge,
            "decision_tree": forecast_tree,
            "mlp": forecast_mlp,
            "ppp": forecast_ppp,
            "irp": forecast_irp
        }

        selected_model_forecast = float(
            forecasts[
                selected_model_key
            ]
        )

        if selected_model_forecast > last_known_price:
            selected_model_direction = "UP"

        elif selected_model_forecast < last_known_price:
            selected_model_direction = "DOWN"

        else:
            selected_model_direction = "FLAT"

    else:

        selected_model_key = None
        selected_model_forecast = None
        selected_model_direction = None

    if forecast_lgbm > last_known_price:
        lightgbm_direction = "UP"

    elif forecast_lgbm < last_known_price:
        lightgbm_direction = "DOWN"

    else:
        lightgbm_direction = "FLAT"

    lightgbm_change_percent = (
        (
            forecast_lgbm -
            last_known_price
        )
        /
        last_known_price
        *
        100
    )

    result = {

        "pair":
            pair_key,

        "pair_label":
            cfg["label"],

        "base_ccy":
            cfg["base"],

        "quote_ccy":
            cfg["quote"],

        "latest_available_date":
            last_known_date.strftime(
                "%Y-%m-%d"
            ),

        "latest_price":
            round(
                last_known_price,
                4
            ),

        "forecast_date":
            forecast_date.strftime(
                "%Y-%m-%d"
            ),

        "random_walk":
            round(
                forecast_rw,
                4
            ),

        "lightgbm":
            round(
                forecast_lgbm,
                4
            ),

        "ridge":
            round(
                forecast_ridge,
                4
            ),

        "decision_tree":
            round(
                forecast_tree,
                4
            ),

        "mlp":
            round(
                forecast_mlp,
                4
            ),

        "ppp":
            round(
                forecast_ppp,
                4
            ),

        "irp":
            round(
                forecast_irp,
                4
            ),

        "ppp_cpi_year":
            latest_ppp_year,

        "irp_rate_year":
            latest_rate_year,

        "lightgbm_change_percent":
            round(
                lightgbm_change_percent,
                4
            ),

        "lightgbm_direction":
            lightgbm_direction,

        "selected_model_key":
            selected_model_key,

        "selected_model_label":
            (
                selection[
                    "selected_model_label"
                ]
                if selection is not None
                else None
            ),

        "selected_model_forecast":
            (
                round(
                    selected_model_forecast,
                    4
                )
                if selected_model_forecast is not None
                else None
            ),

        "selected_model_direction":
            selected_model_direction,

        "selected_model_error":
            (
                selection[
                    "selected_model_error"
                ]
                if selection is not None
                else None
            ),

        "rolling_window_days":
            (
                selection.get(
                    "rolling_window_days"
                )
                if selection is not None
                else None
            ),

        "selection_method":
            (
                selection.get(
                    "selection_method"
                )
                if selection is not None
                else None
            ),

        "selection_mae":
            (
                selection.get(
                    "selection_mae"
                )
                if selection is not None
                else None
            ),

        "generated_at":
            datetime.now().isoformat()
    }

    with open(
        forecast_file_path(pair_key),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    archive_forecast(
        pair_key,
        result
    )

    recent_history = (
        df[
            ["date", "price"]
        ]
        .tail(365)
        .copy()
    )

    recent_history["date"] = (
        recent_history["date"]
        .dt.strftime("%Y-%m-%d")
    )

    history = recent_history.to_dict(
        orient="records"
    )

    with open(
        history_file_path(pair_key),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    print(
        f"[{datetime.now()}] "
        f"{pair_key} | "
        f"Selected: "
        f"{selected_model_key or 'pending'} | "
        f"Next forecast: "
        f"{selected_model_forecast if selected_model_forecast is not None else 'pending'} | "
        f"Random Walk: "
        f"{forecast_rw:.4f}"
    )

    return result


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    static_folder="static"
)

CORS(app)


@app.route("/")
def index():

    return send_from_directory(
        app.static_folder,
        "index.html"
    )


@app.route("/service-worker.js")
def service_worker():

    return send_from_directory(
        app.static_folder,
        "service-worker.js",
        mimetype="application/javascript"
    )


@app.route("/api/pairs")
def pairs():

    return jsonify([

        {
            "key": key,
            "label": cfg["label"],
            "base": cfg["base"],
            "quote": cfg["quote"]
        }

        for key, cfg in PAIRS.items()

    ])


def resolve_pair():

    pair_key = (
        request.args
        .get(
            "pair",
            DEFAULT_PAIR
        )
        .upper()
    )

    if pair_key not in PAIRS:
        return None

    return pair_key


@app.route("/api/predict")
def predict():

    pair_key = resolve_pair()

    if pair_key is None:

        return jsonify({
            "error": "Unknown FX pair."
        }), 400

    path = forecast_file_path(
        pair_key
    )

    try:

        ensure_fresh_forecast(
            pair_key
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    try:

        with open(
            path,
            encoding="utf-8"
        ) as f:

            return jsonify(
                json.load(f)
            )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/history")
def history():

    pair_key = resolve_pair()

    if pair_key is None:

        return jsonify({
            "error": "Unknown FX pair."
        }), 400

    path = history_file_path(
        pair_key
    )

    try:

        ensure_fresh_forecast(
            pair_key
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    try:

        with open(
            path,
            encoding="utf-8"
        ) as f:

            return jsonify(
                json.load(f)
            )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route(
    "/api/refresh",
    methods=["GET", "POST"]
)
def refresh():

    requested_pair = request.args.get(
        "pair"
    )

    if requested_pair:

        pair_key = resolve_pair()

        if pair_key is None:

            return jsonify({
                "error": "Unknown FX pair."
            }), 400

        try:

            return jsonify(
                run_forecast_pipeline(
                    pair_key
                )
            )

        except Exception as e:

            return jsonify({
                "error": str(e)
            }), 500

    results = {}
    errors = {}

    for pair_key in PAIRS:

        try:

            results[pair_key] = (
                run_forecast_pipeline(
                    pair_key
                )
            )

        except Exception as e:

            errors[pair_key] = str(e)

    return jsonify({
        "results": results,
        "errors": errors
    })


@app.route(
    "/api/refresh-cron",
    methods=["GET", "POST"]
)
def refresh_cron():

    requested_pair = request.args.get(
        "pair"
    )

    if not requested_pair:

        return jsonify({

            "message":
                "Provide ?pair=PAIR",

            "pairs":
                list(PAIRS.keys())

        }), 400

    pair_key = resolve_pair()

    if pair_key is None:

        return jsonify({
            "error": "Unknown FX pair."
        }), 400

    try:

        return jsonify(
            run_forecast_pipeline(
                pair_key
            )
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# BACKGROUND SCHEDULER
#
# NOTE: this used to only pre-warm DEFAULT_PAIR ("USDINR"), so
# every other pair in PAIRS (USDJPY, USDCHF, USDCAD, EURUSD,
# GBPUSD, AUDUSD, NZDUSD) still computed cold inside whichever
# user's request happened to find it stale -- exactly the slow
# path this scheduler exists to avoid. refresh_all_pairs() now
# walks every configured pair so the JSON cache is warm for all
# of them, not just the default.
# ============================================================

scheduler = None

if (
    os.environ
    .get(
        "ENABLE_SCHEDULER",
        "false"
    )
    .lower()
    == "true"
):

    def refresh_all_pairs():

        for pair_key in PAIRS:

            try:

                run_forecast_pipeline(
                    pair_key
                )

            except Exception as e:

                print(
                    f"Scheduled refresh failed for "
                    f"{pair_key}: {e}"
                )

    scheduler = BackgroundScheduler(
        timezone="Asia/Kolkata"
    )

    scheduler.add_job(

        refresh_all_pairs,

        "cron",

        day_of_week="mon-fri",

        hour=18,

        minute=0,

        max_instances=1,

        coalesce=True

    )

    scheduler.start()


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )