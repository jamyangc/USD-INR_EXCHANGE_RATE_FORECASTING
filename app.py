import warnings
warnings.filterwarnings("ignore")

import os

# Must be set before numpy/sklearn are imported.
# Pinning BLAS to one thread helps keep results reproducible
# across Render restarts.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import threading


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

START_DATE = "2020-01-01"

PPP_BASE_YEAR = 2020

MIN_TRAIN_DAYS = 500

HORIZON = 1

DAYS_IN_YEAR = 365

MLP_LOGRET_CLIP = 0.02


# ============================================================
# MODEL SELECTION SETTINGS
# ============================================================

SELECTION_WINDOW = 5

ROLLING_WINDOW_DAYS = 7

MIN_ROLLING_DAYS = 3


# ============================================================
# APP SETTINGS
# ============================================================

DEFAULT_PAIR = "USDINR"

APP_NAME = "FX Forecast"

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL FEATURES
# ============================================================

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


# ============================================================
# FX PAIRS
# ============================================================

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


# ============================================================
# FORECAST MODELS
# ============================================================

FORECAST_MODEL_KEYS = [
    "ridge",
    "decision_tree",
    "mlp",
    "ppp",
    "irp"
]


MODEL_LABELS = {

    "ridge": "Ridge",

    "decision_tree": "Decision Tree",

    "mlp": "MLP",

    "ppp": "PPP",

    "irp": "IRP"

}


# ============================================================
# FILE PATHS
# ============================================================

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


# ============================================================
# DATE HELPERS
# ============================================================

def next_business_day(date_value):

    ts = pd.Timestamp(
        date_value
    ).normalize()

    next_day = (
        ts +
        pd.Timedelta(days=1)
    )

    while next_day.weekday() >= 5:

        next_day += pd.Timedelta(
            days=1
        )

    return next_day


def stored_forecast_is_stale(pair_key):

    """
    A stored forecast is stale once the forecast date has
    already passed.

    A forecast remains valid throughout its forecast date.
    """

    path = forecast_file_path(
        pair_key
    )

    if not os.path.exists(path):

        return True

    try:

        with open(
            path,
            encoding="utf-8"
        ) as f:

            saved = json.load(f)

        forecast_date = (
            saved.get("forecast_date")
        )

        if not forecast_date:

            return True

        today = (
            pd.Timestamp.now()
            .normalize()
        )

        saved_forecast_date = (
            pd.Timestamp(
                forecast_date
            ).normalize()
        )

        return (
            saved_forecast_date <
            today
        )

    except Exception:

        return True


# ============================================================
# PER-PAIR PIPELINE LOCKS
# ============================================================

_pipeline_locks = {}

_pipeline_locks_guard = (
    threading.Lock()
)


def get_pipeline_lock(pair_key):

    with _pipeline_locks_guard:

        if pair_key not in _pipeline_locks:

            _pipeline_locks[
                pair_key
            ] = threading.Lock()

        return _pipeline_locks[
            pair_key
        ]


def ensure_fresh_forecast(pair_key):

    if not stored_forecast_is_stale(
        pair_key
    ):

        return

    lock = get_pipeline_lock(
        pair_key
    )

    with lock:

        if stored_forecast_is_stale(
            pair_key
        ):

            run_forecast_pipeline(
                pair_key
            )


# ============================================================
# WORLD BANK DATA
# ============================================================

def get_world_bank_indicator(
    country,
    indicator
):

    url = (
        f"https://api.worldbank.org/v2/country/"
        f"{country}/indicator/{indicator}"
        f"?format=json&per_page=100"
    )

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if (
        len(data) < 2
        or data[1] is None
    ):

        raise RuntimeError(
            f"No World Bank data found for "
            f"{country}/{indicator}"
        )

    records = [

        {
            "year": int(
                item["date"]
            ),

            "value": float(
                item["value"]
            )
        }

        for item in data[1]

        if item["value"] is not None

    ]

    if not records:

        raise RuntimeError(
            f"No usable World Bank data found "
            f"for {country}/{indicator}"
        )

    return (
        pd.DataFrame(records)
        .sort_values("year")
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    df["ret_1"] = (
        df["price"].pct_change(1)
    )

    df["ret_5"] = (
        df["price"].pct_change(5)
    )

    df["ret_10"] = (
        df["price"].pct_change(10)
    )

    df["ret_20"] = (
        df["price"].pct_change(20)
    )


    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    df["vol_5"] = (
        df["ret_1"]
        .rolling(5)
        .std()
    )

    df["vol_20"] = (
        df["ret_1"]
        .rolling(20)
        .std()
    )


    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    df["ma_5"] = (
        df["price"]
        .rolling(5)
        .mean()
    )

    df["ma_20"] = (
        df["price"]
        .rolling(20)
        .mean()
    )

    df["price_ma5_ratio"] = (
        df["price"] /
        df["ma_5"]
    )

    df["price_ma20_ratio"] = (
        df["price"] /
        df["ma_20"]
    )

    df["ma_ratio"] = (
        df["ma_5"] /
        df["ma_20"]
    )


    # --------------------------------------------------------
    # RSI 14
    #
    # Explicitly handle:
    #
    # 1. Normal gain/loss
    # 2. Gain > 0 and loss == 0 -> RSI 100
    # 3. Gain == 0 and loss > 0 -> RSI 0
    # 4. Gain == 0 and loss == 0 -> RSI 50
    #
    # This prevents unnecessary NaN values.
    # --------------------------------------------------------

    delta = (
        df["price"].diff()
    )

    gain = (
        delta
        .clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta
        .clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = pd.Series(
        np.nan,
        index=df.index,
        dtype=float
    )

    normal_loss = (
        loss > 0
    )

    rs.loc[normal_loss] = (
        gain.loc[normal_loss] /
        loss.loc[normal_loss]
    )

    # Gain exists but no loss.
    no_loss = (
        (loss == 0) &
        (gain > 0)
    )

    df.loc[
        no_loss,
        "rsi_14"
    ] = 100.0

    # Loss exists but no gain.
    no_gain = (
        (gain == 0) &
        (loss > 0)
    )

    df.loc[
        no_gain,
        "rsi_14"
    ] = 0.0

    # No movement.
    no_movement = (
        (gain == 0) &
        (loss == 0)
    )

    df.loc[
        no_movement,
        "rsi_14"
    ] = 50.0

    # Normal RSI calculation.
    normal_rsi = (
        normal_loss
    )

    df.loc[
        normal_rsi,
        "rsi_14"
    ] = (
        100 -
        (
            100 /
            (
                1 +
                rs.loc[normal_rsi]
            )
        )
    )


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    df["target_logret"] = np.log(
        df["price"].shift(-HORIZON)
        /
        df["price"]
    )

    return df


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_feature_data(
    df,
    feature_cols=FEATURE_COLS
):

    """
    Remove rows containing NaN or infinite values
    in the required model features.

    Missing values are NOT replaced with zero.
    """

    result = df.copy()

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result = result.dropna(
        subset=feature_cols
    )

    if result.empty:

        raise RuntimeError(
            "No valid rows remain after "
            "feature validation."
        )

    return result


# ============================================================
# MODEL CREATION
# ============================================================

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

    return MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="lbfgs",
        alpha=0.01,
        max_iter=500,
        random_state=RANDOM_STATE
    )


# ============================================================
# PREDICTION HELPERS
# ============================================================

def clip_logret(value):

    return np.clip(
        value,
        -MLP_LOGRET_CLIP,
        MLP_LOGRET_CLIP
    )


# ============================================================
# FORECAST ARCHIVE
# ============================================================

def load_forecast_archive(
    pair_key
):

    path = (
        forecast_archive_file_path(
            pair_key
        )
    )

    if not os.path.exists(path):

        return []

    try:

        with open(
            path,
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return (
            data
            if isinstance(data, list)
            else []
        )

    except Exception:

        return []


def save_forecast_archive(
    pair_key,
    archive
):

    path = (
        forecast_archive_file_path(
            pair_key
        )
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            archive[-120:],
            f,
            indent=2
        )


def archive_forecast(
    pair_key,
    result
):

    archive = (
        load_forecast_archive(
            pair_key
        )
    )

    forecast_date = (
        result.get(
            "forecast_date"
        )
    )

    archive = [

        item

        for item in archive

        if item.get(
            "forecast_date"
        ) != forecast_date

    ]

    archive.append(
        result
    )

    save_forecast_archive(
        pair_key,
        archive
    )


# ============================================================
# ROLLING MODEL SELECTION
# ============================================================

def select_model_rolling_window(
    pair_key,
    df,
    window=ROLLING_WINDOW_DAYS,
    min_days=MIN_ROLLING_DAYS
):

    archive = (
        load_forecast_archive(
            pair_key
        )
    )

    if not archive:

        return None

    date_to_price = dict(
        zip(

            df["date"]
            .dt
            .strftime("%Y-%m-%d"),

            df["price"]

        )
    )

    matched = []

    for item in archive:

        forecast_date = (
            item.get(
                "forecast_date"
            )
        )

        if forecast_date in date_to_price:

            matched.append(
                (
                    forecast_date,
                    item,
                    float(
                        date_to_price[
                            forecast_date
                        ]
                    )
                )
            )

    if len(matched) < min_days:

        return None

    matched.sort(
        key=lambda entry: entry[0]
    )

    matched = matched[-window:]

    errors = {

        model_key: []

        for model_key
        in FORECAST_MODEL_KEYS

    }

    for (
        forecast_date,
        item,
        actual_price
    ) in matched:

        for model_key in FORECAST_MODEL_KEYS:

            value = item.get(
                model_key
            )

            if value is None:

                continue

            try:

                forecast_value = float(
                    value
                )

                if not np.isfinite(
                    forecast_value
                ):

                    continue

                errors[
                    model_key
                ].append(
                    abs(
                        actual_price -
                        forecast_value
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                continue

    mae = {}

    for (
        model_key,
        model_errors
    ) in errors.items():

        if model_errors:

            mae[model_key] = float(
                np.mean(
                    model_errors
                )
            )

    if not mae:

        return None

    selected_model_key = min(
        mae,
        key=mae.get
    )

    return {

        "selected_model_key":
            selected_model_key,

        "selected_model_label":
            MODEL_LABELS[
                selected_model_key
            ],

        "selected_model_error":
            round(
                mae[
                    selected_model_key
                ],
                6
            ),

        "selection_method":
            f"{len(matched)}-day rolling MAE",

        "selection_mae": {
            key: round(
                value,
                6
            )
            for key, value
            in mae.items()
        },

        "rolling_window_days":
            len(matched)

    }


# ============================================================
# HISTORICAL MODEL SELECTION
# ============================================================

def historical_model_selection(
    df,
    ppp_data,
    rates
):

    usable = (
        df
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna(
            subset=
                FEATURE_COLS +
                ["target_logret"]
        )
        .copy()
    )

    if (
        len(usable)
        <
        MIN_TRAIN_DAYS +
        SELECTION_WINDOW
    ):

        return None

    validation_start = (
        len(usable) -
        SELECTION_WINDOW
    )

    errors = {

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

        train = (
            usable
            .iloc[:i]
            .copy()
        )

        test = (
            usable
            .iloc[[i]]
            .copy()
        )

        if len(train) < MIN_TRAIN_DAYS:

            continue

        X_train = (
            train[FEATURE_COLS]
        )

        y_train = (
            train["target_logret"]
        )

        X_test = (
            test[FEATURE_COLS]
        )

        # ----------------------------------------------------
        # Safety validation
        # ----------------------------------------------------

        if not np.isfinite(
            X_train.to_numpy(
                dtype=float
            )
        ).all():

            continue

        if not np.isfinite(
            y_train.to_numpy(
                dtype=float
            )
        ).all():

            continue

        if not np.isfinite(
            X_test.to_numpy(
                dtype=float
            )
        ).all():

            continue

        previous_price = float(
            train["price"].iloc[-1]
        )

        actual_price = float(
            test["price"].iloc[0]
        )

        test_year = int(
            test["year"].iloc[0]
        )


        # ----------------------------------------------------
        # Ridge
        # ----------------------------------------------------

        try:

            scaler = (
                StandardScaler()
            )

            X_train_scaled = (
                scaler.fit_transform(
                    X_train
                )
            )

            X_test_scaled = (
                scaler.transform(
                    X_test
                )
            )

            model = (
                create_ridge_model()
            )

            model.fit(
                X_train_scaled,
                y_train
            )

            pred = (
                model
                .predict(
                    X_test_scaled
                )[0]
            )

            forecast = (
                previous_price *
                np.exp(pred)
            )

            if np.isfinite(
                forecast
            ):

                errors[
                    "ridge"
                ].append(
                    abs(
                        actual_price -
                        forecast
                    )
                )

        except Exception as e:

            print(
                f"Historical Ridge error: {e}"
            )


        # ----------------------------------------------------
        # Decision Tree
        # ----------------------------------------------------

        try:

            model = (
                create_tree_model()
            )

            model.fit(
                X_train,
                y_train
            )

            pred = (
                model
                .predict(
                    X_test
                )[0]
            )

            forecast = (
                previous_price *
                np.exp(pred)
            )

            if np.isfinite(
                forecast
            ):

                errors[
                    "decision_tree"
                ].append(
                    abs(
                        actual_price -
                        forecast
                    )
                )

        except Exception as e:

            print(
                f"Historical Tree error: {e}"
            )


        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        try:

            scaler = (
                StandardScaler()
            )

            X_train_scaled = (
                scaler.fit_transform(
                    X_train
                )
            )

            X_test_scaled = (
                scaler.transform(
                    X_test
                )
            )

            model = (
                create_mlp_model()
            )

            model.fit(
                X_train_scaled,
                y_train
            )

            pred = (
                model
                .predict(
                    X_test_scaled
                )[0]
            )

            pred = clip_logret(
                pred
            )

            forecast = (
                previous_price *
                np.exp(pred)
            )

            if np.isfinite(
                forecast
            ):

                errors[
                    "mlp"
                ].append(
                    abs(
                        actual_price -
                        forecast
                    )
                )

        except Exception as e:

            print(
                f"Historical MLP error: {e}"
            )


        # ----------------------------------------------------
        # PPP
        # ----------------------------------------------------

        try:

            ppp_rows = (
                ppp_data[
                    ppp_data["year"]
                    <= test_year
                ]
                .dropna(
                    subset=[
                        "ppp_rate"
                    ]
                )
                .sort_values(
                    "year"
                )
            )

            if not ppp_rows.empty:

                ppp_forecast = float(
                    ppp_rows
                    .iloc[-1]
                    ["ppp_rate"]
                )

                if np.isfinite(
                    ppp_forecast
                ):

                    errors[
                        "ppp"
                    ].append(
                        abs(
                            actual_price -
                            ppp_forecast
                        )
                    )

        except Exception as e:

            print(
                f"Historical PPP error: {e}"
            )


        # ----------------------------------------------------
        # IRP
        # ----------------------------------------------------

        try:

            rate_rows = (
                rates[
                    rates["year"]
                    <= test_year
                ]
                .dropna(
                    subset=[
                        "quote_rate_decimal",
                        "base_rate_decimal"
                    ]
                )
                .sort_values(
                    "year"
                )
            )

            if not rate_rows.empty:

                row = (
                    rate_rows.iloc[-1]
                )

                quote_rate = float(
                    row[
                        "quote_rate_decimal"
                    ]
                )

                base_rate = float(
                    row[
                        "base_rate_decimal"
                    ]
                )

                irp_forecast = (
                    previous_price *
                    (
                        (1 + quote_rate) /
                        (1 + base_rate)
                    ) **
                    (1 / DAYS_IN_YEAR)
                )

                if np.isfinite(
                    irp_forecast
                ):

                    errors[
                        "irp"
                    ].append(
                        abs(
                            actual_price -
                            irp_forecast
                        )
                    )

        except Exception as e:

            print(
                f"Historical IRP error: {e}"
            )


        # ----------------------------------------------------
        # Memory cleanup
        # ----------------------------------------------------

        gc.collect()


    # ========================================================
    # CALCULATE MAE
    # ========================================================

    mae = {}

    for (
        model_key,
        model_errors
    ) in errors.items():

        if model_errors:

            mae[model_key] = float(
                np.mean(
                    model_errors
                )
            )

    if not mae:

        return None

    selected_model_key = min(
        mae,
        key=mae.get
    )

    return {

        "selected_model_key":
            selected_model_key,

        "selected_model_label":
            MODEL_LABELS[
                selected_model_key
            ],

        "selected_model_error":
            round(
                mae[
                    selected_model_key
                ],
                6
            ),

        "selection_method":
            (
                f"{SELECTION_WINDOW}-day "
                f"walk-forward MAE"
            ),

        "selection_mae": {
            key: round(
                value,
                6
            )
            for key, value
            in mae.items()
        }

    }


# ============================================================
# MAIN FORECAST PIPELINE
# ============================================================

def run_forecast_pipeline(
    pair_key=DEFAULT_PAIR
):

    if pair_key not in PAIRS:

        raise ValueError(
            f"Unknown pair: {pair_key}"
        )

    cfg = PAIRS[
        pair_key
    ]


    # ========================================================
    # DOWNLOAD FX DATA
    # ========================================================

    fx = yf.download(

        cfg["ticker"],

        period="10y",

        interval="1d",

        auto_adjust=False,

        progress=False

    )

    if fx.empty:

        raise RuntimeError(
            f"Could not download "
            f"{pair_key} data."
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
        [
            "date",
            "price"
        ]
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
        df["date"]
        >=
        pd.Timestamp(
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
            f"Not enough data for "
            f"{pair_key}."
        )


    df["year"] = (
        df["date"].dt.year
    )


    # ========================================================
    # BUILD FEATURES
    # ========================================================

    df = build_features(
        df
    )


    # ========================================================
    # WORLD BANK CPI
    # ========================================================

    quote_cpi = (
        get_world_bank_indicator(
            cfg["wb_quote"],
            "FP.CPI.TOTL"
        )
        .rename(
            columns={
                "value":
                    "quote_cpi"
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
                "value":
                    "base_cpi"
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
        .groupby("year")
        ["price"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "price":
                    "average_price"
            }
        )
    )


    ppp_data = pd.merge(

        annual_fx,

        cpi,

        on="year",

        how="inner"

    )


    # ========================================================
    # PPP
    # ========================================================

    base_rows = (
        ppp_data[
            ppp_data["year"]
            ==
            PPP_BASE_YEAR
        ]
    )

    if base_rows.empty:

        raise RuntimeError(
            f"No PPP base-year data "
            f"available for "
            f"{pair_key}."
        )


    base_row = (
        base_rows.iloc[0]
    )


    base_price = float(
        base_row[
            "average_price"
        ]
    )


    base_quote_cpi = float(
        base_row[
            "quote_cpi"
        ]
    )


    base_base_cpi = float(
        base_row[
            "base_cpi"
        ]
    )


    if (
        not np.isfinite(
            base_price
        )
        or
        not np.isfinite(
            base_quote_cpi
        )
        or
        not np.isfinite(
            base_base_cpi
        )
        or
        base_quote_cpi == 0
        or
        base_base_cpi == 0
    ):

        raise RuntimeError(
            f"Invalid PPP base data "
            f"for {pair_key}."
        )


    ppp_data["ppp_rate"] = (

        base_price

        *

        (
            ppp_data[
                "quote_cpi"
            ]
            /
            base_quote_cpi
        )

        /

        (
            ppp_data[
                "base_cpi"
            ]
            /
            base_base_cpi
        )

    )


    # ========================================================
    # WORLD BANK INTEREST RATES
    # ========================================================

    quote_rate = (
        get_world_bank_indicator(
            cfg["wb_quote"],
            "FR.INR.LEND"
        )
        .rename(
            columns={
                "value":
                    "quote_rate"
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
                "value":
                    "base_rate"
            }
        )
    )


    rates = pd.merge(

        quote_rate,

        base_rate,

        on="year",

        how="inner"

    )


    rates[
        "quote_rate_decimal"
    ] = (
        rates[
            "quote_rate"
        ] / 100
    )


    rates[
        "base_rate_decimal"
    ] = (
        rates[
            "base_rate"
        ] / 100
    )


    # ========================================================
    # CLEAN TRAINING DATA
    # ========================================================

    train_df = (

        df

        .replace(
            [np.inf, -np.inf],
            np.nan
        )

        .dropna(
            subset=
                FEATURE_COLS +
                ["target_logret"]
        )

        .copy()

    )


    if len(train_df) < MIN_TRAIN_DAYS:

        raise RuntimeError(
            "Not enough usable "
            "training data."
        )


    X_train = (
        train_df[
            FEATURE_COLS
        ]
    )


    y_train = (
        train_df[
            "target_logret"
        ]
    )


    # ========================================================
    # FINAL TRAINING DATA VALIDATION
    # ========================================================

    if not np.isfinite(
        X_train.to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            f"Training features "
            f"contain invalid values "
            f"for {pair_key}."
        )


    if not np.isfinite(
        y_train.to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            f"Training target "
            f"contains invalid values "
            f"for {pair_key}."
        )


    # ========================================================
    # SCALE FEATURES FOR RIDGE AND MLP
    # ========================================================

    scaler = (
        StandardScaler()
    )


    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )


    # ========================================================
    # RIDGE
    # ========================================================

    ridge_model = (
        create_ridge_model()
    )


    ridge_model.fit(

        X_train_scaled,

        y_train

    )


    # ========================================================
    # DECISION TREE
    # ========================================================

    tree_model = (
        create_tree_model()
    )


    tree_model.fit(

        X_train,

        y_train

    )


    # ========================================================
    # MLP
    # ========================================================

    mlp_model = (
        create_mlp_model()
    )


    mlp_model.fit(

        X_train_scaled,

        y_train

    )


    # ========================================================
    # LATEST VALID FEATURE ROW
    #
    # IMPORTANT:
    # Do not blindly use df.iloc[-1][FEATURE_COLS].
    #
    # Instead, find the latest row where every model feature
    # is valid.
    # ========================================================

    latest_feature_rows = (

        df

        .replace(
            [np.inf, -np.inf],
            np.nan
        )

        .dropna(
            subset=FEATURE_COLS
        )

    )


    if latest_feature_rows.empty:

        raise RuntimeError(
            f"No valid feature row "
            f"available for "
            f"{pair_key}."
        )


    latest_feature_row = (

        latest_feature_rows

        .iloc[[-1]]

        .copy()

    )


    latest_features = (

        latest_feature_row[
            FEATURE_COLS
        ]

    )


    # ========================================================
    # FINAL SAFETY CHECK BEFORE SKLEARN
    # ========================================================

    if not np.isfinite(

        latest_features
        .to_numpy(
            dtype=float
        )

    ).all():

        raise RuntimeError(

            f"Invalid model features "
            f"detected for "
            f"{pair_key}."

        )


    # ========================================================
    # SCALE LATEST FEATURES
    # ========================================================

    latest_features_scaled = (

        scaler.transform(

            latest_features

        )

    )


    # ========================================================
    # LATEST DATE AND PRICE
    # ========================================================

    last_known_date = (

        latest_feature_row[
            "date"
        ].iloc[0]

    )


    last_known_price = float(

        latest_feature_row[
            "price"
        ].iloc[0]

    )


    if not np.isfinite(
        last_known_price
    ):

        raise RuntimeError(
            f"Invalid latest price "
            f"for {pair_key}."
        )


    # ========================================================
    # RIDGE FORECAST
    # ========================================================

    ridge_prediction = (

        ridge_model

        .predict(

            latest_features_scaled

        )[0]

    )


    forecast_ridge = (

        last_known_price *

        np.exp(
            ridge_prediction
        )

    )


    # ========================================================
    # DECISION TREE FORECAST
    # ========================================================

    tree_prediction = (

        tree_model

        .predict(

            latest_features

        )[0]

    )


    forecast_tree = (

        last_known_price *

        np.exp(
            tree_prediction
        )

    )


    # ========================================================
    # MLP FORECAST
    # ========================================================

    mlp_prediction = (

        mlp_model

        .predict(

            latest_features_scaled

        )[0]

    )


    mlp_prediction = (
        clip_logret(
            mlp_prediction
        )
    )


    forecast_mlp = (

        last_known_price *

        np.exp(
            mlp_prediction
        )

    )


    # ========================================================
    # VALIDATE MODEL FORECASTS
    # ========================================================

    model_forecasts = {

        "ridge":
            forecast_ridge,

        "decision_tree":
            forecast_tree,

        "mlp":
            forecast_mlp

    }


    for (
        model_key,
        forecast_value
    ) in model_forecasts.items():

        if not np.isfinite(
            forecast_value
        ):

            raise RuntimeError(

                f"{model_key} produced "
                f"an invalid forecast "
                f"for {pair_key}."

            )


    # ========================================================
    # PPP FORECAST
    # ========================================================

    ppp_rows = (

        ppp_data

        .dropna(
            subset=[
                "ppp_rate"
            ]
        )

        .sort_values(
            "year"
        )

    )


    if ppp_rows.empty:

        raise RuntimeError(
            f"No usable PPP data "
            f"for {pair_key}."
        )


    latest_ppp_row = (
        ppp_rows.iloc[-1]
    )


    forecast_ppp = float(
        latest_ppp_row[
            "ppp_rate"
        ]
    )


    latest_ppp_year = int(
        latest_ppp_row[
            "year"
        ]
    )


    if not np.isfinite(
        forecast_ppp
    ):

        raise RuntimeError(
            f"Invalid PPP forecast "
            f"for {pair_key}."
        )


    # ========================================================
    # IRP FORECAST
    # ========================================================

    rate_rows = (

        rates

        .dropna(
            subset=[
                "quote_rate_decimal",
                "base_rate_decimal"
            ]
        )

        .sort_values(
            "year"
        )

    )


    if rate_rows.empty:

        raise RuntimeError(
            f"No usable interest-rate "
            f"data for {pair_key}."
        )


    latest_rate_row = (
        rate_rows.iloc[-1]
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

        latest_rate_row[
            "year"
        ]

    )


    forecast_irp = (

        last_known_price *

        (

            (1 + quote_rate_value)

            /

            (1 + base_rate_value)

        )

        **

        (1 / DAYS_IN_YEAR)

    )


    if not np.isfinite(
        forecast_irp
    ):

        raise RuntimeError(
            f"Invalid IRP forecast "
            f"for {pair_key}."
        )


    # ========================================================
    # RANDOM WALK BENCHMARK
    # ========================================================

    forecast_rw = (
        last_known_price
    )


    # ========================================================
    # FORECAST DATE
    # ========================================================

    forecast_date = (
        next_business_day(
            last_known_date
        )
    )


    # ========================================================
    # MODEL SELECTION
    # ========================================================

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

        if rolling_selection
        is not None

        else

        historical_selection

    )


    # ========================================================
    # SELECTED MODEL
    # ========================================================

    if selection is not None:

        selected_model_key = (

            selection[
                "selected_model_key"
            ]

        )


        forecasts = {

            "ridge":
                forecast_ridge,

            "decision_tree":
                forecast_tree,

            "mlp":
                forecast_mlp,

            "ppp":
                forecast_ppp,

            "irp":
                forecast_irp

        }


        selected_model_forecast = float(

            forecasts[
                selected_model_key
            ]

        )


        if (
            selected_model_forecast
            >
            last_known_price
        ):

            selected_model_direction = (
                "UP"
            )

        elif (
            selected_model_forecast
            <
            last_known_price
        ):

            selected_model_direction = (
                "DOWN"
            )

        else:

            selected_model_direction = (
                "FLAT"
            )

    else:

        selected_model_key = None

        selected_model_forecast = None

        selected_model_direction = None


    # ========================================================
    # RIDGE DIRECTION
    # ========================================================

    if (
        forecast_ridge
        >
        last_known_price
    ):

        ridge_direction = "UP"

    elif (
        forecast_ridge
        <
        last_known_price
    ):

        ridge_direction = "DOWN"

    else:

        ridge_direction = "FLAT"


    # ========================================================
    # RIDGE CHANGE %
    # ========================================================

    ridge_change_percent = (

        (

            forecast_ridge
            -
            last_known_price

        )

        /

        last_known_price

        *

        100

    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

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

        "ridge_change_percent":
            round(
                ridge_change_percent,
                4
            ),

        "ridge_direction":
            ridge_direction,

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

                if (
                    selected_model_forecast
                    is not None
                )

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


    # ========================================================
    # SAVE LATEST FORECAST
    # ========================================================

    with open(

        forecast_file_path(
            pair_key
        ),

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            result,

            f,

            indent=2

        )


    # ========================================================
    # ARCHIVE FORECAST
    # ========================================================

    archive_forecast(

        pair_key,

        result

    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    recent_history = (

        df[
            [
                "date",
                "price"
            ]
        ]

        .tail(365)

        .copy()

    )


    recent_history[
        "date"
    ] = (

        recent_history[
            "date"
        ]

        .dt

        .strftime(
            "%Y-%m-%d"
        )

    )


    history = (
        recent_history
        .to_dict(
            orient="records"
        )
    )


    with open(

        history_file_path(
            pair_key
        ),

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            history,

            f,

            indent=2

        )


    # ========================================================
    # LOG
    # ========================================================

    print(

        f"[{datetime.now()}] "

        f"{pair_key} | "

        f"Selected: "

        f"{selected_model_key or 'pending'} | "

        f"Next forecast: "

        f"{"pending" if selected_model_forecast is None else selected_model_forecast} | "

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


# ============================================================
# CORS
# ============================================================

CORS(

    app,

    resources={

        r"/api/*": {

            "origins": [

                "https://localhost"

            ]

        }

    }

)


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def index():

    return send_from_directory(

        app.static_folder,

        "index.html"

    )


@app.route(
    "/service-worker.js"
)
def service_worker():

    return send_from_directory(

        app.static_folder,

        "service-worker.js",

        mimetype=
            "application/javascript"

    )


# ============================================================
# AVAILABLE PAIRS
# ============================================================

@app.route("/api/pairs")
def pairs():

    return jsonify([

        {

            "key": key,

            "label": cfg["label"],

            "base": cfg["base"],

            "quote": cfg["quote"]

        }

        for key, cfg
        in PAIRS.items()

    ])


# ============================================================
# RESOLVE PAIR
# ============================================================

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


# ============================================================
# PREDICT
# ============================================================

@app.route("/api/predict")
def predict():

    pair_key = (
        resolve_pair()
    )

    if pair_key is None:

        return jsonify({

            "error":
                "Unknown FX pair."

        }), 400


    path = (
        forecast_file_path(
            pair_key
        )
    )


    try:

        ensure_fresh_forecast(

            pair_key

        )

    except Exception as e:

        print(
            f"Prediction failed "
            f"for {pair_key}: {e}"
        )

        return jsonify({

            "error":
                str(e),

            "pair":
                pair_key

        }), 500


    try:

        with open(

            path,

            encoding="utf-8"

        ) as f:

            data = json.load(f)


        response = jsonify(
            data
        )

        response.headers[
            "Cache-Control"
        ] = "no-store"


        return response


    except FileNotFoundError:

        return jsonify({

            "error":
                "Forecast file not found.",

            "pair":
                pair_key

        }), 404


    except Exception as e:

        return jsonify({

            "error":
                str(e),

            "pair":
                pair_key

        }), 500


# ============================================================
# HISTORY
# ============================================================

@app.route("/api/history")
def history():

    pair_key = (
        resolve_pair()
    )

    if pair_key is None:

        return jsonify({

            "error":
                "Unknown FX pair."

        }), 400


    path = (
        history_file_path(
            pair_key
        )
    )


    try:

        with open(

            path,

            encoding="utf-8"

        ) as f:

            data = json.load(f)


        response = jsonify(
            data
        )

        response.headers[
            "Cache-Control"
        ] = "no-store"


        return response


    except FileNotFoundError:

        return jsonify([])


    except Exception as e:

        return jsonify({

            "error":
                str(e),

            "pair":
                pair_key

        }), 500


# ============================================================
# MANUAL REFRESH
# ============================================================

@app.route(

    "/api/refresh",

    methods=[
        "GET",
        "POST"
    ]

)
def refresh():

    requested_pair = (
        request.args.get(
            "pair"
        )
    )


    pair_key = (

        resolve_pair()

        if requested_pair

        else DEFAULT_PAIR

    )


    if pair_key is None:

        return jsonify({

            "error":
                "Unknown FX pair."

        }), 400


    try:

        return jsonify(

            run_forecast_pipeline(

                pair_key

            )

        )


    except Exception as e:

        print(

            f"Manual refresh failed "
            f"for {pair_key}: {e}"

        )


        return jsonify({

            "error":
                str(e),

            "pair":
                pair_key

        }), 500


# ============================================================
# CRON REFRESH
# ============================================================

@app.route(

    "/api/refresh-cron",

    methods=[
        "GET",
        "POST"
    ]

)
def refresh_cron():

    requested_pair = (
        request.args.get(
            "pair"
        )
    )


    if not requested_pair:

        return jsonify({

            "message":
                "Provide ?pair=PAIR",

            "pairs":
                list(
                    PAIRS.keys()
                )

        }), 400


    pair_key = (
        resolve_pair()
    )


    if pair_key is None:

        return jsonify({

            "error":
                "Unknown FX pair."

        }), 400


    try:

        return jsonify(

            run_forecast_pipeline(

                pair_key

            )

        )


    except Exception as e:

        print(

            f"Cron refresh failed "
            f"for {pair_key}: {e}"

        )


        return jsonify({

            "error":
                str(e),

            "pair":
                pair_key

        }), 500


# ============================================================
# BACKGROUND SCHEDULER
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

                    f"Scheduled refresh failed "
                    f"for {pair_key}: {e}"

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


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

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