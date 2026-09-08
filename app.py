# ============================================================
# FX FORECAST DASHBOARD - FLASK BACKEND (MULTI-PAIR)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

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


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
START_DATE = "2020-01-01"
PPP_BASE_YEAR = 2020
MIN_TRAIN_DAYS = 500
HORIZON = 1
DAYS_IN_YEAR = 365

MLP_LOGRET_CLIP = 0.02

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

DEFAULT_PAIR = "USDINR"


# ============================================================
# PAIR DEFINITIONS
#
# "base" / "quote" follow FX convention: the price is
# quote-currency-per-1-unit-of-base-currency, matching the
# Yahoo Finance ticker (BASEQUOTE=X).
#
# wb_base / wb_quote are World Bank ISO3 country (or
# aggregate) codes used to pull CPI (for PPP) and lending
# rates (for IRP) for each side of the pair.
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


# ============================================================
# WORLD BANK HELPER
# ============================================================

def get_world_bank_indicator(country, indicator):

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

    if len(data) < 2 or data[1] is None:
        raise RuntimeError(
            f"No World Bank data found for "
            f"{country} / {indicator}"
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
            f"No usable World Bank data found for "
            f"{country} / {indicator}"
        )

    return pd.DataFrame(records).sort_values("year")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df):

    df = df.copy()

    df["ret_1"] = df["price"].pct_change(1)
    df["ret_5"] = df["price"].pct_change(5)
    df["ret_10"] = df["price"].pct_change(10)
    df["ret_20"] = df["price"].pct_change(20)

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

    df["ma_50"] = (
        df["price"]
        .rolling(50)
        .mean()
    )

    df["price_ma5_ratio"] = (
        df["price"] / df["ma_5"]
    )

    df["price_ma20_ratio"] = (
        df["price"] / df["ma_20"]
    )

    df["ma_ratio"] = (
        df["ma_5"] / df["ma_20"]
    )

    # RSI(14)
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

    df["rsi_14"] = (
        100 - 100 / (1 + rs)
    )

    # One-day-ahead log return target
    df["target_logret"] = np.log(
        df["price"].shift(-HORIZON)
        / df["price"]
    )

    return df


# ============================================================
# MODEL FACTORIES
# ============================================================

def create_model():

    return lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        objective="regression",
        verbosity=-1,
        n_jobs=-1
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

    return MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="lbfgs",
        alpha=1e-2,
        max_iter=5000,
        random_state=RANDOM_STATE
    )


def clip_logret(
    value,
    limit=MLP_LOGRET_CLIP
):

    if value > limit:
        return limit

    if value < -limit:
        return -limit

    return value


# ============================================================
# MAIN FORECAST PIPELINE
# Runs once per trading day, per currency pair.
# ============================================================

def run_forecast_pipeline(pair_key=DEFAULT_PAIR):

    if pair_key not in PAIRS:
        raise ValueError(f"Unknown pair: {pair_key}")

    cfg = PAIRS[pair_key]

    ticker = cfg["ticker"]
    base_ccy = cfg["base"]
    quote_ccy = cfg["quote"]
    wb_base = cfg["wb_base"]
    wb_quote = cfg["wb_quote"]

    print(
        f"[{datetime.now()}] "
        f"Running daily forecast pipeline for {pair_key}..."
    )

    # ========================================================
    # DAILY PRICE DATA
    # ========================================================

    fx = yf.download(
        ticker,
        period="10y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if fx.empty:
        raise RuntimeError(
            f"Could not download {pair_key} data from Yahoo Finance."
        )

    # Handle MultiIndex columns returned by newer yfinance versions
    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)

    fx = fx.reset_index()

    fx = fx.rename(
        columns={
            "Date": "date",
            "Close": "price"
        }
    )

    if "date" not in fx.columns:
        raise RuntimeError(
            "Yahoo Finance data does not contain a Date column."
        )

    if "price" not in fx.columns:
        raise RuntimeError(
            "Yahoo Finance data does not contain a Close column."
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
        df["date"] >= pd.Timestamp(START_DATE)
    ].copy()

    df = (
        df
        .sort_values("date")
        .drop_duplicates(subset="date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise RuntimeError(
            f"No {pair_key} observations available from 2020 onward."
        )

    if len(df) < MIN_TRAIN_DAYS:
        raise RuntimeError(
            f"Not enough {pair_key} observations. "
            f"Required: {MIN_TRAIN_DAYS}, "
            f"available: {len(df)}."
        )

    df["year"] = df["date"].dt.year

    df = build_features(df)

    # ========================================================
    # PPP
    # (price = quote-per-base, so relative PPP moves the
    # base-year price by the quote/base CPI ratio)
    # ========================================================

    quote_cpi = (
        get_world_bank_indicator(
            wb_quote,
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
            wb_base,
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
        df.groupby("year")["price"]
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

    if PPP_BASE_YEAR not in ppp_data["year"].values:
        raise RuntimeError(
            "PPP base year not available."
        )

    base_row = ppp_data[
        ppp_data["year"] == PPP_BASE_YEAR
    ].iloc[0]

    base_price = base_row["average_price"]
    base_quote_cpi = base_row["quote_cpi"]
    base_base_cpi = base_row["base_cpi"]

    ppp_data["ppp_rate"] = (
        base_price
        * (
            ppp_data["quote_cpi"]
            / base_quote_cpi
        )
        / (
            ppp_data["base_cpi"]
            / base_base_cpi
        )
    )

    # ========================================================
    # IRP
    # ========================================================

    quote_rate = (
        get_world_bank_indicator(
            wb_quote,
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
            wb_base,
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

    # ========================================================
    # TRAINING DATA
    # ========================================================

    train_df = df.dropna(
        subset=FEATURE_COLS + ["target_logret"]
    ).copy()

    if len(train_df) < MIN_TRAIN_DAYS:
        raise RuntimeError(
            f"Not enough usable training observations. "
            f"Required: {MIN_TRAIN_DAYS}, "
            f"available: {len(train_df)}."
        )

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    model = create_model()

    model.fit(
        train_df[FEATURE_COLS],
        train_df["target_logret"]
    )

    feature_scaler = StandardScaler()

    train_features_scaled = (
        feature_scaler.fit_transform(
            train_df[FEATURE_COLS]
        )
    )

    ridge_model = create_ridge_model()

    ridge_model.fit(
        train_features_scaled,
        train_df["target_logret"]
    )

    tree_model = create_tree_model()

    tree_model.fit(
        train_df[FEATURE_COLS],
        train_df["target_logret"]
    )

    mlp_model = create_mlp_model()

    mlp_model.fit(
        train_features_scaled,
        train_df["target_logret"]
    )

    # ========================================================
    # LATEST AVAILABLE DAILY OBSERVATION
    # ========================================================

    last_known_date = df["date"].iloc[-1]

    last_known_price = float(
        df["price"].iloc[-1]
    )

    latest_features = (
        df.iloc[[-1]][FEATURE_COLS]
    )

    if latest_features.isnull().values.any():
        raise RuntimeError(
            "Latest observation is missing required features."
        )

    latest_features_scaled = (
        feature_scaler.transform(
            latest_features
        )
    )

    # ========================================================
    # FORECASTS
    # ========================================================

    predicted_logret = model.predict(
        latest_features
    )[0]

    forecast_lgbm = (
        last_known_price
        * np.exp(predicted_logret)
    )

    predicted_logret_ridge = (
        ridge_model.predict(
            latest_features_scaled
        )[0]
    )

    forecast_ridge = (
        last_known_price
        * np.exp(predicted_logret_ridge)
    )

    predicted_logret_tree = (
        tree_model.predict(
            latest_features
        )[0]
    )

    forecast_tree = (
        last_known_price
        * np.exp(predicted_logret_tree)
    )

    predicted_logret_mlp = (
        mlp_model.predict(
            latest_features_scaled
        )[0]
    )

    predicted_logret_mlp = clip_logret(
        predicted_logret_mlp
    )

    forecast_mlp = (
        last_known_price
        * np.exp(predicted_logret_mlp)
    )

    latest_ppp_row = (
        ppp_data
        .dropna(subset=["ppp_rate"])
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

    latest_quote_rate = float(
        latest_rate_row["quote_rate_decimal"]
    )

    latest_base_rate = float(
        latest_rate_row["base_rate_decimal"]
    )

    latest_rate_year = int(
        latest_rate_row["year"]
    )

    forecast_irp = (
        last_known_price
        * (
            (1 + latest_quote_rate)
            / (1 + latest_base_rate)
        ) ** (1 / DAYS_IN_YEAR)
    )

    forecast_rw = last_known_price

    forecast_date = (
        last_known_date
        + pd.tseries.offsets.BDay(1)
    )

    if forecast_lgbm > last_known_price:

        direction = "UP"

    elif forecast_lgbm < last_known_price:

        direction = "DOWN"

    else:

        direction = "FLAT"

    change_pct = (
        (forecast_lgbm - last_known_price)
        / last_known_price
        * 100
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "pair": pair_key,

        "pair_label": cfg["label"],

        "base_ccy": base_ccy,

        "quote_ccy": quote_ccy,

        "latest_available_date":
            last_known_date.strftime("%Y-%m-%d"),

        "latest_price":
            round(last_known_price, 4),

        "forecast_date":
            forecast_date.strftime("%Y-%m-%d"),

        "random_walk":
            round(forecast_rw, 4),

        "ppp":
            round(forecast_ppp, 4),

        "ppp_cpi_year":
            latest_ppp_year,

        "irp":
            round(forecast_irp, 4),

        "irp_rate_year":
            latest_rate_year,

        "ridge":
            round(forecast_ridge, 4),

        "decision_tree":
            round(forecast_tree, 4),

        "mlp":
            round(forecast_mlp, 4),

        "lightgbm":
            round(forecast_lgbm, 4),

        "lightgbm_change_percent":
            round(change_pct, 4),

        "lightgbm_direction":
            direction,

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

    # ========================================================
    # SAVE LAST 60 DAYS
    # ========================================================

    recent = (
        df[["date", "price"]]
        .tail(60)
        .copy()
    )

    recent["date"] = (
        recent["date"]
        .dt.strftime("%Y-%m-%d")
    )

    history = recent.to_dict(
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
        f"Forecast completed for {pair_key}. "
        f"LightGBM -> "
        f"{forecast_lgbm:.4f} "
        f"({direction}) | "
        f"Ridge -> {forecast_ridge:.4f} | "
        f"Tree -> {forecast_tree:.4f} | "
        f"MLP -> {forecast_mlp:.4f}"
    )

    return result


def run_all_pairs():

    results = {}

    errors = {}

    for pair_key in PAIRS:

        try:

            results[pair_key] = run_forecast_pipeline(pair_key)

        except Exception as e:

            print(
                f"[{datetime.now()}] "
                f"Pipeline failed for {pair_key}: {e}"
            )

            errors[pair_key] = str(e)

    return results, errors


# ============================================================
# FLASK APP
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


# ============================================================
# PAIRS API
# Lets the frontend build a currency-pair selector without
# hardcoding the list.
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
        for key, cfg in PAIRS.items()
    ])


def _resolve_pair_arg():

    pair_key = request.args.get("pair", DEFAULT_PAIR).upper()

    if pair_key not in PAIRS:
        return None

    return pair_key


# ============================================================
# FORECAST API
# ============================================================

@app.route("/api/predict")
def predict():

    pair_key = _resolve_pair_arg()

    if pair_key is None:

        return jsonify({
            "error": f"Unknown pair '{request.args.get('pair')}'"
        }), 400

    path = forecast_file_path(pair_key)

    if not os.path.exists(path):

        try:

            run_forecast_pipeline(pair_key)

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


# ============================================================
# HISTORY API
# ============================================================

@app.route("/api/history")
def history():

    pair_key = _resolve_pair_arg()

    if pair_key is None:

        return jsonify({
            "error": f"Unknown pair '{request.args.get('pair')}'"
        }), 400

    path = history_file_path(pair_key)

    if not os.path.exists(path):

        try:

            run_forecast_pipeline(pair_key)

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


# ============================================================
# MANUAL REFRESH API
# With no ?pair=, refreshes every configured pair.
# ============================================================

@app.route(
    "/api/refresh",
    methods=["POST"]
)
def refresh():

    requested_pair = request.args.get("pair")

    if requested_pair:

        pair_key = _resolve_pair_arg()

        if pair_key is None:

            return jsonify({
                "error": f"Unknown pair '{requested_pair}'"
            }), 400

        try:

            result = run_forecast_pipeline(pair_key)

            return jsonify(result)

        except Exception as e:

            return jsonify({
                "error": str(e)
            }), 500

    results, errors = run_all_pairs()

    return jsonify({
        "results": results,
        "errors": errors
    })


# ============================================================
# CRON-FRIENDLY REFRESH
# Always refreshes every configured pair.
# ============================================================

@app.route(
    "/api/refresh-cron",
    methods=["GET", "POST"]
)
def refresh_cron():

    results, errors = run_all_pairs()

    return jsonify({
        "results": results,
        "errors": errors
    })


# ============================================================
# DAILY TRADING-DAY SCHEDULER
#
# IMPORTANT:
# Do not start APScheduler when using Gunicorn workers.
# Set ENABLE_SCHEDULER=true if you specifically want the
# in-process scheduler enabled.
# ============================================================

scheduler = None

if os.environ.get(
    "ENABLE_SCHEDULER",
    "false"
).lower() == "true":

    scheduler = BackgroundScheduler(
        timezone="Asia/Kolkata"
    )

    scheduler.add_job(
        run_all_pairs,
        "cron",
        day_of_week="mon-fri",
        hour=18,
        minute=0,
        max_instances=1,
        coalesce=True
    )

    scheduler.start()

    print(
        "APScheduler started."
    )


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    # Run once per pair at startup if no forecast exists yet.

    for pair_key in PAIRS:

        if not os.path.exists(
            forecast_file_path(pair_key)
        ):

            try:

                run_forecast_pipeline(pair_key)

            except Exception as e:

                print(
                    f"Startup pipeline run failed for {pair_key}:",
                    e
                )

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