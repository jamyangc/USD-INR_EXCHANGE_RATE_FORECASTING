# ============================================================
# USD/INR FORECAST DASHBOARD - FLASK BACKEND
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

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TICKER = "USDINR=X"
START_DATE = "2020-01-01"
PPP_BASE_YEAR = 2020
MIN_TRAIN_DAYS = 500
HORIZON = 1
DAYS_IN_YEAR = 365

FORECAST_FILE = os.path.join(
    os.path.dirname(__file__),
    "latest_forecast.json"
)

HISTORY_FILE = os.path.join(
    os.path.dirname(__file__),
    "history.json"
)

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
# WORLD BANK HELPER
# ============================================================

def get_world_bank_indicator(country, indicator):

    url = (
        f"https://api.worldbank.org/v2/country/{country}/indicator/"
        f"{indicator}?format=json&per_page=100"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()

    if len(data) < 2:
        raise RuntimeError(
            f"No World Bank data found for {country} / {indicator}"
        )

    records = [
        {
            "year": int(item["date"]),
            "value": float(item["value"])
        }
        for item in data[1]
        if item["value"] is not None
    ]

    return pd.DataFrame(records).sort_values("year")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df):

    df = df.copy()

    df["ret_1"] = df["usdinr"].pct_change(1)
    df["ret_5"] = df["usdinr"].pct_change(5)
    df["ret_10"] = df["usdinr"].pct_change(10)
    df["ret_20"] = df["usdinr"].pct_change(20)

    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["ma_5"] = df["usdinr"].rolling(5).mean()
    df["ma_20"] = df["usdinr"].rolling(20).mean()
    df["ma_50"] = df["usdinr"].rolling(50).mean()

    df["price_ma5_ratio"] = (
        df["usdinr"] / df["ma_5"]
    )

    df["price_ma20_ratio"] = (
        df["usdinr"] / df["ma_20"]
    )

    df["ma_ratio"] = (
        df["ma_5"] / df["ma_20"]
    )

    delta = df["usdinr"].diff()

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

    df["target_logret"] = np.log(
        df["usdinr"].shift(-HORIZON)
        / df["usdinr"]
    )

    return df


# ============================================================
# LIGHTGBM MODEL
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


# ============================================================
# RIDGE / DECISION TREE / MLP MODELS
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
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=2000,
        random_state=RANDOM_STATE
    )


# ============================================================
# MAIN FORECAST PIPELINE
# RUNS ONCE PER TRADING DAY
# ============================================================

def run_forecast_pipeline():

    print(
        f"[{datetime.now()}] "
        "Running daily forecast pipeline..."
    )

    # ========================================================
    # USD/INR DAILY DATA
    # ========================================================

    fx = yf.download(
        TICKER,
        period="10y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if fx.empty:
        raise RuntimeError(
            "Could not download USD/INR data from Yahoo Finance."
        )

    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)

    fx = (
        fx.reset_index()
        .rename(
            columns={
                "Date": "date",
                "Close": "usdinr"
            }
        )
    )

    df = fx[["date", "usdinr"]].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["usdinr"] = pd.to_numeric(
        df["usdinr"],
        errors="coerce"
    )

    df = df.dropna()

    df = df[
        df["date"] >= pd.Timestamp(START_DATE)
    ].copy()

    df = (
        df.sort_values("date")
        .drop_duplicates(subset="date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise RuntimeError(
            "No USD/INR observations available from 2020 onward."
        )

    df["year"] = df["date"].dt.year

    df = build_features(df)


    # ========================================================
    # PPP
    # ========================================================

    india_cpi = (
        get_world_bank_indicator(
            "IND",
            "FP.CPI.TOTL"
        )
        .rename(columns={"value": "india_cpi"})
    )

    usa_cpi = (
        get_world_bank_indicator(
            "USA",
            "FP.CPI.TOTL"
        )
        .rename(columns={"value": "usa_cpi"})
    )

    cpi = pd.merge(
        india_cpi,
        usa_cpi,
        on="year",
        how="inner"
    )

    annual_fx = (
        df.groupby("year")["usdinr"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "usdinr": "average_usdinr"
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

    base_fx = base_row["average_usdinr"]
    base_india_cpi = base_row["india_cpi"]
    base_usa_cpi = base_row["usa_cpi"]

    ppp_data["ppp_rate"] = (
        base_fx
        * (
            ppp_data["india_cpi"]
            / base_india_cpi
        )
        / (
            ppp_data["usa_cpi"]
            / base_usa_cpi
        )
    )


    # ========================================================
    # IRP
    # ========================================================

    india_rate = (
        get_world_bank_indicator(
            "IND",
            "FR.INR.LEND"
        )
        .rename(columns={"value": "india_rate"})
    )

    usa_rate = (
        get_world_bank_indicator(
            "USA",
            "FR.INR.LEND"
        )
        .rename(columns={"value": "usa_rate"})
    )

    rates = pd.merge(
        india_rate,
        usa_rate,
        on="year",
        how="inner"
    )

    rates["india_rate_decimal"] = (
        rates["india_rate"] / 100
    )

    rates["usa_rate_decimal"] = (
        rates["usa_rate"] / 100
    )


    # ========================================================
    # TRAIN LIGHTGBM
    # ========================================================

    train_df = df.dropna(
        subset=FEATURE_COLS + ["target_logret"]
    )

    if len(train_df) < 100:
        raise RuntimeError(
            "Not enough data to train LightGBM."
        )

    model = create_model()

    model.fit(
        train_df[FEATURE_COLS],
        train_df["target_logret"]
    )


    # ========================================================
    # TRAIN RIDGE / DECISION TREE / MLP
    # (Ridge and MLP use standardized features;
    #  the tree is trained on raw features.)
    # ========================================================

    feature_scaler = StandardScaler()

    train_features_scaled = feature_scaler.fit_transform(
        train_df[FEATURE_COLS]
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
        df["usdinr"].iloc[-1]
    )

    latest_features = df.iloc[[-1]][FEATURE_COLS]

    if latest_features.isnull().values.any():
        raise RuntimeError(
            "Latest observation is missing required features."
        )

    latest_features_scaled = feature_scaler.transform(
        latest_features
    )


    # ========================================================
    # LIGHTGBM FORECAST
    # ========================================================

    predicted_logret = model.predict(
        latest_features
    )[0]

    forecast_lgbm = (
        last_known_price
        * np.exp(predicted_logret)
    )


    # ========================================================
    # RIDGE FORECAST
    # ========================================================

    predicted_logret_ridge = ridge_model.predict(
        latest_features_scaled
    )[0]

    forecast_ridge = (
        last_known_price
        * np.exp(predicted_logret_ridge)
    )


    # ========================================================
    # DECISION TREE FORECAST
    # ========================================================

    predicted_logret_tree = tree_model.predict(
        latest_features
    )[0]

    forecast_tree = (
        last_known_price
        * np.exp(predicted_logret_tree)
    )


    # ========================================================
    # MLP FORECAST
    # ========================================================

    predicted_logret_mlp = mlp_model.predict(
        latest_features_scaled
    )[0]

    forecast_mlp = (
        last_known_price
        * np.exp(predicted_logret_mlp)
    )


    # ========================================================
    # PPP FORECAST
    # ========================================================

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


    # ========================================================
    # IRP FORECAST
    # ========================================================

    latest_rate_row = (
        rates
        .dropna(
            subset=[
                "india_rate_decimal",
                "usa_rate_decimal"
            ]
        )
        .sort_values("year")
        .iloc[-1]
    )

    latest_india_rate = float(
        latest_rate_row["india_rate_decimal"]
    )

    latest_usa_rate = float(
        latest_rate_row["usa_rate_decimal"]
    )

    latest_rate_year = int(
        latest_rate_row["year"]
    )

    forecast_irp = (
        last_known_price
        * (
            (1 + latest_india_rate)
            / (1 + latest_usa_rate)
        ) ** (1 / DAYS_IN_YEAR)
    )


    # ========================================================
    # RANDOM WALK
    # ========================================================

    forecast_rw = last_known_price


    # ========================================================
    # TOMORROW'S BUSINESS DAY
    # ========================================================

    forecast_date = (
        last_known_date
        + pd.tseries.offsets.BDay(1)
    )


    # ========================================================
    # DIRECTION
    # ========================================================

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

        "latest_available_date":
            last_known_date.strftime("%Y-%m-%d"),

        "latest_usdinr":
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


    # ========================================================
    # SAVE FORECAST
    # ========================================================

    with open(
        FORECAST_FILE,
        "w"
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
        df[["date", "usdinr"]]
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
        HISTORY_FILE,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )


    print(
        f"[{datetime.now()}] "
        f"Forecast completed. "
        f"LightGBM -> "
        f"{forecast_lgbm:.4f} "
        f"({direction}) | "
        f"Ridge -> {forecast_ridge:.4f} | "
        f"Tree -> {forecast_tree:.4f} | "
        f"MLP -> {forecast_mlp:.4f}"
    )

    return result


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_folder="static"
)

CORS(app)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return send_from_directory(
        "static",
        "index.html"
    )


# ============================================================
# FORECAST API
# ============================================================

@app.route("/api/predict")
def predict():

    if not os.path.exists(FORECAST_FILE):

        try:

            run_forecast_pipeline()

        except Exception as e:

            return jsonify({
                "error": str(e)
            }), 500

    try:

        with open(
            FORECAST_FILE
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

    if not os.path.exists(HISTORY_FILE):

        try:

            run_forecast_pipeline()

        except Exception as e:

            return jsonify({
                "error": str(e)
            }), 500

    try:

        with open(
            HISTORY_FILE
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
# ============================================================

@app.route(
    "/api/refresh",
    methods=["POST"]
)
def refresh():

    try:

        result = run_forecast_pipeline()

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# DAILY TRADING-DAY SCHEDULER
# ============================================================

scheduler = BackgroundScheduler(
    timezone="Asia/Kolkata"
)

scheduler.add_job(
    run_forecast_pipeline,
    "cron",
    day_of_week="mon-fri",
    hour=18,
    minute=0,
    max_instances=1,
    coalesce=True
)

scheduler.start()


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    # Run once when starting the application
    # if no forecast exists yet.
    if not os.path.exists(
        FORECAST_FILE
    ):

        try:

            run_forecast_pipeline()

        except Exception as e:

            print(
                "Startup pipeline run failed:",
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