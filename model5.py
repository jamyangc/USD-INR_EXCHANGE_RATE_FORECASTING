# ============================================================
# USD/INR SHORT-TERM FORECASTING
#
# PERIOD: 2020 TO PRESENT
#
# MODELS:
#   1. Random Walk    -> Benchmark
#   2. PPP             -> Classical economic model
#   3. IRP              -> Classical economic model
#   4. Ridge            -> Machine Learning model (linear)
#   5. Decision Tree    -> Machine Learning model (small tree)
#   6. MLP              -> Machine Learning model (tiny neural net)
#   7. LightGBM         -> Machine Learning model (gradient boosting)
#
# DATA:
#   USD/INR  -> Yahoo Finance
#   CPI      -> World Bank
#   Interest -> World Bank
#
# NO API KEY REQUIRED
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import json
import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import lightgbm as lgb

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TICKER = "USDINR=X"

START_DATE = "2020-01-01"

PPP_BASE_YEAR = 2020

MIN_TRAIN_DAYS = 500

RETRAIN_EVERY = 21

HORIZON = 1

TRANSACTION_COST_BPS = 5

DAYS_IN_YEAR = 365

# --- NEW: safety clip for MLP log-return predictions ---------
# USD/INR essentially never moves >2% in a single day.
# The tiny MLP occasionally mis-converges and predicts a
# log-return several times larger than anything realistic,
# which blows up into an absurd price forecast after exp().
# Clipping keeps the MLP output inside a plausible daily range.
MLP_LOGRET_CLIP = 0.02

# ============================================================
# DASHBOARD JSON EXPORT PATH
# ============================================================
#
# This MUST point at the SAME "latest_forecast.json" that your
# Flask app.py reads from in its FORECAST_FILE setting.
#
# By default this assumes model5.py lives in the SAME FOLDER
# as app.py (e.g. both inside capstone_BEA_409/). If app.py is
# in a different folder, change this path to point there
# directly, e.g.:
#
#   DASHBOARD_JSON_FILE = r"C:\path\to\flask_app_folder\latest_forecast.json"
#
DASHBOARD_JSON_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "latest_forecast.json"
)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("          USD/INR SHORT-TERM FORECASTING")
print("   Random Walk vs PPP vs IRP vs Ridge vs Tree vs MLP vs LightGBM")
print("=" * 70)


# ============================================================
# 1. DOWNLOAD USD/INR DATA
# ============================================================

print("\nDownloading latest USD/INR data from Yahoo Finance...")


fx = yf.download(
    TICKER,
    period="10y",
    interval="1d",
    auto_adjust=False,
    progress=False
)


if fx.empty:
    raise RuntimeError(
        "Could not download USD/INR data from Yahoo Finance. "
        "Check your internet connection."
    )


# Handle MultiIndex from newer yfinance

if isinstance(fx.columns, pd.MultiIndex):
    fx.columns = fx.columns.get_level_values(0)


fx = fx.reset_index()


fx = fx.rename(
    columns={
        "Date": "date",
        "Close": "usdinr"
    }
)


df = fx[
    ["date", "usdinr"]
].copy()


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


df["usdinr"] = pd.to_numeric(
    df["usdinr"],
    errors="coerce"
)


df = df.dropna()


# ============================================================
# USE ONLY 2020 TO PRESENT
# ============================================================

df = df[
    df["date"] >= pd.Timestamp(START_DATE)
].copy()


df = df.sort_values("date")

df = df.drop_duplicates(
    subset="date"
)

df = df.reset_index(
    drop=True
)


if df.empty:
    raise RuntimeError(
        "No USD/INR observations are available from 2020 onward."
    )


print("\nUSD/INR data loaded successfully.")

print(
    "Observations:",
    len(df)
)

print(
    "Date range:",
    df["date"].iloc[0].strftime("%Y-%m-%d"),
    "to",
    df["date"].iloc[-1].strftime("%Y-%m-%d")
)

print(
    "Latest date:",
    df["date"].iloc[-1].strftime("%Y-%m-%d")
)

print(
    "Latest USD/INR:",
    f"{df['usdinr'].iloc[-1]:.4f}"
)


# ============================================================
# 2. LIGHTGBM FEATURE ENGINEERING
# ============================================================

print("\nCreating ML features...")


# ------------------------------------------------------------
# Returns
# ------------------------------------------------------------

df["ret_1"] = (
    df["usdinr"].pct_change(1)
)

df["ret_5"] = (
    df["usdinr"].pct_change(5)
)

df["ret_10"] = (
    df["usdinr"].pct_change(10)
)

df["ret_20"] = (
    df["usdinr"].pct_change(20)
)


# ------------------------------------------------------------
# Volatility
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Moving averages
# ------------------------------------------------------------

df["ma_5"] = (
    df["usdinr"]
    .rolling(5)
    .mean()
)

df["ma_20"] = (
    df["usdinr"]
    .rolling(20)
    .mean()
)

df["ma_50"] = (
    df["usdinr"]
    .rolling(50)
    .mean()
)


# ------------------------------------------------------------
# Price / MA ratios
# ------------------------------------------------------------

df["price_ma5_ratio"] = (
    df["usdinr"] /
    df["ma_5"]
)

df["price_ma20_ratio"] = (
    df["usdinr"] /
    df["ma_20"]
)

df["ma_ratio"] = (
    df["ma_5"] /
    df["ma_20"]
)


# ------------------------------------------------------------
# RSI
# ------------------------------------------------------------

delta = df["usdinr"].diff()


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


rs = (
    gain /
    loss.replace(0, np.nan)
)


df["rsi_14"] = (
    100 -
    100 / (1 + rs)
)


# ============================================================
# FEATURES
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
# TARGET
# ============================================================

df["target_logret"] = np.log(
    df["usdinr"].shift(-HORIZON)
    /
    df["usdinr"]
)


# ============================================================
# 3. RANDOM WALK
# ============================================================

print("\nCreating Random Walk benchmark...")


# Tomorrow's price = today's price

df["fcst_rw"] = df["usdinr"]


# ============================================================
# 4. WORLD BANK FUNCTION (WITH RETRY / BACKOFF)
# ============================================================

def get_world_bank_indicator(
    country,
    indicator,
    max_retries=5,
    timeout=60
):

    url = (
        "https://api.worldbank.org/v2/country/"
        + country
        + "/indicator/"
        + indicator
        + "?format=json&per_page=100"
    )

    session = requests.Session()

    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    response = session.get(
        url,
        timeout=timeout
    )

    response.raise_for_status()

    data = response.json()

    if len(data) < 2:
        raise RuntimeError(
            f"No World Bank data found for "
            f"{country} / {indicator}"
        )

    records = []

    for item in data[1]:

        if item["value"] is not None:

            records.append(
                {
                    "year": int(item["date"]),
                    "value": float(item["value"])
                }
            )

    result = pd.DataFrame(
        records
    )

    result = result.sort_values(
        "year"
    )

    return result


# ============================================================
# 5. PPP DATA
# ============================================================

print("\nDownloading CPI data for PPP...")


# World Bank:
# FP.CPI.TOTL = Consumer Price Index

india_cpi = get_world_bank_indicator(
    "IND",
    "FP.CPI.TOTL"
)


usa_cpi = get_world_bank_indicator(
    "USA",
    "FP.CPI.TOTL"
)


india_cpi = india_cpi.rename(
    columns={
        "value": "india_cpi"
    }
)


usa_cpi = usa_cpi.rename(
    columns={
        "value": "usa_cpi"
    }
)


cpi = pd.merge(
    india_cpi,
    usa_cpi,
    on="year",
    how="inner"
)


# ============================================================
# 6. PPP BASE
# ============================================================

print("\nCalculating PPP...")


# Add year

df["year"] = (
    df["date"].dt.year
)


# Annual average USD/INR

annual_fx = (
    df.groupby("year")["usdinr"]
    .mean()
    .reset_index()
)


annual_fx = annual_fx.rename(
    columns={
        "usdinr":
        "average_usdinr"
    }
)


ppp_data = pd.merge(
    annual_fx,
    cpi,
    on="year",
    how="inner"
)


if PPP_BASE_YEAR not in (
    ppp_data["year"].values
):

    raise RuntimeError(
        "PPP base year 2020 is not available "
        "in the downloaded data."
    )


base_row = ppp_data[
    ppp_data["year"] ==
    PPP_BASE_YEAR
].iloc[0]


base_fx = (
    base_row["average_usdinr"]
)


base_india_cpi = (
    base_row["india_cpi"]
)


base_usa_cpi = (
    base_row["usa_cpi"]
)


print(
    "PPP base year:",
    PPP_BASE_YEAR
)


print(
    "PPP base USD/INR:",
    f"{base_fx:.4f}"
)


# ============================================================
# 7. PPP FORMULA
# ============================================================

ppp_data["ppp_rate"] = (

    base_fx

    *

    (
        ppp_data["india_cpi"]
        /
        base_india_cpi
    )

    /

    (
        ppp_data["usa_cpi"]
        /
        base_usa_cpi
    )
)


# ============================================================
# 8. IMPORTANT:
# PREVENT FUTURE INFORMATION LEAKAGE
# ============================================================
#
# A year's CPI is not known on January 1 of that year.
#
# Therefore:
#
# 2021 FX uses 2020 CPI
# 2022 FX uses 2021 CPI
# 2023 FX uses 2022 CPI
# etc.
#
# This is more appropriate for historical forecasting.
# ============================================================

ppp_data["available_year"] = (
    ppp_data["year"] + 1
)


# Map PPP value from previous year's information

ppp_for_merge = ppp_data[
    [
        "available_year",
        "ppp_rate"
    ]
].copy()


ppp_for_merge = ppp_for_merge.rename(
    columns={
        "available_year":
        "year"
    }
)


df = pd.merge(
    df,
    ppp_for_merge,
    on="year",
    how="left"
)


df["fcst_ppp"] = (
    df["ppp_rate"]
)


print(
    "PPP calculation completed."
)


# ============================================================
# 9. IRP DATA
# ============================================================

print("\nDownloading interest-rate data for IRP...")


# World Bank:
# FR.INR.LEND = Lending interest rate

india_rate = get_world_bank_indicator(
    "IND",
    "FR.INR.LEND"
)


usa_rate = get_world_bank_indicator(
    "USA",
    "FR.INR.LEND"
)


india_rate = india_rate.rename(
    columns={
        "value":
        "india_rate"
    }
)


usa_rate = usa_rate.rename(
    columns={
        "value":
        "usa_rate"
    }
)


rates = pd.merge(
    india_rate,
    usa_rate,
    on="year",
    how="inner"
)


# ============================================================
# 10. IRP
# ============================================================

print("Calculating IRP...")


rates["india_rate_decimal"] = (
    rates["india_rate"] / 100
)


rates["usa_rate_decimal"] = (
    rates["usa_rate"] / 100
)


# ============================================================
# 11. PREVIOUS-YEAR INTEREST RATES
# ============================================================
#
# Example:
#
# 2021 FX uses 2020 interest rates
# 2022 FX uses 2021 interest rates
# etc.
# ============================================================

rates["available_year"] = (
    rates["year"] + 1
)


rates_for_merge = rates[
    [
        "available_year",
        "india_rate_decimal",
        "usa_rate_decimal"
    ]
].copy()


rates_for_merge = rates_for_merge.rename(
    columns={
        "available_year":
        "year"
    }
)


df = pd.merge(
    df,
    rates_for_merge,
    on="year",
    how="left"
)


# ============================================================
# 12. IRP FORMULA
# ============================================================

df["fcst_irp"] = (

    df["usdinr"]

    *

    (

        (
            1 +
            df["india_rate_decimal"]
        )

        /

        (
            1 +
            df["usa_rate_decimal"]
        )

    )

    **

    (
        1 /
        DAYS_IN_YEAR
    )
)


print(
    "IRP calculation completed."
)


# ============================================================
# 13. ML MODELS
# ============================================================
#
# LightGBM   -> gradient boosted trees (primary ML model)
# Ridge      -> linear model, trained on standardized features
# Decision Tree -> small single tree, trained on raw features
# MLP        -> tiny one-hidden-layer neural net, standardized
#               features
# ============================================================

def create_model():

    model = lgb.LGBMRegressor(

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


    return model


def create_ridge_model():

    model = Ridge(
        alpha=1.0,
        random_state=RANDOM_STATE
    )

    return model


def create_tree_model():

    model = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    )

    return model


# --- CHANGED: MLP was diverging (predicting logret ~0.04+,
# i.e. a >4% one-day move, which is not physically plausible
# for USD/INR). Switched solver to "lbfgs", which converges
# far more reliably than "adam" on small tabular datasets like
# this one, and raised regularization slightly. ---
def create_mlp_model():

    model = MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="lbfgs",
        alpha=1e-2,
        max_iter=5000,
        random_state=RANDOM_STATE
    )

    return model


# --- NEW: hard safety clip applied to any predicted log-return
# before it gets exponentiated into a price. Used as a guard
# rail for the MLP specifically, since it's the model most prone
# to occasional bad fits on a small/noisy target. ---
def clip_logret(value, limit=MLP_LOGRET_CLIP):

    if value > limit:
        return limit

    if value < -limit:
        return -limit

    return value


# ============================================================
# 14. WALK-FORWARD LIGHTGBM
# ============================================================

def walk_forward_lightgbm(
    data,
    min_train=MIN_TRAIN_DAYS,
    retrain_every=RETRAIN_EVERY
):

    n = len(data)


    predictions = pd.Series(
        index=data.index,
        dtype=float
    )


    model = None


    for i in range(
        min_train,
        n - HORIZON
    ):


        if (
            model is None

            or

            (
                (i - min_train)
                % retrain_every
                == 0
            )
        ):


            train = data.iloc[
                :i
            ].dropna(
                subset=
                FEATURE_COLS +
                ["target_logret"]
            )


            if len(train) < 100:
                continue


            X_train = train[
                FEATURE_COLS
            ]


            y_train = train[
                "target_logret"
            ]


            model = create_model()


            model.fit(
                X_train,
                y_train
            )


        x_today = data.iloc[
            [i]
        ][FEATURE_COLS]


        if x_today.isnull().values.any():
            continue


        prediction = model.predict(
            x_today
        )[0]


        predictions.iloc[i] = (
            prediction
        )


    return predictions


# ============================================================
# 14b. WALK-FORWARD SKLEARN MODELS (RIDGE / TREE / MLP)
# ============================================================
#
# Same walk-forward logic as LightGBM above, generalized for
# any scikit-learn regressor. When use_scaler=True, a fresh
# StandardScaler is fit on each training window (Ridge, MLP);
# the Decision Tree does not need scaling.
#
# --- CHANGED: added an optional clip_limit parameter so a
# model's predicted log-return can be bounded to a plausible
# daily range before being stored. ---
# ============================================================

def walk_forward_sklearn_model(
    data,
    model_factory,
    use_scaler=False,
    min_train=MIN_TRAIN_DAYS,
    retrain_every=RETRAIN_EVERY,
    clip_limit=None
):

    n = len(data)


    predictions = pd.Series(
        index=data.index,
        dtype=float
    )


    model = None

    scaler = None


    for i in range(
        min_train,
        n - HORIZON
    ):


        if (
            model is None

            or

            (
                (i - min_train)
                % retrain_every
                == 0
            )
        ):


            train = data.iloc[
                :i
            ].dropna(
                subset=
                FEATURE_COLS +
                ["target_logret"]
            )


            if len(train) < 100:
                continue


            X_train = train[
                FEATURE_COLS
            ]


            y_train = train[
                "target_logret"
            ]


            if use_scaler:

                scaler = StandardScaler()

                X_train = scaler.fit_transform(
                    X_train
                )


            model = model_factory()


            model.fit(
                X_train,
                y_train
            )


        x_today = data.iloc[
            [i]
        ][FEATURE_COLS]


        if x_today.isnull().values.any():
            continue


        if use_scaler:

            x_today = scaler.transform(
                x_today
            )


        prediction = model.predict(
            x_today
        )[0]


        if clip_limit is not None:

            prediction = clip_logret(
                prediction,
                clip_limit
            )


        predictions.iloc[i] = (
            prediction
        )


    return predictions


# ============================================================
# 15. RUN LIGHTGBM
# ============================================================

print("\n" + "=" * 70)
print("             RUNNING WALK-FORWARD LIGHTGBM")
print("=" * 70)


print("\nPlease wait...")


lgbm_predictions = (
    walk_forward_lightgbm(df)
)


df["lgbm_logret"] = (
    lgbm_predictions
)


df["fcst_lgbm"] = (

    df["usdinr"]

    *

    np.exp(
        df["lgbm_logret"]
    )
)


print(
    "\nLightGBM testing completed."
)


# ============================================================
# 15b. RUN RIDGE / DECISION TREE / MLP
# ============================================================

print("\n" + "=" * 70)
print("        RUNNING WALK-FORWARD RIDGE / TREE / MLP")
print("=" * 70)


print("\nPlease wait...")


ridge_predictions = (
    walk_forward_sklearn_model(
        df,
        create_ridge_model,
        use_scaler=True
    )
)


df["ridge_logret"] = (
    ridge_predictions
)


df["fcst_ridge"] = (

    df["usdinr"]

    *

    np.exp(
        df["ridge_logret"]
    )
)


tree_predictions = (
    walk_forward_sklearn_model(
        df,
        create_tree_model,
        use_scaler=False
    )
)


df["tree_logret"] = (
    tree_predictions
)


df["fcst_tree"] = (

    df["usdinr"]

    *

    np.exp(
        df["tree_logret"]
    )
)


mlp_predictions = (
    walk_forward_sklearn_model(
        df,
        create_mlp_model,
        use_scaler=True,
        clip_limit=MLP_LOGRET_CLIP
    )
)


df["mlp_logret"] = (
    mlp_predictions
)


df["fcst_mlp"] = (

    df["usdinr"]

    *

    np.exp(
        df["mlp_logret"]
    )
)


print(
    "\nRidge / Decision Tree / MLP testing completed."
)


# ============================================================
# 16. FAIR EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    data,
    forecast_column,
    label,
    start_date="2021-01-01"
):

    temp = data[
        [
            "date",
            "usdinr",
            forecast_column
        ]
    ].copy()


    # Use same evaluation period

    temp = temp[
        temp["date"] >=
        pd.Timestamp(start_date)
    ]


    # Actual next trading day

    temp["actual_next"] = (
        temp["usdinr"]
        .shift(-HORIZON)
    )


    temp = temp.dropna()


    if temp.empty:

        return {
            "Model": label,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Directional Accuracy": np.nan,
            "Total P&L": np.nan
        }


    y_true = (
        temp["actual_next"]
        .values
    )


    y_pred = (
        temp[forecast_column]
        .values
    )


    previous_price = (
        temp["usdinr"]
        .values
    )


    # ========================================================
    # RMSE
    # ========================================================

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


    # ========================================================
    # MAE
    # ========================================================

    mae = mean_absolute_error(
        y_true,
        y_pred
    )


    # ========================================================
    # DIRECTIONAL ACCURACY
    # ========================================================

    actual_direction = np.sign(
        y_true -
        previous_price
    )


    predicted_direction = np.sign(
        y_pred -
        previous_price
    )


    # Ignore exactly-zero cases

    valid_direction = (
        actual_direction != 0
    )


    if valid_direction.sum() > 0:

        directional_accuracy = np.mean(
            predicted_direction[
                valid_direction
            ]
            ==
            actual_direction[
                valid_direction
            ]
        )

    else:

        directional_accuracy = np.nan


    # ========================================================
    # P&L
    # ========================================================

    position = (
        predicted_direction
    )


    actual_change = (
        y_true -
        previous_price
    )


    raw_pnl = (
        position *
        actual_change
    )


    transaction_cost = (

        TRANSACTION_COST_BPS
        /
        10000
        *
        previous_price
    )


    net_pnl = (
        raw_pnl -
        transaction_cost
    )


    total_pnl = (
        net_pnl.sum()
    )


    return {

        "Model": label,

        "RMSE": round(
            rmse,
            5
        ),

        "MAE": round(
            mae,
            5
        ),

        "Directional Accuracy": round(
            directional_accuracy,
            4
        ),

        "Total P&L": round(
            total_pnl,
            3
        )
    }


# ============================================================
# 17. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("                 MODEL COMPARISON")
print("=" * 70)


print(
    "\nEvaluation period: 2021-01-01 to latest available date"
)


results = [

    evaluate_model(
        df,
        "fcst_rw",
        "Random Walk"
    ),

    evaluate_model(
        df,
        "fcst_ppp",
        "PPP"
    ),

    evaluate_model(
        df,
        "fcst_irp",
        "IRP"
    ),

    evaluate_model(
        df,
        "fcst_ridge",
        "Ridge"
    ),

    evaluate_model(
        df,
        "fcst_tree",
        "Decision Tree"
    ),

    evaluate_model(
        df,
        "fcst_mlp",
        "MLP"
    ),

    evaluate_model(
        df,
        "fcst_lgbm",
        "LightGBM"
    )
]


results_df = pd.DataFrame(
    results
)


print(
    "\n"
)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 18. SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)


print(
    "\nSaved: model_comparison.csv"
)


# ============================================================
# 19. FINAL MODELS (TRAINED ON ALL AVAILABLE DATA)
# ============================================================

print("\n" + "=" * 70)
print("             TRAINING FINAL MODELS")
print("=" * 70)


train_df = df.dropna(
    subset=
    FEATURE_COLS +
    ["target_logret"]
)


if len(train_df) < 100:

    raise RuntimeError(
        "Not enough data to train the models."
    )


X_train = train_df[
    FEATURE_COLS
]


y_train = train_df[
    "target_logret"
]


final_model = create_model()


final_model.fit(
    X_train,
    y_train
)


print(
    "\nFinal LightGBM trained using available data."
)


# ------------------------------------------------------------
# Ridge / Decision Tree / MLP
# (Ridge and MLP use standardized features; the tree uses the
# raw features, same as LightGBM.)
# ------------------------------------------------------------

final_scaler = StandardScaler()

X_train_scaled = final_scaler.fit_transform(
    X_train
)


final_ridge_model = create_ridge_model()

final_ridge_model.fit(
    X_train_scaled,
    y_train
)


final_tree_model = create_tree_model()

final_tree_model.fit(
    X_train,
    y_train
)


final_mlp_model = create_mlp_model()

final_mlp_model.fit(
    X_train_scaled,
    y_train
)


print(
    "Final Ridge / Decision Tree / MLP trained using available data."
)


# ============================================================
# 20. LATEST DATA
# ============================================================

last_known_date = (
    df["date"].iloc[-1]
)


last_known_price = float(
    df["usdinr"].iloc[-1]
)


latest_features = (
    df.iloc[[-1]][FEATURE_COLS]
)


if latest_features.isnull().values.any():

    raise RuntimeError(
        "Latest observation does not have all "
        "required ML features."
    )


latest_features_scaled = (
    final_scaler.transform(
        latest_features
    )
)


# ============================================================
# 21. LIGHTGBM NEXT-DAY FORECAST
# ============================================================

predicted_logret = (
    final_model.predict(
        latest_features
    )[0]
)


forecast_lgbm = (

    last_known_price

    *

    np.exp(
        predicted_logret
    )
)


# ============================================================
# 21b. RIDGE / DECISION TREE / MLP NEXT-DAY FORECAST
# ============================================================

predicted_logret_ridge = (
    final_ridge_model.predict(
        latest_features_scaled
    )[0]
)


forecast_ridge = (

    last_known_price

    *

    np.exp(
        predicted_logret_ridge
    )
)


predicted_logret_tree = (
    final_tree_model.predict(
        latest_features
    )[0]
)


forecast_tree = (

    last_known_price

    *

    np.exp(
        predicted_logret_tree
    )
)


# --- CHANGED: clip the live MLP forecast the same way the
# walk-forward evaluation does, so the headline number can't
# blow up either. ---
predicted_logret_mlp = (
    final_mlp_model.predict(
        latest_features_scaled
    )[0]
)


predicted_logret_mlp = clip_logret(
    predicted_logret_mlp,
    MLP_LOGRET_CLIP
)


forecast_mlp = (

    last_known_price

    *

    np.exp(
        predicted_logret_mlp
    )
)


# ============================================================
# 22. RANDOM WALK NEXT-DAY FORECAST
# ============================================================

forecast_rw = (
    last_known_price
)


# ============================================================
# 23. PPP LIVE FORECAST
# ============================================================
#
# Use the most recent available World Bank CPI information.
#
# The latest available annual PPP estimate is carried forward
# to the next trading day because CPI is not daily data.
# ============================================================

latest_ppp_row = (
    ppp_data
    .dropna(
        subset=["ppp_rate"]
    )
    .sort_values("year")
    .iloc[-1]
)


latest_ppp = float(
    latest_ppp_row["ppp_rate"]
)


forecast_ppp = (
    latest_ppp
)


latest_ppp_year = int(
    latest_ppp_row["year"]
)


# ============================================================
# 24. IRP LIVE FORECAST
# ============================================================
#
# Use the latest available World Bank interest rates.
#
# Apply one-day IRP adjustment to today's USD/INR.
# ============================================================

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
    latest_rate_row[
        "india_rate_decimal"
    ]
)


latest_usa_rate = float(
    latest_rate_row[
        "usa_rate_decimal"
    ]
)


latest_rate_year = int(
    latest_rate_row["year"]
)


forecast_irp = (

    last_known_price

    *

    (

        (
            1 +
            latest_india_rate
        )

        /

        (
            1 +
            latest_usa_rate
        )

    )

    **

    (
        1 /
        DAYS_IN_YEAR
    )
)


# ============================================================
# 25. NEXT TRADING DAY
# ============================================================

forecast_date = (

    last_known_date

    +

    pd.tseries.offsets.BDay(1)
)


# ============================================================
# 26. LIGHTGBM DIRECTION
# ============================================================

if forecast_lgbm > last_known_price:

    direction = "UP"

elif forecast_lgbm < last_known_price:

    direction = "DOWN"

else:

    direction = "FLAT"


lightgbm_change = (
    forecast_lgbm -
    last_known_price
)


lightgbm_change_pct = (

    lightgbm_change /
    last_known_price
) * 100


# ============================================================
# 27. FINAL FORECAST DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("             NEXT TRADING-DAY FORECAST")
print("=" * 70)


print(
    "\nLatest available date:"
)

print(
    f"    {last_known_date.strftime('%Y-%m-%d')}"
)


print(
    "\nLatest USD/INR:"
)

print(
    f"    {last_known_price:.4f}"
)


print(
    "\nForecast target date:"
)

print(
    f"    {forecast_date.strftime('%Y-%m-%d')}"
)


print(
    "\n----------------------------------------"
)


print(
    "\nRandom Walk:"
)

print(
    f"    {forecast_rw:.4f}"
)


print(
    "\nPPP:"
)

print(
    f"    {forecast_ppp:.4f}"
)

print(
    f"    Based on latest annual CPI data: {latest_ppp_year}"
)


print(
    "\nIRP:"
)

print(
    f"    {forecast_irp:.4f}"
)

print(
    f"    Based on latest annual interest data: {latest_rate_year}"
)


print(
    "\nRidge:"
)

print(
    f"    {forecast_ridge:.4f}"
)


print(
    "\nDecision Tree:"
)

print(
    f"    {forecast_tree:.4f}"
)


print(
    "\nMLP:"
)

print(
    f"    {forecast_mlp:.4f}"
)


print(
    "\nLightGBM:"
)

print(
    f"    {forecast_lgbm:.4f}"
)


print(
    "\nLightGBM expected change:"
)

print(
    f"    {lightgbm_change_pct:.4f}%"
)


print(
    "\nLightGBM direction:"
)

print(
    f"    {direction}"
)


print(
    "\n" + "=" * 70
)


# ============================================================
# 28. SAVE NEXT-DAY FORECAST (APPENDING HISTORY LOG)
# ============================================================
#
# This keeps a running record instead of overwriting:
#
#   run_timestamp          -> when this script was executed
#   latest_available_date  -> today's actual known date/price
#   latest_usdinr           (this is the "today's assumed rate")
#   forecast_date           -> tomorrow's target date
#   random_walk / ppp / irp / ridge / decision_tree / mlp /
#   lightgbm -> tomorrow's predictions
#
# If you already saved a forecast for the same forecast_date,
# it is replaced with this run's latest version rather than
# duplicated. Different forecast_dates simply accumulate below
# each other, building a history over time.
# ============================================================

FORECAST_LOG_FILE = "next_day_forecast.csv"

new_row = pd.DataFrame(
    [
        {

            "run_timestamp":
                pd.Timestamp.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "latest_available_date":
                last_known_date.strftime(
                    "%Y-%m-%d"
                ),

            "latest_usdinr":
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

            "lightgbm":
                round(
                    forecast_lgbm,
                    4
                ),

            "lightgbm_change_percent":
                round(
                    lightgbm_change_pct,
                    4
                ),

            "lightgbm_direction":
                direction
        }
    ]
)


try:

    existing_log = pd.read_csv(
        FORECAST_LOG_FILE
    )

    # Drop any earlier row with the same forecast_date so this
    # run's prediction replaces it instead of duplicating

    existing_log = existing_log[
        existing_log["forecast_date"]
        != forecast_date.strftime("%Y-%m-%d")
    ]

    forecast_log = pd.concat(
        [existing_log, new_row],
        ignore_index=True
    )

except FileNotFoundError:

    forecast_log = new_row


forecast_log = forecast_log.sort_values(
    "forecast_date"
).reset_index(drop=True)


forecast_log.to_csv(
    FORECAST_LOG_FILE,
    index=False
)


print(
    f"\nSaved / updated: {FORECAST_LOG_FILE}"
)

print(
    f"Total forecast records logged: {len(forecast_log)}"
)


# ============================================================
# 28b. EXPORT TO DASHBOARD JSON (latest_forecast.json)
# ============================================================
#
# This writes the SAME file that app.py's /api/predict route
# reads from (FORECAST_FILE), using identical field names.
# Once this runs, refreshing the dashboard (or just hitting
# /api/predict) will immediately show this run's numbers —
# no need to call /api/refresh or restart Flask.
#
# IMPORTANT: DASHBOARD_JSON_FILE (set near the top of this
# file) must point at the exact same "latest_forecast.json"
# that app.py uses. If model5.py and app.py are in different
# folders, edit DASHBOARD_JSON_FILE above to the correct path.
# ============================================================

dashboard_result = {

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
        round(lightgbm_change_pct, 4),

    "lightgbm_direction":
        direction,

    "generated_at":
        pd.Timestamp.now().isoformat()

}

with open(DASHBOARD_JSON_FILE, "w") as f:
    json.dump(dashboard_result, f, indent=2)

print(
    f"\nSaved / updated dashboard file: {DASHBOARD_JSON_FILE}"
)


# ============================================================
# 29. SAVE COMPLETE DATASET
# ============================================================

df.to_csv(
    "usd_inr_results.csv",
    index=False
)


print(
    "Saved: usd_inr_results.csv"
)


# ============================================================
# 30. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("                       COMPLETE")
print("=" * 70)


print(
    "\nModels:"
)

print(
    "  Random Walk   = Benchmark"
)

print(
    "  PPP           = Classical economic model"
)

print(
    "  IRP           = Classical economic model"
)

print(
    "  Ridge         = Machine Learning model (linear)"
)

print(
    "  Decision Tree = Machine Learning model (small tree)"
)

print(
    "  MLP           = Machine Learning model (tiny neural net)"
)

print(
    "  LightGBM      = Machine Learning model (gradient boosting)"
)


print(
    "\nData period:"
)

print(
    "  USD/INR = 2020 to latest available date"
)


print(
    "\nRun the same program again on another day."
)

print(
    "No date needs to be changed manually."
)


print("=" * 70)