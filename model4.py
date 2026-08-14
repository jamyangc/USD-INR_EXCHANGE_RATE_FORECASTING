"""
Live USD/INR daily forecasting pipeline — price-only.

Install:
    pip install yfinance lightgbm scikit-learn pandas
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from lightgbm import LGBMRegressor

HISTORY_CSV = "usdinr_history.csv"
LOOKBACK_DAYS = 400
TICKER = "INR=X"


# ---------------------------------------------------------
# 1. DOWNLOAD LATEST USD/INR DATA
# ---------------------------------------------------------

def fetch_latest_data():

    end = datetime.today().date()
    start = end - timedelta(days=LOOKBACK_DAYS)

    print(f"Downloading {TICKER} data...")
    print(f"Period: {start} to {end}")

    try:
        raw = yf.download(
            TICKER,
            start=start,
            end=end + timedelta(days=1),
            progress=False,
            auto_adjust=True
        )
    except Exception as e:
        print("Yahoo Finance download failed:")
        print(e)
        return pd.DataFrame(columns=["date", "rate"])

    # Check whether Yahoo returned anything
    if raw.empty:
        print("WARNING: Yahoo Finance returned no USD/INR data.")
        return pd.DataFrame(columns=["date", "rate"])

    # Extract Close
    close = raw["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    df = close.rename("rate").reset_index()

    # Rename date column
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})

    # Make sure date column exists
    if "date" not in df.columns:
        print("WARNING: Could not find date column.")
        return pd.DataFrame(columns=["date", "rate"])

    df["date"] = pd.to_datetime(df["date"])

    df = df[["date", "rate"]]

    df = df.dropna(subset=["rate"])

    print(f"Downloaded {len(df)} rows.")

    return df


# ---------------------------------------------------------
# 2. UPDATE LOCAL HISTORY
# ---------------------------------------------------------

def update_history(new_data):

    # Load existing history
    try:
        history = pd.read_csv(
            HISTORY_CSV,
            parse_dates=["date"]
        )

        print(f"Existing history: {len(history)} rows.")

    except FileNotFoundError:

        print("No existing history file found.")

        history = pd.DataFrame(
            columns=["date", "rate"]
        )

    # If Yahoo returned no data
    if new_data.empty:

        print("No new data downloaded.")
        print("Keeping existing historical data.")

        if history.empty:
            raise ValueError(
                "No historical data exists and Yahoo Finance "
                "did not return any new data."
            )

        return history.sort_values("date").reset_index(drop=True)

    # Combine old + new data
    combined = pd.concat(
        [history, new_data],
        ignore_index=True
    )

    # Remove duplicate dates
    combined = combined.drop_duplicates(
        subset="date",
        keep="last"
    )

    # Sort by date
    combined = combined.sort_values(
        "date"
    ).reset_index(drop=True)

    # Save
    combined.to_csv(
        HISTORY_CSV,
        index=False
    )

    print(f"Updated history: {len(combined)} rows.")

    return combined


# ---------------------------------------------------------
# 3. FEATURES
# ---------------------------------------------------------

FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "ma_5d",
    "ma_20d",
    "ma_ratio",
]


def add_features(df):

    df = df.copy()

    df["ret_1d"] = df["rate"].pct_change(1)

    df["ret_5d"] = df["rate"].pct_change(5)

    df["ret_10d"] = df["rate"].pct_change(10)

    df["ret_20d"] = df["rate"].pct_change(20)

    df["vol_5d"] = (
        df["ret_1d"]
        .rolling(5)
        .std()
    )

    df["vol_20d"] = (
        df["ret_1d"]
        .rolling(20)
        .std()
    )

    df["ma_5d"] = (
        df["rate"]
        .rolling(5)
        .mean()
    )

    df["ma_20d"] = (
        df["rate"]
        .rolling(20)
        .mean()
    )

    df["ma_ratio"] = (
        df["ma_5d"] /
        df["ma_20d"]
    )

    return df


# ---------------------------------------------------------
# 4. LIGHTGBM + RANDOM WALK
# ---------------------------------------------------------

def predict_next_day(df):

    df = add_features(df)

    train = (
        df
        .dropna(subset=FEATURES)
        .iloc[:-1]
        .copy()
    )

    # Next-day return
    train["target"] = (
        train["rate"].shift(-1) /
        train["rate"] - 1
    )

    train = train.dropna(
        subset=["target"]
    )

    if len(train) < 50:
        raise ValueError(
            "Not enough historical observations "
            "to train the model."
        )

    model = LGBMRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )

    model.fit(
        train[FEATURES],
        train["target"]
    )

    latest_row = (
        df
        .dropna(subset=FEATURES)
        .iloc[[-1]]
    )

    pred_ret = model.predict(
        latest_row[FEATURES]
    )[0]

    latest_rate = float(
        latest_row["rate"].values[0]
    )

    # LightGBM prediction
    pred_rate = (
        latest_rate *
        (1 + pred_ret)
    )

    # Random Walk prediction
    rw_pred = latest_rate

    return {

        "as_of_date":
            str(
                latest_row["date"]
                .values[0]
            )[:10],

        "latest_actual_rate":
            round(
                latest_rate,
                4
            ),

        "lgbm_forecast_next_day":
            round(
                float(pred_rate),
                4
            ),

        "random_walk_forecast":
            round(
                float(rw_pred),
                4
            ),

        "lgbm_forecast_return_pct":
            round(
                float(pred_ret) * 100,
                3
            ),
    }


# ---------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n==============================")
    print(" USD/INR FORECASTING PIPELINE")
    print("==============================\n")

    # Download latest data
    new_data = fetch_latest_data()

    # Update local historical file
    full_history = update_history(new_data)

    print(
        f"\nTotal historical observations: "
        f"{len(full_history)}"
    )

    # Make prediction
    result = predict_next_day(
        full_history
    )

    print("\nForecast result:")
    print(result)

    print("\nPipeline completed.")