"""
preprocessing.py
-----------------
Pipeline stage 3: "Data Preprocessing" (cleaning, normalisation, structuring,
feature engineering).

Reads data/prices.csv and computes, per symbol:
  - daily return
  - 10-day and 20-day simple moving averages (as % deviation from price)
  - 14-day RSI
  - 20-day rolling volatility
  - next-day return and next-day direction (the prediction targets)

Output: data/features.csv
"""

import numpy as np
import pandas as pd


def compute_features_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()

    df["return"] = df["close"].pct_change()

    for window in (10, 20):
        sma = df["close"].rolling(window).mean()
        df[f"ma_dev_{window}"] = (df["close"] - sma) / sma * 100

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    df["volatility"] = df["return"].rolling(20).std() * 100

    # Volume relative to its own rolling average (captures unusual activity)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Prediction targets: next day's return and direction
    df["target_return"] = df["return"].shift(-1)
    df["target_direction"] = np.where(df["target_return"] > 0, "Up", "Down")

    return df


def main():
    prices = pd.read_csv("data/prices.csv", parse_dates=["date"])

    # Basic cleaning: drop duplicate rows, forward-fill isolated gaps
    prices = prices.drop_duplicates(subset=["date", "symbol"])
    prices = prices.sort_values(["symbol", "date"])
    prices["close"] = prices.groupby("symbol")["close"].ffill()

    processed = []
    for symbol, grp in prices.groupby("symbol"):
        feat = compute_features_for_symbol(grp)
        processed.append(feat)

    features = pd.concat(processed, ignore_index=True)

    # Drop warm-up rows where rolling indicators aren't valid yet, and the
    # final row of each symbol (no next-day target available)
    features = features.dropna(subset=[
        "ma_dev_20", "rsi", "volatility", "volume_ratio", "target_return"
    ]).reset_index(drop=True)

    features.to_csv("data/features.csv", index=False)
    print(f"Computed features for {features['symbol'].nunique()} symbols, "
          f"{len(features)} total rows -> data/features.csv")
    print(features[["symbol", "close", "ma_dev_20", "rsi", "volatility",
                     "volume_ratio", "target_direction"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
