"""
stock_prediction.py
---------------------
Pipeline stage 4: "AI Stock Prediction" -- machine learning models forecast
future stock price movements and trends.

As specified in the project architecture, this uses:
  - An LSTM neural network (TensorFlow / Keras)   -> predicts next-day return
  - XGBoost                                       -> predicts next-day
                                                      direction (Up / Down)

Automatic fallback: if tensorflow / xgboost are not installed (as in this
offline sandbox), equivalent scikit-learn models are used instead so the
pipeline still runs end-to-end:
  - LSTM      -> MLPRegressor  (a feed-forward neural network)
  - XGBoost   -> GradientBoostingClassifier (same gradient-boosted-trees
                 family XGBoost belongs to)

On your own machine, `pip install tensorflow xgboost` and re-run -- no other
code changes needed, the script detects and uses them automatically.
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

FEATURES = ["ma_dev_10", "ma_dev_20", "rsi", "volatility", "volume_ratio"]

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception as e:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False
    _XGBOOST_IMPORT_ERROR = e

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    HAS_TENSORFLOW = True
except Exception as e:
    from sklearn.neural_network import MLPRegressor
    HAS_TENSORFLOW = False
    _TENSORFLOW_IMPORT_ERROR = e


def make_direction_model():
    if HAS_XGBOOST:
        print("[stock_prediction] Using XGBoost for direction prediction.")
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss", random_state=42,
        )
    else:
        print(f"[stock_prediction] xgboost unavailable ({_XGBOOST_IMPORT_ERROR.__class__.__name__}: "
              f"{str(_XGBOOST_IMPORT_ERROR)[:120]}...) -> using "
              "GradientBoostingClassifier (sklearn) as a stand-in. On Mac, this is usually "
              "fixed with: brew install libomp")
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        )


def make_return_model(n_features):
    if HAS_TENSORFLOW:
        print("[stock_prediction] Using an LSTM (TensorFlow/Keras) for return prediction.")
        model = Sequential([
            LSTM(32, input_shape=(1, n_features)),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model, "keras"
    else:
        print(f"[stock_prediction] tensorflow unavailable ({_TENSORFLOW_IMPORT_ERROR.__class__.__name__}: "
              f"{str(_TENSORFLOW_IMPORT_ERROR)[:120]}...) -> using MLPRegressor "
              "(sklearn) as a stand-in for the LSTM.")
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42,
            learning_rate_init=0.001, alpha=0.01, early_stopping=True,
            n_iter_no_change=20,
        )
        return model, "sklearn"


def main():
    df = pd.read_csv("data/features.csv")

    X = df[FEATURES]
    y_dir = df["target_direction"]
    y_ret = df["target_return"]

    X_train, X_test, ydir_train, ydir_test, yret_train, yret_test = train_test_split(
        X, y_dir, y_ret, test_size=0.20, random_state=42, stratify=y_dir
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- Direction model (XGBoost / GradientBoosting) ----
    dir_model = make_direction_model()
    if HAS_XGBOOST:
        ydir_train_enc = (ydir_train == "Up").astype(int)
        ydir_test_enc = (ydir_test == "Up").astype(int)
        dir_model.fit(X_train_s, ydir_train_enc)
        dir_pred = dir_model.predict(X_test_s)
        acc = accuracy_score(ydir_test_enc, dir_pred)
    else:
        dir_model.fit(X_train_s, ydir_train)
        dir_pred = dir_model.predict(X_test_s)
        acc = accuracy_score(ydir_test, dir_pred)

    print(f"\nDirection model test accuracy: {acc * 100:.2f}%")
    print(classification_report(
        ydir_test if not HAS_XGBOOST else ydir_test_enc,
        dir_pred
    ))

    # ---- Return model (LSTM / MLP) ----
    ret_model, kind = make_return_model(len(FEATURES))
    if kind == "keras":
        X_train_seq = X_train_s.reshape(-1, 1, len(FEATURES))
        X_test_seq = X_test_s.reshape(-1, 1, len(FEATURES))
        ret_model.fit(X_train_seq, yret_train, epochs=30, verbose=0)
        ret_pred = ret_model.predict(X_test_seq, verbose=0).flatten()
    else:
        ret_model.fit(X_train_s, yret_train)
        ret_pred = ret_model.predict(X_test_s)

    rmse = np.sqrt(mean_squared_error(yret_test, ret_pred))
    print(f"Return model RMSE: {rmse:.5f}  (typical daily return magnitude for reference: "
          f"{yret_test.abs().mean():.5f})")

    # ---- Save everything needed by later stages ----
    with open("models/direction_model.pkl", "wb") as f:
        pickle.dump({"model": dir_model, "uses_xgboost": HAS_XGBOOST}, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    if kind == "sklearn":
        with open("models/return_model.pkl", "wb") as f:
            pickle.dump(ret_model, f)
    else:
        ret_model.save("models/return_model.keras")

    # Predict on the FULL dataset (latest row per symbol) for downstream use
    latest = df.sort_values("date").groupby("symbol").tail(1).copy()
    X_latest = scaler.transform(latest[FEATURES])
    if HAS_XGBOOST:
        latest["predicted_direction"] = np.where(dir_model.predict(X_latest) == 1, "Up", "Down")
    else:
        latest["predicted_direction"] = dir_model.predict(X_latest)

    if kind == "keras":
        latest["predicted_return"] = ret_model.predict(
            X_latest.reshape(-1, 1, len(FEATURES)), verbose=0
        ).flatten()
    else:
        latest["predicted_return"] = ret_model.predict(X_latest)

    latest[["symbol", "date", "close", "predicted_direction", "predicted_return"]].to_csv(
        "outputs/latest_predictions.csv", index=False
    )
    print("\nLatest per-symbol predictions:")
    print(latest[["symbol", "close", "predicted_direction", "predicted_return"]].to_string(index=False))
    print("\nSaved to outputs/latest_predictions.csv")


if __name__ == "__main__":
    main()
