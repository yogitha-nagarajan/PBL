"""
dashboard_app.py
------------------
Pipeline stage 9: "Investor Dashboard" -- visualising data, predictions,
and recommendations on a user-friendly interface.

Run with:
    streamlit run dashboard_app.py

Requires the earlier stages to have been run at least once (so the CSV/PNG
outputs exist):
    python market_data.py
    python preprocessing.py
    python stock_prediction.py
    python sentiment_analysis.py
    python risk_assessment.py
    python portfolio_optimization.py
    python recommendation.py
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from recommendation import build_recommendation

st.set_page_config(page_title="AI Investment Assistant", layout="wide")

st.title("AI-Powered Investment Decision Assistant")
st.caption(
    "Stock prediction (LSTM/XGBoost) + sentiment analysis (VADER/FinBERT) + "
    "portfolio optimisation (Modern Portfolio Theory / Sharpe ratio), "
    "combined into a personalised recommendation."
)

# ---------------- load all pipeline outputs ----------------
prices = pd.read_csv("data/prices.csv", parse_dates=["date"])
predictions = pd.read_csv("outputs/latest_predictions.csv")
sentiment = pd.read_csv("outputs/sentiment_by_symbol.csv")
risk = pd.read_csv("outputs/risk_metrics.csv")
portfolios = pd.read_csv("outputs/optimized_portfolios.csv")

# ---------------- sidebar: investor profile ----------------
st.sidebar.header("Your investor profile")
risk_tolerance = st.sidebar.slider(
    "Risk tolerance (0 = conservative, 1 = aggressive)", 0.0, 1.0, 0.5, 0.05
)
if risk_tolerance < 0.33:
    profile_name = "Conservative"
elif risk_tolerance < 0.66:
    profile_name = "Balanced"
else:
    profile_name = "Aggressive"
st.sidebar.write(f"Profile: **{profile_name}**")

symbol_focus = st.sidebar.selectbox("Focus stock (for the price chart)", sorted(prices["symbol"].unique()))

# ---------------- top row: market snapshot ----------------
st.subheader("Market snapshot")
merged = predictions.merge(sentiment, on="symbol").merge(risk, on="symbol")
st.dataframe(
    merged[["symbol", "close", "predicted_direction", "predicted_return",
            "sentiment_score", "mood", "risk_tier", "sharpe_ratio"]]
    .rename(columns={
        "close": "Price", "predicted_direction": "AI Prediction",
        "predicted_return": "Predicted Return", "sentiment_score": "Sentiment",
        "mood": "Mood", "risk_tier": "Risk Tier", "sharpe_ratio": "Sharpe",
    }),
    use_container_width=True, hide_index=True,
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{symbol_focus} — price history")
    hist = prices[prices["symbol"] == symbol_focus].sort_values("date")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(hist["date"], hist["close"])
    ax.set_ylabel("Close price")
    st.pyplot(fig)

with col2:
    st.subheader("Efficient frontier")
    st.image("outputs/efficient_frontier.png", use_container_width=True)

# ---------------- personalized recommendation ----------------
st.subheader(f"Your personalised recommendation — {profile_name} investor")
rec = build_recommendation(profile_name, risk_tolerance, portfolios, predictions, sentiment, risk)

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.pie(rec["recommended_weight"], labels=rec["symbol"], autopct="%1.1f%%", startangle=90)
ax2.set_title("Recommended allocation")
st.pyplot(fig2)

st.dataframe(
    rec.rename(columns={"recommended_weight": "Weight"}).assign(
        Weight=lambda d: (d["Weight"] * 100).round(1).astype(str) + "%"
    ),
    use_container_width=True, hide_index=True,
)

st.caption(
    "This dashboard blends the Min-Volatility and Max-Sharpe portfolios according to your "
    "risk tolerance, then nudges each stock's weight based on its AI-predicted direction "
    "and current news sentiment."
)
