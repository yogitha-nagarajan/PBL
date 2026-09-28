"""
build_dashboard.py
---------------------
Regenerates the interactive HTML dashboard (Stage 9) using YOUR real
pipeline output -- whatever is currently in data/ and outputs/ from your
last run of run_all.py (real Yahoo Finance data if you have internet and
yfinance installed, or the synthetic fallback otherwise).

Unlike dashboard_app.py (which needs Streamlit running as a local server),
this produces a single self-contained .html file with your results baked
in as static data. You can:
  - double-click it to open in any browser, no server needed
  - upload it to GitHub Pages (rename to index.html) to get a public URL
  - email it, since it's just one file

Run this AFTER run_all.py (it reads run_all.py's output files).

Output: investment_dashboard.html
"""

import base64
import json
import os

import pandas as pd

TEMPLATE_PATH = "dashboard_template.html"
OUTPUT_PATH = "investment_dashboard.html"


def build_bundle():
    prices = pd.read_csv("data/prices.csv", parse_dates=["date"])
    predictions = pd.read_csv("outputs/latest_predictions.csv")
    sentiment = pd.read_csv("outputs/sentiment_by_symbol.csv")
    risk = pd.read_csv("outputs/risk_metrics.csv")
    portfolios = pd.read_csv("outputs/optimized_portfolios.csv")

    symbols = sorted(prices["symbol"].unique())

    price_data = {}
    for sym in symbols:
        g = prices[prices["symbol"] == sym].sort_values("date").tail(250).copy()
        g["date"] = g["date"].dt.strftime("%Y-%m-%d")
        price_data[sym] = g[["date", "close"]].to_dict(orient="records")

    min_vol = portfolios[portfolios["portfolio"].str.contains("Min Volatility")].iloc[0]
    max_sharpe = portfolios[portfolios["portfolio"].str.contains("Max Sharpe")].iloc[0]

    anchors = {
        "minVol": {s: float(min_vol[s]) for s in symbols},
        "maxSharpe": {s: float(max_sharpe[s]) for s in symbols},
    }
    anchor_perf = {
        "minVol": {
            "return": float(min_vol["annual_return"]),
            "vol": float(min_vol["annual_volatility"]),
            "sharpe": float(min_vol["sharpe_ratio"]),
        },
        "maxSharpe": {
            "return": float(max_sharpe["annual_return"]),
            "vol": float(max_sharpe["annual_volatility"]),
            "sharpe": float(max_sharpe["sharpe_ratio"]),
        },
    }

    snapshot = predictions.merge(sentiment, on="symbol").merge(risk, on="symbol")

    with open("outputs/efficient_frontier.png", "rb") as f:
        frontier_b64 = base64.b64encode(f.read()).decode("ascii")

    return {
        "symbols": symbols,
        "prices": price_data,
        "anchors": anchors,
        "anchorPerf": anchor_perf,
        "snapshot": snapshot.to_dict(orient="records"),
        "frontierImage": frontier_b64,
    }


def main():
    required = [
        "data/prices.csv", "outputs/latest_predictions.csv",
        "outputs/sentiment_by_symbol.csv", "outputs/risk_metrics.csv",
        "outputs/optimized_portfolios.csv", "outputs/efficient_frontier.png",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Missing required file(s):")
        for m in missing:
            print(f"  - {m}")
        print("\nRun 'python run_all.py' first, then run this script again.")
        return

    bundle = build_bundle()

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("__BUNDLE_JSON__", json.dumps(bundle))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"Built {OUTPUT_PATH} ({size_kb:.0f} KB) using your real pipeline output.")
    print(f"Symbols included: {', '.join(bundle['symbols'])}")
    print("\nOpen it by double-clicking the file, or rename it to index.html "
          "and upload it to GitHub Pages to get a public URL.")


if __name__ == "__main__":
    main()
