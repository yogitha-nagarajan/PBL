"""
recommendation.py
--------------------
Pipeline stage 8: "Personalized Recommendations" -- generate tailored
investment suggestions based on individual investor risk profiles.

Combines the outputs of every earlier stage:
  - predicted direction / return   (stage 4)
  - sentiment score                (stage 5)
  - risk metrics                   (stage 6)
  - Max Sharpe & Min Volatility anchor portfolios (stage 7)

An investor's risk tolerance (0 = fully conservative, 1 = fully aggressive)
blends linearly between the Min Volatility and Max Sharpe portfolios, then
each stock's allocation is nudged slightly based on its predicted direction
and current sentiment -- so two investors with different risk tolerances
looking at the same market get different, personalised allocations, not
just a single "optimal" answer.

Output: outputs/recommendation_<profile>.csv, printed summary
"""

import numpy as np
import pandas as pd

# Nudges applied on top of the risk-blended base weight.
DIRECTION_NUDGE = 0.03   # predicted "Up" adds this many weight points, "Down" subtracts
SENTIMENT_NUDGE_SCALE = 0.02  # multiplied by sentiment score in [-1, 1]


def blend_portfolios(min_vol_row, max_sharpe_row, symbols, risk_tolerance):
    """risk_tolerance: 0.0 (fully conservative) -> 1.0 (fully aggressive)"""
    base = {}
    for sym in symbols:
        base[sym] = (
            (1 - risk_tolerance) * min_vol_row[sym] + risk_tolerance * max_sharpe_row[sym]
        )
    return base


def apply_signal_nudges(base_weights, predictions, sentiment):
    adjusted = {}
    for sym, w in base_weights.items():
        nudge = 0.0
        pred_row = predictions[predictions["symbol"] == sym]
        if not pred_row.empty:
            direction = pred_row.iloc[0]["predicted_direction"]
            nudge += DIRECTION_NUDGE if direction == "Up" else -DIRECTION_NUDGE

        sent_row = sentiment[sentiment["symbol"] == sym]
        if not sent_row.empty:
            nudge += sent_row.iloc[0]["sentiment_score"] * SENTIMENT_NUDGE_SCALE

        adjusted[sym] = max(0.0, w + nudge)

    total = sum(adjusted.values())
    return {sym: w / total for sym, w in adjusted.items()}


def explain(sym, weight, predictions, sentiment, risk):
    pred_row = predictions[predictions["symbol"] == sym]
    sent_row = sentiment[sentiment["symbol"] == sym]
    risk_row = risk[risk["symbol"] == sym]

    direction = pred_row.iloc[0]["predicted_direction"] if not pred_row.empty else "n/a"
    mood = sent_row.iloc[0]["mood"] if not sent_row.empty else "n/a"
    tier = risk_row.iloc[0]["risk_tier"] if not risk_row.empty else "n/a"
    sharpe = risk_row.iloc[0]["sharpe_ratio"] if not risk_row.empty else float("nan")

    return (f"{sym}: allocate {weight*100:.1f}% -- model predicts {direction}, "
            f"news sentiment is {mood}, {tier} risk tier (Sharpe {sharpe:.2f})")


def build_recommendation(profile_name, risk_tolerance, portfolios, predictions, sentiment, risk):
    symbols = [c for c in portfolios.columns if c not in
               ("portfolio", "annual_return", "annual_volatility", "sharpe_ratio")]

    min_vol_row = portfolios[portfolios["portfolio"].str.contains("Min Volatility")].iloc[0]
    max_sharpe_row = portfolios[portfolios["portfolio"].str.contains("Max Sharpe")].iloc[0]

    base = blend_portfolios(min_vol_row, max_sharpe_row, symbols, risk_tolerance)
    final = apply_signal_nudges(base, predictions, sentiment)

    lines = [explain(sym, w, predictions, sentiment, risk) for sym, w in
             sorted(final.items(), key=lambda kv: -kv[1])]

    out = pd.DataFrame([
        {"symbol": sym, "recommended_weight": w} for sym, w in final.items()
    ]).sort_values("recommended_weight", ascending=False)
    out.to_csv(f"outputs/recommendation_{profile_name.lower()}.csv", index=False)

    print(f"\n=== {profile_name} investor (risk tolerance = {risk_tolerance}) ===")
    for line in lines:
        print("  " + line)
    print(f"  (saved to outputs/recommendation_{profile_name.lower()}.csv)")

    return out


def main():
    portfolios = pd.read_csv("outputs/optimized_portfolios.csv")
    predictions = pd.read_csv("outputs/latest_predictions.csv")
    sentiment = pd.read_csv("outputs/sentiment_by_symbol.csv")
    risk = pd.read_csv("outputs/risk_metrics.csv")

    for profile_name, tolerance in [("Conservative", 0.15), ("Balanced", 0.5), ("Aggressive", 0.9)]:
        build_recommendation(profile_name, tolerance, portfolios, predictions, sentiment, risk)


if __name__ == "__main__":
    main()
