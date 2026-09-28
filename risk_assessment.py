"""
risk_assessment.py
--------------------
Pipeline stage 6: "Risk Assessment" -- evaluate market risk, volatility,
and financial health metrics per stock. Fully real computation, no
substitutes needed (these are closed-form statistics, not learned models).

Metrics computed per symbol, from the last 252 trading days (~1 year):
  - Annualised volatility
  - Historical 95% daily Value-at-Risk (VaR)
  - Maximum drawdown
  - Annualised Sharpe ratio (assuming a 4% annual risk-free rate)

Output: outputs/risk_metrics.csv
"""

import numpy as np
import pandas as pd

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252


def compute_risk_metrics(returns: pd.Series, prices: pd.Series) -> dict:
    ann_return = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)

    var_95 = np.percentile(returns, 5)  # historical VaR, 95% confidence

    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    max_drawdown = drawdown.min()

    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan

    return {
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "var_95_daily": var_95,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
    }


def risk_tier(sharpe: float, ann_vol: float) -> str:
    """A simple, explainable tiering used by the recommendation stage."""
    if ann_vol > 0.45:
        return "High"
    if ann_vol > 0.25:
        return "Medium"
    return "Low"


def main():
    prices = pd.read_csv("data/prices.csv", parse_dates=["date"])

    rows = []
    for symbol, grp in prices.groupby("symbol"):
        grp = grp.sort_values("date").tail(TRADING_DAYS + 1)
        returns = grp["close"].pct_change().dropna()
        metrics = compute_risk_metrics(returns, grp["close"])
        metrics["symbol"] = symbol
        metrics["risk_tier"] = risk_tier(metrics["sharpe_ratio"], metrics["annual_volatility"])
        rows.append(metrics)

    risk_df = pd.DataFrame(rows)[[
        "symbol", "annual_return", "annual_volatility", "var_95_daily",
        "max_drawdown", "sharpe_ratio", "risk_tier"
    ]]

    risk_df.to_csv("outputs/risk_metrics.csv", index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("Risk assessment (trailing ~1 year):")
    print(risk_df.to_string(index=False))
    print("\nSaved to outputs/risk_metrics.csv")


if __name__ == "__main__":
    main()
