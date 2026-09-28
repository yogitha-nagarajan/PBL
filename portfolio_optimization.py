"""
portfolio_optimization.py
----------------------------
Pipeline stage 7: "Portfolio Optimization" -- define asset allocations,
balancing return maximisation against risk tolerance.

Fully real implementation of Modern Portfolio Theory (no substitutes
needed -- this is closed-form / numerical optimisation, not a trained
model):
  - Estimates expected returns and the covariance matrix from historical data
  - Runs a Monte Carlo simulation of thousands of random portfolios to
    visualise the efficient frontier
  - Uses scipy.optimize to find the portfolio that maximises the Sharpe
    ratio (the "tangency portfolio")
  - Also finds the minimum-volatility portfolio
  - Blends between them according to the investor's risk tolerance
    (Conservative / Balanced / Aggressive), feeding stage 8

Output:
  outputs/efficient_frontier.png
  outputs/optimized_portfolios.csv
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
N_MONTE_CARLO = 8000
RANDOM_STATE = 42


def get_returns_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    pivot = prices.pivot(index="date", columns="symbol", values="close")
    return pivot.pct_change().dropna()


def portfolio_performance(weights, mean_returns, cov_matrix):
    ann_return = np.dot(weights, mean_returns) * TRADING_DAYS
    ann_vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else -np.inf
    return ann_return, ann_vol, sharpe


def monte_carlo_portfolios(mean_returns, cov_matrix, n_assets, rng, n=N_MONTE_CARLO):
    results = np.zeros((n, 3))
    all_weights = np.zeros((n, n_assets))
    for i in range(n):
        w = rng.random(n_assets)
        w /= w.sum()
        ret, vol, sharpe = portfolio_performance(w, mean_returns, cov_matrix)
        results[i] = [ret, vol, sharpe]
        all_weights[i] = w
    return results, all_weights


def optimize_for_target(mean_returns, cov_matrix, n_assets, objective):
    bounds = tuple((0.0, 0.40) for _ in range(n_assets))  # cap any single stock at 40%
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    init_guess = np.repeat(1 / n_assets, n_assets)

    if objective == "max_sharpe":
        def neg_sharpe(w):
            _, _, sharpe = portfolio_performance(w, mean_returns, cov_matrix)
            return -sharpe
        result = minimize(neg_sharpe, init_guess, method="SLSQP",
                           bounds=bounds, constraints=constraints)
    elif objective == "min_vol":
        def vol(w):
            return portfolio_performance(w, mean_returns, cov_matrix)[1]
        result = minimize(vol, init_guess, method="SLSQP",
                           bounds=bounds, constraints=constraints)
    else:
        raise ValueError(objective)

    return result.x


def main():
    prices = pd.read_csv("data/prices.csv", parse_dates=["date"])
    returns = get_returns_matrix(prices)
    symbols = list(returns.columns)
    n_assets = len(symbols)

    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    rng = np.random.default_rng(RANDOM_STATE)
    mc_results, mc_weights = monte_carlo_portfolios(mean_returns, cov_matrix, n_assets, rng)

    max_sharpe_w = optimize_for_target(mean_returns, cov_matrix, n_assets, "max_sharpe")
    min_vol_w = optimize_for_target(mean_returns, cov_matrix, n_assets, "min_vol")

    max_sharpe_perf = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix)
    min_vol_perf = portfolio_performance(min_vol_w, mean_returns, cov_matrix)
    equal_weight_perf = portfolio_performance(
        np.repeat(1 / n_assets, n_assets), mean_returns, cov_matrix
    )

    # ---- Save the two anchor portfolios (stage 8 blends between these
    #      according to the investor's risk tolerance) ----
    out_rows = []
    for name, w, perf in [
        ("Max Sharpe (Aggressive anchor)", max_sharpe_w, max_sharpe_perf),
        ("Min Volatility (Conservative anchor)", min_vol_w, min_vol_perf),
        ("Equal Weight (baseline)", np.repeat(1 / n_assets, n_assets), equal_weight_perf),
    ]:
        row = {"portfolio": name, "annual_return": perf[0], "annual_volatility": perf[1], "sharpe_ratio": perf[2]}
        for sym, wt in zip(symbols, w):
            row[sym] = wt
        out_rows.append(row)
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv("outputs/optimized_portfolios.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("Optimised portfolios:")
    print(out_df.to_string(index=False))

    # ---- Efficient frontier plot ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(mc_results[:, 1], mc_results[:, 0], c=mc_results[:, 2],
                     cmap="viridis", s=6, alpha=0.5)
    plt.colorbar(sc, label="Sharpe ratio")
    ax.scatter(max_sharpe_perf[1], max_sharpe_perf[0], marker="*", s=350,
               color="red", edgecolor="black", label="Max Sharpe (tangency) portfolio", zorder=5)
    ax.scatter(min_vol_perf[1], min_vol_perf[0], marker="*", s=350,
               color="blue", edgecolor="black", label="Min volatility portfolio", zorder=5)
    ax.set_xlabel("Annualised volatility (risk)")
    ax.set_ylabel("Annualised expected return")
    ax.set_title(f"Efficient Frontier ({N_MONTE_CARLO:,} simulated portfolios)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/efficient_frontier.png", dpi=150)
    print("\nSaved outputs/efficient_frontier.png and outputs/optimized_portfolios.csv")


if __name__ == "__main__":
    main()
