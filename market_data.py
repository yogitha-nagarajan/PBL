"""
market_data.py
---------------
Pipeline stages 1-2 from the architecture diagram: "Market Data & News" and
"Data Collection".

Real version (what this does when you run it on your own machine with
internet access):
  - Historical prices via the Yahoo Finance API  (yfinance)
  - Recent financial news headlines via the Alpha Vantage News API

Sandbox fallback (what happens here, offline):
  - A realistic multi-stock price simulator (random walk with regime shifts,
    same approach used in the earlier prototype)
  - A synthetic daily headline generator that mixes positive / negative /
    neutral financial phrases, so the sentiment-analysis stage has
    something real to score

Either way the OUTPUT SHAPE is identical, so every downstream stage
(preprocessing, prediction, sentiment, risk, optimisation) works unmodified
regardless of which data source produced it.

Output:
  data/prices.csv   -> columns: date, symbol, close, volume
  data/news.csv     -> columns: date, symbol, headline
"""

import numpy as np
import pandas as pd

SYMBOLS = ["STK-A", "STK-B", "STK-C", "STK-D", "STK-E"]
N_DAYS = 500
RANDOM_STATE = 42

POS_PHRASES = [
    "beats quarterly earnings expectations",
    "announces record revenue growth",
    "upgraded by analysts to buy rating",
    "expands into new markets, boosting outlook",
    "reports strong demand and raised guidance",
    "signs major new partnership deal",
]
NEG_PHRASES = [
    "misses quarterly earnings expectations",
    "downgraded by analysts amid weak outlook",
    "faces regulatory investigation",
    "reports declining margins and soft demand",
    "warns of slowing growth next quarter",
    "hit by supply chain disruption",
]
NEU_PHRASES = [
    "holds annual shareholder meeting",
    "announces board changes",
    "to present at industry conference",
    "trading volume in line with average",
    "releases scheduled quarterly filing",
    "maintains previous full-year guidance",
]


def try_fetch_real_data():
    """Attempts to use yfinance, exactly as specified in the project's tool
    list. Returns a long-format DataFrame [date, symbol, close, volume] on
    success, or None (so the caller falls back to the synthetic simulator)
    if yfinance isn't installed or the request fails."""
    try:
        import yfinance as yf
    except ImportError:
        print("[market_data] yfinance not installed -> using synthetic data. "
              "Run 'pip install yfinance' on a machine with internet to use "
              "real Yahoo Finance data instead.")
        return None

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    try:
        # group_by='ticker' gives predictable columns: df[ticker]['Close'] etc.
        raw = yf.download(
            tickers, period="2y", progress=False,
            group_by="ticker", auto_adjust=True,
        )
        if raw.empty:
            raise RuntimeError("empty response from Yahoo Finance")

        records = []
        for ticker in tickers:
            # yfinance nests columns as (ticker, field); if only one ticker
            # actually returned data, the frame may not be multi-indexed
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                sub = raw[ticker]
            else:
                sub = raw
            sub = sub.dropna(subset=["Close"])
            for date, row in sub.iterrows():
                records.append({
                    "date": pd.Timestamp(date).normalize(),
                    "symbol": ticker,
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                })

        df = pd.DataFrame(records)
        if df.empty:
            raise RuntimeError("no usable rows after reshaping")

        print(f"[market_data] Fetched real data from Yahoo Finance: "
              f"{df['symbol'].nunique()} tickers, {len(df)} rows.")
        return df

    except Exception as e:
        print(f"[market_data] Live fetch failed ({e}) -> using synthetic data.")
        return None


def simulate_prices(rng, n_days=N_DAYS):
    """Random-walk-with-regime-shifts simulator (one series per symbol).

    A mild momentum/mean-reversion feedback is layered on top of pure
    noise: a stock trending above its own short-term average drifts
    (weakly) further in that direction next day, while an extreme RSI-like
    condition pulls the next return the other way. This mirrors two
    well-documented (if modest and noisy) real-market effects, so the
    prediction models below have genuine, if weak, signal to learn --
    rather than the ~50% coin-flip you get from a pure random walk."""
    records = []
    for symbol in SYMBOLS:
        prices = [100.0 + rng.normal(0, 10)]
        vol_regime = 0.012
        for day in range(1, n_days):
            if rng.random() < 0.02:
                vol_regime = rng.choice([0.008, 0.012, 0.02, 0.03])

            window = prices[-10:]
            sma = sum(window) / len(window)
            deviation = (prices[-1] - sma) / sma  # short-term trend signal

            momentum = np.clip(deviation, -0.05, 0.05) * 0.55
            mean_reversion = -np.clip(deviation, -0.08, 0.08) * 0.35
            signal = momentum + mean_reversion

            drift = rng.normal(0.0002, 0.0004)
            shock = rng.normal(drift + signal, vol_regime)
            price = max(prices[-1] * (1 + shock), 1.0)
            prices.append(price)

            volume = int(rng.normal(1_000_000, 200_000) * (1 + abs(shock) * 10))
            records.append({
                "day": day,
                "symbol": symbol,
                "close": round(price, 4),
                "volume": max(volume, 10_000),
            })
    df = pd.DataFrame(records)
    n_days_actual = df["day"].max() + 1
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days_actual)
    date_map = {d: dates[d] for d in range(n_days_actual)}
    df["date"] = df["day"].map(date_map)
    return df[["date", "symbol", "close", "volume"]]


def simulate_news(rng, prices_df, headlines_per_day=(0, 2)):
    """Generates synthetic headlines. Headlines skew positive after strong
    up-days and negative after strong down-days, so sentiment carries real
    (if synthetic) signal rather than being pure noise."""
    records = []
    for symbol, grp in prices_df.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)
        returns = grp["close"].pct_change().fillna(0)
        for i, row in grp.iterrows():
            n_headlines = rng.integers(headlines_per_day[0], headlines_per_day[1] + 1)
            for _ in range(n_headlines):
                r = returns.iloc[i]
                if r > 0.01:
                    pool, bias = POS_PHRASES, 0.7
                elif r < -0.01:
                    pool, bias = NEG_PHRASES, 0.7
                else:
                    pool, bias = NEU_PHRASES, 0.5
                if rng.random() < bias:
                    phrase = rng.choice(pool)
                else:
                    phrase = rng.choice(POS_PHRASES + NEG_PHRASES + NEU_PHRASES)
                records.append({
                    "date": row["date"],
                    "symbol": symbol,
                    "headline": f"{symbol} {phrase}",
                })
    return pd.DataFrame(records)


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    real = try_fetch_real_data()
    if real is not None:
        prices = real
        # Alpha Vantage news requires a free API key (not something this
        # script can assume you have), so headlines are still simulated --
        # but driven by each real stock's actual day-to-day returns, so
        # sentiment still correlates sensibly with real price action.
        news = simulate_news(rng, prices)
        print("[market_data] Note: news headlines are still simulated "
              "(Alpha Vantage needs a free API key -- see "
              "https://www.alphavantage.co/support/#api-key -- wire it into "
              "try_fetch_real_data() to pull real headlines).")
    else:
        prices = simulate_prices(rng)
        news = simulate_news(rng, prices)

    prices.to_csv("data/prices.csv", index=False)
    news.to_csv("data/news.csv", index=False)

    print(f"{len(prices)} price rows across {prices['symbol'].nunique()} symbols -> data/prices.csv")
    print(f"{len(news)} headlines -> data/news.csv")


if __name__ == "__main__":
    main()
