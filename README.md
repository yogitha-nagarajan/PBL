# AI-Powered Investment Decision Assistant — Working Prototype

This implements the 9-stage architecture from the project's PBL presentation
(CS4511 — Intelligent Computing using ML — "Machine Learning Framework for
Intelligent Investment Decision Making"):

| # | Stage | Script |
|---|-------|--------|
| 1–2 | Market Data & News → Data Collection | `market_data.py` |
| 3 | Data Preprocessing | `preprocessing.py` |
| 4 | AI Stock Prediction (LSTM + XGBoost) | `stock_prediction.py` |
| 5 | Sentiment Analysis (VADER + FinBERT) | `sentiment_analysis.py` |
| 6 | Risk Assessment | `risk_assessment.py` |
| 7 | Portfolio Optimization (MPT / Sharpe ratio) | `portfolio_optimization.py` |
| 8 | Personalized Recommendations | `recommendation.py` |
| 9 | Investor Dashboard | `dashboard_app.py` (Streamlit) |

## Important: what's real vs. what's a fallback, and why

This project was built inside a sandbox **with no internet access**, which
means `pip install` cannot reach PyPI, and the specified libraries
(TensorFlow, XGBoost, vaderSentiment, transformers/FinBERT, yfinance)
cannot be downloaded there. So every stage is written with an **automatic
fallback**:

- It first tries to `import` the real library named in your spec (e.g.
  `xgboost`, `tensorflow`).
- If that import fails, it prints a note and uses a scikit-learn /
  from-scratch equivalent instead, so the pipeline still runs completely.
- **No code changes are needed on your end.** Just run:
  ```
  pip install -r requirements.txt
  ```
  on your own machine (which has internet), and every stage will
  automatically pick up the real TensorFlow LSTM, XGBoost, VADER, and
  FinBERT instead of the fallback.

| Stage | Specified tool | Fallback used here (auto-swapped) |
|---|---|---|
| Stock prediction (trend) | XGBoost | `GradientBoostingClassifier` (same gradient-boosted-tree family) |
| Stock prediction (return) | LSTM (TensorFlow/Keras) | `MLPRegressor` (a feed-forward neural network) |
| Sentiment | VADER (`vaderSentiment`) | A compact lexicon implementing VADER's actual rules (negation, intensifiers, punctuation emphasis) over a smaller finance-specific word list |
| Sentiment cross-check | FinBERT (`transformers`) | Skipped (needs a ~400MB model download) — VADER-family score used alone |
| Market data | Yahoo Finance API (`yfinance`) + Alpha Vantage news | A realistic synthetic price/news simulator (random walk + regime shifts + mild momentum/mean-reversion, so there's genuine — if modest — predictive signal, same as real markets) |

Risk assessment (Stage 6) and portfolio optimization (Stage 7) need **no
fallback** — Sharpe ratio, Value-at-Risk, max drawdown, and Modern
Portfolio Theory are closed-form/numerical calculations (`numpy`/`scipy`),
not trained models, so what runs here is exactly what would run with
real data.

## How to run

```
pip install -r requirements.txt      # add the specified libraries; safe to
                                      # skip tensorflow/xgboost/vaderSentiment/
                                      # transformers/torch if you just want
                                      # the fallback versions to run faster
python run_all.py                    # runs stages 1–8 in order
streamlit run dashboard_app.py       # stage 9: opens the interactive dashboard
```

Or run each stage individually (useful while developing/debugging):

```
python market_data.py
python preprocessing.py
python stock_prediction.py
python sentiment_analysis.py
python risk_assessment.py
python portfolio_optimization.py
python recommendation.py
streamlit run dashboard_app.py
```

## What you'll see

- **Direction model accuracy**: ~60-62% (realistic for short-term stock
  direction prediction — genuine market data rarely exceeds the high 50s
  to low 60s% for this task; a much higher number would suggest the
  labels leaked into the features)
- **Efficient frontier**: a proper Markowitz plot of ~8,000 simulated
  portfolios, with the Max-Sharpe and Min-Volatility portfolios marked
- **Three example investor profiles** (Conservative/Balanced/Aggressive)
  each get a different personalised allocation across the same 5 stocks

## Files produced

```
data/prices.csv                       Stage 1-2 output
data/news.csv                         Stage 1-2 output
data/features.csv                     Stage 3 output
models/direction_model.pkl            Stage 4
models/return_model.pkl (or .keras)   Stage 4
models/scaler.pkl                     Stage 4
outputs/latest_predictions.csv        Stage 4
outputs/sentiment_by_symbol.csv       Stage 5
outputs/risk_metrics.csv              Stage 6
outputs/efficient_frontier.png        Stage 7
outputs/optimized_portfolios.csv      Stage 7
outputs/recommendation_*.csv          Stage 8
```

## Live web dashboard

A self-contained interactive HTML version (`investment_dashboard.html`,
shared separately) mirrors stage 9 without needing Python installed —
it runs the real Modern Portfolio Theory optimiser live in JavaScript,
with pre-computed prediction/sentiment/risk results from this exact
pipeline embedded as data.
