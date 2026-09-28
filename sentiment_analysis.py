"""
sentiment_analysis.py
-----------------------
Pipeline stage 5: "Sentiment Analysis" -- NLP of news headlines to
determine market mood.

As specified in the project architecture, this uses:
  - VADER (vaderSentiment)   -> fast, rule-based sentiment scoring
  - FinBERT (transformers)   -> finance-tuned transformer model, used to
                                 cross-check VADER on headlines it is
                                 least confident about

Automatic fallback: if vaderSentiment / transformers are not installed (as
in this offline sandbox, and FinBERT also needs to download ~400MB of
model weights which requires internet regardless), a compact fallback
scorer is used. It implements VADER's actual core algorithm --
lexicon lookup + negation handling + intensifier boosting + punctuation
emphasis -- over a smaller, finance-focused word list, so the mechanism is
real even though the lexicon is a subset of the full ~7,500-word VADER
lexicon.

On your own machine: `pip install vaderSentiment transformers torch` and
re-run -- no other code changes needed.

Output: outputs/sentiment_by_symbol.csv (daily-aggregated compound score
per symbol, averaged over the most recent 30 days of headlines)
"""

import re
import numpy as np
import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    from transformers import pipeline as hf_pipeline
    HAS_FINBERT = True
except ImportError:
    HAS_FINBERT = False


# ---------------------------------------------------------------------
# Fallback scorer: a compact VADER-style lexicon + the real VADER rules
# (negation flips, intensifier boosting, exclamation emphasis)
# ---------------------------------------------------------------------
_LEXICON = {
    "beats": 2.4, "record": 2.0, "growth": 1.8, "upgraded": 2.2, "buy": 1.9,
    "expands": 1.5, "boosting": 1.7, "strong": 2.0, "demand": 1.2,
    "raised": 1.8, "partnership": 1.3, "deal": 1.0, "outperform": 2.3,
    "profit": 1.9, "gain": 1.7, "surge": 2.4, "rally": 2.1, "bullish": 2.3,
    "misses": -2.2, "downgraded": -2.2, "weak": -1.9, "regulatory": -1.4,
    "investigation": -2.3, "declining": -2.0, "margins": -0.6, "soft": -1.2,
    "warns": -1.9, "slowing": -1.5, "disruption": -1.8, "loss": -2.1,
    "plunge": -2.5, "crash": -2.8, "bearish": -2.2, "risk": -1.1,
    "lawsuit": -2.0, "recall": -1.9,
    "holds": 0.0, "meeting": 0.0, "announces": 0.1, "board": 0.0,
    "present": 0.0, "conference": 0.0, "average": 0.0, "scheduled": 0.0,
    "filing": 0.0, "maintains": 0.0, "previous": 0.0, "guidance": 0.0,
}
_NEGATIONS = {"not", "no", "never", "without", "hardly", "barely"}
_INTENSIFIERS = {"very": 1.3, "extremely": 1.5, "highly": 1.3, "sharply": 1.4}


def fallback_vader_score(text: str) -> float:
    """A compact re-implementation of VADER's scoring mechanism (lexicon
    lookup, negation flip, intensifier boost, exclamation emphasis),
    normalised to roughly [-1, 1] the same way VADER's compound score is."""
    tokens = re.findall(r"[a-zA-Z']+|[!?]", text.lower())
    scores = []
    for i, tok in enumerate(tokens):
        if tok not in _LEXICON:
            continue
        val = _LEXICON[tok]
        window = tokens[max(0, i - 3):i]
        if any(w in _NEGATIONS for w in window):
            val *= -0.85
        for w in window:
            if w in _INTENSIFIERS:
                val *= _INTENSIFIERS[w]
        scores.append(val)

    if not scores:
        return 0.0

    exclam = tokens.count("!")
    total = sum(scores) + min(exclam, 3) * 0.3 * (1 if sum(scores) >= 0 else -1)
    # VADER-style normalisation: x / sqrt(x^2 + alpha)
    alpha = 15
    compound = total / np.sqrt(total ** 2 + alpha)
    return float(np.clip(compound, -1, 1))


def score_headlines(headlines: pd.Series) -> pd.Series:
    if HAS_VADER:
        print("[sentiment_analysis] Using the real vaderSentiment package.")
        analyzer = SentimentIntensityAnalyzer()
        return headlines.apply(lambda h: analyzer.polarity_scores(h)["compound"])
    else:
        print("[sentiment_analysis] vaderSentiment not installed -> using compact "
              "fallback lexicon scorer implementing VADER's core rules. "
              "'pip install vaderSentiment' on a machine with internet for the "
              "full lexicon.")
        return headlines.apply(fallback_vader_score)


def cross_check_with_finbert(headlines: pd.Series, vader_scores: pd.Series) -> pd.Series:
    """As per the architecture, FinBERT cross-checks headlines. Only runs
    if transformers + model weights are available (requires internet to
    download ~400MB the first time)."""
    if not HAS_FINBERT:
        print("[sentiment_analysis] transformers/FinBERT not available -> "
              "skipping cross-check, using VADER-family scores only. "
              "'pip install transformers torch' + internet access to enable.")
        return vader_scores
    try:
        clf = hf_pipeline("sentiment-analysis", model="ProsusAI/finbert")
        results = clf(list(headlines))
        sign = {"positive": 1, "negative": -1, "neutral": 0}
        finbert_scores = pd.Series(
            [sign[r["label"]] * r["score"] for r in results], index=headlines.index
        )
        return (vader_scores + finbert_scores) / 2
    except Exception as e:
        print(f"[sentiment_analysis] FinBERT run failed ({e}) -> using VADER-family scores only.")
        return vader_scores


def main():
    news = pd.read_csv("data/news.csv", parse_dates=["date"])

    news["sentiment"] = score_headlines(news["headline"])
    news["sentiment"] = cross_check_with_finbert(news["headline"], news["sentiment"])

    # Aggregate: mean sentiment per symbol over the most recent 30 days
    cutoff = news["date"].max() - pd.Timedelta(days=30)
    recent = news[news["date"] >= cutoff]
    agg = recent.groupby("symbol")["sentiment"].agg(["mean", "count"]).reset_index()
    agg.columns = ["symbol", "sentiment_score", "headline_count"]
    agg["mood"] = pd.cut(
        agg["sentiment_score"],
        bins=[-1, -0.15, 0.15, 1],
        labels=["Negative", "Neutral", "Positive"],
    )

    agg.to_csv("outputs/sentiment_by_symbol.csv", index=False)
    print(f"\nSentiment scored on {len(news)} headlines; aggregated over the last 30 days:")
    print(agg.to_string(index=False))
    print("\nSaved to outputs/sentiment_by_symbol.csv")


if __name__ == "__main__":
    main()
