"""
run_all.py
-----------
Runs the entire 9-stage pipeline in order, end to end.
Equivalent to running each stage script manually, one after another.
"""
import subprocess
import sys

STAGES = [
    ("market_data.py", "Stage 1-2: Market Data & Collection"),
    ("preprocessing.py", "Stage 3: Data Preprocessing"),
    ("stock_prediction.py", "Stage 4: AI Stock Prediction"),
    ("sentiment_analysis.py", "Stage 5: Sentiment Analysis"),
    ("risk_assessment.py", "Stage 6: Risk Assessment"),
    ("portfolio_optimization.py", "Stage 7: Portfolio Optimization"),
    ("recommendation.py", "Stage 8: Personalized Recommendations"),
]

for script, label in STAGES:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n[run_all] {script} failed — stopping.")
        sys.exit(1)

print(f"\n{'='*70}\nAll stages complete. Run 'streamlit run dashboard_app.py' for "
      f"Stage 9 (Investor Dashboard).\n{'='*70}")
