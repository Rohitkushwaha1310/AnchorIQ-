"""Run from the project root:  python tests/smoke_test.py"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from services.loader import load_any
from services.pipeline import run_analysis

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: sys.exit(1)

# 1) data WITH an outcome column
if os.path.exists("telco_churn.csv"):
    df, meta = load_any("telco_churn.csv", open("telco_churn.csv", "rb").read())
    r = run_analysis(df, "telco_churn.csv", "churn", base_dir="tests/_out")
    check("churn target detected", r["profile"]["target"] == "Churn")
    check("model trained (AUC > 0.7)", r["model_results"].get("auc", 0) > 0.7)
    check("per-row predictions saved", os.path.exists(r["enriched_file"]))
    check("recommendations produced", len(r["recommendations"]) >= 3)

# 2) data WITHOUT an outcome column, with dates
rng = np.random.RandomState(1); n = 400
d = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n), "region": rng.choice(list("NSEW"), n),
                  "units": rng.poisson(20, n), "revenue": rng.gamma(5, 40, n)})
buf = d.to_csv(index=False).encode()
df, meta = load_any("sales.csv", buf)
r = run_analysis(df, "sales.csv", "auto", base_dir="tests/_out")
check("no-target data still gives insights", len(r["facts"]["findings"]) > 0)
print("\nAll smoke tests passed.")
