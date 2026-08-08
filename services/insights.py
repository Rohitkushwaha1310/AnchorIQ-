# services/insights.py
from google import genai
from google.genai import types
import json
import os

# ---- CONFIGURE GEMINI ----
# Get free key: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = "Geminni api"

client = genai.Client(api_key=GEMINI_API_KEY)

def generate_insights(inspection: dict,
                      model_results: dict,
                      target: str) -> str:
    try:
        prompt = f"""
You are a senior data analyst presenting to a CEO.

DATASET:
- Records  : {inspection['rows']:,}
- Features : {inspection['columns']}
- Target   : {target}
- Missing  : {inspection['missing_total']}

ML RESULTS:
- Best Model : {model_results.get('best_model','N/A')}
- AUC Score  : {model_results.get('auc','N/A')}
- Accuracy   : {model_results.get('accuracy','N/A')}%

Write:
## Executive Summary (2-3 sentences)
## Top 3 Key Findings (with numbers)
## Top 3 Recommendations (actionable)
## Risk Assessment (1-2 sentences)

Professional tone. Under 300 words.
"""
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        print("✅ Gemini insights generated!")
        return response.text

    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")
        return _rule_based_insights(
            inspection, model_results, target)


def _rule_based_insights(inspection: dict,
                          model_results: dict,
                          target: str) -> str:
    """
    Fallback insights if Gemini API fails.
    Rule-based but still professional!
    """
    auc      = model_results.get('auc', 0)
    accuracy = model_results.get('accuracy', 0)
    best     = model_results.get('best_model', 'ML Model')
    rows     = inspection['rows']
    cols     = inspection['columns']
    missing  = inspection['missing_total']
    dupes    = inspection['duplicates']

    # Model quality assessment
    if auc >= 0.85:
        model_quality = "excellent"
        confidence    = "high confidence"
    elif auc >= 0.75:
        model_quality = "good"
        confidence    = "reasonable confidence"
    else:
        model_quality = "moderate"
        confidence    = "limited confidence"

    return f"""
## Executive Summary
Analysis of **{rows:,} records** across **{cols} features** 
has been completed successfully. The {best} model achieved 
an **AUC of {auc}** — indicating {model_quality} predictive 
power for **{target}** with {confidence}.

## Top 3 Key Findings
1. **Data Quality**: Dataset contained {missing} missing 
   values and {dupes} duplicates — all automatically 
   resolved before analysis.

2. **Model Performance**: {best} achieved {accuracy}% 
   accuracy with AUC of {auc} — {'✅ Strong' if auc >= 0.8 
   else '⚠️ Needs improvement'} for business deployment.

3. **Predictive Power**: Cross-validated AUC of 
   {model_results.get('cv_mean', 'N/A')} confirms model 
   stability across different data splits.

## Top 3 Business Recommendations
1. **Deploy the model** for real-time {target} prediction 
   — flag high-risk cases before they occur.

2. **Focus retention efforts** on customers identified 
   as high-risk by the model — proactive beats reactive.

3. **Retrain monthly** as new data arrives to maintain 
   model accuracy and capture changing patterns.

## Risk Assessment
- Model confidence: {'High' if auc >= 0.85 else 
  'Medium' if auc >= 0.75 else 'Low'}
- Data freshness: Ensure monthly data updates
- Bias check: Validate model fairness across segments
"""





if __name__ == "__main__":
    # Mock data to test
    inspection = {
        'rows'         : 7043,
        'columns'      : 21,
        'missing_total': 11,
        'duplicates'   : 0,
        'stats'        : {
            'tenure': {'mean': 32.4, 'median': 29,
                       'std': 24.6, 'min': 0,
                       'max': 72, 'skewness': 0.24,
                       'nulls': 0}
        }
    }
    model_results = {
        'problem_type': 'classification',
        'best_model'  : 'Logistic Regression',
        'auc'         : 0.8189,
        'accuracy'    : 80.62,
        'cv_mean'     : 0.8123
    }

    insights = generate_insights(
        inspection, model_results, 'Churn')
    print(insights)