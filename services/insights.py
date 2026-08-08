import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env")

client = Groq(api_key=GROQ_API_KEY)

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
- Duplicates: {inspection['duplicates']}

ML RESULTS:
- Best Model : {model_results.get('best_model','N/A')}
- AUC Score  : {model_results.get('auc','N/A')}
- Accuracy   : {model_results.get('accuracy','N/A')}%
- CV AUC     : {model_results.get('cv_mean','N/A')}

Write a professional business report:

## Executive Summary
(2-3 sentences about data health and model performance)

## Top 3 Key Findings
(Specific findings with numbers)

## Top 3 Business Recommendations
(Actionable steps)

## Risk Assessment
(1-2 sentences on risks)

Professional tone. Under 300 words. No technical jargon.
"""
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # free & fast!
            messages=[
                {"role": "system",
                 "content": "You are a senior data analyst."},
                {"role": "user",
                 "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        insights = response.choices[0].message.content
        print("✅ Groq insights generated!")
        return insights

    except Exception as e:
        print(f"⚠️ Groq failed: {e}")
        return _rule_based_insights(
            inspection, model_results, target)

def _rule_based_insights(inspection: dict,
                          model_results: dict,
                          target: str) -> str:
    auc      = model_results.get('auc', 0)
    accuracy = model_results.get('accuracy', 0)
    best     = model_results.get('best_model', 'ML Model')
    rows     = inspection['rows']
    cols     = inspection['columns']
    missing  = inspection['missing_total']
    dupes    = inspection['duplicates']

    quality = "excellent" if auc >= 0.85 else \
              "good" if auc >= 0.75 else "moderate"

    return f"""
## Executive Summary
Analysis of **{rows:,} records** across **{cols} features**
completed successfully. {best} achieved **AUC of {auc}**
— {quality} predictive power for **{target}**.

## Top 3 Key Findings
1. **Data Quality**: {missing} missing values and
   {dupes} duplicates — all automatically resolved.
2. **Model Performance**: {best} achieved {accuracy}%
   accuracy with AUC {auc} —
   {'✅ Strong' if auc >= 0.8 else '⚠️ Needs improvement'}.
3. **Stability**: CV AUC {model_results.get('cv_mean','N/A')}
   confirms consistent performance across data splits.

## Top 3 Business Recommendations
1. Deploy model for real-time {target} prediction.
2. Focus retention on high-risk flagged customers.
3. Retrain monthly with fresh data.

## Risk Assessment
Model confidence: {'High' if auc >= 0.85 else
'Medium' if auc >= 0.75 else 'Low'}.
Validate fairness across customer segments monthly.
"""




# Add at bottom to test
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