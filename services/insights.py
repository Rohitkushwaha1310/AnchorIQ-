"""
services/insights.py (v3)
The LLM receives verified FACTS (numbers computed from the data) and only writes them up.
If no API key / the API fails, a data-driven fallback renders the same facts - no generic templates.
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODELS = [m.strip() for m in os.getenv(
    "GROQ_MODELS", "llama-3.3-70b-versatile,llama-3.1-70b-versatile,llama-3.1-8b-instant,llama3-70b-8192,llama3-8b-8192,mixtral-8x7b-32768,gemma2-9b-it").split(",") if m.strip()]


def generate_insights(facts: dict, analysis_type: str = "general") -> str:
    compact = {k: facts[k] for k in ("overview", "findings", "recommendations", "warnings")}
    prompt = f"""You are a senior data analyst writing for a business owner.
Use ONLY the verified FACTS below. Do not invent numbers, columns or causes. If something is uncertain, say so.
Explain in plain language, no jargon. Keep under 450 words.

FACTS (JSON):
{json.dumps(compact, default=str)[:9000]}

Write markdown with exactly these sections:
## 🎯 Executive Summary
## 📊 Key Findings   (3-5 numbered points, each with the concrete numbers)
## 💡 Recommendations   (3-4 actions, ordered by priority; each says WHY (evidence) and expected impact; keep stated assumptions)
## ⚠️ Risks & Caveats
## 🚀 Next Steps   (2-3 bullets)"""
    if GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            available_models = list(GROQ_MODELS)
            try:
                listed = [m.id for m in client.models.list().data if getattr(m, "active", True)]
                if listed:
                    available_models = [m for m in GROQ_MODELS if m in listed] + [m for m in listed if m not in GROQ_MODELS]
            except Exception:
                pass

            for m in available_models:
                try:
                    r = client.chat.completions.create(
                        model=m, temperature=0.3, max_tokens=900,
                        messages=[{"role": "system", "content": "You write accurate, evidence-based business analysis."},
                                  {"role": "user", "content": prompt}])
                    text = r.choices[0].message.content
                    if text and len(text) > 200:
                        return text
                except Exception:
                    continue
        except Exception as e:
            print(f"[WARN] Groq unavailable: {e}")
    return render_from_facts(facts)


def render_from_facts(facts: dict) -> str:
    o, F, R, W = facts["overview"], facts["findings"], facts["recommendations"], facts["warnings"]
    lines = ["## 🎯 Executive Summary",
             f"Analysed **{o['rows']:,} records** x **{o['columns']} columns**"
             + (f", predicting **{o['target']}**." if o["target"] else " (no outcome column found, so we looked for segments, anomalies and trends).")]
    top = next((f["text"] for f in F if f["kind"] in ("gains", "trend", "base_rate")), None)
    if top:
        lines.append(top)
    lines += ["", "## 📊 Key Findings"]
    lines += [f"{i}. {f['text']}" for i, f in enumerate(F[:5], 1)] or ["No strong patterns were found."]
    lines += ["", "## 💡 Recommendations"]
    for r in R[:4]:
        lines.append(f"{r['priority']}. **{r['title']}** - {r['action']} _Impact: {r['impact']}_")
    if not R:
        lines.append("Not enough signal for a specific recommendation.")
    lines += ["", "## ⚠️ Risks & Caveats"]
    lines += [f"- {w}" for w in W] or ["- Findings are correlations in your historical data, not proof of cause. Re-run as new data arrives."]
    lines += ["", "## 🚀 Next Steps",
              "- Download the predictions file and start with the highest-priority group." if o["target"] else "- Review the segments and flagged records.",
              "- Re-run this analysis monthly with fresh data to track whether actions work."]
    return "\n".join(lines)
