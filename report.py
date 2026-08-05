from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                 Spacer, Table, TableStyle,
                                 HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import pandas as pd


df= pd.read_csv('reports/churn_with_predictions.csv')

#Calculate metrics
total_customers = len(df)
churned = df['Churn_Binary'].sum()
churn_rate = df['Churn_Binary'].mean()*100

at_risk = df['Churn_Prediction'].sum()
monthly_revenue  = df['MonthlyCharges'].sum()
revenue_at_risk  = df[df['Churn_Prediction']==1]['MonthlyCharges'].sum()
avg_tenure_churn = df[df['Churn_Binary']==1]['tenure'].mean()
avg_tenure_stay  = df[df['Churn_Binary']==0]['tenure'].mean()

#COLORS
NAVY  = colors.HexColor('#1B3A6B')
BLACK = colors.HexColor('#1a1a1a')
GRAY  = colors.HexColor('#555555')
RED   = colors.HexColor('#e74c3c')
GREEN = colors.HexColor('#2ecc71')


#styles
def style(name, **kw):
    return ParagraphStyle(name, **kw)

title_s = style('t', fontsize =22, textColor = NAVY,
                alignment = TA_CENTER,
                fontName = 'Helvetica-Bold')   

sub_s    = style('s', fontSize=11, textColor=GRAY,
                  alignment=TA_CENTER,
                  fontName='Helvetica', spaceAfter=5*mm)
sec_s    = style('sec', fontSize=13, textColor=NAVY,
                  fontName='Helvetica-Bold',
                  spaceBefore=5*mm, spaceAfter=2*mm)
body_s   = style('b', fontSize=10, textColor=BLACK,
                  fontName='Helvetica', spaceAfter=2*mm,
                  leading=14, alignment=TA_JUSTIFY)
bullet_s = style('bul', fontSize=10, textColor=BLACK,
                  fontName='Helvetica', spaceAfter=2*mm,
                  leading=14, leftIndent=10,
                  firstLineIndent=-10)                 


def hr():
    return HRFlowable(width = "100%", thickness =1, color = NAVY, spaceAfter = 3*mm)           

def sec(t):
    return [Paragraph(t, sec_s), hr()]

def bul(t):
    return Paragraph(f"• {t}", bullet_s)


#build report
doc = SimpleDocTemplate(
    'reports/AnchorIQ_Churn_Report.pdf',
    pagesize=A4,
    topMargin=15*mm, bottomMargin=15*mm,
    leftMargin=20*mm, rightMargin=20*mm)
 

story = []

#title page 
story.append(Spacer(1, 10*mm))
story.append(Paragraph("AnchorIQ", title_s))
story.append(Paragraph(
    "Customer Churn Prediction — Business Report",
    style('st', fontSize=14, textColor=NAVY,
          alignment=TA_CENTER,
          fontName='Helvetica-Bold', spaceAfter=2*mm)))
story.append(Paragraph(
    f"TeleConnect India  |  {datetime.now().strftime('%B %d, %Y')}",
    sub_s))
story.append(Spacer(1, 5*mm))
story.append(hr())

#executive summary 
story += sec("1. EXECUTIVE SUMMARY")
story.append(Paragraph(
    f"TeleConnect India is experiencing a critical churn rate of "
    f"<b>{churn_rate:.1f}%</b>, meaning <b>{churned:,} out of "
    f"{total_customers:,} customers</b> have left the service. "
    f"This analysis identifies the key drivers of churn and provides "
    f"actionable recommendations to reduce customer attrition and "
    f"protect <b>${monthly_revenue:,.0f}</b> in monthly recurring revenue.",
    body_s))


#key matrics table 
metrics_data = [
    ['Metric', 'Value', 'Status'],
    ['Total Customers', f'{total_customers:,}', '—'],
    ['Churned Customers', f'{churned:,}', '🔴 Critical'],
    ['Churn Rate', f'{churn_rate:.2f}%', '🔴 High'],
    ['At Risk Customers', f'{int(at_risk):,}', '🟡 Monitor'],
    ['Monthly Revenue', f'${monthly_revenue:,.0f}', '✅ Stable'],
    ['Revenue at Risk', f'${revenue_at_risk:,.0f}', '🔴 Urgent'],
    ['Avg Tenure (Churned)', f'{avg_tenure_churn:.1f} months', '🔴 Low'],
    ['Avg Tenure (Stayed)', f'{avg_tenure_stay:.1f} months', '✅ Good'],
    ['Model AUC Score', '0.8463', '✅ Strong'],
    ['Optimal Threshold', '0.15', '✅ Tuned'],
]

table = Table(metrics_data, colWidths=['50%','30%','20%'])
table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), NAVY),
    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',   (0,0), (-1,-1), 10),
    ('ALIGN',      (0,0), (-1,-1), 'LEFT'),
    ('ROWBACKGROUNDS', (0,1), (-1,-1),
     [colors.HexColor('#f8f9fa'), colors.white]),
    ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
    ('PADDING',    (0,0), (-1,-1), 6),
]))
story.append(table)
story.append(Spacer(1, 5*mm))

# ---- KEY FINDINGS ----
story += sec("3. KEY FINDINGS")
story.append(bul(
    f"<b>Month-to-month contracts</b> have the highest churn rate "
    f"at ~43% compared to only 3% for two-year contracts — "
    f"contract type is the single strongest churn predictor."))
story.append(bul(
    f"<b>Electronic check payment</b> users churn at the highest "
    f"rate among all payment methods — auto-pay customers are "
    f"significantly more loyal."))
story.append(bul(
    f"<b>New customers (tenure &lt; 12 months)</b> churn at "
    f"48.3% — the first year is the most critical retention window."))
story.append(bul(
    f"<b>High monthly charges</b> correlate positively with churn "
    f"— price-sensitive customers need targeted retention offers."))
story.append(bul(
    f"<b>Low service adoption</b> (ServiceCount = 0-1) leads to "
    f"higher churn — customers with more services stay longer."))

# ---- ML MODEL RESULTS ----
story += sec("4. ML MODEL PERFORMANCE")
story.append(Paragraph(
    "A Logistic Regression model was trained on 5,634 customer records "
    "and evaluated on 1,409 held-out customers. The model achieved "
    "an AUC of <b>0.8463</b> — outperforming Random Forest (0.8267) "
    "and XGBoost (0.8407). Threshold optimization at <b>0.15</b> "
    "reduced business cost from $92,850 to $36,400 — saving an "
    "additional <b>$21,900</b> purely through threshold tuning.",
    body_s))

model_data = [
    ['Model', 'Accuracy', 'AUC', 'F1', 'Winner'],
    ['Logistic Regression', '80.62%', '0.8463', '—', '🏆 Best'],
    ['XGBoost',             '—',      '0.8407', '—', '2nd'],
    ['Random Forest',       '—',      '0.8267', '—', '3rd'],
]
model_table = Table(model_data, colWidths=['35%','20%','15%','15%','15%'])
model_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), NAVY),
    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',   (0,0), (-1,-1), 10),
    ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
    ('ROWBACKGROUNDS', (0,1), (-1,-1),
     [colors.HexColor('#f8f9fa'), colors.white]),
    ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
    ('PADDING',    (0,0), (-1,-1), 6),
]))
story.append(model_table)
story.append(Spacer(1, 5*mm))

# ---- RECOMMENDATIONS ----
story += sec("5. BUSINESS RECOMMENDATIONS")
recs = [
    ("Convert Month-to-Month Customers",
     "Offer 20% discount for switching to annual contracts. "
     "Converting just 500 customers saves ~$120,000 annually."),
    ("Launch Auto-Pay Incentive Program",
     "Offer $5/month discount for switching to auto-pay. "
     "Expected churn reduction: 18% among manual pay customers."),
    ("90-Day New Customer Onboarding",
     "Assign dedicated support for first 90 days. "
     "Target the 48.3% new customer churn rate directly."),
    ("Service Bundle Promotions",
     "Customers with 4+ services churn significantly less. "
     "Offer free trial of TechSupport or StreamingTV."),
    ("High-Risk Customer Alerts",
     f"Use ML model to flag {int(at_risk):,} at-risk customers "
     "monthly. Proactive outreach before churn occurs."),
]
for title, desc in recs:
    story.append(Paragraph(
        f"<b>{title}:</b> {desc}", bullet_s))
    story.append(Spacer(1, 1*mm))

# ---- CONCLUSION ----
story += sec("6. CONCLUSION")
story.append(Paragraph(
    f"This analysis demonstrates that TeleConnect India's "
    f"<b>${monthly_revenue:,.0f}</b> monthly revenue is at significant "
    f"risk from a <b>{churn_rate:.1f}%</b> churn rate. The AnchorIQ "
    f"ML pipeline identifies at-risk customers with an AUC of 0.8463, "
    f"enabling proactive retention before revenue is lost. "
    f"Implementing the five recommendations above could reduce churn "
    f"by an estimated 30-40%, protecting <b>${revenue_at_risk*0.35:,.0f}</b> "
    f"in monthly recurring revenue.",
    body_s))

story.append(Spacer(1, 5*mm))
story.append(hr())
story.append(Paragraph(
    "Generated by AnchorIQ ",
    style('f', fontSize=8, textColor=GRAY,
          alignment=TA_CENTER, fontName='Helvetica')))

doc.build(story)
print("✅ Report saved: reports/AnchorIQ_Churn_Report.pdf")

    