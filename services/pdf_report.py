"""
services/pdf_report.py
Dynamic Executive PDF Report Generator for AnchorIQ — Universal, Crisp, Professional.
"""

import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT


# ── Colour Palette ────────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#0f172a")
DARK    = colors.HexColor("#1e293b")
BLUE    = colors.HexColor("#3B82F6")
INDIGO  = colors.HexColor("#6366F1")
GREEN   = colors.HexColor("#10B981")
RED     = colors.HexColor("#EF4444")
ORANGE  = colors.HexColor("#F59E0B")
PURPLE  = colors.HexColor("#8B5CF6")
BLACK   = colors.HexColor("#1e293b")
GRAY    = colors.HexColor("#64748B")
LIGHT   = colors.HexColor("#F8FAFC")
WHITE   = colors.white

# Analysis type → accent colour
TYPE_COLORS = {
    "churn"  : RED,
    "sales"  : GREEN,
    "fraud"  : ORANGE,
    "hr"     : PURPLE,
    "general": INDIGO,
    "auto"   : BLUE,
}

TYPE_LABELS = {
    "churn"  : "Customer Churn & Retention Analysis",
    "sales"  : "Sales & Revenue Performance Report",
    "fraud"  : "Fraud & Anomaly Detection Analysis",
    "hr"     : "Workforce & Talent Analytics Report",
    "general": "Exploratory Data Science Report",
    "auto"   : "Autonomous Machine Learning Report",
}


def _style(name, **kw):
    return ParagraphStyle(name, **kw)


def _hr(accent=INDIGO, thickness=1.5):
    return HRFlowable(width="100%", thickness=thickness, color=accent, spaceAfter=3 * mm, spaceBefore=1 * mm)


def _section(title, accent=INDIGO):
    st = _style("sec", fontSize=13, textColor=accent,
                fontName="Helvetica-Bold",
                spaceBefore=5 * mm, spaceAfter=2 * mm)
    return [Paragraph(title, st), _hr(accent, thickness=1.2)]


def _bullet(text, accent=INDIGO):
    st = _style("bul", fontSize=9.5, textColor=BLACK,
                fontName="Helvetica", spaceAfter=1.8 * mm,
                leading=13.5, leftIndent=12, firstLineIndent=-10)
    return Paragraph(f"<font color='#{accent.hexval()[2:]}'>&bull;</font> {text}", st)


def generate_pdf_report(
    result: dict,
    chart_paths: list | None = None,
    output_path: str = "reports/AnchorIQ_Report.pdf",
) -> str:
    """
    Generate a full executive PDF analysis report from the result dict.
    Returns the path to the saved PDF.
    """
    if chart_paths is None:
        chart_paths = result.get("chart_paths", [])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    analysis_type = result.get("inspection", {}).get("analysis_type", "auto")
    accent        = TYPE_COLORS.get(analysis_type, BLUE)
    label         = TYPE_LABELS.get(analysis_type, "Autonomous Data Analysis")
    filename      = result.get("filename", "Dataset")
    insp          = result.get("inspection", {})
    clean         = result.get("cleaning", {})
    ml            = result.get("model_results", {})
    insights_text = result.get("insights", "")
    now           = datetime.now().strftime("%B %d, %Y - %H:%M")

    # ── Styles ────────────────────────────────────────────────────────────────
    title_s  = _style("t",  fontSize=24, textColor=NAVY,
                      alignment=TA_CENTER, fontName="Helvetica-Bold", leading=28)
    sub_s    = _style("s",  fontSize=11, textColor=GRAY,
                      alignment=TA_CENTER, fontName="Helvetica", spaceAfter=2 * mm, leading=14)
    meta_s   = _style("m",  fontSize=8.5, textColor=GRAY,
                      alignment=TA_CENTER, fontName="Helvetica", spaceAfter=3 * mm)
    body_s   = _style("b",  fontSize=9.5, textColor=BLACK,
                      fontName="Helvetica", spaceAfter=2 * mm,
                      leading=14, alignment=TA_JUSTIFY)
    cap_s    = _style("c",  fontSize=8,  textColor=GRAY,
                      alignment=TA_CENTER, fontName="Helvetica-Oblique",
                      spaceAfter=3 * mm)
    footer_s = _style("f",  fontSize=8,  textColor=GRAY,
                      alignment=TA_CENTER, fontName="Helvetica")

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=12 * mm, bottomMargin=12 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
    )

    story = []

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 1: COVER & QUICK STATS
    # ════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 15 * mm))
    story.append(Paragraph("AnchorIQ", title_s))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(f"<b>{label}</b>", sub_s))
    story.append(Paragraph(f"Dataset: <b>{filename}</b>", sub_s))
    story.append(Spacer(1, 2 * mm))
    story.append(_hr(accent, thickness=2))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(f"Generated: {now} | Powered by AnchorIQ v2.0", meta_s))
    story.append(Spacer(1, 6 * mm))

    # Quick KPI Banner
    target_name = str(insp.get("target_column", "N/A"))
    banner_data = [
        [
            Paragraph(f"<font size='14'><b>{insp.get('rows', 0):,}</b></font><br/><font color='#cbd5e1'>Records</font>", sub_s),
            Paragraph(f"<font size='14'><b>{insp.get('columns', 0)}</b></font><br/><font color='#cbd5e1'>Features</font>", sub_s),
            Paragraph(f"<font size='14'><b>{target_name[:12]}</b></font><br/><font color='#cbd5e1'>Target</font>", sub_s),
            Paragraph(f"<font size='14'><b>{insp.get('missing_total', 0)}</b></font><br/><font color='#cbd5e1'>Fixed Nulls</font>", sub_s),
            Paragraph(f"<font size='14'><b>{insp.get('duplicates', 0)}</b></font><br/><font color='#cbd5e1'>Dupes Cleared</font>", sub_s),
        ]
    ]
    banner = Table(banner_data, colWidths=["20%"] * 5)
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, -1), WHITE),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",    (0, 0), (-1, -1), 8),
    ]))
    story.append(banner)
    story.append(Spacer(1, 6 * mm))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1: EXECUTIVE SUMMARY & AI INSIGHTS
    # ════════════════════════════════════════════════════════════════════════
    story += _section("1. EXECUTIVE SUMMARY & AI INSIGHTS", accent)

    if insights_text:
        for line in insights_text.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 1 * mm))
                continue
            if line.startswith("## "):
                st = _style("h2", fontSize=11, textColor=accent,
                            fontName="Helvetica-Bold", spaceBefore=3 * mm, spaceAfter=1 * mm)
                story.append(Paragraph(line[3:], st))
            elif line.startswith("### "):
                st = _style("h3", fontSize=10, textColor=NAVY,
                            fontName="Helvetica-Bold", spaceAfter=1 * mm)
                story.append(Paragraph(line[4:], st))
            elif line.startswith(("1.", "2.", "3.", "4.", "5.")):
                story.append(Paragraph(line, body_s))
            elif line.startswith("- ") or line.startswith("* "):
                story.append(_bullet(line[2:], accent))
            else:
                story.append(Paragraph(line, body_s))
    else:
        story.append(Paragraph("Executive insights compiled from exploratory and predictive engines.", body_s))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2: DATASET OVERVIEW & QUALITY AUDIT
    # ════════════════════════════════════════════════════════════════════════
    story += _section("2. DATASET OVERVIEW & QUALITY AUDIT", accent)

    quality_pct = max(0, 100 - (insp.get("missing_total", 0) / max(insp.get("rows", 1), 1) * 100))
    quality_label = "Optimal" if quality_pct >= 95 else "Good" if quality_pct >= 85 else "Action Needed"

    overview_data = [
        ["Audit Metric", "Value", "Status"],
        ["Total Records",       f"{insp.get('rows', 0):,}",        "Validated"],
        ["Total Feature Count", str(insp.get("columns", 0)),        "Validated"],
        ["Numerical Columns",   str(len(insp.get("numerical_cols", []))), "Active"],
        ["Categorical Columns", str(len(insp.get("categorical_cols", []))), "Active"],
        ["Target Column",       str(insp.get("target_column", "N/A")), "Identified"],
        ["Missing Cells Imputed",str(insp.get("missing_total", 0)), "Resolved"],
        ["Duplicate Rows Purged",str(insp.get("duplicates", 0)),    "Resolved"],
        ["Data Quality Index",  f"{quality_pct:.1f}%",              quality_label],
    ]
    ov_table = Table(overview_data, colWidths=["50%", "30%", "20%"])
    ov_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
        ("ALIGN",      (2, 0), (2, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING",    (0, 0), (-1, -1), 5),
    ]))
    story.append(ov_table)
    story.append(Spacer(1, 3 * mm))

    # Cleaning steps
    steps = clean.get("steps_applied", [])
    if steps:
        story.append(Paragraph("<b>Preprocessing Pipeline Execution:</b>", body_s))
        for step in steps[:8]:
            story.append(_bullet(step, accent))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3: ML MODEL PERFORMANCE & BENCHMARKS
    # ════════════════════════════════════════════════════════════════════════
    if ml and "error" not in ml:
        story += _section("3. MACHINE LEARNING BENCHMARKS", accent)
        is_reg = ml.get("problem_type") == "regression"

        story.append(Paragraph(
            f"Problem Formulation: <b>{ml.get('problem_type', 'N/A').title()}</b> | "
            f"Best Champion: <b>{ml.get('best_model', 'N/A')}</b> | "
            f"Training: <b>{ml.get('train_samples', 0):,}</b> samples | "
            f"Evaluation: <b>{ml.get('test_samples', 0):,}</b> samples | "
            f"Engineered Features: <b>{ml.get('n_features', 0)}</b>",
            body_s,
        ))

        if is_reg:
            model_rows = [
                ["Model Candidate", "Metric Score", "Validation Status"],
                ["Champion: " + str(ml.get("best_model", "N/A")), f"R² = {ml.get('r2_score','N/A')}", "Top Performer"],
                ["Ridge Regression",  f"R² = {ml.get('lr_r2','N/A')}",  "Baseline"],
                ["Random Forest",     f"R² = {ml.get('rf_r2','N/A')}",  "Ensemble"],
                ["XGBoost Regressor", f"R² = {ml.get('xgb_r2','N/A')}", "Gradient Boosted"],
                ["RMSE (Error)",      str(ml.get("rmse", "N/A")),       "Lower is better"],
                ["Cross-Validation",  f"{ml.get('cv_mean','N/A')} ± {ml.get('cv_std','0')}", "5-Fold CV"],
            ]
        else:
            model_rows = [
                ["Model Candidate", "AUC Score", "Validation Status"],
                ["Champion: " + str(ml.get("best_model", "N/A")), f"AUC = {ml.get('auc','N/A')}", "Top Performer"],
                ["Logistic Regression", f"AUC = {ml.get('lr_auc','N/A')}",  "Linear"],
                ["Random Forest",       f"AUC = {ml.get('rf_auc','N/A')}",  "Ensemble"],
                ["XGBoost Classifier",  f"AUC = {ml.get('xgb_auc','N/A')}", "Gradient Boosted"],
                ["Accuracy",            f"{ml.get('accuracy', 'N/A')}%",     "Overall Match"],
                ["Cross-Validation",    f"{ml.get('cv_mean','N/A')} ± {ml.get('cv_std','0')}", "5-Fold CV"],
            ]

        ml_table = Table(model_rows, colWidths=["45%", "30%", "25%"])
        ml_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), accent),
            ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING",    (0, 0), (-1, -1), 5),
        ]))
        story.append(ml_table)
        story.append(Spacer(1, 3 * mm))

        # Top feature importance
        fi = ml.get("feature_importance", {})
        if fi:
            story.append(Paragraph("<b>Key Drivers & Influential Signals:</b>", body_s))
            fi_data = [["Rank", "Feature Variable", "Importance Weight"]]
            for i, (feat, score) in enumerate(list(fi.items())[:8], 1):
                fi_data.append([str(i), str(feat)[:35], f"{score:.4f}"])
            fi_table = Table(fi_data, colWidths=["12%", "58%", "30%"])
            fi_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
                ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
                ("ALIGN",      (2, 0), (2, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("PADDING",    (0, 0), (-1, -1), 4),
            ]))
            story.append(fi_table)

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4: VISUAL EDA CHARTS
    # ════════════════════════════════════════════════════════════════════════
    if chart_paths:
        story += _section("4. EXPLORATORY DATA ANALYSIS (EDA CHARTS)", accent)

        chart_titles = [
            "Feature Distributions",
            "Feature Correlation Heatmap",
            "Target Variable Distribution",
            "Categorical Feature Impact",
            "Numerical Distributions by Target",
            "Domain-Specific Analysis",
        ]

        valid_charts = [p for p in chart_paths if os.path.exists(p)]
        for i, chart_path in enumerate(valid_charts):
            title = chart_titles[i] if i < len(chart_titles) else f"Visual Insight {i+1}"
            chart_block = []
            chart_block.append(Paragraph(f"<b>Figure {i+1}: {title}</b>", body_s))
            try:
                img = Image(chart_path, width=170 * mm, height=88 * mm)
                chart_block.append(img)
                chart_block.append(Spacer(1, 4 * mm))
                story.append(KeepTogether(chart_block))
            except Exception:
                pass

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5: STATISTICAL SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    stats = insp.get("stats", {})
    if stats:
        story.append(PageBreak())
        story += _section("5. FEATURE STATISTICAL SUMMARY", accent)

        stat_header = ["Feature", "Mean", "Median", "Std Dev", "Min", "Max", "Skewness"]
        stat_data = [stat_header]
        for col, vals in list(stats.items())[:15]:
            stat_data.append([
                str(col)[:22],
                str(vals.get("mean", "—")),
                str(vals.get("median", "—")),
                str(vals.get("std", "—")),
                str(vals.get("min", "—")),
                str(vals.get("max", "—")),
                str(vals.get("skewness", "—")),
            ])

        stat_table = Table(stat_data, colWidths=["26%", "12%", "12%", "13%", "12%", "12%", "13%"])
        stat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("ALIGN",      (0, 0), (0, -1), "LEFT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING",    (0, 0), (-1, -1), 4.5),
        ]))
        story.append(stat_table)

    # ════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 8 * mm))
    story.append(_hr(GRAY, thickness=1))
    story.append(Paragraph(
        f"Generated by <b>AnchorIQ</b> — Autonomous Data Analysis Platform | {now}",
        footer_s,
    ))

    doc.build(story)
    print(f"[OK] PDF report saved: {output_path}")
    return output_path
