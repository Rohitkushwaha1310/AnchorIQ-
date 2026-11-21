import io
import os
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
from reportlab.pdfgen import canvas


# Brand Palette
NAVY   = colors.HexColor("#1B3A6B")
TEAL   = colors.HexColor("#0EA5E9")
DARK   = colors.HexColor("#1E293B")
GRAY   = colors.HexColor("#64748B")
LIGHT  = colors.HexColor("#F8FAFC")
WHITE  = colors.white
GREEN  = colors.HexColor("#10B981")
RED    = colors.HexColor("#EF4444")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for 'AnchorIQ | Autonomous Business Intelligence | Page X of Y' footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(GRAY)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(18 * mm, 12 * mm, 192 * mm, 12 * mm)
        footer_text = f"AnchorIQ | Autonomous Business Intelligence | Page {self._pageNumber} of {page_count}"
        self.drawCentredString(105 * mm, 8 * mm, footer_text)
        self.restoreState()


def _style(name, **kw):
    return ParagraphStyle(name, **kw)


def generate_pdf_report(
    result: dict,
    chart_paths: list | None = None,
    output_path: str | None = None,
    chart_images: list | None = None,
) -> any:
    """
    Generate professional executive PDF report using ReportLab with exact pages:
    Page 1 — Cover: AnchorIQ logo, analysis type, dataset, date, KPI banner
    Page 2 — Dataset Overview: Inspection table, target info, numerical stats
    Page 3 — Cleaning Report: Steps applied list, before/after shape comparison
    Pages 4-5 — EDA Charts: Embed all generated chart PNGs with captions (2 per page)
    Page 6 — ML Results: Model comparison table (best highlighted green), confusion matrix, CV scores
    Page 7 — AI Insights: Full insights markdown rendered, navy headers, bullet points
    Footer on every page: AnchorIQ | Autonomous Business Intelligence | Page X of Y
    If output_path is None, generates and returns the PDF bytes in memory.
    """
    in_memory = output_path is None
    target_dest = io.BytesIO() if in_memory else output_path
    if not in_memory and os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    insp = result.get("inspection", {})
    clean = result.get("cleaning", {})
    ml = result.get("model_results", {})
    insights_text = result.get("insights", "")
    analysis_type = result.get("analysis_type", insp.get("analysis_type", "auto")).upper()
    filename = result.get("filename", "Business Dataset")
    timestamp = datetime.now().strftime("%B %d, %Y - %H:%M")

    doc = SimpleDocTemplate(
        target_dest,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = _style("CovTitle", fontSize=28, fontName="Helvetica-Bold", textColor=NAVY, alignment=TA_CENTER, leading=34)
    subtitle_style = _style("CovSub", fontSize=12, fontName="Helvetica", textColor=TEAL, alignment=TA_CENTER, leading=16)
    section_style = _style("SecTitle", fontSize=14, fontName="Helvetica-Bold", textColor=NAVY, spaceBefore=4*mm, spaceAfter=2*mm)
    body_style = _style("Body", fontSize=9, fontName="Helvetica", textColor=DARK, leading=13)
    body_bold = _style("BodyB", fontSize=9, fontName="Helvetica-Bold", textColor=DARK, leading=13)
    table_cell = _style("TCell", fontSize=8.5, fontName="Helvetica", textColor=DARK, leading=11)
    table_cell_bold = _style("TCellB", fontSize=8.5, fontName="Helvetica-Bold", textColor=DARK, leading=11)
    caption_style = _style("Cap", fontSize=8.5, fontName="Helvetica-Oblique", textColor=GRAY, alignment=TA_CENTER, spaceBefore=1.5*mm, spaceAfter=3*mm)

    story = []

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER PAGE
    # ════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 25 * mm))
    story.append(Paragraph("⚓ AnchorIQ", title_style))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Autonomous Business Intelligence Platform", subtitle_style))
    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=4*mm))

    badge_table = Table([[Paragraph(f"<b>ANALYSIS TYPE: {analysis_type}</b>", _style("Bdg", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE, alignment=TA_CENTER))]], colWidths=["100%"])
    badge_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TEAL),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(badge_table)
    story.append(Spacer(1, 6 * mm))

    meta_info = [
        ["Dataset Name:", filename],
        ["Report Generated:", timestamp],
        ["Target Variable:", str(insp.get("target_column") or "N/A")],
        ["Best ML Algorithm:", str(ml.get("best_model") or "Auto-Selected Model")],
        ["Validation Score:", f"AUC/R²: {ml.get('auc', ml.get('r2', 'N/A'))} | Accuracy: {ml.get('accuracy', 'N/A')}%"],
    ]
    meta_table = Table(
        [[Paragraph(k, body_bold), Paragraph(v, body_style)] for k, v in meta_info],
        colWidths=["35%", "65%"]
    )
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10 * mm))

    # Executive KPI Summary Tiles
    kpi_data = [
        [
            Paragraph(f"<font size='15'><b>{insp.get('rows', 0):,}</b></font><br/><font color='#64748B'>Rows</font>", _style("K1", alignment=TA_CENTER)),
            Paragraph(f"<font size='15'><b>{insp.get('columns', 0)}</b></font><br/><font color='#64748B'>Columns</font>", _style("K2", alignment=TA_CENTER)),
            Paragraph(f"<font size='15'><b>{insp.get('missing_total', 0)}</b></font><br/><font color='#64748B'>Fixed Nulls</font>", _style("K3", alignment=TA_CENTER)),
            Paragraph(f"<font size='15'><b>{insp.get('duplicates', 0)}</b></font><br/><font color='#64748B'>Duplicates</font>", _style("K4", alignment=TA_CENTER)),
        ]
    ]
    kpi_t = Table(kpi_data, colWidths=["25%"]*4)
    kpi_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#BFDBFE")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(kpi_t)
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 2 — DATASET OVERVIEW
    # ════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Dataset Overview & Profile", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=3*mm))

    t_info = insp.get("target_info", {})
    ov_data = [
        ["Audit Metric", "Discovered Value"],
        ["Total Row Count", f"{insp.get('rows', 0):,}"],
        ["Total Column Count", str(insp.get("columns", 0))],
        ["Numerical Feature Columns", f"{len(insp.get('numerical_cols', []))} ({', '.join(insp.get('numerical_cols', [])[:5])}...)"],
        ["Categorical Feature Columns", f"{len(insp.get('categorical_cols', []))} ({', '.join(insp.get('categorical_cols', [])[:5])}...)"],
        ["Identified High-Cardinality IDs", f"{len(insp.get('id_cols', []))} ({', '.join(insp.get('id_cols', [])) or 'None'})"],
        ["Auto-Detected Target Column", f"{insp.get('target_column')} (Unique Values: {t_info.get('unique_count', 'N/A')})"],
        ["Formulated Problem Type", str(insp.get("problem_type", "Classification")).title()],
        ["Missing Cells / Null Values", f"{insp.get('missing_total', 0)} total across {len(insp.get('missing_cols', {}))} columns"],
        ["Duplicate Row Records", f"{insp.get('duplicates', 0)} exact rows"],
    ]
    ov_tbl = Table([[Paragraph(c, table_cell_bold if i == 0 else table_cell) for c in row] for i, row in enumerate(ov_data)], colWidths=["38%", "62%"])
    ov_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 4.5),
    ]))
    story.append(ov_tbl)
    story.append(Spacer(1, 4 * mm))

    # Top Numerical Stats Table
    stats = insp.get("stats", {})
    if stats:
        story.append(Paragraph("Numerical Attributes Distribution Summary", _style("SubSec", fontSize=11, fontName="Helvetica-Bold", textColor=NAVY, spaceAfter=2*mm)))
        st_data = [["Feature", "Mean", "Median", "Std Dev", "Min", "Max", "Skewness"]]
        for col, s in list(stats.items())[:12]:
            st_data.append([
                str(col)[:20],
                f"{s.get('mean', 0):.2f}",
                f"{s.get('median', 0):.2f}",
                f"{s.get('std', 0):.2f}",
                f"{s.get('min', 0):.2f}",
                f"{s.get('max', 0):.2f}",
                f"{s.get('skewness', 0):.2f}",
            ])
        st_tbl = Table([[Paragraph(c, table_cell_bold if i == 0 else table_cell) for c in row] for i, row in enumerate(st_data)], colWidths=["28%", "12%", "12%", "12%", "12%", "12%", "12%"])
        st_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 3.5),
        ]))
        story.append(st_tbl)

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 3 — CLEANING REPORT
    # ════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Data Hygiene & Cleaning Report", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=3*mm))

    sh_data = [
        ["Phase", "Dataset Dimensions", "Status"],
        ["Original Raw Ingestion", f"{clean.get('original_shape', (0,0))[0]:,} rows × {clean.get('original_shape', (0,0))[1]} cols", "Ingested"],
        ["Cleaned & Standardized", f"{clean.get('cleaned_shape', (0,0))[0]:,} rows × {clean.get('cleaned_shape', (0,0))[1]} cols", "Optimized"],
        ["Residual Null Cells", f"{clean.get('nulls_remaining', 0)} nulls remaining", "Ready for ML"],
    ]
    sh_tbl = Table([[Paragraph(c, table_cell_bold if i == 0 else table_cell) for c in row] for i, row in enumerate(sh_data)], colWidths=["30%", "45%", "25%"])
    sh_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(sh_tbl)
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("Automated Pipeline Steps Executed:", _style("StpHeader", fontSize=11, fontName="Helvetica-Bold", textColor=NAVY, spaceAfter=3*mm)))
    steps = clean.get("steps_applied", [])
    if steps:
        step_rows = []
        for idx, stp in enumerate(steps, 1):
            step_rows.append([Paragraph(f"<b>Step {idx}</b>", table_cell_bold), Paragraph(f"✅ {stp}", table_cell)])
        stp_tbl = Table(step_rows, colWidths=["15%", "85%"])
        stp_tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT]),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(stp_tbl)
    else:
        story.append(Paragraph("Dataset conformed to high quality thresholds upon arrival — zero transformations needed.", body_style))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # PAGES 4-5 — EDA CHARTS (2 charts per page)
    # ════════════════════════════════════════════════════════════════════════
    # Collect candidate chart image objects
    all_charts = chart_images if chart_images is not None else (chart_paths or result.get("chart_paths", []))
    valid_charts = []
    for item in all_charts:
        if isinstance(item, (bytes, io.BytesIO)):
            valid_charts.append(item)
        elif isinstance(item, str):
            if os.path.exists(item):
                valid_charts.append(item)
            elif item.startswith("http://") or item.startswith("https://"):
                try:
                    import requests
                    r = requests.get(item, timeout=8)
                    if r.status_code == 200:
                        valid_charts.append(io.BytesIO(r.content))
                except Exception:
                    pass

    chart_names = [
        "Feature Numerical Distributions",
        "Feature Correlation Heatmap",
        "Target Variable Distribution",
        "Categorical Signals vs Target",
        "Feature Boxplots by Target Class",
        "Outlier Detection & IQR Boundaries",
        "Missing Values Heatmap Matrix",
        "Top Predictive Feature Importances",
    ]

    for idx, c_item in enumerate(valid_charts):
        if idx % 2 == 0:
            if idx > 0:
                story.append(PageBreak())
            page_num = 4 + (idx // 2)
            story.append(Paragraph(f"Exploratory Data Analysis — Visual Insights (Part {idx//2 + 1})", section_style))
            story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=2*mm))

        c_name = chart_names[idx] if idx < len(chart_names) else f"Visual Analysis Chart {idx+1}"
        story.append(Paragraph(f"<b>Figure {idx+1}: {c_name}</b>", body_bold))
        try:
            img_src = io.BytesIO(c_item) if isinstance(c_item, bytes) else c_item
            img = Image(img_src, width=170 * mm, height=92 * mm)
            story.append(img)
            caption_ref = f"Figure {idx+1}" if not isinstance(c_item, str) else os.path.basename(c_item)
            story.append(Paragraph(f"Chart automatically generated by AnchorIQ EDA Engine ({caption_ref})", caption_style))
        except Exception as e:
            story.append(Paragraph(f"[Image Rendering Failed: {e}]", body_style))
        story.append(Spacer(1, 2 * mm))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 6 — ML RESULTS
    # ════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Machine Learning Benchmarking & Performance", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=3*mm))

    is_reg = ml.get("problem_type") == "regression"
    score_col_name = "R² Score" if is_reg else "ROC AUC"
    best_name = ml.get("best_model", "Random Forest")

    m_rows = [
        ["Model Architecture", score_col_name, "Evaluation Status"],
        ["Logistic Regression / Ridge", f"{ml.get('lr_score', 0):.4f}", "Baseline Linear"],
        ["Random Forest (Ensemble)", f"{ml.get('rf_score', 0):.4f}", "Bagging Non-Linear"],
        ["XGBoost (Gradient Boosted)", f"{ml.get('xgb_score', 0):.4f}", "Gradient Boosting"],
    ]

    tbl_rows = []
    for i, r in enumerate(m_rows):
        tbl_rows.append([Paragraph(c, table_cell_bold if i == 0 else table_cell) for c in r])
    ml_tbl = Table(tbl_rows, colWidths=["45%", "25%", "30%"])

    t_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]

    # Highlight best model row green
    for idx_r, row in enumerate(m_rows[1:], 1):
        if best_name.lower() in row[0].lower() or (best_name.lower().startswith("xgb") and "xgboost" in row[0].lower()):
            t_styles.append(("BACKGROUND", (0, idx_r), (-1, idx_r), colors.HexColor("#D1FAE5")))
            t_styles.append(("TEXTCOLOR", (0, idx_r), (-1, idx_r), colors.HexColor("#065F46")))

    ml_tbl.setStyle(TableStyle(t_styles))
    story.append(ml_tbl)
    story.append(Spacer(1, 5 * mm))

    # Details
    details_data = [
        ["Cross-Validation Mean (5-Fold)", f"{ml.get('cv_mean', 0):.4f} (± {ml.get('cv_std', 0):.4f})"],
        ["Training / Test Sample Split", f"{ml.get('train_samples', 0):,} train / {ml.get('test_samples', 0):,} test"],
        ["Engineered Feature Count", f"{ml.get('n_features', 0)} active dimensions"],
        ["Optimal Decision Threshold", f"{ml.get('optimal_threshold', 0.5)} (Cost-optimized: FN=500, FP=50)" if not is_reg else "N/A (Continuous Regression)"],
    ]
    det_tbl = Table([[Paragraph(k, table_cell_bold), Paragraph(v, table_cell)] for k, v in details_data], colWidths=["45%", "55%"])
    det_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 4.5),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
    ]))
    story.append(det_tbl)
    story.append(Spacer(1, 5 * mm))

    # Confusion Matrix
    cm = ml.get("confusion_matrix", [])
    if cm and len(cm) == 2:
        story.append(Paragraph("Holdout Evaluation Confusion Matrix:", _style("CMH", fontSize=10, fontName="Helvetica-Bold", textColor=NAVY, spaceAfter=2*mm)))
        cm_tbl_data = [
            ["", "Predicted Negative (0)", "Predicted Positive (1)"],
            ["Actual Negative (0)", f"TN: {cm[0][0]:,}", f"FP: {cm[0][1]:,}"],
            ["Actual Positive (1)", f"FN: {cm[1][0]:,}", f"TP: {cm[1][1]:,}"],
        ]
        cm_tbl = Table([[Paragraph(c, table_cell_bold if (i == 0 or j == 0) else table_cell) for j, c in enumerate(r)] for i, r in enumerate(cm_tbl_data)], colWidths=["34%", "33%", "33%"])
        cm_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(cm_tbl)

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 7 — AI INSIGHTS
    # ════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Strategic Business Intelligence & Insights", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=4*mm))

    if insights_text:
        lines = insights_text.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str:
                story.append(Spacer(1, 1.5 * mm))
                continue
            if line_str.startswith("## "):
                sec_head = line_str[3:].strip()
                story.append(Paragraph(f"<b>{sec_head}</b>", _style("AiHead", fontSize=12, fontName="Helvetica-Bold", textColor=NAVY, spaceBefore=3.5*mm, spaceAfter=1.5*mm)))
            elif line_str.startswith("### "):
                sub_head = line_str[4:].strip()
                story.append(Paragraph(f"<b>{sub_head}</b>", _style("AiSub", fontSize=10, fontName="Helvetica-Bold", textColor=TEAL, spaceBefore=2*mm, spaceAfter=1*mm)))
            elif line_str.startswith(("- ", "* ")):
                bullet_text = line_str[2:].strip()
                story.append(Paragraph(f"<font color='#0EA5E9'>&bull;</font> {bullet_text}", _style("AiBul", fontSize=9, fontName="Helvetica", textColor=DARK, leading=13, leftIndent=8)))
            elif line_str[0].isdigit() and (line_str[1:3] in [". ", ") "]):
                num_text = line_str[2:].strip() if line_str[1] in [".", ")"] else line_str[3:].strip()
                num_idx = line_str.split()[0]
                story.append(Paragraph(f"<b>{num_idx}</b> {num_text}", _style("AiNum", fontSize=9, fontName="Helvetica", textColor=DARK, leading=13, leftIndent=8)))
            else:
                story.append(Paragraph(line_str, body_style))
    else:
        story.append(Paragraph("AI-generated business analysis not available for this run.", body_style))

    # Build PDF with custom NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    if in_memory:
        target_dest.seek(0)
        return target_dest.getvalue()
    return output_path
