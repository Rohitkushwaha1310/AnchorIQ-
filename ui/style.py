"""
AnchorIQ UI System — Minimalist, Human-Made Editorial Design System.
Defines design tokens, helper functions, and custom CSS overrides strictly avoiding neon, violet, glow, and gradients.
"""

import streamlit as st

# ── DESIGN TOKENS ─────────────────────────────────────────────────────────────
PAGE_BG = "#F2F1EC"
SURFACE = "#FFFFFF"
BORDER = "#E7E4DB"
PRIMARY_TEXT = "#16201C"
SECONDARY_TEXT = "#707670"
ACCENT = "#1F6B58"
ACCENT_SOFT = "#E4EFEA"

SIDEBAR_BG = "#14201C"
SIDEBAR_TEXT = "#C9D1CC"
SIDEBAR_MUTED = "#7E8A84"
SIDEBAR_ACTIVE = "#1F2E29"
SIDEBAR_BUTTON = "#2E8A71"

WARNING = "#B4532A"
WARNING_SOFT = "#F6E7DF"
CAUTION = "#B8892B"
NEUTRAL_CHIP = "#EFEDE6"
METER_TRACK = "#ECE9E0"
NEUTRAL_BAR = "#8A9A92"

CHART_PALETTE = [
    "#1F6B58",
    "#B8892B",
    "#B4532A",
    "#6F7C5E",
    "#8A8F88",
    "#4B3F36",
]


def get_custom_css() -> str:
    """Return the unified minimalist stylesheet matching editorial design standards."""
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* ── Typography & Global Reset ── */
    html, body, [class*="css"], .stApp {{
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: {PAGE_BG} !important;
        color: {PRIMARY_TEXT} !important;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }}

    /* Hide Streamlit default chrome */
    header, footer, [data-testid="stToolbar"], #MainMenu, .stDeployButton {{
        visibility: hidden !important;
        display: none !important;
    }}

    /* ── Main Layout & Containers ── */
    [data-testid="stAppViewContainer"] {{
        background-color: {PAGE_BG} !important;
    }}

    [data-testid="block-container"] {{
        padding: 36px 44px 54px 44px !important;
        max-width: 1280px !important;
        margin: 0 auto !important;
    }}

    /* ── Headings & Titles ── */
    h1, .page-title {{
        font-size: 30px !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: {PRIMARY_TEXT} !important;
        margin: 0 0 8px 0 !important;
        line-height: 1.2 !important;
        text-align: left !important;
    }}

    h2, h3, .section-title {{
        font-size: 15px !important;
        font-weight: 600 !important;
        color: {PRIMARY_TEXT} !important;
        letter-spacing: -0.01em !important;
        margin: 16px 0 10px 0 !important;
        text-align: left !important;
    }}

    h4, h5, h6 {{
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: {PRIMARY_TEXT} !important;
        margin: 12px 0 6px 0 !important;
    }}

    p, span, label, div {{
        font-size: 14.5px;
        color: {PRIMARY_TEXT};
        line-height: 1.5;
    }}

    a {{
        color: {ACCENT} !important;
        text-decoration: none !important;
    }}

    a:hover {{
        text-decoration: underline !important;
    }}

    code, pre {{
        font-family: 'IBM Plex Mono', monospace !important;
        background-color: {NEUTRAL_CHIP} !important;
        color: {PRIMARY_TEXT} !important;
        border-radius: 6px !important;
    }}

    /* ── Dark Minimalist Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG} !important;
        border-right: 1px solid #1F2E29 !important;
        box-shadow: none !important;
        padding-top: 1.5rem !important;
    }}

    [data-testid="stSidebar"] * {{
        color: {SIDEBAR_TEXT} !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] strong {{
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: #1F2E29 !important;
        margin: 18px 0 !important;
    }}

    [data-testid="stSidebar"] [data-baseweb="radio"] div {{
        border-color: #3B5249 !important;
    }}

    [data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {{
        background-color: {SIDEBAR_BUTTON} !important;
        border-color: {SIDEBAR_BUTTON} !important;
    }}

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] div[data-baseweb="select"] {{
        background-color: {SIDEBAR_ACTIVE} !important;
        border: 1px solid #2B3D36 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }}

    .sidebar-brand {{
        padding: 4px 0 16px 0;
        border-bottom: 1px solid #1F2E29;
        margin-bottom: 20px;
    }}

    .sidebar-brand-name {{
        font-size: 18px;
        font-weight: 600;
        color: #FFFFFF !important;
        letter-spacing: -0.01em;
        margin: 0;
    }}

    .sidebar-brand-sub {{
        font-size: 12px;
        color: {SIDEBAR_MUTED} !important;
        margin-top: 2px;
        font-weight: 400;
    }}

    /* ── Clean Hero / Page Header ── */
    .hero-editorial {{
        padding: 4px 0 20px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid {BORDER};
        text-align: left;
    }}

    .hero-editorial .hero-title {{
        font-size: 30px !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: {PRIMARY_TEXT} !important;
        margin: 0 0 6px 0 !important;
        line-height: 1.2 !important;
    }}

    .hero-editorial .hero-sub {{
        font-size: 14.5px !important;
        color: {SECONDARY_TEXT} !important;
        font-weight: 400 !important;
        margin: 0 !important;
        line-height: 1.5 !important;
    }}

    /* ── Cards & Surfaces ── */
    .clean-card, .panel-3d {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 2px rgba(22, 32, 28, 0.04) !important;
        padding: 20px 22px !important;
        margin-bottom: 16px !important;
    }}

    /* ── KPI Grid & Metrics ── */
    .kpi-grid, .kpi-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 16px;
        margin: 16px 0 20px 0;
    }}

    .kpi-card, .kpi-box {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 2px rgba(22, 32, 28, 0.04) !important;
        padding: 18px 20px !important;
        text-align: left !important;
        transition: border-color 0.15s ease;
    }}

    .kpi-card:hover, .kpi-box:hover {{
        border-color: #D3CEBE !important;
    }}

    .kpi-lbl, .kpi-box-label {{
        font-size: 12px !important;
        font-weight: 500 !important;
        color: {SECONDARY_TEXT} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        margin-bottom: 6px !important;
    }}

    .kpi-val, .kpi-box-value {{
        font-size: 30px !important;
        font-weight: 600 !important;
        color: {PRIMARY_TEXT} !important;
        line-height: 1.1 !important;
        font-feature-settings: "tnum" 1 !important;
    }}

    .kpi-sub, .kpi-box-subtext {{
        font-size: 12px !important;
        font-weight: 500 !important;
        color: {SECONDARY_TEXT} !important;
        margin-top: 4px !important;
    }}

    /* Native Streamlit Metric Overrides */
    [data-testid="stMetric"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 2px rgba(22, 32, 28, 0.04) !important;
        padding: 16px 18px !important;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 28px !important;
        font-weight: 600 !important;
        color: {PRIMARY_TEXT} !important;
        font-feature-settings: "tnum" 1 !important;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 12px !important;
        font-weight: 500 !important;
        color: {SECONDARY_TEXT} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }}

    /* ── Reliability Card ── */
    .reliability-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 4px solid {ACCENT};
        border-radius: 14px;
        padding: 18px 22px;
        margin: 16px 0 24px 0;
        box-shadow: 0 1px 2px rgba(22, 32, 28, 0.04);
    }}

    .reliability-card.verdict-strong {{
        border-left-color: {ACCENT};
    }}

    .reliability-card.verdict-moderate {{
        border-left-color: {CAUTION};
    }}

    .reliability-card.verdict-weak {{
        border-left-color: {WARNING};
    }}

    .reliability-meta {{
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: {SECONDARY_TEXT};
        margin-bottom: 4px;
    }}

    .reliability-heading {{
        font-size: 16px;
        font-weight: 600;
        color: {PRIMARY_TEXT};
        margin: 0 0 6px 0;
    }}

    .reliability-desc {{
        font-size: 14px;
        color: {SECONDARY_TEXT};
        margin: 0;
        line-height: 1.5;
    }}

    /* ── Minimalist Clean Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent !important;
        border-bottom: 1px solid {BORDER} !important;
        gap: 4px !important;
        margin-bottom: 20px !important;
        padding-bottom: 0 !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {SECONDARY_TEXT} !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        padding: 10px 16px !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        transition: color 0.15s ease, border-color 0.15s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {PRIMARY_TEXT} !important;
        background-color: {ACCENT_SOFT} !important;
        border-radius: 6px 6px 0 0 !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: {PRIMARY_TEXT} !important;
        font-weight: 600 !important;
        border-bottom: 2px solid {ACCENT} !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }}

    /* ── Interactive Elements & Buttons ── */
    button[kind="primary"], .stButton > button[kind="primary"], .stFormSubmitButton > button {{
        background-color: {ACCENT} !important;
        color: #FFFFFF !important;
        border: 1px solid {ACCENT} !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 8px 18px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease;
    }}

    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {{
        background-color: #185546 !important;
        border-color: #185546 !important;
        color: #FFFFFF !important;
    }}

    button[kind="secondary"], .stButton > button, .stButton > button[kind="secondary"] {{
        background-color: {SURFACE} !important;
        color: {PRIMARY_TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 8px 18px !important;
        box-shadow: none !important;
    }}

    button[kind="secondary"]:hover, .stButton > button:hover {{
        background-color: {PAGE_BG} !important;
        border-color: #D3CEBE !important;
        color: {PRIMARY_TEXT} !important;
    }}

    /* Form Inputs */
    input, select, textarea, div[data-baseweb="select"] {{
        background-color: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        color: {PRIMARY_TEXT} !important;
        font-size: 14px !important;
    }}

    input:focus, textarea:focus, div[data-baseweb="select"]:focus-within {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 1px {ACCENT} !important;
    }}

    /* Radios & Checkboxes */
    [data-baseweb="radio"] input:checked + div,
    [data-baseweb="checkbox"] input:checked + div {{
        background-color: {ACCENT} !important;
        border-color: {ACCENT} !important;
    }}

    [data-baseweb="radio"] div,
    [data-baseweb="checkbox"] div {{
        border-color: {SECONDARY_TEXT} !important;
    }}

    /* Sliders & Progress */
    .stProgress > div > div > div > div {{
        background-color: {ACCENT} !important;
    }}

    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {ACCENT} !important;
        border-color: {ACCENT} !important;
    }}

    .stSlider [data-baseweb="slider"] div[data-testid="stSliderTickBar"] {{
        background-color: {ACCENT} !important;
    }}

    /* File Uploader */
    [data-testid="stFileUploader"] {{
        background-color: {SURFACE} !important;
        border: 1px dashed #D3CEBE !important;
        border-radius: 14px !important;
        padding: 16px !important;
        box-shadow: 0 1px 2px rgba(22, 32, 28, 0.04) !important;
    }}

    [data-testid="stFileUploader"]:hover {{
        border-color: {ACCENT} !important;
    }}

    /* Tables & DataFrames */
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(22, 32, 28, 0.03) !important;
        margin-bottom: 12px !important;
    }}

    /* Chart Containers */
    .chart-container, .chart-frame {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 1px 2px rgba(22, 32, 28, 0.04) !important;
    }}

    /* Executive Insights Box */
    .insight-container {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(22, 32, 28, 0.04);
        line-height: 1.65;
        color: {PRIMARY_TEXT};
    }}

    .insight-container h1, .insight-container h2, .insight-container h3, .insight-container h4 {{
        color: {PRIMARY_TEXT} !important;
        font-weight: 600 !important;
        margin-top: 18px !important;
        margin-bottom: 8px !important;
        border-bottom: 1px solid {BORDER} !important;
        padding-bottom: 6px !important;
    }}

    .insight-container strong {{
        color: {ACCENT} !important;
        font-weight: 600 !important;
    }}

    .insight-container ul, .insight-container ol {{
        margin: 10px 0 14px 20px !important;
        color: {PRIMARY_TEXT} !important;
    }}

    .insight-container li {{
        margin-bottom: 6px !important;
        color: {PRIMARY_TEXT} !important;
    }}

    /* Alerts */
    .stAlert {{
        border-radius: 10px !important;
        border: 1px solid {BORDER} !important;
        font-size: 13.5px !important;
        background-color: {SURFACE} !important;
        color: {PRIMARY_TEXT} !important;
    }}

    /* Chips & Badges */
    .chip {{
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.02em;
    }}

    .chip-accent {{
        background-color: {ACCENT_SOFT};
        color: {ACCENT};
        border: 1px solid rgba(31, 107, 88, 0.25);
    }}

    .chip-warning {{
        background-color: {WARNING_SOFT};
        color: {WARNING};
        border: 1px solid rgba(180, 83, 42, 0.25);
    }}

    .chip-neutral {{
        background-color: {NEUTRAL_CHIP};
        color: {PRIMARY_TEXT};
        border: 1px solid {BORDER};
    }}

    /* Results Callout */
    .result-callout {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
        box-shadow: 0 1px 2px rgba(22, 32, 28, 0.04);
    }}

    .callout-sig {{
        border-left: 4px solid {ACCENT};
    }}

    .callout-nonsig {{
        border-left: 4px solid {WARNING};
    }}

    .callout-caution {{
        border-left: 4px solid {CAUTION};
    }}

    /* Empty state */
    .empty-state-box {{
        background: {SURFACE};
        border: 1px dashed #D3CEBE;
        border-radius: 14px;
        padding: 48px 32px;
        text-align: center;
        margin: 24px 0;
    }}

    .empty-state-title {{
        font-size: 16px;
        font-weight: 600;
        color: {PRIMARY_TEXT};
        margin-bottom: 6px;
    }}

    .empty-state-sub {{
        font-size: 13.5px;
        color: {SECONDARY_TEXT};
    }}
    """


def inject_custom_css():
    """Inject the unified design system into Streamlit app."""
    st.markdown(f"<style>{get_custom_css()}</style>", unsafe_allow_html=True)


def render_header(title: str, subtitle: str) -> None:
    """Render a clean editorial header without gradients or emojis."""
    st.markdown(
        f"""
        <div class="hero-editorial">
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_header(title: str = "AnchorIQ", subtitle: str = "Autonomous Analytics") -> None:
    """Render a dark minimalist sidebar brand element without emojis or icons."""
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-name">{title}</div>
            <div class="sidebar-brand-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reliability_card(verdict: str, winner: str, why_won: str, ensemble_details: str = None) -> None:
    """Render the model reliability verification card."""
    v_clean = str(verdict or "moderate").lower()
    if "strong" in v_clean:
        v_class = "verdict-strong"
        v_label = "High Reliability"
    elif "weak" in v_clean:
        v_class = "verdict-weak"
        v_label = "Low Reliability"
    else:
        v_class = "verdict-moderate"
        v_label = "Moderate Reliability"

    details_html = f"<div style='margin-top: 6px; font-size: 13px; color: {SECONDARY_TEXT};'>{ensemble_details}</div>" if ensemble_details else ""

    st.markdown(
        f"""
        <div class="reliability-card {v_class}">
            <div class="reliability-meta">Model Validation &bull; {v_label}</div>
            <div class="reliability-heading">Champion: {winner}</div>
            <p class="reliability-desc">{why_won}</p>
            {details_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str = "No dataset imported", subtitle: str = "Upload a CSV, Excel, or JSON file to begin analysis.") -> None:
    """Render clean minimalist empty state."""
    st.markdown(
        f"""
        <div class="empty-state-box">
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
