"""
Global theme and styling for Tabular ML Lab.

Modern glassmorphism design system with Inter typography.
Inject via inject_custom_css() at the top of each page.
"""
import streamlit as st


def inject_custom_css():
    """Inject global design system CSS."""
    st.markdown("""
    <style>
    /* ── Load Inter Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── CSS Variables ───────────────────────────────────────── */
    :root {
        --accent: #667eea;
        --accent-dark: #5a6fd6;
        --accent-light: #8b9cf7;
        --accent-glow: rgba(102, 126, 234, 0.25);
        --accent-subtle: rgba(102, 126, 234, 0.08);
        --success: #22c55e;
        --success-bg: #f0fdf4;
        --warning: #f59e0b;
        --warning-bg: #fffbeb;
        --error: #ef4444;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --surface: rgba(255, 255, 255, 0.72);
        --surface-solid: #ffffff;
        --surface-raised: rgba(255, 255, 255, 0.85);
        --border: rgba(148, 163, 184, 0.2);
        --border-hover: rgba(102, 126, 234, 0.3);
        --bg-page: #f1f5f9;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.04), 0 1px 3px rgba(0,0,0,0.06);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.04), 0 2px 4px rgba(0,0,0,0.06);
        --shadow-lg: 0 10px 25px rgba(0,0,0,0.06), 0 4px 10px rgba(0,0,0,0.04);
        --shadow-glow: 0 0 20px var(--accent-glow);
    }

    /* ── Global ──────────────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background: var(--bg-page) !important;
    }

    /* Smoother text rendering */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* ── Hide Streamlit Chrome ───────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    /* Hide deploy button */
    .stDeployButton { display: none !important; }
    /* Hide "Made with Streamlit" */
    .viewerBadge_container__r5tak { display: none !important; }

    /* ── Typography ──────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em;
    }
    h1 { font-weight: 800 !important; }
    h2 { font-weight: 700 !important; font-size: 1.5rem !important; }
    h3 { font-weight: 600 !important; font-size: 1.2rem !important; }
    /* Apply Inter to text elements only — avoid overriding Streamlit's
       Material Symbols Rounded icon font used for expander arrows, etc. */
    p, li, label, td, th, caption,
    .stMarkdown, .stText, .stCaption,
    .stSelectbox label, .stMultiSelect label,
    .stRadio label, .stCheckbox label,
    .stTextInput label, .stNumberInput label,
    .stSlider label, .stTextArea label,
    button, input, textarea, select {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Main Content Area ───────────────────────────────────── */
    .block-container {
        padding-top: 2rem !important;
        max-width: 1200px !important;
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown a {
        color: var(--accent-light) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
    }
    /* Sidebar navigation links */
    section[data-testid="stSidebar"] .stPageLink,
    section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
        border-radius: var(--radius-sm) !important;
        transition: background 0.15s ease !important;
    }
    section[data-testid="stSidebar"] .stPageLink:hover,
    section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {
        background: rgba(255,255,255,0.06) !important;
    }
    /* Sidebar selectbox / inputs */
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
    }
    /* Sidebar checkbox */
    section[data-testid="stSidebar"] .stCheckbox label span {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
    /* Sidebar progress bar */
    section[data-testid="stSidebar"] .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent) 0%, var(--accent-light) 100%) !important;
        border-radius: 4px !important;
    }
    section[data-testid="stSidebar"] .stProgress > div {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 4px !important;
    }
    /* Sidebar caption */
    section[data-testid="stSidebar"] .stCaption, 
    section[data-testid="stSidebar"] small {
        color: #64748b !important;
    }

    /* ── Cards / Containers ──────────────────────────────────── */
    .info-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: white;
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-lg), 0 0 30px rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg), 0 0 40px rgba(102, 126, 234, 0.25);
    }
    .info-card h3 { color: white !important; margin-top: 0; font-size: 1.05rem; font-weight: 600; }
    .info-card p { color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem; line-height: 1.5; }

    .glass-card {
        background: var(--surface);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        border-color: var(--border-hover);
    }

    .guidance-card {
        background: var(--surface);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-left: 3px solid var(--accent);
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        font-size: 0.9rem;
        line-height: 1.65;
        box-shadow: var(--shadow-sm);
    }
    .guidance-card strong { color: var(--text-primary); }

    .warning-card {
        background: var(--warning-bg);
        border-left: 3px solid var(--warning);
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-sm);
    }

    .success-card {
        background: var(--success-bg);
        border-left: 3px solid var(--success);
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-sm);
    }

    /* ── Metric Cards ────────────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: var(--surface);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.25rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--border-hover);
    }
    .metric-value {
        font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace !important;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent) !important;
        margin: 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }
    .metric-ci {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }

    /* ── Streamlit Native Metric Override ─────────────────────── */
    div[data-testid="stMetric"] {
        background: var(--surface);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1rem;
        box-shadow: var(--shadow-sm);
        transition: border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--border-hover);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted) !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--accent) !important;
    }

    /* ── Section Dividers ────────────────────────────────────── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
        margin: 2rem 0 !important;
    }

    /* ── Buttons ─────────────────────────────────────────────── */
    .stButton > button {
        border-radius: var(--radius-sm) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.01em;
        transition: all 0.2s ease !important;
        border: 1px solid var(--border) !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: var(--shadow-sm), 0 0 0 0 var(--accent-glow) !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md), 0 0 20px var(--accent-glow) !important;
    }
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {
        background: var(--surface) !important;
        backdrop-filter: blur(8px);
        color: var(--text-primary) !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background: var(--surface-raised) !important;
        border-color: var(--border-hover) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Inputs ──────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border) !important;
        background: var(--surface-solid) !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }

    /* ── Select boxes ────────────────────────────────────────── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border) !important;
        transition: border-color 0.2s ease !important;
    }
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover {
        border-color: var(--border-hover) !important;
    }

    /* ── Expanders ───────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        border-radius: var(--radius-sm) !important;
    }
    details[data-testid="stExpander"] {
        background: var(--surface) !important;
        backdrop-filter: blur(8px);
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-sm);
        overflow: hidden;
    }
    details[data-testid="stExpander"] summary {
        padding: 0.8rem 1rem !important;
    }
    details[data-testid="stExpander"][open] {
        border-color: var(--border-hover) !important;
    }
    /* Expander toggle icon — ensure visible on both light and dark backgrounds */
    details[data-testid="stExpander"] summary svg {
        color: var(--text-secondary) !important;
        fill: var(--text-secondary) !important;
    }
    /* Sidebar expander icons need light color */
    section[data-testid="stSidebar"] details[data-testid="stExpander"] summary svg {
        color: #94a3b8 !important;
        fill: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] details[data-testid="stExpander"] {
        background: rgba(255,255,255,0.04) !important;
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ── DataFrames / Tables ─────────────────────────────────── */
    .stDataFrame {
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-sm) !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Tabs ────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background: var(--surface) !important;
        border-radius: var(--radius-md) !important;
        padding: 4px !important;
        border: 1px solid var(--border) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface-solid) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* ── Progress Bar ────────────────────────────────────────── */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent) 0%, var(--accent-light) 100%) !important;
        border-radius: 4px !important;
    }
    .stProgress > div {
        background: var(--accent-subtle) !important;
        border-radius: 4px !important;
    }

    /* ── Alerts / Info boxes ─────────────────────────────────── */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border: none !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* ── Code blocks ─────────────────────────────────────────── */
    code {
        font-family: 'JetBrains Mono', 'SF Mono', monospace !important;
        font-size: 0.85em !important;
    }

    /* ── Section Headers ─────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border);
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.4rem;
    }

    /* ── Reviewer Concern Badge ──────────────────────────────── */
    .reviewer-concern {
        background: var(--warning-bg);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: var(--radius-md);
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        box-shadow: var(--shadow-sm);
    }
    .reviewer-concern::before {
        content: "⚠️ Reviewer concern: ";
        font-weight: 600;
    }

    /* ── Step Indicator / Breadcrumb ─────────────────────────── */
    .step-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--surface);
        backdrop-filter: blur(8px);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 0.4rem 1.1rem;
        font-size: 0.82rem;
        color: var(--accent);
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
    }

    /* ── Workflow Stepper (sidebar) ──────────────────────────── */
    .workflow-step {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.4rem 0;
        position: relative;
    }
    .workflow-step-number {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.75rem;
        flex-shrink: 0;
    }
    .workflow-step-active .workflow-step-number {
        background: var(--accent);
        color: white;
        box-shadow: 0 0 12px var(--accent-glow);
    }
    .workflow-step-complete .workflow-step-number {
        background: var(--success);
        color: white;
    }
    .workflow-step-pending .workflow-step-number {
        background: rgba(255,255,255,0.08);
        color: #64748b;
    }

    /* ── Tooltips / Why badges ───────────────────────────────── */
    .why-badge {
        display: inline-block;
        background: var(--accent-subtle);
        color: var(--accent);
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        font-size: 0.72rem;
        font-weight: 500;
        cursor: help;
        margin-left: 0.25rem;
    }

    /* ── Plotly chart containers ─────────────────────────────── */
    .stPlotlyChart {
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    /* ── Scrollbar ───────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.3);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.5); }

    /* ── Hero Section (home page only) ───────────────────────── */
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-container {
        background: linear-gradient(-45deg, #667eea, #764ba2, #5b86e5, #36d1dc);
        background-size: 400% 400%;
        animation: gradient-shift 15s ease infinite;
        border-radius: var(--radius-xl);
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg), 0 0 60px rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(255,255,255,0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.15) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(255,255,255,0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero-container h1 {
        color: white !important;
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.03em !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.15);
        margin: 0 0 0.5rem 0 !important;
        position: relative;
    }
    .hero-container .hero-sub {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        line-height: 1.6;
        max-width: 550px;
        margin: 0 auto;
        position: relative;
        font-weight: 400;
    }
    .hero-container .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.78rem;
        color: rgba(255,255,255,0.9);
        font-weight: 500;
        margin-bottom: 1rem;
        position: relative;
        letter-spacing: 0.02em;
    }

    /* ── Workflow Steps (home page) ──────────────────────────── */
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 0.75rem 1rem;
        border-radius: var(--radius-md);
        transition: background 0.15s ease;
        margin: 0.15rem 0;
    }
    .step-item:hover {
        background: var(--accent-subtle);
    }
    .step-num {
        min-width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent), var(--accent-dark));
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.82rem;
        flex-shrink: 0;
        box-shadow: 0 2px 8px var(--accent-glow);
    }
    .step-content strong {
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    .step-content span {
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* ── Sidebar Stepper ─────────────────────────────────────── */
    .sidebar-step {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.3rem 0;
        font-size: 0.85rem;
    }
    .sidebar-step-done {
        color: #4ade80 !important;
    }
    .sidebar-step-pending {
        color: #475569 !important;
    }
    .sidebar-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .sidebar-dot-done {
        background: #4ade80;
        box-shadow: 0 0 6px rgba(74, 222, 128, 0.4);
    }
    .sidebar-dot-pending {
        background: rgba(255,255,255,0.12);
    }
    </style>
    """, unsafe_allow_html=True)


def render_guidance(text: str, icon: str = "💡"):
    """Render a guidance card with actionable advice."""
    st.markdown(f"""
    <div class="guidance-card">
        {icon} {text}
    </div>
    """, unsafe_allow_html=True)


def render_reviewer_concern(text: str):
    """Render a reviewer concern badge."""
    st.markdown(f"""
    <div class="reviewer-concern">{text}</div>
    """, unsafe_allow_html=True)


def render_step_indicator(step_number: int, step_name: str, total_steps: int = 9):
    """Render a step indicator breadcrumb badge."""
    st.markdown(f"""
    <div class="step-indicator">
        Step {step_number} of {total_steps} · {step_name}
    </div>
    """, unsafe_allow_html=True)


def render_info_card(title: str, body: str):
    """Render a glassmorphism info card."""
    st.markdown(f"""
    <div class="info-card">
        <h3>{title}</h3>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


def render_glass_card(content: str):
    """Render a glass-effect card with arbitrary HTML content."""
    st.markdown(f'<div class="glass-card">{content}</div>', unsafe_allow_html=True)


def render_metric_card(label: str, value: str, ci: str = ""):
    """Render a single metric card with optional CI."""
    ci_html = f'<div class="metric-ci">{ci}</div>' if ci else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {ci_html}
    </div>
    """


def render_metric_row(metrics: list):
    """Render a row of metric cards.

    metrics: list of (label, value, ci_text) tuples
    """
    cards = "".join(render_metric_card(l, v, c) for l, v, c in metrics)
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)
