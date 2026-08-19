"""Theme definitions and CSS generation for the Streamlit control-room dashboard.

One static CSS template (below) is populated with CSS custom-property values
drawn from the active theme dictionary, so all four themes share a single
stylesheet implementation rather than four duplicated ones.
"""
from __future__ import annotations

import streamlit as st

DEFAULT_THEME = "Industrial Slate"

THEMES: dict[str, dict[str, str]] = {
    "Industrial Slate": {
        "background": "#161B22",
        "sidebar": "#1C222B",
        "card": "#222A35",
        "card_elevated": "#283240",
        "border": "#364152",
        "border_strong": "#455467",
        "text_primary": "#F3F6FA",
        "text_secondary": "#AAB6C5",
        "text_muted": "#78879A",
        "accent": "#55B8E8",
        "accent_soft": "rgba(85, 184, 232, 0.12)",
        "healthy": "#4FD17B",
        "healthy_soft": "rgba(79, 209, 123, 0.10)",
        "warning": "#F2B84B",
        "warning_soft": "rgba(242, 184, 75, 0.10)",
        "critical": "#F26464",
        "critical_soft": "rgba(242, 100, 100, 0.10)",
        "grid": "rgba(170, 182, 197, 0.10)",
        "shadow": "rgba(0, 0, 0, 0.20)",
    },
    "Deep Navy": {
        "background": "#0F172A",
        "sidebar": "#111C31",
        "card": "#1E293B",
        "card_elevated": "#263449",
        "border": "#334155",
        "border_strong": "#475569",
        "text_primary": "#F8FAFC",
        "text_secondary": "#CBD5E1",
        "text_muted": "#94A3B8",
        "accent": "#38BDF8",
        "accent_soft": "rgba(56, 189, 248, 0.12)",
        "healthy": "#22C55E",
        "healthy_soft": "rgba(34, 197, 94, 0.10)",
        "warning": "#F59E0B",
        "warning_soft": "rgba(245, 158, 11, 0.10)",
        "critical": "#EF4444",
        "critical_soft": "rgba(239, 68, 68, 0.10)",
        "grid": "rgba(203, 213, 225, 0.10)",
        "shadow": "rgba(0, 0, 0, 0.24)",
    },
    "Light Operations": {
        "background": "#F3F6F9",
        "sidebar": "#E8EDF3",
        "card": "#FFFFFF",
        "card_elevated": "#F8FAFC",
        "border": "#D7DEE7",
        "border_strong": "#BBC6D2",
        "text_primary": "#17212B",
        "text_secondary": "#52606D",
        "text_muted": "#7B8794",
        "accent": "#1976D2",
        "accent_soft": "rgba(25, 118, 210, 0.09)",
        "healthy": "#198754",
        "healthy_soft": "rgba(25, 135, 84, 0.09)",
        "warning": "#D97706",
        "warning_soft": "rgba(217, 119, 6, 0.09)",
        "critical": "#DC3545",
        "critical_soft": "rgba(220, 53, 69, 0.09)",
        "grid": "rgba(82, 96, 109, 0.12)",
        "shadow": "rgba(23, 33, 43, 0.08)",
    },
    "Steel Blue": {
        "background": "#18232D",
        "sidebar": "#202E3A",
        "card": "#263746",
        "card_elevated": "#304657",
        "border": "#405666",
        "border_strong": "#536B7C",
        "text_primary": "#F1F5F7",
        "text_secondary": "#B8C5CE",
        "text_muted": "#8699A6",
        "accent": "#4DB6AC",
        "accent_soft": "rgba(77, 182, 172, 0.12)",
        "healthy": "#52C878",
        "healthy_soft": "rgba(82, 200, 120, 0.10)",
        "warning": "#E8AD46",
        "warning_soft": "rgba(232, 173, 70, 0.10)",
        "critical": "#E85D5D",
        "critical_soft": "rgba(232, 93, 93, 0.10)",
        "grid": "rgba(184, 197, 206, 0.10)",
        "shadow": "rgba(0, 0, 0, 0.18)",
    },
}

THEME_NAMES: list[str] = list(THEMES.keys())

# Semantic theme key -> CSS custom property name.
_CSS_VARS: list[tuple[str, str]] = [
    ("background", "--scada-bg"),
    ("sidebar", "--scada-sidebar"),
    ("card", "--scada-card"),
    ("card_elevated", "--scada-card-elevated"),
    ("border", "--scada-border"),
    ("border_strong", "--scada-border-strong"),
    ("text_primary", "--scada-text-primary"),
    ("text_secondary", "--scada-text-secondary"),
    ("text_muted", "--scada-text-muted"),
    ("accent", "--scada-accent"),
    ("accent_soft", "--scada-accent-soft"),
    ("healthy", "--scada-healthy"),
    ("healthy_soft", "--scada-healthy-soft"),
    ("warning", "--scada-warning"),
    ("warning_soft", "--scada-warning-soft"),
    ("critical", "--scada-critical"),
    ("critical_soft", "--scada-critical-soft"),
    ("grid", "--scada-grid"),
    ("shadow", "--scada-shadow"),
]

REQUIRED_THEME_KEYS: frozenset[str] = frozenset(key for key, _ in _CSS_VARS)


def get_theme(name: str | None) -> dict[str, str]:
    """Return the named theme's palette, falling back safely to the default."""
    return THEMES.get(name, THEMES[DEFAULT_THEME])


def _root_variables(theme: dict[str, str]) -> str:
    declarations = "\n".join(f"    {css_var}: {theme[key]};" for key, css_var in _CSS_VARS)
    return f":root {{\n{declarations}\n}}"


_STATIC_CSS = """
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--scada-bg);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1360px; }

html, body, [class*="css"] { color: var(--scada-text-primary); }
p, span, label, div { color: inherit; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background-color: var(--scada-sidebar);
    border-right: 1px solid var(--scada-border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }
[data-testid="stSidebar"] label { color: var(--scada-text-secondary); }

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: var(--scada-card);
    border: 1px solid var(--scada-border);
    border-radius: 10px;
    padding: .9rem 1.1rem;
    box-shadow: 0 1px 3px var(--scada-shadow);
}
[data-testid="stMetricLabel"] { color: var(--scada-text-secondary); font-size: .8rem; letter-spacing: .03em; }
[data-testid="stMetricValue"] { color: var(--scada-text-primary); font-size: 1.55rem; font-weight: 700; }
[data-testid="stMetricDelta"] { color: var(--scada-accent) !important; }

/* ---------- Expanders ---------- */
[data-testid="stExpander"] {
    background: var(--scada-card);
    border: 1px solid var(--scada-border);
    border-radius: 10px;
}
[data-testid="stExpander"] summary { color: var(--scada-text-primary); font-weight: 600; }

/* ---------- Tabs ---------- */
[data-testid="stTabs"] button[role="tab"] {
    color: var(--scada-text-secondary);
    font-weight: 600;
    letter-spacing: .01em;
    font-size: .92rem;
}
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--scada-accent); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--scada-accent); }
[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: var(--scada-border); }

/* ---------- Buttons ---------- */
.stButton button, .stDownloadButton button {
    background: var(--scada-card-elevated);
    border: 1px solid var(--scada-border-strong);
    color: var(--scada-text-primary);
    border-radius: 8px;
    font-weight: 600;
}
.stButton button:hover:not(:disabled), .stDownloadButton button:hover:not(:disabled) {
    border-color: var(--scada-accent);
    color: var(--scada-accent);
}
.stButton button:disabled { color: var(--scada-text-muted); opacity: .55; }

/* ---------- Select / dropdown widgets ---------- */
[data-baseweb="select"] > div {
    background-color: var(--scada-card-elevated) !important;
    border-color: var(--scada-border-strong) !important;
    color: var(--scada-text-primary) !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] input { color: var(--scada-text-primary) !important; }
[data-baseweb="select"] span { color: var(--scada-text-primary) !important; }
[data-baseweb="select"] svg { fill: var(--scada-text-secondary) !important; }
[data-baseweb="select"]:hover > div { border-color: var(--scada-accent) !important; }
[data-testid="stWidgetLabel"] p { color: var(--scada-text-secondary); }

div[data-baseweb="popover"] ul[role="listbox"],
div[data-baseweb="popover"] div[data-baseweb="menu"] {
    background-color: var(--scada-card) !important;
    border: 1px solid var(--scada-border) !important;
}
li[role="option"] { color: var(--scada-text-primary) !important; }
li[role="option"]:hover, li[aria-selected="true"] { background-color: var(--scada-card-elevated) !important; }

/* ---------- Radio / checkbox ---------- */
[data-testid="stRadio"] label span, [data-testid="stCheckbox"] label span { color: var(--scada-text-primary); }

/* ---------- Dataframes ---------- */
[data-testid="stDataFrame"] { border: 1px solid var(--scada-border); border-radius: 10px; overflow: hidden; }

hr { border-color: var(--scada-border); margin: 1.1rem 0; }
[data-testid="stCaptionContainer"] p, .stCaption { color: var(--scada-text-muted) !important; }

/* ---------- Section headings (main content) ---------- */
.scada-eyebrow {
    color: var(--scada-text-primary);
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: .01em;
    margin: 20px 0 10px 0;
}
.scada-eyebrow:first-child { margin-top: 0; }

/* ---------- Sidebar micro-labels ---------- */
.scada-eyebrow-sub {
    color: var(--scada-text-muted);
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: .04em;
    margin: 16px 0 6px 0;
}
.scada-eyebrow-sub:first-child { margin-top: 0; }

/* ---------- Sidebar: product identity ---------- */
.scada-nav-title {
    color: var(--scada-text-primary);
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: .01em;
    margin: 0;
    line-height: 1.2;
}
.scada-nav-subtitle {
    color: var(--scada-text-secondary);
    font-size: .78rem;
    margin: 2px 0 14px 0;
}

/* ---------- Sidebar: navigation rail ---------- */
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 2px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 10px;
    border-radius: 8px;
    width: 100%;
    margin: 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: var(--scada-card-elevated); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: var(--scada-accent-soft);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: var(--scada-accent);
    font-weight: 700;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input { accent-color: var(--scada-accent); }

/* ---------- Sidebar: system status ---------- */
.scada-sidebar-status {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--scada-text-primary);
    font-size: .84rem;
    font-weight: 600;
    margin: 18px 0 10px 0;
}
.scada-sidebar-status .dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--scada-healthy);
    box-shadow: 0 0 0 3px var(--scada-healthy-soft);
}

/* ---------- Summary cards (Model & System page) ---------- */
.scada-summary-card .heading { font-size: 1.05rem; font-weight: 800; color: var(--scada-text-primary); margin-bottom: 10px; }
.scada-summary-card .row { display: flex; justify-content: space-between; gap: 12px; padding: 6px 0; border-bottom: 1px solid var(--scada-border); font-size: .86rem; }
.scada-summary-card .row:last-child { border-bottom: none; }
.scada-summary-card .row .rl { color: var(--scada-text-secondary); }
.scada-summary-card .row .rv { color: var(--scada-text-primary); font-weight: 700; text-align: right; }

/* ---------- Checklist (leakage-safe inputs) ---------- */
.scada-checklist { list-style: none; margin: 8px 0 0 0; padding: 0; }
.scada-checklist li {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
    color: var(--scada-text-primary);
    font-size: .88rem;
}
.scada-checklist .mark { font-weight: 800; width: 16px; text-align: center; }
.scada-checklist-approved .mark { color: var(--scada-healthy); }
.scada-checklist-excluded .mark { color: var(--scada-critical); }

/* ---------- Header / title ---------- */
.scada-title {
    font-size: 2.05rem;
    font-weight: 800;
    letter-spacing: .01em;
    color: var(--scada-text-primary);
    margin: 0;
    line-height: 1.2;
}
.scada-title .accent { color: var(--scada-accent); }
.scada-subtitle {
    color: var(--scada-text-secondary);
    font-size: .95rem;
    margin-top: 2px;
}

.scada-badge {
    display: inline-block;
    padding: .25rem .65rem;
    border-radius: 999px;
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    border: 1px solid var(--scada-border);
    color: var(--scada-text-secondary);
    background: var(--scada-card);
}
.scada-badge-online { color: var(--scada-healthy); border-color: var(--scada-healthy); background: var(--scada-healthy-soft); }

/* ---------- Status strip ---------- */
.scada-status-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    background: var(--scada-card);
    border: 1px solid var(--scada-border);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 14px 0 20px 0;
}
.scada-status-item { flex: 1 1 150px; min-width: 130px; }
.scada-status-item .k { color: var(--scada-text-muted); font-size: .72rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }
.scada-status-item .v { color: var(--scada-text-primary); font-size: .98rem; font-weight: 700; margin-top: 3px; }
.scada-status-item .v.on { color: var(--scada-healthy); }

/* ---------- KPI cards ---------- */
.scada-card-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.scada-card {
    flex: 1 1 180px;
    min-width: 160px;
    border-radius: 12px;
    padding: 16px 18px;
    border: 1px solid var(--scada-border);
    background: var(--scada-card);
    box-shadow: 0 1px 4px var(--scada-shadow);
}
.scada-card .k { color: var(--scada-text-secondary); font-size: .74rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
.scada-card .v { font-size: 1.65rem; font-weight: 800; margin-top: 6px; color: var(--scada-text-primary); }
.scada-card .s { font-size: .8rem; color: var(--scada-text-secondary); margin-top: 4px; }
.scada-card-normal { border-color: var(--scada-healthy); background: linear-gradient(180deg, var(--scada-healthy-soft), var(--scada-card) 65%); }
.scada-card-normal .v { color: var(--scada-healthy); }
.scada-card-warning { border-color: var(--scada-warning); background: linear-gradient(180deg, var(--scada-warning-soft), var(--scada-card) 65%); }
.scada-card-warning .v { color: var(--scada-warning); }
.scada-card-critical { border-color: var(--scada-critical); background: linear-gradient(180deg, var(--scada-critical-soft), var(--scada-card) 65%); }
.scada-card-critical .v { color: var(--scada-critical); }
.scada-card-info { border-color: var(--scada-border); }
.scada-card-info .v { color: var(--scada-accent); }

/* ---------- AI risk banner ---------- */
.scada-banner {
    border-radius: 12px;
    padding: 18px 22px;
    border: 1px solid var(--scada-border);
    background: var(--scada-card);
    margin: 6px 0 20px 0;
}
.scada-banner-normal { border-color: var(--scada-healthy); background: linear-gradient(180deg, var(--scada-healthy-soft), var(--scada-card) 55%); }
.scada-banner-warning { border-color: var(--scada-warning); background: linear-gradient(180deg, var(--scada-warning-soft), var(--scada-card) 55%); }
.scada-banner-critical { border-color: var(--scada-critical); background: linear-gradient(180deg, var(--scada-critical-soft), var(--scada-card) 55%); }
.scada-banner .eyebrow-label { font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--scada-text-muted); margin-bottom: 4px; }
.scada-banner .title { font-size: 1.1rem; font-weight: 800; letter-spacing: .01em; color: var(--scada-text-primary); margin-bottom: 8px; }
.scada-banner-normal .title { color: var(--scada-healthy); }
.scada-banner-warning .title { color: var(--scada-warning); }
.scada-banner-critical .title { color: var(--scada-critical); }
.scada-banner .msg { color: var(--scada-text-primary); font-size: .94rem; margin-bottom: 12px; }
.scada-banner .stats { display: flex; gap: 28px; flex-wrap: wrap; font-size: .84rem; color: var(--scada-text-secondary); }
.scada-banner .stats b { color: var(--scada-text-primary); font-size: .92rem; }

/* ---------- Pipeline schematic ---------- */
.scada-panel-card {
    background: var(--scada-card);
    border: 1px solid var(--scada-border);
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 4px;
}
.scada-pipeline { display: flex; align-items: center; gap: 6px; overflow-x: auto; padding: 6px 0; }
.scada-seg { flex: 0 0 auto; text-align: center; }
.scada-seg .box {
    min-width: 84px;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid var(--scada-border);
    background: var(--scada-card-elevated);
    font-size: .85rem;
    font-weight: 700;
    color: var(--scada-text-secondary);
}
.scada-seg .box.normal { border-color: var(--scada-healthy); color: var(--scada-healthy); background: var(--scada-healthy-soft); }
.scada-seg .box.warning { border-color: var(--scada-warning); color: var(--scada-warning); background: var(--scada-warning-soft); }
.scada-seg .box.critical { border-color: var(--scada-critical); color: var(--scada-critical); background: var(--scada-critical-soft); }
.scada-seg .box.selected { outline: 2px solid var(--scada-accent); outline-offset: 1px; }
.scada-seg .tag { font-size: .66rem; color: var(--scada-accent); font-weight: 700; letter-spacing: .05em; margin-top: 4px; visibility: hidden; }
.scada-seg .tag.show { visibility: visible; }
.scada-connector { flex: 0 0 auto; color: var(--scada-border-strong); font-size: 1.1rem; }

/* ---------- Fault intelligence ---------- */
.scada-fault-card {
    border-radius: 12px;
    border: 1px solid var(--scada-border);
    background: var(--scada-card);
    padding: 16px 20px;
    margin-bottom: 12px;
}
.scada-fault-card .name { font-size: 1.15rem; font-weight: 800; color: var(--scada-warning); letter-spacing: .02em; }
.scada-fault-card .conf { color: var(--scada-text-secondary); font-size: .85rem; margin-top: 4px; }
.scada-fault-card .desc { color: var(--scada-text-primary); font-size: .9rem; margin-top: 8px; }

/* ---------- Architecture card ---------- */
.scada-arch {
    background: var(--scada-card);
    border: 1px solid var(--scada-border);
    border-radius: 12px;
    padding: 22px 26px;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: .88rem;
    line-height: 2;
    color: var(--scada-text-secondary);
    text-align: center;
    white-space: pre-wrap;
}
.scada-arch b { color: var(--scada-accent); }
"""


def get_dashboard_css(theme: dict[str, str]) -> str:
    """Build the complete stylesheet for one theme: variables + shared static rules."""
    return f"<style>\n{_root_variables(theme)}\n{_STATIC_CSS}\n</style>"


def inject_global_styles(theme: dict[str, str]) -> None:
    """Apply the active theme's CSS once per render."""
    st.markdown(get_dashboard_css(theme), unsafe_allow_html=True)
