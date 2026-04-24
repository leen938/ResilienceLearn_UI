# streamlit_app_ui_only.py
# ResilienceLearn — student check-in dashboard.
# Run: pip install -r requirements.txt
#      uvicorn backend.app:app --reload --app-dir .
#      python -m streamlit run streamlit_app_ui_only.py
#
# Set RESILIENCE_API_BASE to your API (default http://127.0.0.1:8000).
# Set RESILIENCELEARN_ADMIN=1 to show technical / developer panels.

from __future__ import annotations

import html
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.ui_mapping import ui_checkin_to_predict_body  # noqa: E402

st.set_page_config(
    page_title="ResilienceLearn",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# App mode & config
# ---------------------------------------------------------------------------
ADMIN_MODE: bool = os.environ.get("RESILIENCELEARN_ADMIN", "").lower() in (
    "1",
    "true",
    "yes",
)
DEFAULT_API_BASE = os.environ.get("RESILIENCE_API_BASE", "http://127.0.0.1:8000").rstrip(
    "/"
)

# ---------------------------------------------------------------------------
# Theme: colors, copy (single place for student-facing strings)
# ---------------------------------------------------------------------------
# Student-facing palette (original navy / teal / semantic accents)
T = {
    "page_bg": "#F5F7FB",
    "sidebar_bg": "#FAFBFD",
    "card_bg": "#FFFFFF",
    "primary": "#1E3A5F",
    "primary_dark": "#152A45",
    "secondary": "#1E3A5F",
    "accent": "#0D9488",
    "accent_teal": "#0D9488",
    "accent_amber": "#D97706",
    "accent_green": "#059669",
    "accent_red": "#DC2626",
    "warning": "#D97706",
    "danger": "#DC2626",
    "heading": "#1E3A5F",
    "body": "#475569",
    "muted": "#64748B",
    "text_muted": "#64748B",
    "border": "#E2E8F0",
    "neutral_bg": "#F5F7FB",
    "chip_bg": "#F0F2F6",
    "card_radius": "20px",
    "card_shadow": "0 1px 3px rgba(30, 58, 95, 0.06)",
    # Primary buttons, tabs, sliders: same as accent_amber; hover/pressed one step darker
    "control_accent": "#D97706",
    "control_accent_dark": "#B45309",
}

COPY = {
    "subtitle": "A quick, private check-in to understand your study and well-being patterns—"
    "and get ideas that may help you stay on track.",
    "reassurance": "Your answers are used only in this session to build your summary, unless your school sets up saving differently.",
    "hero_badge": "Private • Student support check-in",
    "hero_title": "ResilienceLearn",
    "connect_error": "We could not connect to the service. Please check your connection and try again, or try again in a few minutes.",
    "connect_error_admin": "Could not reach the API. Set RESILIENCE_API_BASE or start the server: uvicorn backend.app:app --reload --app-dir .",
    "interpret_on_track": "This check-in suggests you are currently **on track** academically, based on what you shared.",
    "interpret_attention": "This check-in suggests you may **benefit from extra support** right now. That is a signal to notice—not a label about you.",
    "interpret_moderate": "Your result is in a **middle range**—worth noticing a few small changes that could help you feel more steady.",
    "uncertainty_plain": "The score is less certain (your answers were closer to a middle zone). Use it as one signal alongside how you actually feel day to day.",
    "suggestions_header": "Ideas for you",
    "factors_header": "What may be influencing your result",
    "factors_caption": "These factors stood out most for this check-in. Small steps in one area can still help.",
    "summary_header": "Your check-in summary",
    "progress_header": "Your progress over time",
    "empty_history": "When you save a check-in, you will see your history and a simple trend here.",
    "one_point_history": "You have one saved check-in. Save another to see a trend over time.",
    "insights_title": "Insights",
    "insights_blurb": "Deeper views use the same check-in you entered in the sidebar. "
    "Run an insight when you want to see **what may be driving** your result.",
    "support_title": "Support",
    "support_blurb": "Short supportive suggestions based on what you type—not therapy or a crisis service. If you are in danger, contact local emergency or crisis lines.",
    "chat_privacy": "What you type here stays in this browser session unless your organization configures it otherwise.",
    "include_context": "Use my last check-in summary to personalize replies",
    "insight_drivers_button": "See what may be driving my result",
    "insight_common_button": "See common factors across many check-ins",
    "no_factors": "No factor breakdown is available for this check-in yet. Try again after your connection is back.",
    "admin_chat_ctx": "Prediction context (diagnostic, optional attachment)",
    "tab_checkin": "Check-in",
    "tab_insights": "Insights",
    "tab_support": "Support",
    "run_cta": "Update my results",
    "save_cta": "Save this check-in",
    "sidebar_title": "Your check-in",
    "academic": "Academic habits",
    "attendance": "Attendance",
    "wellbeing": "Well-being",
    "context": "Your context",
    "crisis_caption": "Context where you live and study (sliders; all optional to think about as a whole).",
    "update_ok": "Your summary is up to date.",
    "update_fail": "We could not refresh your summary. Try again shortly.",
    "save_ok": "Check-in saved. You can review it below in your progress section.",
    "save_fail": "Save needs a working connection and a completed check-in. Try again when the service is available.",
    "y_axis_risk": "Level of concern (0 = lower, 1 = higher)",
    "x_checkins": "Check-in number",
    "table_date": "Date",
    "table_status": "Status",
    "table_score": "Score",
    "table_note": "Summary",
    "trend_up": "Trend: higher concern than before",
    "trend_down": "Trend: lower concern than before",
    "trend_flat": "Trend: similar to your last check-in",
    "latest": "Latest",
    "n_checkins": "Check-ins saved",
    "trend_dir": "Recent direction",
    "chat_unavailable": "We could not reach support just now. Please try again in a moment.",
}


def is_admin() -> bool:
    return ADMIN_MODE


def get_api_base() -> str:
    if is_admin():
        return str(st.session_state.get("api_base_admin_input", DEFAULT_API_BASE)).rstrip(
            "/"
        )
    return DEFAULT_API_BASE


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "prediction_context" not in st.session_state:
    st.session_state.prediction_context = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "support_draft_key" not in st.session_state:
    st.session_state.support_draft_key = 0
if "explain_cache" not in st.session_state:
    st.session_state.explain_cache = None
if "importance_cache" not in st.session_state:
    st.session_state.importance_cache = None
def format_api_error_technical(exc: BaseException) -> str:
    if isinstance(exc, requests.exceptions.HTTPError):
        r = exc.response
        if r is not None:
            try:
                body = r.json()
                detail = body.get("detail", body)
            except Exception:
                detail = (r.text or "")[:400]
            return f"HTTP {r.status_code}: {detail}"
    if isinstance(exc, requests.exceptions.RequestException):
        return COPY["connect_error_admin"] if is_admin() else COPY["connect_error"]
    return f"{type(exc).__name__}: {exc}"


def format_api_error_user(exc: BaseException) -> str:
    if is_admin():
        return format_api_error_technical(exc)
    if isinstance(exc, requests.exceptions.RequestException):
        return COPY["connect_error"]
    if isinstance(exc, requests.exceptions.HTTPError):
        return "The service could not complete this request. Please try again in a moment."
    return "Something went wrong. Please try again in a few minutes."


def call_predict(api_base: str, payload: dict) -> dict:
    url = api_base.rstrip("/") + "/predict"
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def call_feature_importance(api_base: str) -> dict:
    url = api_base.rstrip("/") + "/feature-importance"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def call_explain(api_base: str, payload: dict) -> dict:
    url = api_base.rstrip("/") + "/explain"
    r = requests.post(url, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


def call_support(api_base: str, message: str, context: dict | None) -> dict:
    url = api_base.rstrip("/") + "/chat/support"
    r = requests.post(
        url,
        json={"message": message, "context": context},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def apply_plot_style() -> None:
    plt.rcParams["figure.facecolor"] = T["page_bg"]
    plt.rcParams["axes.facecolor"] = T["card_bg"]
    plt.rcParams["axes.edgecolor"] = T["border"]
    plt.rcParams["axes.labelcolor"] = T["body"]
    plt.rcParams["text.color"] = T["heading"]
    plt.rcParams["xtick.color"] = T["muted"]
    plt.rcParams["ytick.color"] = T["muted"]
    plt.rcParams["font.size"] = 10
    # Do not set figure.tight_layout here — removed as a valid rcParam in newer matplotlib.
    # Figures call plt.tight_layout() where needed.


# ---------------------------------------------------------------------------
# Suggestions: categorized for student UI
# ---------------------------------------------------------------------------
def suggestions_categorized(inputs: dict) -> dict[str, list[str]]:
    """Group tips for display (student-friendly)."""
    out: dict[str, list[str]] = {
        "Study habits & learning": [],
        "Sleep & routine": [],
        "Attendance & courses": [],
        "Well-being & support": [],
    }
    if inputs["electricity_hours"] < 8:
        out["Study habits & learning"].append(
            "Plan two short focus blocks when you have power (e.g. 25 minutes, then a short break)."
        )
    if inputs["internet_quality"] < 6:
        out["Study habits & learning"].append(
            "When internet is shaky, download readings ahead of time and keep offline notes."
        )
    if inputs["sleep_quality"] < 6:
        out["Sleep & routine"].append(
            "Try a steady sleep and wake time for a week, and ease off caffeine in the late afternoon."
        )
    if inputs["absences"] > 6:
        out["Attendance & courses"].append(
            "Pick one course to prioritize showing up to for the next two weeks—small steps count."
        )
    if inputs["financial_stress"] > 7:
        out["Well-being & support"].append(
            "Look for campus or community support for basics; prioritize meals and rest where you can."
        )
    if inputs["motivation"] < 5:
        out["Study habits & learning"].append(
            "Start with a 10-minute ‘starter’ task to make beginning easier (tiny goals build momentum)."
        )
    if inputs["war_stress"] > 7 or inputs["mental_health"] < 5:
        out["Well-being & support"].append(
            "Consider talking to a counselor, student services, or someone you trust."
        )
    if not any(out.values()):
        out["Well-being & support"].append(
            "Keep a simple weekly check-in with yourself: what went okay, and one thing to try next week."
        )
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# Risk presentation helpers
# ---------------------------------------------------------------------------
def status_tone(
    risk_label: int | None, risk_p: float | None
) -> tuple[str, str, str]:
    """
    Return (css_accent, friendly_status, level_key).
    level_key: low | moderate | high
    """
    if risk_label is None or risk_p is None:
        return T["muted"], "—", "low"

    p = float(risk_p)
    if risk_label == 0 and p < 0.4:
        return T["accent_green"], "On track", "low"
    if risk_label == 0:
        return T["accent_teal"], "Mostly on track", "moderate"
    if p < 0.55:
        return T["accent_amber"], "Needs attention", "moderate"
    return T["accent_red"], "Needs attention", "high"


def confidence_phrase(uncertainty_score: float) -> str | None:
    if uncertainty_score >= 0.25:
        return "The result is a bit less certain—your answers were closer to a middle zone, so use this as one signal among many."
    if uncertainty_score >= 0.15:
        return "Confidence in this score is fairly typical for check-ins like yours."
    return None


def interpret_sentence(
    risk_label: int | None, risk_p: float | None, level: str
) -> str:
    if risk_label is None or risk_p is None:
        return "Complete your check-in to see a short interpretation here."
    if level == "low":
        return COPY["interpret_on_track"]
    if level == "moderate":
        return COPY["interpret_moderate"]
    return COPY["interpret_attention"]


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    st.markdown(
        f"""
<style>
  html, body, [data-testid="stAppViewContainer"] {{
    font-family: "Segoe UI", "Inter", system-ui, -apple-system, sans-serif;
  }}
  [data-testid="stAppViewContainer"] .main .block-container {{
    padding-top: 1.75rem; padding-bottom: 3.5rem; max-width: 1120px;
    background: linear-gradient(180deg, {T["neutral_bg"]} 0%, #eef1f6 100%) !important;
  }}
  [data-testid="stAppViewContainer"] .main .block-container > * {{
    margin-bottom: 0.2rem;
  }}
  [data-testid="stAppViewContainer"] [data-testid="stHeader"] {{
    background: {T["neutral_bg"]};
  }}
  div[data-testid="stSidebar"] {{
    background: {T["sidebar_bg"]} !important;
    border-right: 1px solid {T["border"]};
  }}
  section[data-testid="stSidebar"] > div {{
    background: {T["sidebar_bg"]} !important;
  }}
  div[data-testid="stToolbar"] button {{
    color: {T["muted"]};
  }}
  .stButton > button, div[data-testid="stButton"] > button, button[kind] {{
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
    border: 1px solid {T["border"]} !important;
    color: {T["heading"]} !important;
    background: {T["card_bg"]} !important;
  }}
  .stButton > button[kind="primary"] {{
    background: {T["control_accent"]} !important;
    color: #ffffff !important;
    border: 1px solid {T["control_accent"]} !important;
  }}
  .stButton > button[kind="primary"]:hover,
  .stButton > button[kind="primary"]:active {{
    background: {T["control_accent_dark"]} !important;
    border-color: {T["control_accent_dark"]} !important;
    color: #ffffff !important;
  }}
  .stButton > button[kind="primary"]:focus-visible {{
    box-shadow: 0 0 0 2px {T["card_bg"]}, 0 0 0 4px {T["control_accent"]} !important;
  }}
  /* Sliders: thumb and track tints = same orange as primary buttons (accent_amber family) */
  [data-baseweb="slider"] [role="slider"] {{
    background: {T["control_accent"]} !important;
    border: 2px solid #ffffff !important;
    box-shadow: 0 0 0 1px {T["control_accent"]} !important;
  }}
  [data-baseweb="slider"] [role="slider"]::before,
  [data-baseweb="slider"] [role="slider"]::after {{ color: {T["body"]} !important; }}
  div[data-testid="stSlider"] [data-baseweb="slider"] > div {{
    background: rgba(217, 119, 6, 0.16) !important;
    border-radius: 999px;
  }}
  [data-baseweb="select"] > div, [data-baseweb="input"] input {{
    border-color: {T["border"]} !important;
    color: {T["body"]} !important;
  }}
  [data-baseweb="checkbox"] [role="checkbox"] {{
    background: {T["card_bg"]} !important;
    border-color: {T["border"]} !important;
  }}
  [data-baseweb="checkbox"] [data-state="checked"] [role="checkbox"] {{
    background: {T["primary"]} !important; border: none !important;
  }}
  .stExpander, details {{
    background: {T["card_bg"]} !important;
    border: 1px solid {T["border"]} !important;
    border-radius: 14px !important;
  }}
  .stExpander [data-baseweb="accordion"] {{
    border: none;
  }}
  [data-baseweb="tab-list"] {{ gap: 8px; background: {T["card_bg"]}; padding: 8px; border-radius: 16px; border: 1px solid {T["border"]}; box-shadow: {T["card_shadow"]}; }}
  [data-baseweb="tab"] {{
    border-radius: 12px; padding: 0.5rem 1.15rem; font-weight: 600; color: {T["body"]} !important;
  }}
  [data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, {T["control_accent"]} 0%, {T["control_accent_dark"]} 100%) !important;
    color: #ffffff !important;
    border: 1px solid {T["control_accent_dark"]} !important;
    box-shadow: 0 1px 3px rgba(180, 83, 9, 0.28) !important;
  }}
  [data-baseweb="tab"][aria-selected="true"]:hover, [data-baseweb="tab"][aria-selected="true"]:active {{
    background: {T["control_accent_dark"]} !important;
    color: #ffffff !important;
    border-color: {T["control_accent_dark"]} !important;
  }}
  [data-baseweb="tab"][aria-selected="true"] * {{
    color: #ffffff !important;
  }}
  [data-baseweb="tab-panel"] {{ padding-top: 1.5rem; }}
  div[data-testid="stAlert"] {{
    border-radius: 14px !important;
    border: 1px solid {T["border"]} !important;
    color: {T["body"]} !important;
  }}
  .stErrorAlert, [data-baseweb="notification"] {{
    border-radius: 14px;
  }}
  .stInfo {{
    background: rgba(30, 58, 95, 0.04) !important;
    border: 1px solid rgba(30, 58, 95, 0.12) !important;
  }}
  [data-baseweb="notification"] {{ border-radius: 12px; }}
  /* Error / warning: calm surface, danger only as border accent (not full harsh fill) */
  div[data-testid="stNotificationContent"] p, div[data-testid="stAlert"] p
  {{ color: {T["body"]} !important; }}
  [data-baseweb="notification"] [class*="stError"] {{
    background: rgba(220, 38, 38, 0.08) !important;
    border: 1px solid rgba(220, 38, 38, 0.4) !important;
  }}
  [data-baseweb="notification"] [class*="stWarning"] {{
    background: rgba(217, 119, 6, 0.1) !important;
    border: 1px solid rgba(217, 119, 6, 0.45) !important;
  }}
  /* Progress: primary → teal (original app accent) */
  [data-baseweb="progress-bar"] {{ background: {T["border"]} !important; border-radius: 6px; }}
  [data-baseweb="progress-bar"] [role="progressbar"] {{ background: linear-gradient(90deg, {T["primary"]}, {T["accent_teal"]}) !important; border-radius: 6px; }}
  div[data-testid="stProgress"] [data-baseweb="progress-bar"] [role="progressbar"] + div
  {{ background: linear-gradient(90deg, {T["primary"]}, {T["accent_teal"]}) !important; }}
  @keyframes rl-hero-fade-up {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  @keyframes rl-hero-blob-in {{
    from {{ opacity: 0; }}
    to {{ opacity: 1; }}
  }}
  .rl-hero {{
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    border: 1px solid rgba(226, 232, 240, 0.95);
    margin-bottom: 2.25rem;
    padding: 0;
    isolation: isolate;
    background: linear-gradient(
      150deg,
      {T["card_bg"]} 0%,
      #f1f5fb 38%,
      #eef5f2 100%
    );
    box-shadow:
      0 1px 2px rgba(30, 58, 95, 0.04),
      0 10px 40px -12px rgba(30, 58, 95, 0.09);
  }}
  .rl-hero__blobs {{
    position: absolute;
    inset: 0;
    z-index: 0;
    pointer-events: none;
  }}
  .rl-hero__blob {{
    position: absolute;
    border-radius: 50%;
    filter: blur(56px);
    mix-blend-mode: multiply;
    animation: rl-hero-blob-in 1s ease-out 0s both;
  }}
  .rl-hero__blob--1 {{
    width: min(55vw, 320px);
    height: min(40vw, 220px);
    right: -8%;
    top: -25%;
    background: radial-gradient(ellipse at center, rgba(13, 148, 136, 0.22) 0%, rgba(13, 148, 136, 0) 70%);
  }}
  .rl-hero__blob--2 {{
    width: min(50vw, 280px);
    height: min(45vw, 200px);
    left: -12%;
    bottom: -30%;
    background: radial-gradient(ellipse at center, rgba(30, 58, 95, 0.14) 0%, rgba(30, 58, 95, 0) 70%);
  }}
  .rl-hero__inner {{
    position: relative;
    z-index: 1;
    padding: 1.75rem 2rem 1.65rem 2rem;
    max-width: 46rem;
  }}
  .rl-hero__badge {{
    display: inline-flex;
    align-items: center;
    margin: 0 0 0.9rem 0;
    padding: 0.35rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {T["primary"]};
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(30, 58, 95, 0.1);
    border-radius: 999px;
    box-shadow: 0 1px 2px rgba(30, 58, 95, 0.05);
    animation: rl-hero-fade-up 0.7s cubic-bezier(0.22, 1, 0.36, 1) 0.08s both;
  }}
  .rl-hero__title {{
    color: {T["heading"]};
    font-size: 2.15rem;
    font-weight: 700;
    margin: 0 0 0.65rem 0;
    letter-spacing: -0.03em;
    line-height: 1.12;
    text-wrap: balance;
    animation: rl-hero-fade-up 0.75s cubic-bezier(0.22, 1, 0.36, 1) 0.14s both;
  }}
  .rl-hero p.lead {{
    color: {T["body"]};
    font-size: 1.125rem;
    margin: 0 0 0.7rem 0;
    line-height: 1.65;
    font-weight: 400;
    letter-spacing: -0.01em;
    animation: rl-hero-fade-up 0.75s cubic-bezier(0.22, 1, 0.36, 1) 0.22s both;
  }}
  .rl-hero p.small {{
    color: {T["muted"]};
    font-size: 0.875rem;
    margin: 0;
    line-height: 1.55;
    max-width: 40rem;
    animation: rl-hero-fade-up 0.75s cubic-bezier(0.22, 1, 0.36, 1) 0.3s both;
  }}
  @media (prefers-reduced-motion: reduce) {{
    .rl-hero__blob, .rl-hero__badge, .rl-hero__title, .rl-hero p.lead, .rl-hero p.small {{
      animation: none !important;
    }}
  }}
  .rl-card {{
    background: {T["card_bg"]};
    border: 1px solid {T["border"]};
    border-radius: {T["card_radius"]};
    padding: 1.35rem 1.5rem 1.25rem 1.5rem;
    margin-bottom: 1.75rem;
    box-shadow: {T["card_shadow"]};
  }}
  .rl-card h3 {{ margin: 0 0 0.7rem 0; font-size: 1.08rem; color: {T["heading"]}; font-weight: 600; letter-spacing: -0.01em; }}
  .rl-metric-row {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 0.6rem; }}
  .rl-metric {{
    flex: 1 1 160px;
    min-width: 140px;
    background: {T["neutral_bg"]};
    border-radius: 16px;
    padding: 1rem 1.1rem;
    border: 1px solid {T["border"]};
    border-left: 4px solid {T["accent_teal"]};
  }}
  .rl-metric .lab {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: {T["muted"]}; margin-bottom: 0.25rem; font-weight: 600; }}
  .rl-metric .val {{ font-size: 1.42rem; font-weight: 700; color: {T["heading"]}; line-height: 1.2; }}
  .rl-metric .sub {{ font-size: 0.84rem; color: {T["body"]}; margin-top: 0.3rem; line-height: 1.4; }}
  .rl-pill {{
    display: inline-block; padding: 0.28rem 0.7rem; border-radius: 999px; font-size: 0.8rem;
    font-weight: 600; margin-right: 0.4rem; margin-bottom: 0.4rem; background: {T["neutral_bg"]};
    color: {T["primary"]}; border: 1px solid {T["border"]};
  }}
  p, .stCaption {{ color: {T["body"]}; }}
  .stCaption {{ color: {T["muted"]} !important; }}
  section.main [data-testid="stLabel"] p {{ color: {T["heading"]} !important; font-weight: 500; }}
</style>
        """,
        unsafe_allow_html=True,
    )


def card_metric_html(label: str, value: str, sub: str = "", border: str = T["accent_teal"]) -> str:
    return (
        f'<div class="rl-metric" style="border-left-color: {border};">'
        f'<div class="lab">{html.escape(label)}</div>'
        f'<div class="val">{html.escape(value)}</div>'
        f'<div class="sub">{html.escape(sub) if sub else "&nbsp;"}</div></div>'
    )


def render_hero() -> None:
    st.markdown(
        f"""
<div class="rl-hero" role="region" aria-label="Introduction">
  <div class="rl-hero__blobs" aria-hidden="true">
    <div class="rl-hero__blob rl-hero__blob--1"></div>
    <div class="rl-hero__blob rl-hero__blob--2"></div>
  </div>
  <div class="rl-hero__inner">
    <span class="rl-hero__badge">{html.escape(COPY["hero_badge"])}</span>
    <h1 class="rl-hero__title">{html.escape(COPY["hero_title"])}</h1>
    <p class="lead">{html.escape(COPY["subtitle"])}</p>
    <p class="small">{html.escape(COPY["reassurance"])}</p>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_result_summary(
    risk_p: float | None,
    risk_label: int | None,
    uncertainty_score: float,
    uncertainty_message: str,
) -> str:
    """Renders cards; returns level key for use elsewhere. Call only when API succeeded."""
    acc, friendly_status, level = status_tone(risk_label, risk_p)
    p_pct = f"{100 * float(risk_p):.0f}%" if risk_p is not None else "—"
    st.markdown(
        f'<div class="rl-card"><h3>{html.escape(COPY["summary_header"])}</h3></div>',
        unsafe_allow_html=True,
    )
    sub_status = "Based on this check-in only—not a judgment about you as a person."
    c_html = f'<div class="rl-metric-row">{card_metric_html("Status", friendly_status, sub_status, acc)}'
    c_html += card_metric_html(
        "Check-in score",
        p_pct,
        "Higher = more need for support in this snapshot",
        acc,
    )
    if is_admin() and risk_label is not None:
        c_html += card_metric_html(
            "Internal label",
            f"{risk_label}",
            "Admin: model class",
            T["muted"],
        )
    c_html += "</div>"
    st.markdown(c_html, unsafe_allow_html=True)

    interp = interpret_sentence(risk_label, risk_p, level)
    st.markdown(f"**What this means**  \n{interp}")
    st.caption(
        "We combine your answers into a single view of how much **extra support** could help right now. "
        "It is a snapshot, not a grade."
    )

    ph = confidence_phrase(uncertainty_score)
    if ph:
        st.info(ph)
    elif is_admin() and (uncertainty_message or "").strip():
        st.caption(uncertainty_message)

    st.progress(
        min(1.0, max(0.0, float(risk_p) if risk_p is not None else 0.0)),
    )
    return level


def render_guidance(suggestions: dict[str, list[str]]) -> None:
    st.markdown(
        f'<div class="rl-card"><h3>{html.escape(COPY["suggestions_header"])}</h3></div>',
        unsafe_allow_html=True,
    )
    for cat, items in suggestions.items():
        st.markdown(f"**{html.escape(cat)}**")
        for it in items:
            st.markdown(f"- {it}")


def render_top_factors_badge(top_factors: list) -> None:
    if not top_factors:
        return
    for i, tf in enumerate(top_factors[:3], start=1):
        label = tf.get("label", "")
        if i == 1:
            tag = "Strongest signal"
        elif i == 2:
            tag = "Also matters"
        else:
            tag = "Contributes"
        st.markdown(
            f'<span class="rl-pill">{html.escape(tag)}: {html.escape(str(label))}</span>',
            unsafe_allow_html=True,
        )


def render_drivers(
    api_ok: bool, top_factors: list, use_wide_chart: bool = True
) -> None:
    st.markdown(
        f'<div class="rl-card"><h3>{html.escape(COPY["factors_header"])}</h3><p style="color:{T["text_muted"]}; font-size:0.92rem; margin:0 0 0.75rem 0;">{html.escape(COPY["factors_caption"])}</p></div>',
        unsafe_allow_html=True,
    )
    if not api_ok or not top_factors:
        st.caption(COPY["no_factors"])
        return

    df_exp = pd.DataFrame(
        [
            {"Factor": tf["label"], "How much (%)": float(tf["share_percent"])}
            for tf in top_factors
        ]
    )
    w = 10 if use_wide_chart else 8
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(w, max(2.5, 0.45 * len(df_exp) + 1.2)))
    y = list(df_exp["Factor"])[::-1]
    x = [float(v) for v in list(df_exp["How much (%)"])[::-1]]
    color = T["accent_teal"]
    ax.barh(y, x, color=color, height=0.55, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(
        "How much each area showed up in this check-in (%)"
    )
    ax.set_title("Areas that stood out in your check-in")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    render_top_factors_badge(top_factors)
    # Longer factor notes are reserved for admin; students see the chart and badges above.
    if is_admin():
        for tf in top_factors[:5]:
            note = (tf.get("note") or "").strip()
            if note:
                st.caption(f"**{tf['label']}:** {note}")


def history_friendly_table(hist: list) -> pd.DataFrame:
    rows = []
    for h in hist:
        p = h.get("risk")
        p_str = f"{100 * float(p):.0f}%" if p is not None else "—"
        status_s = str(h.get("status", "—"))
        note = f"Check-in: {status_s.lower() if status_s else 'saved'}"
        rows.append(
            {
                COPY["table_date"]: h.get("timestamp", "—"),
                COPY["table_status"]: status_s,
                COPY["table_score"]: p_str,
                COPY["table_note"]: note,
            }
        )
    return pd.DataFrame(rows)


def trend_direction(hist: list) -> str:
    if len(hist) < 2:
        return "—"
    a = float(hist[-2].get("risk", 0))
    b = float(hist[-1].get("risk", 0))
    if b > a + 0.03:
        return COPY["trend_up"]
    if b < a - 0.03:
        return COPY["trend_down"]
    return COPY["trend_flat"]


def render_progress() -> None:
    st.markdown(
        f'<div class="rl-card"><h3>{html.escape(COPY["progress_header"])}</h3></div>',
        unsafe_allow_html=True,
    )
    h = st.session_state.history
    n = len(h)
    c1, c2, c3 = st.columns(3)
    last_ts = h[-1].get("timestamp", "—") if n else "—"
    c1.metric(COPY["latest"], last_ts)
    c2.metric(COPY["n_checkins"], str(n))
    c3.metric(COPY["trend_dir"], trend_direction(h) if n >= 2 else "—")

    if n == 0:
        st.info(COPY["empty_history"])
        return
    st.dataframe(
        history_friendly_table(h),
        use_container_width=True,
        hide_index=True,
    )
    if n == 1:
        st.success(COPY["one_point_history"])
        return
    apply_plot_style()
    hist_df = pd.DataFrame(h)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    y_val = hist_df["risk"].astype(float).tolist()
    x = list(range(1, len(y_val) + 1))
    ax.fill_between(
        x,
        y_val,
        alpha=0.12,
        color=T["accent_teal"],
    )
    ax.plot(
        x,
        y_val,
        marker="o",
        color=T["accent_teal"],
        linewidth=2,
        markersize=8,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel(COPY["x_checkins"])
    ax.set_ylabel("Score (0 = lower concern, 1 = higher concern)")
    ax.set_title("How your check-in score changed over time")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _insight_direction_plain(direction: str | None) -> str:
    d = (direction or "").strip()
    if d == "increases_estimated_risk":
        return "In this check-in, this area nudges the result toward needing more support."
    if d == "decreases_estimated_risk":
        return "In this check-in, this area nudges the result toward a steadier picture."
    if d == "neutral":
        return "This area did not swing the result very much either way here."
    return d or "—"


def explain_factors_to_dataframe(ex: dict) -> pd.DataFrame:
    """Display table for the Insights tab: plain language, no internal codes as labels."""
    rows: list[dict[str, object]] = []
    for f in ex.get("factors") or []:
        label = f.get("label", "—")
        val = f.get("feature_value", "—")
        if f.get("method") == "shap_tree":
            rows.append(
                {
                    "Area": label,
                    "What it suggests for you": _insight_direction_plain(
                        str(f.get("direction"))
                    ),
                    "The level in your check-in": val,
                }
            )
        else:
            n = (f.get("note") or "").strip()
            rows.append(
                {
                    "Area": label,
                    "What it suggests for you": n
                    if n
                    else "This area shows up in this view of your answers.",
                    "The level in your check-in": val,
                }
            )
    return pd.DataFrame(rows)


def render_insights_tab(api_base: str, feature_payload: dict) -> None:
    st.markdown(f"### {COPY['insights_title']}")
    st.markdown(COPY["insights_blurb"])

    st.markdown("#### What may be driving your result")
    if st.button(COPY["insight_drivers_button"], key="btn_expl_main"):
        with st.spinner("Preparing your insight…"):
            try:
                st.session_state.explain_cache = call_explain(api_base, feature_payload)
            except Exception as e:
                st.error(format_api_error_user(e))
    ex = st.session_state.explain_cache
    if ex:
        rp = ex.get("risk_probability")
        st.metric(
            "Check-in score (this view)",
            f"{float(rp):.0%}" if rp is not None else "—",
        )
        if is_admin():
            with st.expander("Technical details (admin)"):
                st.code(
                    f"method={ex.get('method')}\n{ex.get('message', '')}",
                    language="text",
                )
        df_ex = explain_factors_to_dataframe(ex)
        if not df_ex.empty:
            st.dataframe(df_ex, use_container_width=True, hide_index=True)
        else:
            st.caption("No rows returned for this insight.")

    st.divider()
    st.markdown("#### Patterns that often matter")
    st.caption(
        "A general view across many check-ins—use it as context, not a personal diagnosis."
    )
    if st.button(COPY["insight_common_button"], key="btn_imp_main"):
        with st.spinner("Loading…"):
            try:
                st.session_state.importance_cache = call_feature_importance(api_base)
            except Exception as e:
                st.error(format_api_error_user(e))
    g = st.session_state.importance_cache
    if g:
        if is_admin():
            st.caption(g.get("source", ""))
        feat_df = pd.DataFrame(g.get("features", []))
        if feat_df.empty or "label" not in feat_df.columns or "importance" not in feat_df.columns:
            st.caption("No pattern data to display right now.")
            if is_admin() and not feat_df.empty:
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
        else:
            n_show = min(12, max(1, len(feat_df)))
            plot_df = feat_df.nlargest(n_show, "importance").iloc[::-1]
            apply_plot_style()
            x_label = (
                "Relative importance (technical)" if is_admin() else "How often each area tends to show up"
            )
            title = (
                "Factors that often show up in scores"
                if not is_admin()
                else "Global feature importance (12 largest)"
            )
            fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(plot_df))))
            ax.barh(
                plot_df["label"].astype(str),
                plot_df["importance"].astype(float),
                color=T["accent_teal"],
                height=0.55,
            )
            ax.set_xlabel(x_label)
            ax.set_title(title)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def render_support_tab(api_base: str) -> None:
    st.markdown(f"### {COPY['support_title']}")
    st.markdown(COPY["support_blurb"])
    st.caption(COPY["chat_privacy"])

    ctx = st.session_state.prediction_context
    if ctx and is_admin():
        with st.expander(COPY["admin_chat_ctx"]):
            st.json(ctx)

    use_ctx = st.checkbox(COPY["include_context"], value=True)

    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg.get('content', '')}")
        else:
            st.markdown("**Support:**")
            st.write(msg.get("content", ""))
            if msg.get("actions"):
                st.caption("Ideas: " + " · ".join(msg["actions"]))

    _draft_id = int(st.session_state.support_draft_key)
    prompt = st.text_area(
        "How are you feeling today? (This stays in your browser session.)",
        height=100,
        key=f"support_draft_v{_draft_id}",
    )
    c_send, c_clear = st.columns(2)
    with c_send:
        send = st.button("Send", type="primary", use_container_width=True)
    with c_clear:
        clear = st.button("Clear conversation", use_container_width=True)

    if send and prompt.strip():
        user_msg = prompt.strip()
        st.session_state.chat_messages.append({"role": "user", "content": user_msg})
        try:
            body_ctx = st.session_state.prediction_context if use_ctx else None
            out = call_support(api_base, user_msg, body_ctx)
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": out["reply"],
                    "actions": out.get("supportive_actions") or [],
                }
            )
        except Exception as e:
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        format_api_error_user(e)
                        if is_admin()
                        else COPY["chat_unavailable"]
                    ),
                    "actions": [],
                }
            )
        st.session_state.support_draft_key = _draft_id + 1
        st.rerun()

    if clear:
        st.session_state.chat_messages = []
        st.session_state.support_draft_key = int(st.session_state.support_draft_key) + 1
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[dict, bool, bool]:
    with st.sidebar:
        st.markdown(
            f"### {COPY['sidebar_title']}",
        )
        if is_admin():
            st.text_input(
                "API base URL (admin)",
                value=DEFAULT_API_BASE,
                key="api_base_admin_input",
                help="Set RESILIENCE_API_BASE in the environment for non-admin runs.",
            )
            with st.expander("Service status (admin)", expanded=False):
                if st.button("Check health", use_container_width=True):
                    try:
                        base = get_api_base()
                        r = requests.get(f"{base}/health", timeout=5)
                        r.raise_for_status()
                        st.success("Backend responded.")
                        st.json(r.json())
                    except Exception as e:
                        st.error(format_api_error_technical(e))
        st.caption(
            "Adjust the sliders, then use **Update my results** to refresh your summary."
        )

        with st.expander(f"**{COPY['academic']}**", expanded=True):
            study_hours = st.slider(
                "Study hours per day",
                min_value=0.0,
                max_value=16.0,
                value=3.0,
                step=0.5,
                help="Rough average on a typical day.",
            )
        with st.expander(f"**{COPY['attendance']}**", expanded=True):
            absences = st.number_input(
                "Absences in the last four weeks",
                min_value=0,
                max_value=30,
                value=2,
                step=1,
            )
        with st.expander(f"**{COPY['wellbeing']}**", expanded=True):
            sleep_quality = st.slider("Sleep quality", 1, 10, 6)
            mental_health = st.slider("How you are doing emotionally", 1, 10, 6)
            motivation = st.slider("Motivation for studying", 1, 10, 6)
        with st.expander(f"**{COPY['context']}**", expanded=True):
            st.caption(COPY["crisis_caption"])
            financial_stress = st.slider("Money stress", 1, 10, 6)
            electricity_hours = st.slider("Stable electricity (hours per day)", 0, 24, 8)
            internet_quality = st.slider("Internet for learning", 1, 10, 6)
            war_stress = st.slider("Stress from conflict or safety concerns", 1, 10, 5)

        st.divider()
        run = st.button(COPY["run_cta"], type="primary", use_container_width=True)
        save = st.button(COPY["save_cta"], use_container_width=True)

    inputs = {
        "study_hours": float(study_hours),
        "absences": int(absences),
        "sleep_quality": int(sleep_quality),
        "mental_health": int(mental_health),
        "financial_stress": int(financial_stress),
        "electricity_hours": int(electricity_hours),
        "internet_quality": int(internet_quality),
        "war_stress": int(war_stress),
        "motivation": int(motivation),
    }
    return inputs, run, save


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
inject_custom_css()
apply_plot_style()

inputs, run, save = render_sidebar()
api_base = get_api_base()
feature_payload = ui_checkin_to_predict_body(inputs)
suggestions = suggestions_categorized(inputs)

api_ok = False
risk_label: int | None = None
risk_probability: float | None = None
status = "—"
uncertainty_score = 0.0
uncertainty_message = ""
top_factors: list = []
caught_exc: BaseException | None = None

try:
    result = call_predict(api_base, feature_payload)
    risk_label = int(result["risk_label"])
    risk_probability = float(result["risk_probability"])
    status = str(result["status_label"])
    uncertainty_score = float(result["uncertainty_score"])
    uncertainty_message = str(result["uncertainty_message"])
    top_factors = list(result.get("top_factors") or [])
    api_ok = True
except Exception as e:
    caught_exc = e
    if isinstance(e, requests.exceptions.RequestException):
        status = "Unavailable"
    else:
        status = "Error"
    risk_probability = None

if api_ok:
    st.session_state.prediction_context = {
        "risk_label": risk_label,
        "risk_probability": risk_probability,
        "status_label": status,
        "uncertainty_score": uncertainty_score,
        "uncertainty_message": uncertainty_message,
        "top_factors": top_factors,
    }

render_hero()
if not api_ok and caught_exc is not None:
    st.error(format_api_error_user(caught_exc))

tab_checkin, tab_insights, tab_support = st.tabs(
    [COPY["tab_checkin"], COPY["tab_insights"], COPY["tab_support"]]
)

with tab_checkin:
    if api_ok:
        render_result_summary(
            risk_probability,
            risk_label,
            uncertainty_score,
            uncertainty_message,
        )
    col_g, col_f = st.columns([1.05, 1.0], gap="large")
    with col_g:
        render_guidance(suggestions)
    with col_f:
        render_drivers(api_ok, top_factors)
    render_progress()

with tab_insights:
    render_insights_tab(api_base, feature_payload)

with tab_support:
    render_support_tab(api_base)

if is_admin():
    with st.expander("Developer notes (admin)"):
        st.markdown(
            "- API: `RESILIENCE_API_BASE` / sidebar URL; run FastAPI with "
            "`uvicorn backend.app:app --reload --app-dir .`.\n"
            "- Endpoints used: predict, explain, feature-importance, chat/support, health."
        )

if run:
    with st.spinner("Updating your check-in…"):
        time.sleep(0.08)
    if api_ok:
        st.success(COPY["update_ok"])
    else:
        st.warning(COPY["update_fail"])

if save:
    if api_ok and risk_probability is not None:
        st.session_state.history.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "status": status,
                "risk_label": risk_label,
                "risk": round(risk_probability, 4),
                "uncertainty": round(uncertainty_score, 4),
                **inputs,
            }
        )
        st.success(COPY["save_ok"])
    else:
        st.warning(COPY["save_fail"])