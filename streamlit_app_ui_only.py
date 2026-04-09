# streamlit_app_ui_only.py
# Dashboard: check-in, explainability (API), support chat (API). Run:
#   pip install -r requirements.txt
#   uvicorn backend.app:app --reload --app-dir .
#   python -m streamlit run streamlit_app_ui_only.py

from __future__ import annotations

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

st.set_page_config(page_title="ResilienceLearn", layout="wide")


def format_api_error(exc: BaseException) -> str:
    """Human-readable messages for API failures (HTTP detail vs network)."""
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
        return (
            f"Could not reach the API ({type(exc).__name__}). "
            "Confirm the URL in the sidebar and that `uvicorn backend.app:app --reload --app-dir .` is running."
        )
    return f"{type(exc).__name__}: {exc}"

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


def suggestions_from_inputs(inputs: dict):
    """Action tips based on thresholds (supportive, non-punitive)."""
    tips = []
    if inputs["electricity_hours"] < 8:
        tips.append("Plan 2×25-min focus blocks during power hours (Pomodoro).")
    if inputs["internet_quality"] < 6:
        tips.append("Download lecture material when internet is stable; use offline notes.")
    if inputs["sleep_quality"] < 6:
        tips.append("Try a fixed sleep/wake time for 5 days + reduce caffeine after 4 PM.")
    if inputs["absences"] > 6:
        tips.append("Pick 1 course and commit to zero absences next 2 weeks (small win strategy).")
    if inputs["financial_stress"] > 7:
        tips.append("Check campus/NGO support options; prioritize essentials + low-cost meals.")
    if inputs["motivation"] < 5:
        tips.append("Use 10-minute starter tasks (micro-goals) to lower resistance.")
    if inputs["war_stress"] > 7 or inputs["mental_health"] < 5:
        tips.append("Consider reaching out to a counselor/support service or a trusted person.")
    if not tips:
        tips.append("Keep your current routine; add weekly review + small adjustments.")
    return tips[:6]


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


st.title("ResilienceLearn — Student check-in")
st.caption(
    "Risk scores come from **POST /predict**. Deeper breakdowns use **POST /explain** (SHAP when available). "
    "Support chat uses **POST /chat/support** (rule-based, not therapy)."
)

with st.sidebar:
    st.header("Student Check-in")
    default_api = os.environ.get("RESILIENCE_API_BASE", "http://127.0.0.1:8000").rstrip("/")
    api_base = st.text_input("API base URL", value=default_api)

    with st.expander("Backend status"):
        if st.button("Check GET /health", use_container_width=True):
            try:
                r = requests.get(f"{api_base.rstrip('/')}/health", timeout=5)
                r.raise_for_status()
                st.success("Backend responded.")
                st.json(r.json())
            except Exception as e:
                st.error(format_api_error(e))

    study_hours = st.number_input(
        "Study hours/day", min_value=0.0, max_value=16.0, value=3.0, step=0.5
    )
    absences = st.number_input(
        "Absences (last 4 weeks)", min_value=0, max_value=30, value=2, step=1
    )

    st.divider()
    st.subheader("Well-being (1–10)")
    sleep_quality = st.slider("Sleep quality", 1, 10, 6)
    mental_health = st.slider("Mental health self-rating", 1, 10, 6)
    motivation = st.slider("Motivation", 1, 10, 6)

    st.divider()
    st.subheader("Crisis context (Lebanon)")
    financial_stress = st.slider("Financial stress", 1, 10, 6)
    electricity_hours = st.slider("Electricity availability (hours/day)", 0, 24, 8)
    internet_quality = st.slider("Internet access quality", 1, 10, 6)
    war_stress = st.slider("War-related stress exposure", 1, 10, 5)

    st.divider()
    run = st.button("Generate Report", use_container_width=True)
    save = st.button("Save this check-in", use_container_width=True)

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

feature_payload = ui_checkin_to_predict_body(inputs)
tips = suggestions_from_inputs(inputs)

api_ok = False
risk_label: int | None = None
risk_probability: float | None = None
status = "—"
uncertainty_score = 0.0
uncertainty_message = ""
top_factors: list = []
api_error = ""

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
    api_error = format_api_error(e)
    if isinstance(e, requests.RequestException):
        status = "API UNAVAILABLE"
    else:
        status = "ERROR"
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

risk_display = risk_probability if risk_probability is not None else 0.0

tab_main, tab_explain, tab_chat = st.tabs(
    ["Check-in & results", "Explainability", "Support chat"]
)

with tab_main:
    colA, colB = st.columns([1.2, 1.0], gap="large")

    with colA:
        st.subheader("Result")
        if not api_ok:
            st.error(
                f"Could not reach the model API: {api_error}\n\n"
                "Start the backend: `uvicorn backend.app:app --reload --app-dir .`"
            )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", status)
        c2.metric(
            "Risk label",
            f"{risk_label}" if risk_label is not None else "—",
            help="1 = at risk, 0 = on track (from API)",
        )
        c3.metric(
            "P(at risk)",
            f"{risk_probability:.2f}" if risk_probability is not None else "—",
        )
        c4.metric("Uncertainty (proxy)", f"{uncertainty_score:.2f}" if api_ok else "—")

        if api_ok and uncertainty_message:
            st.info(uncertainty_message)

        st.progress(risk_display)

        st.subheader("Top factors (with /predict — fast approximate)")
        st.caption("For SHAP-based local drivers, open the **Explainability** tab.")
        if api_ok and top_factors:
            df_exp = pd.DataFrame(
                [
                    {
                        "Factor": tf["label"],
                        "Share %": tf["share_percent"],
                        "Note": tf["note"],
                    }
                    for tf in top_factors
                ]
            )
            st.dataframe(df_exp, use_container_width=True, hide_index=True)

            fig = plt.figure()
            factors = df_exp["Factor"].tolist()
            impact = [float(x) for x in df_exp["Share %"].tolist()]
            plt.barh(factors[::-1], impact[::-1])
            plt.xlabel("Approx. share (%) - /predict")
        else:
            st.caption("No factor ranking (API unavailable or older artifact).")
            fig = plt.figure()
            plt.text(0.1, 0.5, "—")
        plt.tight_layout()
        st.pyplot(fig)

    with colB:
        st.subheader("Personalized suggestions")
        for t in tips:
            st.write(f"- {t}")

        st.subheader("Progress tracking")
        if len(st.session_state.history) == 0:
            st.info("No saved check-ins yet. Click **Save this check-in** to start tracking.")
        else:
            hist_df = pd.DataFrame(st.session_state.history)
            cols_hist = [
                c
                for c in ("timestamp", "status", "risk_label", "risk", "uncertainty")
                if c in hist_df.columns
            ]
            st.dataframe(hist_df[cols_hist], use_container_width=True, hide_index=True)

            fig2 = plt.figure()
            x = list(range(len(hist_df)))
            y_val = hist_df["risk"].astype(float).tolist()
            plt.plot(x, y_val, marker="o")
            plt.ylim(0, 1)
            plt.xlabel("Check-in #")
            plt.ylabel("P(at risk)")
            plt.tight_layout()
            st.pyplot(fig2)

        st.subheader("About this score")
        st.caption(
            "The API returns an estimated risk probability from your check-in features. "
            "There is **no live peer cohort benchmark** in this build — avoid treating the score as a class rank."
        )

with tab_explain:
    st.subheader("Global feature importance")
    st.caption("**GET /feature-importance** — Gini importances from the saved Random Forest.")
    if st.button("Load global importance", key="btn_imp"):
        try:
            st.session_state.importance_cache = call_feature_importance(api_base)
        except Exception as e:
            st.error(format_api_error(e))
    if st.session_state.importance_cache:
        g = st.session_state.importance_cache
        st.write(g.get("source", ""))
        st.dataframe(
            pd.DataFrame(g.get("features", [])),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Local explanation (one prediction)")
    st.caption(
        "**POST /explain** — uses SHAP TreeExplainer when `shap` is installed; otherwise approximate ranking."
    )
    if st.button("Run local explanation for current sidebar inputs", key="btn_expl"):
        try:
            st.session_state.explain_cache = call_explain(api_base, feature_payload)
        except Exception as e:
            st.error(format_api_error(e))
    ex = st.session_state.explain_cache
    if ex:
        st.success(f"Method: **{ex.get('method')}** — {ex.get('message', '')}")
        st.metric("Aligned P(at risk)", f"{ex.get('risk_probability', 0):.3f}")
        facts = ex.get("factors") or []
        if facts:
            rows = []
            for f in facts:
                if f.get("method") == "shap_tree":
                    rows.append(
                        {
                            "Feature": f.get("label"),
                            "SHAP": f.get("shap_value"),
                            "Direction": f.get("direction"),
                            "Value": f.get("feature_value"),
                        }
                    )
                else:
                    rows.append(
                        {
                            "Feature": f.get("label"),
                            "Share %": f.get("approx_share_percent"),
                            "Note": f.get("note"),
                            "Value": f.get("feature_value"),
                        }
                    )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_chat:
    st.subheader("Support chat")
    st.caption(
        "Rule-based supportive replies only — **not** therapy or diagnosis. "
        "Optional context from your last **/predict** is sent if enabled."
    )
    ctx = st.session_state.prediction_context
    if ctx:
        with st.expander("Prediction context (optional attachment)"):
            st.json(ctx)
    use_ctx = st.checkbox("Include check-in context with messages", value=True)

    for msg in st.session_state.chat_messages:
        role = msg["role"].title()
        st.markdown(f"**{role}:**")
        st.write(msg["content"])
        if msg.get("actions"):
            st.caption("Ideas: " + " · ".join(msg["actions"]))

    # New key after each send/clear so the box resets without mutating session_state
    # after the widget is instantiated (that raises StreamlitAPIException).
    _draft_id = int(st.session_state.support_draft_key)
    prompt = st.text_area(
        "How are you feeling? (Anything you type stays in this browser session only unless you change that.)",
        height=90,
        key=f"support_draft_v{_draft_id}",
    )
    c_send, c_clear = st.columns(2)
    with c_send:
        send = st.button("Send message", type="primary", use_container_width=True)
    with c_clear:
        clear = st.button("Clear chat history", use_container_width=True)

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
                    "content": f"(Support chat request failed: {format_api_error(e)})",
                    "actions": [],
                }
            )
        st.session_state.support_draft_key = _draft_id + 1
        st.rerun()

    if clear:
        st.session_state.chat_messages = []
        st.session_state.support_draft_key = int(st.session_state.support_draft_key) + 1
        st.rerun()

if run:
    with st.spinner("Contacting API…"):
        time.sleep(0.1)
    if api_ok:
        st.success("Report updated from the API.")
    else:
        st.warning("Report could not be refreshed.")

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
        st.success("Saved check-in.")
    else:
        st.warning("Save skipped — model API not available.")

st.divider()
with st.expander("How this works"):
    st.write(
        "- **Predict:** `POST /predict` → risk label, probability, uncertainty, fast approximate top factors.\n"
        "- **Explain:** `GET /feature-importance`, `POST /explain` (SHAP TreeExplainer for sklearn RF when possible).\n"
        "- **Chat:** `POST /chat/support` — empathetic rule-based replies; optional last predict context."
    )
