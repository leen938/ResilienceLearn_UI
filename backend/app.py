"""
ResilienceLearn API — inference, explainability, supportive chat.

Run from project root:
    uvicorn backend.app:app --reload --app-dir .
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from ml.explainability import global_importance_list, local_explanation  # noqa: E402
from ml.prediction_extras import top_local_factors, uncertainty_from_probability  # noqa: E402
from ml.support_chat import generate_support_reply  # noqa: E402

_dotenv_path = ROOT / ".env"
_dotenv_loaded = load_dotenv(dotenv_path=_dotenv_path, override=False)
logger = logging.getLogger("resiliencelearn")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Startup diagnostics (no secrets).
_raw_use_openai = os.environ.get("USE_OPENAI_CHAT", "")
_raw_llm_enabled = os.environ.get("LLM_SUPPORT_ENABLED", "")
_raw_llm_provider = os.environ.get("LLM_PROVIDER", "")
_parsed_use_openai = _raw_use_openai.strip().lower() in {"1", "true", "yes", "on"} or (
    _raw_llm_enabled.strip().lower() in {"1", "true", "yes", "on"}
    and _raw_llm_provider.strip().lower() == "openai"
)
_api_key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
_model_name = (os.environ.get("OPENAI_MODEL", "") or os.environ.get("LLM_MODEL_NAME", "") or "gpt-4o-mini").strip()
logger.info(
    "startup dotenv_loaded=%s dotenv_path=%s exists=%s",
    _dotenv_loaded,
    str(_dotenv_path),
    _dotenv_path.is_file(),
)
logger.info(
    "startup openai_config use_openai_raw=%r llm_support_enabled_raw=%r llm_provider_raw=%r use_openai_parsed=%s api_key_present=%s model=%r openai_sdk_available=%s",
    _raw_use_openai,
    _raw_llm_enabled,
    _raw_llm_provider,
    _parsed_use_openai,
    _api_key_present,
    _model_name,
    OpenAI is not None,
)

ARTIFACT_PATH = ROOT / "artifacts" / "model.joblib"

_loaded: dict | None = None


def _load_artifacts() -> dict:
    global _loaded
    if _loaded is not None:
        return _loaded
    path = Path(os.environ.get("RESILIENCE_MODEL_PATH", str(ARTIFACT_PATH)))
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    _loaded = joblib.load(path)
    return _loaded


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        _load_artifacts()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(title="ResilienceLearn API", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Feature vector after ``ml.preprocessing`` (same order as training)."""

    model_config = ConfigDict(extra="forbid")

    year_of_study: float = Field(..., ge=1, le=5)
    gpa: float = Field(..., ge=1, le=4)
    study_hours: float = Field(..., ge=1, le=4)
    attendance: float = Field(..., ge=1, le=4)
    motivation: float = Field(..., ge=1, le=10)
    quiet_environment: float = Field(..., ge=1, le=3)
    mental_health: float = Field(..., ge=1, le=10)
    sleep_quality: float = Field(..., ge=1, le=4)
    stress_academic: float = Field(..., ge=1, le=4)
    breaks_relaxation: float = Field(..., ge=1, le=4)
    electricity_stability: float = Field(..., ge=0, le=4)
    internet_stability: float = Field(..., ge=0, le=4)
    economic_political_impact: float = Field(..., ge=0, le=4)
    war_conflict_impact: float = Field(..., ge=0, le=4)
    crisis_exposure_index: float = Field(..., ge=0, le=1)


class TopFactor(BaseModel):
    feature: str
    label: str
    share_percent: float
    note: str


class PredictResponse(BaseModel):
    risk_label: int = Field(..., description="1 = at risk, 0 = on track")
    risk_probability: float = Field(..., description="P(at risk), positive class")
    status_label: str
    uncertainty_score: float = Field(..., description="Higher = closer to 0.5 probability (proxy)")
    uncertainty_message: str
    top_factors: list[TopFactor] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    artifact_path: str


class GlobalImportanceRow(BaseModel):
    feature: str
    label: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    source: str = Field(..., description="Model-based global importance")
    features: list[GlobalImportanceRow]


class ExplainFactorShap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    label: str
    display_name: str
    method: Literal["shap_tree"] = "shap_tree"
    shap_value: float
    direction: str
    strength: str
    feature_value: float
    explanation_text: str
    rank: int


class ExplainFactorApprox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    label: str
    display_name: str
    method: Literal["approximate"] = "approximate"
    approx_share_percent: float
    note: str
    feature_value: float
    direction: str
    strength: str
    explanation_text: str
    rank: int


ExplainFactorItem = ExplainFactorShap | ExplainFactorApprox


class ExplainResponse(BaseModel):
    method: str = Field(..., description="shap_tree or approximate")
    risk_label: int | None = None
    risk_probability: float | None = None
    status_label: str | None = None
    message: str = Field(
        default="",
        description="Method note for thesis / UI",
    )
    factors: list[ExplainFactorItem] = Field(default_factory=list)


def _coerce_explain_factors(raw: list[dict[str, Any]]) -> list[ExplainFactorItem]:
    out: list[ExplainFactorItem] = []
    for item in raw:
        m = item.get("method")
        if m == "shap_tree":
            out.append(ExplainFactorShap.model_validate(item))
        elif m == "approximate":
            out.append(ExplainFactorApprox.model_validate(item))
        else:
            raise ValueError(f"Unknown explanation factor method: {m!r}")
    return out


class SupportChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=6000)
    context: dict[str, Any] | None = None
    history: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional recent chat messages for conversational continuity. "
            "Each item should look like {'role': 'user'|'assistant', 'content': '...'}."
        ),
    )


class SupportChatResponse(BaseModel):
    reply: str
    mood_detected: str | None = None
    supportive_actions: list[str] = Field(default_factory=list)
    check_in_prompt: str | None = None
    safety_note: str
    provider: Literal["openai", "fallback"] = "fallback"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    path = Path(os.environ.get("RESILIENCE_MODEL_PATH", str(ARTIFACT_PATH)))
    ok = path.is_file()
    try:
        if ok:
            _load_artifacts()
            ok = _loaded is not None
    except Exception:
        ok = False
    return HealthResponse(
        status="ok" if ok else "degraded",
        model_loaded=ok,
        artifact_path=str(path.resolve()),
    )


@app.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance() -> FeatureImportanceResponse:
    try:
        bundle = _load_artifacts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    rows = global_importance_list(bundle)
    if not rows:
        raise HTTPException(
            status_code=503,
            detail="Artifact missing feature_importances; retrain with scripts/train_and_save_model.py",
        )
    return FeatureImportanceResponse(
        source="sklearn RandomForestClassifier.feature_importances_ (Gini)",
        features=[GlobalImportanceRow(**r) for r in rows],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    try:
        bundle = _load_artifacts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    model = bundle["model"]
    names: list[str] = list(bundle["feature_names"])

    row = body.model_dump()
    X = pd.DataFrame([row], columns=names)

    missing = [c for c in names if c not in X.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    if 1 in classes:
        pos_idx = int(np.argmax(classes == 1))
    else:
        pos_idx = int(np.argmax(classes))
    p_risk = float(proba[0, pos_idx])
    pred = int(model.predict(X)[0])

    status = "AT RISK" if pred == 1 else "ON TRACK"
    u_score, u_msg = uncertainty_from_probability(p_risk)

    imps = bundle.get("feature_importances")
    meds = bundle.get("feature_medians")
    raw_factors = top_local_factors(row, names, imps, meds, top_k=3)
    factors = [TopFactor(**item) for item in raw_factors]

    return PredictResponse(
        risk_label=pred,
        risk_probability=round(p_risk, 4),
        status_label=status,
        uncertainty_score=round(u_score, 4),
        uncertainty_message=u_msg,
        top_factors=factors,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(
    body: PredictRequest,
    top_k: int = Query(5, ge=1, le=20, description="Number of top local factors to return"),
    include_hidden: bool = Query(
        False,
        description="Include features not explicitly collected in the UI (e.g., year_of_study).",
    ),
) -> ExplainResponse:
    """Local explanation for one row: SHAP TreeExplainer when available, else approximate ranker."""
    try:
        bundle = _load_artifacts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    model = bundle["model"]
    names: list[str] = list(bundle["feature_names"])
    row = body.model_dump()
    X = pd.DataFrame([row], columns=names)

    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    if 1 in classes:
        pos_idx = int(np.argmax(classes == 1))
    else:
        pos_idx = int(np.argmax(classes))
    p_risk = float(proba[0, pos_idx])
    pred = int(model.predict(X)[0])
    status = "AT RISK" if pred == 1 else "ON TRACK"

    method, raw_factors = local_explanation(bundle, row, top_k=top_k, include_hidden=include_hidden)
    try:
        factors = _coerce_explain_factors(raw_factors)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    msg = (
        "Detailed breakdown: which parts of your check-in had the most influence on the score for this run."
        if method == "shap_tree"
        else "Simpler ranking of which answers stood out for this run; a more detailed view may be available in some setups."
    )
    return ExplainResponse(
        method=method,
        risk_label=pred,
        risk_probability=round(p_risk, 4),
        status_label=status,
        message=msg,
        factors=factors,
    )


@app.post("/chat/support", response_model=SupportChatResponse)
def chat_support(body: SupportChatRequest) -> SupportChatResponse:
    """Supportive, non-diagnostic chat. Not a substitute for professional care."""
    raw_use_openai = os.environ.get("USE_OPENAI_CHAT", "")
    raw_llm_enabled = os.environ.get("LLM_SUPPORT_ENABLED", "")
    raw_llm_provider = os.environ.get("LLM_PROVIDER", "")
    use_openai = raw_use_openai.strip().lower() in {"1", "true", "yes", "on"} or (
        raw_llm_enabled.strip().lower() in {"1", "true", "yes", "on"}
        and raw_llm_provider.strip().lower() == "openai"
    )
    api_key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    model_name = (os.environ.get("OPENAI_MODEL", "") or os.environ.get("LLM_MODEL_NAME", "") or "gpt-4o-mini").strip()

    logger.info(
        "chat_support gating use_openai_raw=%r llm_support_enabled_raw=%r llm_provider_raw=%r use_openai=%s api_key_present=%s openai_sdk_available=%s model=%r",
        raw_use_openai,
        raw_llm_enabled,
        raw_llm_provider,
        use_openai,
        api_key_present,
        OpenAI is not None,
        model_name,
    )

    if use_openai and api_key_present and OpenAI is not None:
        logger.info("chat_support attempting provider=openai")
        try:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

            system = (
                "You are a warm, emotionally intelligent, human-like supportive chat companion for students.\n"
                "\n"
                "Tone + style:\n"
                "- Sound natural, kind, and grounded (like a good friend who listens well).\n"
                "- Keep responses concise but meaningful: 3–6 sentences.\n"
                "- Avoid robotic phrasing, bullet-point lectures, and repeated templates.\n"
                "- Reflect the user's emotion in your first sentence (validate without exaggerating).\n"
                "- Prefer simple language over clinical terms.\n"
                "\n"
                "Conversation behavior:\n"
                "- Ask at most ONE gentle follow-up question when it would help.\n"
                "- If the user seems overwhelmed, help narrow to one small next step.\n"
                "- If the user is venting, prioritize listening over advice.\n"
                "\n"
                "Safety:\n"
                "- You are NOT a therapist or doctor. Do not diagnose.\n"
                "- Do not make medical/clinical claims or provide treatment instructions.\n"
                "- If the user expresses self-harm or suicidal intent, respond with calm urgency and encourage contacting local emergency services, "
                "a crisis hotline, or a trusted person right now.\n"
            )

            messages: list[dict[str, str]] = [{"role": "system", "content": system}]

            # Optional check-in context (keep short).
            ctx = body.context or {}
            ctx_lines: list[str] = []
            if ctx.get("status_label"):
                ctx_lines.append(f"Latest check-in status label (informational): {ctx['status_label']}")
            if ctx.get("uncertainty_message"):
                ctx_lines.append(f"Uncertainty note: {ctx['uncertainty_message']}")
            if ctx.get("top_factors") and isinstance(ctx.get("top_factors"), list):
                labels = []
                for tf in (ctx.get("top_factors") or [])[:2]:
                    if isinstance(tf, dict) and tf.get("label"):
                        labels.append(str(tf["label"]))
                if labels:
                    ctx_lines.append("Themes from last snapshot: " + ", ".join(labels))
            if ctx_lines:
                messages.append({"role": "system", "content": "Optional context:\n" + "\n".join(ctx_lines)})

            # Short recent history for conversational continuity.
            if body.history:
                for item in body.history[-12:]:
                    role = str(item.get("role", "")).lower().strip()
                    content = str(item.get("content", "")).strip()
                    if role in {"user", "assistant"} and content:
                        messages.append({"role": role, "content": content})

            messages.append({"role": "user", "content": body.message})

            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.8,
                max_tokens=260,
            )
            reply = (resp.choices[0].message.content or "").strip()
            if reply:
                logger.info("chat_support provider=openai model=%s", model_name)
                return SupportChatResponse(
                    reply=reply,
                    mood_detected=None,
                    supportive_actions=[],
                    check_in_prompt=None,
                    safety_note=(
                        "I am not a therapist or doctor and cannot diagnose or treat anything. "
                        "If you are in immediate danger, contact local emergency services or someone you trust right away."
                    ),
                    provider="openai",
                )
        except Exception as e:
            logger.exception("chat_support provider=openai failed: %s", str(e))
    else:
        reasons: list[str] = []
        if not use_openai:
            reasons.append("OpenAI not enabled (USE_OPENAI_CHAT or LLM_SUPPORT_ENABLED/LLM_PROVIDER)")
        if not api_key_present:
            reasons.append("OPENAI_API_KEY missing/empty")
        if OpenAI is None:
            reasons.append("OpenAI SDK not importable (missing dependency?)")
        logger.info("chat_support skipping openai reasons=%s", "; ".join(reasons) if reasons else "unknown")

    out = generate_support_reply(body.message, body.context, history=body.history)
    logger.info("chat_support provider=fallback")
    return SupportChatResponse(
        reply=out["reply"],
        mood_detected=out.get("mood_detected"),
        supportive_actions=list(out.get("supportive_actions") or []),
        check_in_prompt=out.get("check_in_prompt"),
        safety_note=str(out.get("safety_note") or ""),
        provider="fallback",
    )
