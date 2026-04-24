"""
ResilienceLearn API — inference, explainability, supportive chat.

Run from project root:
    uvicorn backend.app:app --reload --app-dir .
"""
from __future__ import annotations

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

from ml.explainability import global_importance_list, local_explanation  # noqa: E402
from ml.prediction_extras import top_local_factors, uncertainty_from_probability  # noqa: E402
from ml.support_chat import generate_support_reply  # noqa: E402

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
    method: Literal["shap_tree"] = "shap_tree"
    shap_value: float
    direction: str
    feature_value: float
    rank: int


class ExplainFactorApprox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    label: str
    method: Literal["approximate"] = "approximate"
    approx_share_percent: float
    note: str
    feature_value: float
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


class SupportChatResponse(BaseModel):
    reply: str
    mood_detected: str | None = None
    supportive_actions: list[str] = Field(default_factory=list)
    check_in_prompt: str | None = None
    safety_note: str


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

    method, raw_factors = local_explanation(bundle, row, top_k=top_k)
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
    """Rule-based supportive chat. Not a substitute for professional care."""
    out = generate_support_reply(body.message, body.context)
    return SupportChatResponse(
        reply=out["reply"],
        mood_detected=out.get("mood_detected"),
        supportive_actions=list(out.get("supportive_actions") or []),
        check_in_prompt=out.get("check_in_prompt"),
        safety_note=str(out.get("safety_note") or ""),
    )
