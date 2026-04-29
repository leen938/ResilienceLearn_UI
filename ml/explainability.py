"""
Explainability utilities: global feature importance from artifacts, local SHAP when available.

Design: SHAP is attempted first for /explain; on failure (missing shap, explainer error),
callers fall back to importance x |x - median| (same idea as prediction_extras.top_local_factors).
"""
#Implements global importance and local explanation (SHAP when available, else approximate).
#Called by backend endpoints /feature-importance and /explain.
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ml.prediction_extras import FEATURE_LABELS, top_local_factors

# Features that are required by the model but not explicitly collected/controlled
# in the current Streamlit UI. Keep them for prediction, but avoid presenting them
# as "personal drivers" by default.
HIDDEN_FROM_USER_INSIGHTS: set[str] = {"year_of_study"}


def _display_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def _strength_bucket(abs_value: float, max_abs: float) -> str:
    """Convert absolute contribution into a simple high/medium/low bucket."""
    denom = max(max_abs, 1e-12)
    r = abs_value / denom
    if r >= 0.66:
        return "high"
    if r >= 0.33:
        return "medium"
    return "low"


def _explanation_text(feature: str, feature_value: float, row: dict[str, float]) -> str:
    """
    Plain-language, feature-specific explanations (non-diagnostic).
    Keep these thesis-friendly: explain *why this area can matter* in general.
    """
    name = _display_name(feature)

    if feature == "sleep_quality":
        return (
            "Sleep affects energy, focus, and consistency. When sleep is lower or irregular, it can make studying and keeping routines harder."
        )
    if feature == "attendance":
        return (
            "Attendance often supports understanding and staying on top of coursework. Lower attendance can mean missing explanations, practice, or announcements."
        )
    if feature == "study_hours":
        return (
            "Study time is a proxy for time-on-task. Very low study time can make it harder to keep up, while steady study time can protect against last‑minute pressure."
        )
    if feature == "motivation":
        return (
            "Motivation helps you start and sustain effort, especially when work feels boring or stressful. Lower motivation can make it harder to begin tasks."
        )
    if feature == "mental_health":
        return (
            "How you are feeling mentally can affect concentration, memory, and persistence. When this area is strained, tasks can feel heavier even if ability is unchanged."
        )
    if feature == "stress_academic":
        return (
            "Academic stress can reduce focus and sleep, and make it harder to plan calmly. High stress may push the model toward “needs more support.”"
        )
    if feature == "breaks_relaxation":
        return (
            "Regular breaks and recovery help prevent burnout and keep attention steady. When breaks are rare, it’s easier to hit overload."
        )
    if feature == "quiet_environment":
        return (
            "A quieter, stable study environment can make it easier to concentrate and complete tasks. Noisy or unstable spaces can increase friction and time needed."
        )
    if feature == "gpa":
        return (
            "GPA acts like a summary of recent academic momentum. In the model, stronger past performance often links with lower predicted risk, but it’s not destiny."
        )
    if feature == "internet_stability":
        return (
            "Stable internet supports access to lectures, assignments, and online resources. When internet is disrupted, studying and submitting work can become harder."
        )
    if feature == "electricity_stability":
        return (
            "Stable electricity supports consistent study time and device access. Frequent outages can interrupt routines and reduce available study windows."
        )
    if feature == "economic_political_impact":
        return (
            "Economic or political pressures can consume time, attention, and resources. Higher disruption can make academic routines harder to maintain."
        )
    if feature == "war_conflict_impact":
        return (
            "Conflict-related disruption can affect safety, stress, and daily routines. Higher disruption may make focusing and planning much harder."
        )
    if feature == "crisis_exposure_index":
        elec = row.get("electricity_stability")
        net = row.get("internet_stability")
        econ = row.get("economic_political_impact")
        war = row.get("war_conflict_impact")
        parts = []
        if elec is not None:
            parts.append(f"electricity={elec:g}")
        if net is not None:
            parts.append(f"internet={net:g}")
        if econ is not None:
            parts.append(f"economic/political impact={econ:g}")
        if war is not None:
            parts.append(f"war/conflict impact={war:g}")
        comp = "; ".join(parts)
        return (
            "Your Crisis Exposure Index (CEI) summarizes external barriers such as electricity, internet, economic/political disruption, and war/conflict impact. "
            "Higher CEI means stronger crisis pressure (more barriers), while lower CEI means fewer crisis-related barriers."
            + (f" (Components: {comp})" if comp else "")
        )

    return f"{name} is one of the areas the model considers for this score."

logger = logging.getLogger(__name__)


def global_importance_list(bundle: dict[str, Any]) -> list[dict[str, float | str]]:
    """Sorted global importances from stored ``feature_importances`` (Gini-based RF importance)."""
    imps = bundle.get("feature_importances") or {}
    if not imps:
        return []
    out: list[dict[str, float | str]] = []
    for name, val in sorted(imps.items(), key=lambda x: -float(x[1])):
        out.append(
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name.replace("_", " ").title()),
                "importance": round(float(val), 6),
            }
        )
    return out


def _shap_values_for_positive_class(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    """Return shape (n_features,) SHAP values for row 0 toward positive class, or None."""
    try:
        import shap  # type: ignore[import-not-found]  # noqa: WPS433 — optional dependency
    except ImportError:
        logger.info("shap not installed; local explanations will use approximate method.")
        return None
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            raw = np.asarray(sv[1] if len(sv) > 1 else sv[0])
        else:
            raw = np.asarray(sv)
        if raw.ndim == 3:
            # e.g. (n_samples, n_features, n_classes) for sklearn RF
            cls_idx = 1 if raw.shape[2] > 1 else 0
            arr = np.ravel(raw[0, :, cls_idx])
        elif raw.ndim == 2:
            arr = np.ravel(raw[0])
        else:
            arr = np.ravel(raw)
        return arr.astype(float)
    except Exception as exc:
        logger.warning("SHAP TreeExplainer failed: %s", exc)
        return None


def local_explanation(
    bundle: dict[str, Any],
    row: dict[str, float],
    *,
    top_k: int = 5,
    include_hidden: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Returns (method, factors).

    - method: ``shap_tree`` or ``approximate``
    - factors: sorted by |contribution| descending, length <= top_k
    """
    names: list[str] = list(bundle["feature_names"])
    model = bundle["model"]
    X = pd.DataFrame([row], columns=names)

    vec = _shap_values_for_positive_class(model, X)
    if vec is not None and vec.shape[0] == len(names):
        pairs = list(zip(names, vec, strict=True))
        pairs.sort(key=lambda x: -abs(x[1]))

        # Filter hidden features unless explicitly included.
        if not include_hidden:
            pairs = [(n, v) for (n, v) in pairs if n not in HIDDEN_FROM_USER_INSIGHTS]

        picked = pairs[:top_k]
        out: list[dict[str, Any]] = []
        max_abs = max((abs(float(v)) for _, v in picked), default=0.0)
        for i, (name, shap_v) in enumerate(picked, start=1):
            direction = (
                "increases_estimated_risk"
                if shap_v > 1e-9
                else "decreases_estimated_risk"
                if shap_v < -1e-9
                else "neutral"
            )
            out.append(
                {
                    "feature": name,
                    "label": _display_name(name),
                    "display_name": _display_name(name),
                    "method": "shap_tree",
                    "shap_value": round(float(shap_v), 6),
                    "direction": direction,
                    "strength": _strength_bucket(abs(float(shap_v)), max_abs),
                    "feature_value": round(float(row[name]), 4),
                    "explanation_text": _explanation_text(name, float(row[name]), row),
                    "rank": i,
                }
            )
        return "shap_tree", out

    imps = bundle.get("feature_importances")
    meds = bundle.get("feature_medians")
    approx = top_local_factors(row, names, imps, meds, top_k=top_k)
    out2: list[dict[str, Any]] = []
    # Filter hidden features unless explicitly included.
    approx_items = approx
    if not include_hidden:
        approx_items = [it for it in approx if str(it.get("feature")) not in HIDDEN_FROM_USER_INSIGHTS]

    # Strength proxy from share percent in this fallback view.
    max_share = max((float(it.get("share_percent", 0.0)) for it in approx_items), default=0.0)
    for i, item in enumerate(approx_items, start=1):
        feat = str(item["feature"])
        val = float(row[feat])
        share = float(item["share_percent"])
        out2.append(
            {
                "feature": feat,
                "label": str(item["label"]),
                "display_name": _display_name(feat),
                "method": "approximate",
                "approx_share_percent": share,
                "note": str(item["note"]),
                "feature_value": round(float(val), 4),
                "direction": "neutral",
                "strength": _strength_bucket(share, max_share),
                "explanation_text": _explanation_text(feat, val, row),
                "rank": i,
            }
        )
    return "approximate", out2
