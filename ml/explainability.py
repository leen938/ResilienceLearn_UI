"""
Explainability utilities: global feature importance from artifacts, local SHAP when available.

Design: SHAP is attempted first for /explain; on failure (missing shap, explainer error),
callers fall back to importance x |x - median| (same idea as prediction_extras.top_local_factors).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ml.prediction_extras import FEATURE_LABELS, top_local_factors

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
        import shap  # noqa: WPS433 — optional dependency
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
        picked = pairs[:top_k]
        out: list[dict[str, Any]] = []
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
                    "label": FEATURE_LABELS.get(name, name.replace("_", " ").title()),
                    "method": "shap_tree",
                    "shap_value": round(float(shap_v), 6),
                    "direction": direction,
                    "feature_value": round(float(row[name]), 4),
                    "rank": i,
                }
            )
        return "shap_tree", out

    imps = bundle.get("feature_importances")
    meds = bundle.get("feature_medians")
    approx = top_local_factors(row, names, imps, meds, top_k=top_k)
    out2: list[dict[str, Any]] = []
    for i, item in enumerate(approx, start=1):
        out2.append(
            {
                "feature": str(item["feature"]),
                "label": str(item["label"]),
                "method": "approximate",
                "approx_share_percent": float(item["share_percent"]),
                "note": str(item["note"]),
                "feature_value": round(float(row[str(item["feature"])]), 4),
                "rank": i,
            }
        )
    return "approximate", out2
