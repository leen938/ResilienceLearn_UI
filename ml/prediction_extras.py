"""Helpers for API responses: uncertainty messaging and approximate top factors (not SHAP)."""

from __future__ import annotations

FEATURE_LABELS: dict[str, str] = {
    "year_of_study": "Year of study",
    "gpa": "GPA band",
    "study_hours": "Study hours (ordinal)",
    "attendance": "Class attendance",
    "motivation": "Motivation",
    "quiet_environment": "Quiet study environment",
    "mental_health": "Mental health self-rating",
    "sleep_quality": "Sleep pattern",
    "stress_academic": "Academic stress",
    "breaks_relaxation": "Breaks / relaxation",
    "electricity_stability": "Electricity access",
    "internet_stability": "Internet access",
    "economic_political_impact": "Economic / political impact",
    "war_conflict_impact": "War / conflict impact",
    "crisis_exposure_index": "Crisis exposure index (CEI)",
}


def uncertainty_from_probability(p_risk: float) -> tuple[float, str]:
    """Proxy: high when probability is near 0.5. Returns (score 0–1, supportive note)."""
    p = max(0.0, min(1.0, float(p_risk)))
    u = float(1.0 - abs(p - 0.5) * 2.0)
    if u >= 0.75:
        msg = (
            "This score is fairly close to the decision boundary - treat it as a gentle signal "
            "for reflection or support, not a final judgment."
        )
    elif u >= 0.5:
        msg = "The model is moderately confident; your own situation still matters most."
    else:
        msg = "The model leans clearer here - still combine with how you feel day to day."
    return u, msg


def top_local_factors(
    row: dict[str, float],
    feature_names: list[str],
    importances: dict[str, float] | None,
    medians: dict[str, float] | None,
    *,
    top_k: int = 3,
) -> list[dict[str, str | float]]:
    """
    Rank features by importance × |value − training median| for this row.
    Falls back to global importance-only ranking if medians/importances missing.
    """
    if not importances:
        return []

    if medians:
        scores: list[tuple[str, float]] = []
        for name in feature_names:
            imp = float(importances.get(name, 0.0))
            med = float(medians.get(name, row.get(name, 0.0)))
            val = float(row[name])
            scores.append((name, imp * abs(val - med)))
        scores.sort(key=lambda x: -x[1])
        picked = scores[:top_k]
    else:
        picked = sorted(importances.items(), key=lambda x: -x[1])[:top_k]
        picked = [(n, float(v)) for n, v in picked]

    total = sum(s for _, s in picked) or 1.0
    out: list[dict[str, str | float]] = []
    for name, score in picked:
        label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
        pct = round(100.0 * score / total, 1)
        out.append(
            {
                "feature": name,
                "label": label,
                "share_percent": pct,
                "note": "This theme stands out relative to typical responses in the training data - "
                "small, realistic steps still count.",
            }
        )
    return out
