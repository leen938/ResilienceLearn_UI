"""Helpers for API responses: uncertainty messaging and approximate top factors (not SHAP)."""

from __future__ import annotations

# Short, unique lines for the dashboard—no training-data or model-internals wording.
FEATURE_STUDENT_NOTES: dict[str, str] = {
    "year_of_study": "Your place in the program is part of what you shared here.",
    "gpa": "How your overall grades are doing is part of this picture.",
    "study_hours": "The time you put into studying shows up in this check-in.",
    "attendance": "Showing up to class is one thread in this snapshot.",
    "motivation": "How driven you feel about school is reflected here.",
    "quiet_environment": "Having a calmer place to work is in the mix.",
    "mental_health": "How you are doing emotionally is one important piece.",
    "sleep_quality": "Rest and sleep patterns are part of the story you told us.",
    "stress_academic": "How heavy school pressure feels is showing up here.",
    "breaks_relaxation": "Whether you can take breaks and unwind matters in this view.",
    "electricity_stability": "Reliable power for studying feeds into this summary.",
    "internet_stability": "Steady internet for learning is part of what you shared.",
    "economic_political_impact": "Day-to-day money stress is reflected in this check-in.",
    "war_conflict_impact": "Stress from conflict or safety concerns is in the picture.",
    "crisis_exposure_index": "Overall strain from the pressures around you is captured here.",
}

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
    """Proxy: high when probability is near 0.5. Returns (score 0–1, short supportive note)."""
    p = max(0.0, min(1.0, float(p_risk)))
    u = float(1.0 - abs(p - 0.5) * 2.0)
    if u >= 0.75:
        msg = (
            "This result sits in a middle range—use it as a nudge to notice patterns, not a final "
            "verdict on you."
        )
    elif u >= 0.5:
        msg = "Treat this as one snapshot. How you are actually doing day to day still comes first."
    else:
        msg = "The picture is a bit one-sided for this check-in—still line it up with how you feel in real life."
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
    Rank features by (global importance) × (distance from stored reference) for this row.
    Falls back to global importance only if medians/importances are missing.
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
                "note": FEATURE_STUDENT_NOTES.get(
                    name,
                    "This is one of the areas from your check-in that stood out most in this list.",
                ),
            }
        )
    return out
