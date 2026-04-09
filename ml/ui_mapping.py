"""
Map Streamlit / dashboard inputs to the 15-feature JSON expected by ``POST /predict``.

Training uses survey-derived ordinals; the UI uses different scales, so we convert with
transparent, documentable rules and recompute CEI with the same formula as ``ml.features``.
"""

from __future__ import annotations

import pandas as pd

from ml.features import compute_crisis_exposure_index
from ml.preprocessing import MODEL_FEATURE_ORDER


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _study_hours_ordinal(hours_per_day: float) -> float:
    """Survey buckets: 1–4 ordinal (less than 1h … more than 4h)."""
    if hours_per_day < 1:
        return 1.0
    if hours_per_day < 3:
        return 2.0
    if hours_per_day < 5:
        return 3.0
    return 4.0


def _absences_to_attendance(absences: int) -> float:
    """More absences → lower attendance ordinal (1–4)."""
    step = absences / 7.5
    att = 4.0 - step
    return round(_clamp(att, 1.0, 4.0))


def _slider_1_10_to_sleep_ordinal(slider: int) -> float:
    """Map 1–10 sleep *quality* to sleep-duration ordinal 1–4 used in training."""
    return float(round(_clamp(1.0 + (slider - 1) * 3.0 / 9.0, 1.0, 4.0)))


def _slider_1_10_to_1_5_scale(slider: int) -> float:
    """Rough alignment: survey motivation / mental health were 1–5; UI uses 1–10."""
    return _clamp(1.0 + (slider - 1) * 4.0 / 9.0, 1.0, 5.0)


def ui_checkin_to_predict_body(
    inputs: dict,
    *,
    year_of_study: float = 3.0,
    gpa: float = 3.0,
    quiet_environment: float = 3.0,
    stress_academic: float = 2.0,
    breaks_relaxation: float = 3.0,
) -> dict[str, float]:
    """
    ``inputs`` keys match ``streamlit_app_ui_only.py`` sidebar:
    study_hours, absences, sleep_quality, mental_health, financial_stress,
    electricity_hours, internet_quality, war_stress, motivation.
    """
    sh = _study_hours_ordinal(float(inputs["study_hours"]))
    att = _absences_to_attendance(int(inputs["absences"]))
    sleep_ord = _slider_1_10_to_sleep_ordinal(int(inputs["sleep_quality"]))
    mental = _slider_1_10_to_1_5_scale(int(inputs["mental_health"]))
    motiv = _slider_1_10_to_1_5_scale(int(inputs["motivation"]))

    elec = round(_clamp(float(inputs["electricity_hours"]) / 24.0 * 4.0, 0.0, 4.0))
    net = round(_clamp((int(inputs["internet_quality"]) - 1) / 9.0 * 4.0, 0.0, 4.0))
    econ = round(_clamp((int(inputs["financial_stress"]) - 1) / 9.0 * 4.0, 0.0, 4.0))
    war = round(_clamp((int(inputs["war_stress"]) - 1) / 9.0 * 4.0, 0.0, 4.0))

    cei_series = compute_crisis_exposure_index(
        pd.Series([elec]),
        pd.Series([net]),
        pd.Series([econ]),
        pd.Series([war]),
    )
    cei = float(cei_series.iloc[0])

    row = {
        "year_of_study": float(year_of_study),
        "gpa": float(gpa),
        "study_hours": float(sh),
        "attendance": float(att),
        "motivation": float(motiv),
        "quiet_environment": float(quiet_environment),
        "mental_health": float(mental),
        "sleep_quality": float(sleep_ord),
        "stress_academic": float(stress_academic),
        "breaks_relaxation": float(breaks_relaxation),
        "electricity_stability": float(elec),
        "internet_stability": float(net),
        "economic_political_impact": float(econ),
        "war_conflict_impact": float(war),
        "crisis_exposure_index": float(cei),
    }

    ordered = {k: row[k] for k in MODEL_FEATURE_ORDER}
    assert list(ordered.keys()) == list(MODEL_FEATURE_ORDER)
    return ordered
