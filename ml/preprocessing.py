"""
Survey → numeric feature matrix for academic risk classification.

Used by notebooks and the FastAPI backend so encoding, CEI, and feature order
stay identical between training and inference.

Target: ``risk_label`` (0 = on track, 1 = at risk) from self-rated academic
performance (1–5): {1,2} → 1, {3,4,5} → 0.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ml.features import add_crisis_exposure_index

TARGET_COLUMN = "risk_label"

MODEL_FEATURE_ORDER: tuple[str, ...] = (
    "year_of_study",
    "gpa",
    "study_hours",
    "attendance",
    "motivation",
    "quiet_environment",
    "mental_health",
    "sleep_quality",
    "stress_academic",
    "breaks_relaxation",
    "electricity_stability",
    "internet_stability",
    "economic_political_impact",
    "war_conflict_impact",
    "crisis_exposure_index",
)


def clean_survey_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("?", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return df


def drop_high_cardinality_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_candidates = (
        "timestamp",
        "1university_name",
        "2field_of_study__major",
        "2what_are_the_top_3_factors_affecting_your_academic_performance_the_most",
        "3would_you_be_interested_in_receiving_ai-based_feedback_or_advice_on_improving_your_performance",
    )
    to_drop = [c for c in drop_candidates if c in df.columns]
    return df.drop(columns=to_drop, errors="ignore")


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _survey_column_map(df: pd.DataFrame) -> dict[str, str]:
    m: dict[str, str] = {}
    pairs: dict[str, tuple[str, ...]] = {
        "year": ("1year_of_study", "3year_of_study"),
        "gpa": ("2average_gpa", "4average_gpa"),
        "study_hours": ("1how_many_hours_do_you_study_daily_on_average",),
        "attendance": ("2how_often_do_you_attend_your_classes_or_lectures",),
        "motivation": ("3how_would_you_rate_your_current_motivation_for_studying",),
        "quiet_env": ("4do_you_usually_study_in_a_quiet_and_stable_environment",),
        "mental_health": ("1how_would_you_rate_your_mental_health_lately",),
        "sleep": ("2how_many_hours_of_sleep_do_you_get_per_night_on_average",),
        "stress": ("3how_often_do_you_feel_stressed_or_anxious_due_to_academic_work",),
        "breaks": ("12how_often_do_you_take_breaks_or_practice_relaxation_exercise,_hobbies,_etc",),
        "electricity": (
            "1_how_stable_is_your_access_to_electricity_at_home",
            "1how_stable_is_your_access_to_electricity_at_home",
        ),
        "internet": ("2how_stable_is_your_internet_connection_for_study_purposes",),
        "economic": (
            "3have_you_been_directly_affected_by_ongoing_economic_or_political_crises_job_loss,_displacement,_etc",
        ),
        "war": ("4have_recent_war_or_conflict_situations_impacted_your_focus_or_ability_to_study",),
        "performance": ("1how_would_you_rate_your_academic_performance_this_semester",),
    }
    for key, cands in pairs.items():
        found = _first_existing(df, cands)
        if found is not None:
            m[key] = found
    return m


def norm_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _map_series(s: pd.Series, mapping: Mapping[str, float]) -> pd.Series:
    return norm_text(s).map(mapping)


YEAR_MAP: dict[str, int] = {
    "1st year": 1,
    "2nd year": 2,
    "3rd year": 3,
    "4th year": 4,
    "5th year": 5,
    "graduate": 5,
}

GPA_MAP: dict[str, float] = {
    "below 2.0": 1,
    "2.0 - 2.99": 2,
    "2.0-2.99": 2,
    "3.0 - 3.49": 3,
    "3.0-3.49": 3,
    "3.5 - 4.3": 4,
    "3.5-4.3": 4,
    "nan": np.nan,
}

STUDY_HOURS_MAP: dict[str, int] = {
    "less than 1 hour": 1,
    "1-2 hours": 2,
    "3-4 hours": 3,
    "more than 4 hours": 4,
}

ATTENDANCE_MAP: dict[str, int] = {
    "rarely": 1,
    "sometimes": 2,
    "often": 3,
    "almost always": 4,
}

SLEEP_MAP: dict[str, int] = {
    "less than 5 hours": 1,
    "5-6 hours": 2,
    "7-8 hours": 3,
    "more than 8 hours": 4,
}

FREQ4_MAP: dict[str, int] = {
    "rarely": 1,
    "sometimes": 2,
    "often": 3,
    "always": 4,
}

QUIET_ENV_MAP: dict[str, int] = {
    "rarely": 1,
    "sometimes": 2,
    "yes": 3,
    "often": 3,
    "always": 3,
}

STABILITY_MAP: dict[str, int] = {
    "always available": 4,
    "mostly available": 3,
    "sometimes available": 2,
    "rarely available": 1,
    "severe instability": 0,
    "very stable": 4,
    "stable": 3,
    "somewhat stable": 2,
    "unstable": 1,
    "very unstable": 0,
    "frequently disrupted": 1,
    "occasionally disrupted": 2,
    "almost always avaiable": 4,
    "almost always available": 4,
    "long outages (4-8 hours/day)": 1,
    "long outages (4-8 hours day)": 1,
    "few daily outages": 2,
}

IMPACT_MAP: dict[str, int] = {
    "not at all": 0,
    "not really": 1,
    "somewhat": 2,
    "yes, significantly": 3,
    "yes, severely": 4,
    "barely": 1,
    "slightly": 2,
    "moderately": 3,
}


def _electricity_stability_fallback(t: str) -> float | None:
    if not t or t == "nan":
        return None
    tl = t.lower()
    if "long outage" in tl or "4-8" in tl or "4–8" in tl:
        return 1.0
    if "few daily" in tl:
        return 2.0
    if "frequent" in tl and "disrupt" in tl:
        return 1.0
    if "occasional" in tl and "disrupt" in tl:
        return 2.0
    if "almost always" in tl or "avaiable" in tl:
        return 4.0
    return None


def encode_ordinal_columns(df: pd.DataFrame, cmap: Mapping[str, str]) -> pd.DataFrame:
    out = df.copy()

    ycol = cmap.get("year")
    if ycol and ycol in out.columns:
        ym = {k: float(v) for k, v in YEAR_MAP.items()}
        ym["nan"] = np.nan
        out[ycol] = _map_series(out[ycol], ym)

    gcol = cmap.get("gpa")
    if gcol and gcol in out.columns:
        out[gcol] = _map_series(out[gcol], GPA_MAP)

    key_map_pairs = (
        ("study_hours", STUDY_HOURS_MAP),
        ("attendance", ATTENDANCE_MAP),
        ("sleep", SLEEP_MAP),
        ("stress", FREQ4_MAP),
        ("breaks", FREQ4_MAP),
        ("quiet_env", QUIET_ENV_MAP),
        ("internet", STABILITY_MAP),
        ("economic", IMPACT_MAP),
        ("war", IMPACT_MAP),
    )
    for key, mapping in key_map_pairs:
        c = cmap.get(key)
        if not c or c not in out.columns:
            continue
        mm = {k: float(v) for k, v in mapping.items()}
        out[c] = norm_text(out[c]).map(mm)

    ecol = cmap.get("electricity")
    if ecol and ecol in out.columns:
        raw = norm_text(out[ecol])
        smap = {k: float(v) for k, v in STABILITY_MAP.items()}
        mapped = raw.map(smap)
        fb = raw.map(_electricity_stability_fallback)
        merged = mapped.copy()
        mask = merged.isna() & fb.notna()
        merged.loc[mask] = fb.loc[mask]
        out[ecol] = pd.to_numeric(merged, errors="coerce")

    mh = cmap.get("mental_health")
    if mh and mh in out.columns:
        out[mh] = pd.to_numeric(out[mh], errors="coerce")
    mot = cmap.get("motivation")
    if mot and mot in out.columns:
        out[mot] = pd.to_numeric(out[mot], errors="coerce")

    return out


def _rename_to_canonical(df: pd.DataFrame, cmap: Mapping[str, str]) -> pd.DataFrame:
    rev = {}
    for key, canon in (
        ("year", "year_of_study"),
        ("gpa", "gpa"),
        ("study_hours", "study_hours"),
        ("attendance", "attendance"),
        ("motivation", "motivation"),
        ("quiet_env", "quiet_environment"),
        ("mental_health", "mental_health"),
        ("sleep", "sleep_quality"),
        ("stress", "stress_academic"),
        ("breaks", "breaks_relaxation"),
        ("electricity", "electricity_stability"),
        ("internet", "internet_stability"),
        ("economic", "economic_political_impact"),
        ("war", "war_conflict_impact"),
    ):
        if key in cmap and cmap[key] in df.columns:
            rev[cmap[key]] = canon
    return df.rename(columns=rev)


def impute_medians(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            continue
        if out[c].isna().all():
            continue
        med = out[c].median()
        if pd.isna(med):
            continue
        out[c] = out[c].fillna(med)
    return out


def add_risk_label(df: pd.DataFrame, cmap: Mapping[str, str] | None = None) -> pd.DataFrame:
    out = df.copy()
    col = cmap.get("performance") if cmap else None
    if not col or col not in out.columns:
        col = _first_existing(
            out,
            ("1how_would_you_rate_your_academic_performance_this_semester",),
        )
    if col is None:
        out[TARGET_COLUMN] = pd.Series(np.nan, index=out.index, dtype="float")
        return out
    perf = pd.to_numeric(out[col], errors="coerce")
    risk_map = {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}
    out[TARGET_COLUMN] = perf.map(risk_map).astype("Int64")
    return out


def preprocess_survey_dataframe(
    df: pd.DataFrame,
    *,
    create_label: bool = True,
    drop_meta: bool = True,
    impute: bool = True,
) -> pd.DataFrame:
    out = clean_survey_column_names(df)
    if drop_meta:
        out = drop_high_cardinality_columns(out)

    cmap = _survey_column_map(out)
    out = encode_ordinal_columns(out, cmap)

    if create_label:
        out = add_risk_label(out, cmap)
        pcol = cmap.get("performance")
        if pcol and pcol in out.columns:
            out = out.drop(columns=[pcol])

    out = _rename_to_canonical(out, cmap)
    out = add_crisis_exposure_index(out)

    feature_cols = [c for c in MODEL_FEATURE_ORDER if c in out.columns]
    if impute:
        out = impute_medians(out, feature_cols)

    return out


def prepare_X_y_classification(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    proc = preprocess_survey_dataframe(df, create_label=True)
    if TARGET_COLUMN not in proc.columns:
        raise ValueError("risk_label missing after preprocessing")

    feature_cols = [c for c in MODEL_FEATURE_ORDER if c in proc.columns]
    X = proc[feature_cols].copy()
    y = proc[TARGET_COLUMN].copy()

    valid = y.notna() & X.notna().all(axis=1)
    X = X.loc[valid].copy()
    y = y.loc[valid].astype(int).copy()
    proc_kept = proc.loc[valid].copy()

    return X, y, proc_kept
