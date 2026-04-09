#Preprocessing transforms raw survey data into a structured numerical format suitable for machine learning. 
# This includes cleaning inconsistent text, encoding categorical variables into ordinal values, handling missing data, and engineering additional features to enhance predictive performance
import pandas as pd
import numpy as np

# -------------------- NORMALIZATION --------------------

def norm_text(s):
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .str.replace(r"\s+", " ", regex=True)
    )

# -------------------- MAPS --------------------

year_map = {
    "1st year": 1,
    "2nd year": 2,
    "3rd year": 3,
    "4th year": 4,
    "5th year": 5,
    "graduate": 5
}

gpa_map = {
    "below 2.0": 1,
    "2.0 - 2.99": 2,
    "2.0-2.99": 2,
    "3.0 - 3.49": 3,
    "3.0-3.49": 3,
    "3.5 - 4.3": 4,
    "3.5-4.3": 4,
    "nan": np.nan
}

study_hours_map = {
    "less than 1 hour": 1,
    "1-2 hours": 2,
    "3-4 hours": 3,
    "more than 4 hours": 4
}

attendance_map = {
    "rarely": 1,
    "sometimes": 2,
    "often": 3,
    "almost always": 4
}

sleep_map = {
    "less than 5 hours": 1,
    "5-6 hours": 2,
    "7-8 hours": 3,
    "more than 8 hours": 4
}

freq4_map = {
    "rarely": 1,
    "sometimes": 2,
    "often": 3,
    "always": 4
}

quiet_env_map = {
    "rarely": 1,
    "sometimes": 2,
    "yes": 3,
    "often": 3,
    "always": 3
}

stability_map = {
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
    "occasionally disrupted": 2
}

impact_map = {
    "not at all": 0,
    "not really": 1,
    "somewhat": 2,
    "yes, significantly": 3,
    "yes, severely": 4,
    "barely": 1,
    "slightly": 2,
    "moderately": 3
}

# -------------------- COLUMN GROUPS --------------------

ENCODED_COLUMNS = [
    "3year_of_study",
    "4average_gpa",
    "1how_many_hours_do_you_study_daily_on_average",
    "2how_often_do_you_attend_your_classes_or_lectures",
    "2how_many_hours_of_sleep_do_you_get_per_night_on_average",
    "3how_often_do_you_feel_stressed_or_anxious_due_to_academic_work",
    "12how_often_do_you_take_breaks_or_practice_relaxation_exercise,_hobbies,_etc",
    "4do_you_usually_study_in_a_quiet_and_stable_environment",
    "1how_stable_is_your_access_to_electricity_at_home",
    "2how_stable_is_your_internet_connection_for_study_purposes",
    "3have_you_been_directly_affected_by_ongoing_economic_or_political_crises_job_loss,_displacement,_etc",
    "4have_recent_war_or_conflict_situations_impacted_your_focus_or_ability_to_study",
    "3how_would_you_rate_your_current_motivation_for_studying",
    "1how_would_you_rate_your_mental_health_lately",
]

TARGET_COLUMN = "1how_would_you_rate_your_mental_health_lately"

# -------------------- HELPERS --------------------

def map_with_report(df, col, mapping, verbose=True):
    if col not in df.columns:
        if verbose:
            print(f"Skipped (missing column): {col}")
        return df

    original = norm_text(df[col])
    df[col] = original.map(mapping)

    unmapped = original[df[col].isna() & original.ne("nan")].unique()
    if len(unmapped) > 0 and verbose:
        print(f"\nUnmapped values in {col}:")
        print(unmapped)

    return df

# -------------------- MAIN PREPROCESSING --------------------

def preprocess_dataframe(df, verbose=True):
    df = df.copy()

    # apply ordinal encoding
    df = map_with_report(df, "3year_of_study", year_map, verbose)
    df = map_with_report(df, "4average_gpa", gpa_map, verbose)
    df = map_with_report(df, "1how_many_hours_do_you_study_daily_on_average", study_hours_map, verbose)
    df = map_with_report(df, "2how_often_do_you_attend_your_classes_or_lectures", attendance_map, verbose)
    df = map_with_report(df, "2how_many_hours_of_sleep_do_you_get_per_night_on_average", sleep_map, verbose)
    df = map_with_report(df, "3how_often_do_you_feel_stressed_or_anxious_due_to_academic_work", freq4_map, verbose)
    df = map_with_report(df, "12how_often_do_you_take_breaks_or_practice_relaxation_exercise,_hobbies,_etc", freq4_map, verbose)
    df = map_with_report(df, "4do_you_usually_study_in_a_quiet_and_stable_environment", quiet_env_map, verbose)
    df = map_with_report(df, "1how_stable_is_your_access_to_electricity_at_home", stability_map, verbose)
    df = map_with_report(df, "2how_stable_is_your_internet_connection_for_study_purposes", stability_map, verbose)
    df = map_with_report(df, "3have_you_been_directly_affected_by_ongoing_economic_or_political_crises_job_loss,_displacement,_etc", impact_map, verbose)
    df = map_with_report(df, "4have_recent_war_or_conflict_situations_impacted_your_focus_or_ability_to_study", impact_map, verbose)

    # numeric conversion
    for c in [
        "3how_would_you_rate_your_current_motivation_for_studying",
        "1how_would_you_rate_your_mental_health_lately",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep only present encoded columns
    present = [c for c in ENCODED_COLUMNS if c in df.columns]

    # impute median for columns that are not entirely NaN
    impute_cols = [c for c in present if not df[c].isna().all()]
    for c in impute_cols:
        df[c] = df[c].fillna(df[c].median())
    # remove rows where target is missing
    if TARGET_COLUMN in df.columns:
     df = df[df[TARGET_COLUMN].notna()].copy()
    # optional engineered feature
    if (
        "3how_often_do_you_feel_stressed_or_anxious_due_to_academic_work" in df.columns
        and "2how_many_hours_of_sleep_do_you_get_per_night_on_average" in df.columns
    ):
        df["stress_sleep"] = (
            df["3how_often_do_you_feel_stressed_or_anxious_due_to_academic_work"] *
            df["2how_many_hours_of_sleep_do_you_get_per_night_on_average"]
        )

    return df

def prepare_X_y(df, target_col=TARGET_COLUMN):
    df = preprocess_dataframe(df, verbose=False)

    feature_cols = [
        c for c in ENCODED_COLUMNS
        if c in df.columns and c != target_col and not df[c].isna().all()
    ]

    if "stress_sleep" in df.columns and not df["stress_sleep"].isna().all():
        feature_cols.append("stress_sleep")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    valid_idx = y.notna()
    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    return X, y, df