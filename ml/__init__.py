"""Shared ML utilities for ResilienceLearn (preprocessing, CEI, feature contract)."""

from ml.preprocessing import (
    MODEL_FEATURE_ORDER,
    TARGET_COLUMN,
    clean_survey_column_names,
    drop_high_cardinality_columns,
    add_risk_label,
    preprocess_survey_dataframe,
    prepare_X_y_classification,
)

from ml.features import add_crisis_exposure_index, compute_crisis_exposure_index

__all__ = [
    "MODEL_FEATURE_ORDER",
    "TARGET_COLUMN",
    "clean_survey_column_names",
    "drop_high_cardinality_columns",
    "add_risk_label",
    "preprocess_survey_dataframe",
    "prepare_X_y_classification",
    "add_crisis_exposure_index",
    "compute_crisis_exposure_index",
]
