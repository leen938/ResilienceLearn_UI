"""
Crisis Exposure Index (CEI) and related derived features.

CEI summarizes external crisis burden on [0, 1]: higher means more exposure
(bad infrastructure + higher reported crisis impacts). Uses only engineered
numeric columns after ordinal encoding in preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_crisis_exposure_index(
    electricity_stability: pd.Series,
    internet_stability: pd.Series,
    economic_political_impact: pd.Series,
    war_conflict_impact: pd.Series,
) -> pd.Series:
    """
    Build CEI as the mean of normalized components:

    - electricity / internet: 0–4 where higher = better access → burden = 1 - v/4
    - economic / war impact: 0–4 where higher = worse impact → burden = v/4

    Rows where all four are NaN yield NaN; partial rows use nanmean (numpy).
    """
    e = pd.to_numeric(electricity_stability, errors="coerce").astype(float)
    n = pd.to_numeric(internet_stability, errors="coerce").astype(float)
    c = pd.to_numeric(economic_political_impact, errors="coerce").astype(float)
    w = pd.to_numeric(war_conflict_impact, errors="coerce").astype(float)

    b_elec = 1.0 - np.clip(e / 4.0, 0.0, 1.0)
    b_net = 1.0 - np.clip(n / 4.0, 0.0, 1.0)
    b_crisis_econ = np.clip(c / 4.0, 0.0, 1.0)
    b_crisis_war = np.clip(w / 4.0, 0.0, 1.0)

    mat = np.column_stack([b_elec, b_net, b_crisis_econ, b_crisis_war])
    cei = np.nanmean(mat, axis=1)
    return pd.Series(cei, index=electricity_stability.index, name="crisis_exposure_index")


def add_crisis_exposure_index(df: pd.DataFrame) -> pd.DataFrame:
    """In-place style: adds ``crisis_exposure_index`` if crisis columns exist."""
    need = (
        "electricity_stability",
        "internet_stability",
        "economic_political_impact",
        "war_conflict_impact",
    )
    if not all(c in df.columns for c in need):
        return df
    df = df.copy()
    df["crisis_exposure_index"] = compute_crisis_exposure_index(
        df["electricity_stability"],
        df["internet_stability"],
        df["economic_political_impact"],
        df["war_conflict_impact"],
    )
    return df
