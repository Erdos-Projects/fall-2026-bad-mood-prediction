"""Load the hourly LifeSnaps split files and build the modeling table.

One row per mood report (BAD_MOOD not missing). Predictors are restricted to
information available *before* the report:
  - sleep: the night that ended on the report date (Fitbit dates sleep by wake day)
  - activity: the PREVIOUS day's totals (same-day totals include activity after the report)
  - person-level: age group, gender, BMI
  - context: hour of the report

Edit the feature lists below to change what goes into the models; everything
downstream reads from FEATURES.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID, DATE, TARGET = "id", "date", "BAD_MOOD"

SLEEP_FEATURES = [
    "daily_minutesAsleep",
    "daily_sleep_efficiency",
    "daily_minutesToFallAsleep",
    "daily_minutesAwake",
    "daily_minutesAfterWakeup",
    "daily_sleep_deep_ratio",
    "daily_sleep_light_ratio",
    "daily_sleep_rem_ratio",
    "daily_sleep_wake_ratio",
    "daily_sleep_points_percentage",
    "daily_nremhr",
    "daily_rmssd",
    "daily_full_sleep_breathing_rate",
    "daily_nightly_temperature",
]

# Taken from the previous calendar day (suffix _lag1 in the model table)
ACTIVITY_SOURCE = [
    "daily_lightly_active_minutes",
    "daily_moderately_active_minutes",
    "daily_very_active_minutes",
    "daily_sedentary_minutes",
    "distance_perday",
    "daily_exertion_points_percentage",
    "daily_resting_hr",
]
ACTIVITY_FEATURES = [f"{c}_lag1" for c in ACTIVITY_SOURCE]

PERSON_FEATURES = ["age_30plus", "male", "bmi_num"]
CONTEXT_FEATURES = ["hour"]

FEATURES = SLEEP_FEATURES + ACTIVITY_FEATURES + PERSON_FEATURES + CONTEXT_FEATURES

# Columns that must never be used as predictors (outcome or computed from future moods)
LEAKAGE_COLUMNS = [
    "GOOD_MOOD", "ALERT", "HAPPY", "NEUTRAL", "RESTED/RELAXED", "SAD", "TENSE/ANXIOUS", "TIRED",
    "time_until_badmood", "time_until_goodmood", "time_since_badmood", "time_since_goodmood",
]

_BMI_MAP = {"<19": 18.0, ">=25": 25.0, ">=30": 30.0}
_ACTIVITY_MINUTES = [
    "daily_lightly_active_minutes", "daily_moderately_active_minutes",
    "daily_very_active_minutes", "daily_sedentary_minutes",
]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.drop_duplicates([ID, "timestamp"])
    df[DATE] = pd.to_datetime(df[DATE])
    return df


def _person_level(df: pd.DataFrame) -> pd.DataFrame:
    """Demographics are constant per person; fill within person, then encode."""
    p = df.groupby(ID)[["age", "gender", "bmi"]].first()
    out = pd.DataFrame(index=p.index)
    out["age_30plus"] = p["age"].map({">=30": 1.0, "<30": 0.0})
    out["male"] = p["gender"].map({"MALE": 1.0, "FEMALE": 0.0})
    bmi = p["bmi"].astype(str).replace(_BMI_MAP)
    out["bmi_num"] = pd.to_numeric(bmi, errors="coerce")
    return out


def _daily_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per person-day with daily features; non-wear days -> NaN activity."""
    day = df.sort_values([ID, DATE]).groupby([ID, DATE], as_index=False)[
        SLEEP_FEATURES + ACTIVITY_SOURCE
    ].first()
    nonwear = day["daily_sedentary_minutes"] >= 1440
    day.loc[nonwear, ACTIVITY_SOURCE] = np.nan

    # previous calendar day's activity (only if it is literally the day before)
    prev = day[[ID, DATE] + ACTIVITY_SOURCE].copy()
    prev[DATE] = prev[DATE] + pd.Timedelta(days=1)
    prev = prev.rename(columns={c: f"{c}_lag1" for c in ACTIVITY_SOURCE})
    day = day.merge(prev, on=[ID, DATE], how="left")
    return day[[ID, DATE] + SLEEP_FEATURES + ACTIVITY_FEATURES]


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per mood report with FEATURES, TARGET, ID, DATE."""
    reports = df.loc[df[TARGET].notna(), [ID, DATE, "hour", TARGET]].copy()
    reports[TARGET] = reports[TARGET].astype(int)
    out = reports.merge(_daily_table(df), on=[ID, DATE], how="left")
    out = out.merge(_person_level(df), left_on=ID, right_index=True, how="left")
    return out[[ID, DATE, TARGET] + FEATURES].reset_index(drop=True)


CENTER_SOURCE = SLEEP_FEATURES + ACTIVITY_FEATURES
PC_FEATURES = [f"{c}_pc" for c in CENTER_SOURCE]


def add_person_centered(table: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add <feature>_pc = value minus that person's own average.

    The average uses only the person's Fitbit (sensor) days, never their mood
    labels, so it can be computed for a brand-new person from their device data.
    Caveat: it averages over the whole study period, including days after the
    report (a mild temporal look-ahead; an expanding mean would remove it).
    """
    day = _daily_table(df)
    means = day.groupby(ID)[CENTER_SOURCE].mean()
    out = table.copy()
    m = means.reindex(out[ID]).to_numpy()
    out[PC_FEATURES] = out[CENTER_SOURCE].to_numpy() - m
    return out


def load_xy(path: str, person_centered: bool = False):
    """Convenience: path -> (X, y, groups, full table).

    person_centered=True appends the *_pc features to the raw ones.
    """
    raw = load_raw(path)
    table = build_model_table(raw)
    feats = FEATURES
    if person_centered:
        table = add_person_centered(table, raw)
        feats = FEATURES + PC_FEATURES
    return table[feats], table[TARGET], table[ID], table
