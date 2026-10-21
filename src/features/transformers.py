"""Engineered features used by the nitrate models.

Each function takes the integrated dataset as a DataFrame and returns a copy
with the engineered columns added. Original columns are not modified.
Decisions behind these features are documented in notebooks/feature_selection.ipynb.
"""

import pandas as pd

# Seven SSURGO drainage classes grouped by drainage speed.
# Values are % of the well buffer, so each group is the sum of its classes.
DRAINAGE_GROUPS = {
    "soil_drainage_rapid_pct": ["excessively_drained", "somewhat_excessively_drained"],
    "soil_drainage_moderate_pct": ["well_drained", "moderately_well_drained"],
    "soil_drainage_slow_pct": ["somewhat_poorly_drained", "poorly_drained", "very_poorly_drained"],
}

# Dual hydrologic group C/D behaves as D when undrained (USDA convention).
HYDGROUP_D = ["soil_hydgrpdcd_pct_c_d", "soil_hydgrpdcd_pct_d"]


def add_soil_groups(df):
    """Add grouped drainage classes and the combined hydrologic group D.

    min_count=1 keeps a row missing (NaN) when none of its classes has data,
    instead of filling unknown soil coverage with 0.
    """
    new_cols = {}
    for new_col, classes in DRAINAGE_GROUPS.items():
        original = [f"soil_drclassdcd_pct_{c}" for c in classes]
        new_cols[new_col] = df[original].sum(axis=1, min_count=1)
    new_cols["soil_hydgrp_d_pct"] = df[HYDGROUP_D].sum(axis=1, min_count=1)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def encode_well_category(df):
    """Convert well category to 0/1 (1 = Domestic, 0 = Irrigation / Industrial)."""
    out = df.copy()
    out["gm_well_category"] = (out["gm_well_category"] == "Domestic").astype(int)
    return out


def build_features(df):
    """Apply all engineered features in order."""
    return encode_well_category(add_soil_groups(df))
