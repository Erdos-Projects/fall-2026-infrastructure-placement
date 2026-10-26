"""Engineered features used by the nitrate models.

Each function takes the integrated dataset as a DataFrame and returns a copy
with the engineered columns added. Original columns are not modified.
Decisions behind these features are documented in notebooks/feature_selection.ipynb.
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUBBASIN_PATH = ROOT / "notebooks" / "models" / "salinas_valley_subbasins.geojson"
OUTSIDE = "OUTSIDE"


# Subbasin data

def subbasin_categories(subbasin_path=SUBBASIN_PATH):
    """All subbasin names in the file, plus OUTSIDE, in a fixed order."""
    names = gpd.read_file(subbasin_path)["Basin_Subbasin_Name"]
    return sorted(names.str.replace("SALINAS VALLEY - ", "", regex=False).unique()) + [OUTSIDE]


def add_subbasin(df, subbasin_path=SUBBASIN_PATH, lat_col="gm_latitude", lon_col="gm_longitude",
                 crs="EPSG:4326"):
    """Add `subbasin`: the DWR Bulletin 118 subbasin each well lies in (OUTSIDE if none)."""
    subbasins = gpd.read_file(subbasin_path)
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=crs, index=df.index
    ).to_crs(subbasins.crs)
    joined = gpd.sjoin(points, subbasins[["Basin_Subbasin_Name", "geometry"]],
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated()]          # a point on a shared border: keep one

    names = (joined["Basin_Subbasin_Name"]
             .str.replace("SALINAS VALLEY - ", "", regex=False)
             .fillna(OUTSIDE))
    out = df.copy()
    out["subbasin"] = pd.Categorical(names.reindex(df.index),
                                     categories=subbasin_categories(subbasin_path))
    return out


def subbasin_dummies(df):
    """One 0/1 column per subbasin category, always the same columns."""
    return pd.get_dummies(df["subbasin"], prefix="subbasin", dtype=int)



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
