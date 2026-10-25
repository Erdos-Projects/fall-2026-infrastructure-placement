import geopandas as gpd
import numpy as np
import pandas as pd

from src.features.preprocessing import (ROOT, DATA_PATH, SELECTION_PATH, TARGET, META_COLUMNS, load_selected_features)
from src.features.transformers import build_features


SUBBASIN_PATH = ROOT / "notebooks" / "models" / "salinas_valley_subbasins.geojson"
OUTSIDE = "OUTSIDE"


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



def load_model_data(data_path=DATA_PATH, selection_path=SELECTION_PATH, log_target=True,
                    subbasin=False, drop_elevation=None):
    """Load the integrated dataset and return model inputs.

    subbasin=True adds one 0/1 column per subbasin. drop_elevation defaults to the
    same value as `subbasin`, so subbasin=True swaps elevation for subbasin;
    pass drop_elevation=False to keep both.
    """
    df = pd.read_csv(data_path, parse_dates=["gm_samp_collection_date"])
    df = build_features(df)

    features = load_selected_features(selection_path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Selected features not found in the dataset: {missing}")

    X = df[features]
    if subbasin:
        df = add_subbasin(df)
        X = pd.concat([X, subbasin_dummies(df)], axis=1)
    if drop_elevation if drop_elevation is not None else subbasin:
        X = X.drop(columns="elevation_m")

    y = np.log1p(df[TARGET]) if log_target else df[TARGET]
    meta = df[META_COLUMNS + (["subbasin"] if subbasin else [])]
    return X, y, meta
