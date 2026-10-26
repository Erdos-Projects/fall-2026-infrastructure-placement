"""Prepare model-ready data from the integrated lag-1 dataset.

The selected features are read from results/feature_selection.csv (rows with
decision == "keep"), so the modeling stage always uses the decisions made in
notebooks/feature_selection.ipynb.

Example
-------
from src.features.preprocessing import load_model_data

X, y, meta = load_model_data()
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.transformers import build_features, add_subbasin, subbasin_dummies



ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "notebooks" / "merge" / "groundwater_ml_integration_lag1_2021_2024_with_climate.csv"
SELECTION_PATH = ROOT / "results" / "feature_selection.csv"

TARGET = "nitrate_mg_L"

# Kept alongside X (not as predictors) because the spatial and temporal
# splits in src/evaluation/splits.py need them.
META_COLUMNS = ["gm_well_id", "gm_latitude", "gm_longitude", "gm_samp_collection_date", "year"]


def load_selected_features(selection_path=SELECTION_PATH):
    """Return the list of features marked "keep" in the feature selection log."""
    selection = pd.read_csv(selection_path)
    return selection.loc[selection["decision"] == "keep", "feature"].tolist()


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
