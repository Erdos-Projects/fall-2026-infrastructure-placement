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

from src.features.transformers import build_features

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


def load_model_data(data_path=DATA_PATH, selection_path=SELECTION_PATH, log_target=True):
    """Load the integrated dataset and return model inputs.

    Returns
    -------
    X : DataFrame
        Selected predictors, with engineered features applied.
    y : Series
        Nitrate (mg/L), log(1 + nitrate) when log_target=True.
    meta : DataFrame
        Well ID, coordinates, sample date and year, for building splits.
    """
    df = pd.read_csv(data_path, parse_dates=["gm_samp_collection_date"])
    df = build_features(df)

    features = load_selected_features(selection_path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Selected features not found in the dataset: {missing}")

    X = df[features]
    y = np.log1p(df[TARGET]) if log_target else df[TARGET]
    meta = df[META_COLUMNS]
    return X, y, meta
