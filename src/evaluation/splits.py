import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import StratifiedGroupKFold


EQUAL_AREA_CRS = "EPSG:3310"


def _to_equal_area_points(df, lat_col, lon_col, df_crs, eq_crs=EQUAL_AREA_CRS):
    if isinstance(df, gpd.GeoDataFrame):
        if df.crs is None:
            raise ValueError("GeoDataFrame has no CRS; set one with .set_crs(...) first.")
        return df.geometry.to_crs(eq_crs)
    points = gpd.points_from_xy(df[lon_col], df[lat_col])
    return gpd.GeoSeries(points, crs=df_crs, index=df.index).to_crs(eq_crs)


def assign_grid_cells(df, lat_col="gm_latitude", lon_col="gm_longitude", cell_size_m=10_000, eq_crs=EQUAL_AREA_CRS, df_crs="EPSG:4326"):
    """Takes in a dataframe with indicated latitude and longitude columns and returns a dataframe assigning each row to a cell."""  
    points = _to_equal_area_points(df, lat_col, lon_col, df_crs=df_crs, eq_crs=eq_crs)
    cell_x = np.floor(points.x / cell_size_m).astype(int)
    cell_y = np.floor(points.y / cell_size_m).astype(int)
    return pd.DataFrame({
        "cell_x": cell_x,
        "cell_y": cell_y,
        "cell": cell_x.astype(str) + "_" + cell_y.astype(str)
    }, index=df.index)


def spatial_folds(cells, exceeds, n_splits=5, seed=0):
    """One fold number per row. Whole cells stay together, and folds are balanced on the
    labels in `exceeds`: exceedance True/False, or any other categories
    (e.g. subbasin x exceedance)."""
    cells = np.asarray(cells)
    folds = np.empty(len(cells), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(np.zeros(len(cells)), exceeds, groups=cells)):
        folds[test] = k
    return folds


def iter_folds(folds):
    """One (train, test) pair of True/False masks per fold."""
    folds = np.asarray(folds)
    pairs = []
    for k in np.unique(folds):
        test = (folds == k)
        pairs.append((~test, test))
    return pairs


def declustering_weights(cells):
    """Assigns a weight to each row so that each cell is counted equally in the score, regardless of how many samples it contains. (See `Estimation of aquifer scale proportion using equal area grids: Assessment of regional scale groundwater quality` by Belitz et al.)"""
    cells = pd.Series(cells)
    w = 1.0 / cells.map(cells.value_counts())
    return (w / w.mean()).to_numpy()


def check_split(df, train, test, cols=("gm_well_id", "cell")):
    """Check to make sure that the splits are done correctly and there are no rows assigned to both train and test."""
    for col in cols:
        shared = set(df.loc[train, col]) & set(df.loc[test,col])
        if shared:
            raise AssertionError(f"... e.g. {next(iter(shared))}")
    if (train & test).any():
        raise AssertionError("Some rows are in both train and test.")



def time_split(dates, test_years):
    """Splits data by year for time series prediction for known wells."""
    years = pd.to_datetime(dates).dt.year
    test = years.isin(test_years).to_numpy()
    train = (years < min(test_years)).to_numpy()
    return train, test


def space_time_split(dates, cells, test_years, test_cells):
    """Splits data both spatially and temporally to predict future values at previously unsampled locations."""
    train_t, test_t = time_split(dates, test_years)
    in_test_cells = pd.Series(cells).isin(test_cells).to_numpy()
    return train_t & ~in_test_cells, test_t & in_test_cells




def well_fold_table(df, stratify_by, cell_size_m=10_000, n_splits=5, seed=0, well_col="gm_well_id", **cell_kwargs):
    """One row per well: its grid cell and fold. Folds are computed on the samples (balanced on `stratify_by`, one label per row of df) and reduced to one fold per well."""
    cells = assign_grid_cells(df, cell_size_m=cell_size_m, **cell_kwargs)["cell"]
    folds = spatial_folds(cells, stratify_by, n_splits=n_splits, seed=seed)
    table = (pd.DataFrame({well_col: df[well_col].to_numpy(), "cell": cells.to_numpy(), "fold": folds})
        .drop_duplicates()
        .sort_values(well_col)
        .reset_index(drop=True))
    if table[well_col].duplicated().any():
        raise ValueError("A well was assigned to more than one cell or fold.")
    return table



def fold_balance(folds, subbasins, exceeds):
    """Worst deviation of a subbasin's share in any fold, and the spread of exceedance shares."""
    folds = np.asarray(folds)
    subbasins = pd.Series(np.asarray(subbasins))
    exceeds = pd.Series(np.asarray(exceeds))
    shares = pd.crosstab(folds, subbasins.to_numpy(), normalize="index")
    deviation = (shares - subbasins.value_counts(normalize=True)).abs().max().max()
    by_fold = exceeds.groupby(folds).mean()
    return {"max_subbasin_deviation": round(float(deviation), 3),"exceedance_spread": round(float(by_fold.max() - by_fold.min()), 3)}




def attach_folds(df, fold_path, well_col="gm_well_id"):
    """Add the shared `cell` and `fold` columns to any table with a well-ID column."""
    folds = pd.read_csv(fold_path, dtype={well_col: "string"})
    out = df.merge(folds, on=well_col, how="left", validate="many_to_one")
    missing = out["fold"].isna().sum()
    if missing:
        raise ValueError(f"{missing} rows have wells that are not in the fold table.")
    out["fold"] = out["fold"].astype(int)
    return out
