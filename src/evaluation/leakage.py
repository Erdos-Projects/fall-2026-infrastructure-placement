import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsRegressor


THRESHOLD_MG_L = 10


def projected_xy(df, lat_col, lon_col, eq_crs="EPSG:3310", df_crs="EPSG:4326"):
    pts = gpd.GeoSeries(gpd.points_from_xy(df[lon_col], df[lat_col]), crs=df_crs).to_crs(eq_crs)
    return np.column_stack([pts.x,pts.y])


def cross_val_predict_both(points, y, folds, k=5):
    loc_pred = np.empty(len(y), dtype=float)
    mean_pred = np.empty(len(y), dtype=float)
    for fold in np.unique(folds):
        test = (folds==fold)
        train = ~test
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        loc_pred[test] = knn.fit(points[train], y[train]).predict(points[test])
        mean_pred[test] = y[train].mean()
    return loc_pred, mean_pred


def summarize(
    y, loc_pred, mean_pred, 
    to_mg_l=None, scale_name="mg_l", 
    threshold=THRESHOLD_MG_L, sample_weight=None
):
    identity = (lambda v: v)
    to_mg_l = to_mg_l or identity
    exceeds = to_mg_l(y) > threshold

    rmse_loc = root_mean_squared_error(y, loc_pred, sample_weight=sample_weight)
    rmse_mean = root_mean_squared_error(y, mean_pred, sample_weight=sample_weight)
    share = np.average(exceeds, weights=sample_weight)
    return {
        f"rmse_{scale_name}_location_only": rmse_loc,
        f"rmse_{scale_name}_no_information": rmse_mean,
        "skill": 1 - (rmse_loc / rmse_mean),
        "accuracy_location_only": accuracy_score(exceeds, to_mg_l(loc_pred) > threshold, sample_weight=sample_weight),
        "accuracy_majority_class": max(share, 1-share),
    }


def leakage_table(
    df, fold_schemes, 
    lat_col="gm_latitude", lon_col="gm_longitude", 
    df_crs="EPSG:4326", eq_crs = "EPSG:3310",
    target_col="nitrate_mg_L", 
    transform=np.log1p, inverse=np.expm1, scale_name="log", 
    k=5, threshold=THRESHOLD_MG_L, sample_weight=None
):
    identity = (lambda v: v)
    transform = transform or identity
    inverse = inverse or identity
    raw = df[target_col].to_numpy()
    y = transform(raw)

    if not np.allclose(inverse(y), raw, equal_nan=True):
        raise ValueError("`inverse` does not undo `transform`; pass a matching pair.")

    xy = projected_xy(df, lat_col, lon_col, df_crs=df_crs, eq_crs=eq_crs)
    
    rows = {}
    for name, folds in fold_schemes.items():
        loc_pred, mean_pred = cross_val_predict_both(xy, y, np.asarray(folds), k=k)
        rows[name] = summarize(y, loc_pred, mean_pred, to_mg_l=inverse, scale_name=scale_name, threshold=threshold, sample_weight=sample_weight)
    return pd.DataFrame(rows).T