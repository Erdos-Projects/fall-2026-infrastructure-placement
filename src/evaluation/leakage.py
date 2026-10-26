import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import cKDTree


THRESHOLD_MG_L = 10

DISTANCE_BINS = (0, 500, 1_000, 2_000, 5_000, np.inf)
DISTANCE_LABELS = ("<0.5 km", "0.5-1 km", "1-2 km", "2-5 km", ">5 km")


def nearest_training_distance(xy, train, test):
    """Distance (m) from each test row to the closest training row; xy from projected_xy."""
    return cKDTree(xy[train]).query(xy[test])[0]


def distance_breakdown(y, pred, baseline, distance_m, bins=DISTANCE_BINS, labels=DISTANCE_LABELS):
    """RMSE of the model and the baseline, and improvement over mean, per band of distance to the nearest training well."""
    table = pd.DataFrame({
        "band": pd.cut(distance_m, bins=bins, labels=labels, right=False),
        "err2": (np.asarray(pred) - np.asarray(y)) ** 2,
        "base_err2": (np.asarray(baseline) - np.asarray(y)) ** 2,
    })
    grouped = table.groupby("band", observed=False)
    out = pd.DataFrame({
        "n": grouped.size(),
        "rmse_model": np.sqrt(grouped["err2"].mean()),
        "rmse_mean": np.sqrt(grouped["base_err2"].mean()),
    })
    out["improvement_over_mean"] = 1 - out["rmse_model"] / out["rmse_mean"]
    out["share_of_test"] = out["n"] / out["n"].sum()
    return out.round(3)


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
        "estimated_leakage": 1 - (rmse_loc / rmse_mean),
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