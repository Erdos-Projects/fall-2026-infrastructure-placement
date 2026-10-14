import pandas as pd
import geopandas as gpd
import numpy as np

def to_reports(gdf):
    N_COLS = ["total_nitrogen_applied__lbs_ran", "total_nitrogen_removed__lbs_ran", "a_r__lbs_ranch_acre_", "a_r_ratio", "outlier_status"]
    KEY = ["member_id", "field_id", "year"] + N_COLS
    group_by = "reported_crop_name"
    reports = gdf.groupby(KEY, dropna=False, as_index=False).agg(
        crops=(group_by, lambda s: sorted(set(s))),
        n_crops=(group_by, "nunique"),
        geometry=("geometry", "first")
    )
    return gpd.GeoDataFrame(reports, geometry="geometry", crs=gdf.crs)


def flag_implausible(gdf, whis=3, min_removed=5, ratio_cap=50):
    reports = gdf.copy()
    applied = reports["total_nitrogen_applied__lbs_ran"]
    removed = -reports["total_nitrogen_removed__lbs_ran"]

    q1_a, q3_a = applied.quantile([0.25,0.75])
    iqr_a = q3_a-q1_a
    fence_a = q3_a+whis*iqr_a
    reports["flag_box_app"] = (applied > fence_a)

    q1_r, q3_r = removed.quantile([0.25,0.75])
    iqr_r = q3_r-q1_r
    fence_r = q3_r+whis*iqr_r
    reports["flag_box_rem"] = (removed > fence_r)

    reports["flag_box"] = (reports[["flag_box_app", "flag_box_rem"]]).any(axis=1)

    reports["flag_ratio"] = (removed >= min_removed) & (reports["a_r_ratio"] > ratio_cap)
    
    gdf["implausible"] = reports[["flag_ratio", "flag_box"]].any(axis=1)
    return gdf