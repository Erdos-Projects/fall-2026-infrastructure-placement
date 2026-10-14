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
