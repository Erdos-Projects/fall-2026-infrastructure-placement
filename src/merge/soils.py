from pathlib import Path
import geopandas as gpd
import pandas as pd


def build_soil_polygons(ssurgo_dir, attributes_csv, soil_cols, boundary):
    """SSURGO map-unit polygons inside `boundary`, with the cleaned soil attributes."""
    shps = sorted(Path(ssurgo_dir).glob("*/spatial/soilmu_a_*.shp"))
    polys = pd.concat([gpd.read_file(p) for p in shps], ignore_index=True)
    polys = gpd.GeoDataFrame(polys, crs=gpd.read_file(shps[0], rows=1).crs)

    # Keep polygons touching the watershed (spatial index: fast, no cutting)
    idx = polys.sindex.query(boundary.to_crs(polys.crs).union_all(), predicate="intersects")
    polys = polys.iloc[sorted(idx)].copy()

    attrs = pd.read_csv(attributes_csv).drop_duplicates("MUKEY")
    polys["MUKEY"] = polys["MUKEY"].astype(str)
    attrs["MUKEY"] = attrs["MUKEY"].astype(str)
    out = polys.merge(attrs[["MUKEY"] + soil_cols], on="MUKEY", how="left")

    assert len(out) == len(polys), "MUKEY merge changed the row count"
    return out
