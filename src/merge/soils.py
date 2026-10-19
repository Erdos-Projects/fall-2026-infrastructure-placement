from pathlib import Path
import geopandas as gpd
import pandas as pd


def build_soil_polygons(ssurgo_dir, attributes_csv, soil_cols, boundary):
    """SSURGO map-unit polygons clipped to `boundary`, with the cleaned soil attributes.

    Matches notebooks/soils/Soil_data_cleanup_team.ipynb when `boundary` is the WBDHU8 layer of the NHD HU8 geodatabase."""
    shps = sorted(Path(ssurgo_dir).glob("*/spatial/soilmu_a_*.shp"))
    polys = pd.concat([gpd.read_file(p) for p in shps], ignore_index=True)
    polys = gpd.GeoDataFrame(polys, crs=gpd.read_file(shps[0], rows=1).crs)

    boundary = boundary.to_crs(polys.crs)
    polys = gpd.clip(polys, boundary)              # cut polygons at the watershed boundary

    attrs = pd.read_csv(attributes_csv).drop_duplicates("MUKEY")
    polys["MUKEY"] = polys["MUKEY"].astype(str)
    attrs["MUKEY"] = attrs["MUKEY"].astype(str)
    out = polys.merge(attrs[["MUKEY"] + soil_cols], on="MUKEY", how="left")

    assert len(out) == len(polys), "MUKEY merge changed the row count"
    return out
