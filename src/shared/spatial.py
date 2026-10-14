import numpy as np
import pandas as pd
import json
import geopandas as gpd
import requests
from pathlib import Path
from shapely.geometry import box

def fetch_huc8(URL, huc8_code):
    response = requests.get(f"{URL}/4/query", params={
        "where": f"huc8 = '{huc8_code}'",
        "outFields": "huc8,name,areasqkm",
        "outSR": 4326,
        "f": "geojson",
    }, timeout=60)
    response.raise_for_status()
    return gpd.GeoDataFrame.from_features(response.json()["features"], crs="EPSG:4326")

# def load_boundary(path):
#     gdb = Path(path)
#     return gpd.read_file(gdb, layer="WBDHU8")

def load_boundary(path):
    gdb = Path(path)
    return gpd.read_file(gdb)

def clip_layer(gdf,boundary):
    return gdf.clip(boundary.to_crs(gdf.crs))


def make_grid(boundary, cell_size_m):
    b = boundary.to_crs(3310)                          # meters, so cell size means something
    xmin, ymin, xmax, ymax = b.total_bounds
    cells = [box(x, y, x + cell_size_m, y + cell_size_m)
             for x in np.arange(xmin, xmax, cell_size_m)
             for y in np.arange(ymin, ymax, cell_size_m)]
    grid = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=3310)
    return grid[grid.intersects(b.union_all())]       # keep only cells touching the watershed
