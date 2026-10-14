import pandas as pd
import json
import geopandas as gpd
import requests
from pathlib import Path

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