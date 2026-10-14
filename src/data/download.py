import pandas as pd
import json
import geopandas as gpd
import requests
from . import io
from pathlib import Path


# Feth individual layer from url

def fetch_layer(url, params, crs="EPSG:4326", page_size=2000):
    feats, offset = [], 0
    while True:
        response = requests.get(url, params={**params, "resultOffset": offset, "resultRecordCount": page_size}, timeout=60)
        response.raise_for_status()
        data=response.json()
        feats += data["features"]
        if not data["features"] or not data.get("properties",{}).get("exceededTransferLimit"):
            break
        offset += len(data["features"])
    return gpd.GeoDataFrame.from_features(feats, crs=crs)

