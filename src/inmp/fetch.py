from src.data import io, download, clip

def fetch_INMP_data(base_url, layer_key):
    crs = "EPSG:4326"
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "outSR": 4326,
        "orderByFields": "objectid",
    }
    return {name: download.fetch_layer(f"{base_url}/{layer_id}/query", params, crs=crs) for name, layer_id in layer_key.items()}