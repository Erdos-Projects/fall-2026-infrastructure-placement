import pandas as pd
import json
import geopandas as gpd
from pathlib import Path

def layer_paths(directory, names):
    return {name: Path(directory) / f"{name}.gpkg" for name in names}

def layers_exist(paths):
    return all(path.exists() for path in paths.values())

def save_layers(directory, layer_dict):
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_paths = {}
    for name, gdf in layer_dict.items():
        path = out_dir / f"{name}.gpkg"
        tmp = path.with_suffix(".tmp.gpkg")
        tmp.unlink(missing_ok=True)
        gdf.to_file(tmp)
        tmp.replace(path)
        layer_paths[name] = path
    return layer_paths

def load_data(path):
    file_path = Path(path)
    return gpd.read_file(path)

def load_layers(layer_paths):
    return {name: load_data(path) for name, path in layer_paths.items()}


def load_or_build(directory, names, build, force=False):
    paths = layer_paths(directory, names)
    if layers_exist(paths) and not force:
        return load_layers(paths)
    layers = build()
    save_layers(directory, layers)
    return layers