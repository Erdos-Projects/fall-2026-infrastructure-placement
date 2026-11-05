"""Project package.

Disable PROJ's network datum grids for every coordinate conversion in the project. Without grids,
WGS84 -> NAD83 / California Albers (EPSG:3310) uses the null transformation, which is how the
committed datasets were created. With grids (PROJ_NETWORK=ON, which conda sets when the
proj-data package is missing), points shift by ~0.3 m and rebuilt files no longer match exactly.

Exception: data/processed/groundwater_ml_integration_lag1_2021_2024_with_climate.csv (MODEL_DATA) was
built with the grids on; all_merge_final.ipynb turns them back on so a rebuild matches.
"""
import pyproj

pyproj.network.set_network_enabled(False)
