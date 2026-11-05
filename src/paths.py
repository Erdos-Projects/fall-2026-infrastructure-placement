"""Every file and folder the pipeline reads or writes, defined in one place.

data/              raw inputs, downloaded or exported (not tracked by git; see the README's data sources)
data/processed/    datasets made by the data_cleaning and data_merge notebooks (tracked)
results/           splits, model runs, predictions, tables and figures (tracked)

Import in a notebook with `from src import paths`, then use e.g. `paths.NITRATE_CLEAN`.
This module only uses pathlib, so any other module can import it without circular imports.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
NOTEBOOKS = ROOT / "notebooks"
RESULTS = ROOT / "results"

# ---- raw inputs (data/, not in git) ----
RAW_GROUNDWATER = DATA / "groundwaterdata.csv"                                    # GAMA ILRP export, data.ca.gov
RAW_PRISM = DATA / "PRISM_ppt_tmean_stable_4km_201301_202603.csv"                 # PRISM monthly ppt/tmean export
RAW_SSURGO = DATA / "datasets" / "ssurgo_soildata"                                # SSURGO survey areas (CA### folders)
HUC8_GDB = DATA / "NHD_H_18060005_HU8_GDB" / "NHD_H_18060005_HU8_GDB.gdb"         # USGS NHD, HU8 18060005; layer "WBDHU8"
HUC8_BOUNDARY = DATA / "watershed_boundary" / "salinas_huc8.gpkg"                 # same HU8 from the WBD web service (maps)
WBD_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer"      # where HUC8_BOUNDARY comes from
DEM = DATA / "elevation" / "dem_watershed_10m.tif"                                # 3DEP 1/3 arc-second
SUBBASINS = DATA / "salinas_valley_subbasins.geojson"                             # DWR Bulletin 118 (tracked)
RAW_CROPS = DATA / "crops"                                                        # DWR statewide crop mapping, 2020–2024 (src/data/crop_mapping.py)
CROP_RUNS = DATA / "crop_runs"                                                    # timestamped crop notebook runs (local only)


# ---- derived but local only (too large for git) ----
SOILS = DATA / "soils" / "soils_huc8_attributes.gpkg"                             # 46 MB; Soil_data_cleanup_team.ipynb
PRISM_POINTS = DATA / "prism_points.gpkg"

# ---- processed datasets (data/processed/, tracked) ----
NITRATE_CLEAN = PROCESSED / "groundwater_nitrate_clean.csv"                       # gwater_clean.ipynb
NITRATE_CLEAN_GPKG = PROCESSED / "groundwater_nitrate_clean.gpkg"                 # gwater_clean.ipynb
WELL_BUFFERS = PROCESSED / "groundwater_well_buffers_500m.gpkg"                   # gwater_clean.ipynb
PRISM_CLEAN = PROCESSED / "prism_climate_monthly_clean.csv"                       # prism_climate_cleaning_final.ipynb
CROP_FEATURES = PROCESSED / "crop_features_lag1_2021_2024.csv"                    # crop_mapping_annual_2021_2024.ipynb
MODEL_DATA = PROCESSED / "groundwater_ml_integration_lag1_2021_2024_with_climate.csv"   # all_merge_final.ipynb

# Supporting outputs of the crop notebook (dictionary, validation files) stay next to it
CROP_MAPPING_DIR = NOTEBOOKS / "data_cleaning" / "crop_mapping"

# ---- results ----
FEATURE_SELECTION = RESULTS / "feature_selection.csv"                             # feature_selection.ipynb
SPLITS = RESULTS / "splits"                                                       # src/evaluation/make_splits.py
MODEL_RUNS = RESULTS / "model_runs"                                               # src/evaluation/results_log.py
PREDICTIONS = RESULTS / "predictions"                                             # src/models/evaluate.py
MODEL_COMPARISON = RESULTS / "model_comparison.csv"                               # regression/classification notebooks
FIGURES = {name: RESULTS / name for name in ["evaluation", "regression", "classification"]}


def require(*files):
    """Stop with a clear message if any input is missing (raw inputs are not in git)."""
    missing = [Path(f) for f in files if not Path(f).exists()]
    if missing:
        listing = "\n".join(f"  {f.relative_to(ROOT)}" for f in missing)
        raise FileNotFoundError(f"Missing input files:\n{listing}\nSee the data sources section of the README.")
