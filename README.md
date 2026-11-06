# Predicting Areas of High-Risk Nitrate Pollution in Salinas Valley

Agricultural runoff carries pollutants into rivers and coastal waters, harming ecosystems and water quality. Vegetated Treatment Systems (VTS), such as bioreactors, constructed wetlands and vegetated ditches, can help reduce this pollution, but they are not equally effective everywhere. Their success depends on where they are placed, how they are designed, and how well they are maintained.

This project will predict areas with higher risk of nitrate pollution in the Salinas Valley and determine where VTS could be most suitable. We will train our model on nitrate pollutant measurements in groundwater wells throughout the Salinas Valley and use soil characteristics, precipitation, temperature, and types of cropland as predictor variables.  Additionally, we want to identify where additional nitrate monitoring is most needed in the Salinas Valley.

# Repo Structure


Numbers (1.–9.) give the order to run the pipeline.


`ROOT files`
```
 ├── .gitignore
 ├── README.md
 ├── environment.yml
 ├── evaluation_plan.md
 ├── modeling_plan.md
 ├── run.sh
 ├── run_pipeline.md
 └── summary.md
```

`data/`

```
├── data
│   ├── salinas_valley_subbasins.geojson                     # DWR Bulletin 118 subbasins
│   ├── processed                                            # cleaned and merged datasets
│   │   ├── groundwater_nitrate_clean.csv                    # from 1.
│   │   ├── groundwater_nitrate_clean.gpkg                   # from 1.
│   │   ├── groundwater_well_buffers_500m.gpkg               # from 1.
│   │   ├── prism_climate_monthly_clean.csv                  # from 3.
│   │   ├── crop_features_lag1_2021_2024.csv                 # from 4.
│   │   └── groundwater_ml_integration_lag1_2021_2024_with_climate.csv
│   │                                                        # from 5.; the model dataset
│   │
│   │   # Not in GitHub (too large); see "Data not included in the repository" below
│   ├── groundwaterdata.csv                                  # GAMA ILRP nitrate results (97 MB)
│   ├── PRISM_ppt_tmean_stable_4km_201301_202603.csv         # PRISM monthly climate (2.5 MB)
│   ├── NHD_H_18060005_HU8_GDB
│   │   └── NHD_H_18060005_HU8_GDB.gdb                       # USGS NHD, Salinas watershed (65 MB)
│   ├── datasets
│   │   └── ssurgo_soildata                                  # SSURGO soil surveys (952 MB)
│   │       ├── CA053                                        # Monterey County
│   │       ├── CA069                                        # San Benito County
│   │       ├── CA665                                        # San Luis Obispo County, Paso Robles Area
│   │       ├── CA772                                        # Los Padres National Forest Area
│   │       └── CA788                                        # Pinnacles National Monument
│   ├── elevation
│   │   └── dem_watershed_10m.tif                            # 3DEP 1/3 arc-second DEM (401 MB)
│   ├── crops                                                # DWR crop maps, from 0. (1.4 GB)
│   │
│   │   # Created by the notebooks; not in GitHub
│   ├── soils
│   │   └── soils_huc8_attributes.gpkg                       # from 2. (46 MB)
│   ├── prism_points.gpkg                                    # from 3.
│   └── crop_runs                                            # from 4.; one timestamped folder per run
│                                                            # (local only) timestamped runs of 4.
```

`notebooks/`

```
├── notebooks
│   ├── data_cleaning
│   │   ├── groundwater
│   │   │   ├── gwater_clean.ipynb                           # 1. nitrate samples and 500 m well buffers
│   │   │   └── groundwater_spatial_eda.ipynb
│   │   ├── soils
│   │   │   └── Soil_data_cleanup_team.ipynb                 # 2. SSURGO soil attributes
│   │   ├── precipitation
│   │   │   └── prism_climate_cleaning_final.ipynb           # 3. PRISM monthly climate
│   │   ├── crop_mapping
│   │   │   ├── crop_mapping_annual_2021_2024.ipynb          # 4. annual DWR crop features
│   │   │   └── crop_mapping_eda.ipynb                       # original 2023 crop analysis
│   │
│   ├── data_merge
│   │   └── all_merge_final.ipynb                            # 5. builds the model dataset
│   │
│   ├── feature_selection.ipynb                              # 6. selects model features
│   ├── regression_models.ipynb                              # 8. regression models
│   ├── classification_models.ipynb                          # 8. classification models (above 10 mg/L)
│   ├── data_splitting.ipynb                                 # 9. leakage analysis and choice of split
│   └── final_results.ipynb                                  # 10. final models: hold-out evaluation and feature importance

```

`results/`

```
├── results
│   ├── feature_selection.csv                                # from 6.; keep/drop decision for every feature
│   ├── model_comparison.csv                                 # from 8.; mean and std of scores across folds
│   ├── splits                                               # from 7.; fold per well (.csv) and settings (.json)
│   │
│   ├── model_runs                                           # from 8.; per-fold scores, author__model__task__split__subbasin__noelev.csv
│   │
│   ├── predictions                                          # from 8.; out-of-fold predictions, one column per model
│   │
│   ├── evaluation                                           # from 9.
│   │
│   ├── regression                                           # from 8.
│   │
│   └── classification                                       # from 8.
```


`src/`

```
└── src
    ├── __init__.py                                          # turns off PROJ network grids project-wide
    ├── paths.py                                             # every input and output location
    ├── data
    │   └── crop_mapping.py                                  # 0. downloads the DWR crop maps
    ├── evaluation
    │   ├── __init__.py
    │   ├── leakage.py                                       # spatial leakage diagnostics
    │   ├── make_splits.py                                   # 7. builds and loads the registered splits
    │   ├── results_log.py                                   # saves and summarizes model scores
    │   └── splits.py                                        # grid-cell fold assignment
    ├── features
    │   ├── __init__.py
    │   ├── preprocessing.py                                 # loads model-ready X and y
    │   └── transformers.py                                  # derived features and subbasin columns
    ├── models
    │   ├── __init__.py
    │   └── evaluate.py                                      # shared spatial cross-validation for all models
    └── shared
        ├── __init__.py
        └── datetime.py                                      # date parsing for the groundwater data
```


### Steps
```
Step 0: `python -m src.data.crop_mapping` downloads the DWR crop maps.
Step 7: `python -m src.evaluation.make_splits <split name>` for `10km`, `10km_subbasin`, `7.5km_subbasin` and `5km_subbasin`.
Steps 1–3 can run in any order; the two step-8 notebooks can run in either order.
```

# Data not included in the repository

Some input files are too large for GitHub. To reproduce the results, place them at the paths below,
relative to the repository root. Every path is also defined in `src/paths.py`, and each notebook stops
with a list of any missing files.

| Path | Size | Source | How to get it |
|---|---|---|---|
| `data/groundwaterdata.csv` | 97 MB | GAMA Irrigated Lands Regulatory Program [8] | Download the ILRP groundwater results CSV and save it under this name. |
| `data/PRISM_ppt_tmean_stable_4km_201301_202603.csv` | 2.5 MB | PRISM Climate Group [7] | Monthly precipitation and mean temperature (AN91m, 4 km) at 347 grid points, January 2013 – March 2026, exported 2026-10-07. Use the copy in the shared folder below. |
| `data/NHD_H_18060005_HU8_GDB/NHD_H_18060005_HU8_GDB.gdb` | 65 MB | USGS National Hydrography Dataset [5] | Download https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/GDB/NHD_H_18060005_HU8_GDB.zip and unzip it into `data/NHD_H_18060005_HU8_GDB/`. |
| `data/datasets/ssurgo_soildata/` | 952 MB | USDA NRCS Web Soil Survey [4] | Download the SSURGO data for survey areas CA053, CA069, CA665, CA772 and CA788, and unzip each into its own folder here (for example `ssurgo_soildata/CA053/`). The data used were the September 2026 releases. |
| `data/elevation/dem_watershed_10m.tif` | 401 MB | USGS 3D Elevation Program, 1/3 arc-second | A mosaic of tiles `n36w121`, `n36w122`, `n37w121` and `n37w122` (from https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/), cropped to the watershed. Use the copy in the shared folder below to get identical elevation values. |
| `data/crops/` | 1.4 GB | California DWR Statewide Crop Mapping [6] | Run `python -m src.data.crop_mapping` from the repository root. It downloads the 2020–2024 maps into this folder. |

Files created by the notebooks (`data/soils/`, `data/prism_points.gpkg` and `data/crop_runs/`) are
also not in GitHub; running the pipeline creates them.

**Shared folder:** [Link to Google Drive](https://drive.google.com/drive/folders/1aQjNUVLgbEiMHWpoKIv3kPRP2gz4hIri?usp=sharing). It has the PRISM export and the DEM, which cannot be recreated exactly
from public downloads.


# Selected scientific references

[1] Ransom et al. (2017), A hybrid machine learning model to predict and visualize nitrate concentration throughout the Central Valley aquifer: https://pubs.usgs.gov/publication/70187880

[2] Nolan, Fienen & Lorenz (2015), A statistical learning framework for groundwater nitrate models of the Central Valley: https://pubs.usgs.gov/publication/70157562

[3] Belitz et al., 2010, Estimation of aquifer scale proportion using equal area grids: Assessment of regional scale groundwater quality: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010WR009321
