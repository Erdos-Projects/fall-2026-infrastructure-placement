# Problem Statement - Predict Areas of High-Risk Nitrate Pollution in Salinas Valley 

Agricultural runoff carries nitrogen and other pollutants into rivers and coastal waters, harming ecosystems and water quality. Vegetated Treatment Systems (VTS), such as bioreactors, constructed wetlands and vegetated ditches, can help reduce this pollution, but they are not equally effective everywhere. Their success depends on where they are placed, how they are designed, and how well they are maintained.

This project will predict areas with higher risk of nitrate pollution in the Salinas Valley and determine where VTS could be most suitable. 

# Repo Structure


Numbers (1.–9.) give the order to run the pipeline.

```
├── README.md
├── evaluation_plan.md
├── modeling_plan.md
├── .gitignore
├── data
│   ├── salinas_valley_subbasins.geojson                     # DWR Bulletin 118 subbasins
│   ├── processed                                            # cleaned and merged datasets
│   │   ├── groundwater_nitrate_clean.csv                    # from 1.
│   │   ├── groundwater_nitrate_clean.gpkg                   # from 1.
│   │   ├── groundwater_well_buffers_500m.gpkg               # from 1.
│   │   ├── prism_climate_monthly_clean.csv                  # from 3.
│   │   ├── crop_features_lag1_2021_2024.csv                 # from 4.
│   │   └── groundwater_ml_integration_lag1_2021_2024_with_climate.csv   # from 5.; the model dataset
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
│   │   ├── i15_Crop_Mapping_2020.gdb
│   │   ├── i15_Crop_Mapping_2021.gdb
│   │   ├── i15_crop_mapping_2022.gdb
│   │   ├── i15_Crop_Mapping_2023_Final.gdb
│   │   └── i15_Crop_Mapping_2024_Provisional_20251208.gdb
│   │
│   │   # Created by the notebooks; not in GitHub
│   ├── soils
│   │   └── soils_huc8_attributes.gpkg                       # from 2. (46 MB)
│   ├── prism_points.gpkg                                    # from 3.
│   └── crop_runs                                            # from 4.; one timestamped folder per run                                           # (local only) timestamped runs of 4.
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
│   │   │   ├── crop_mapping_eda.ipynb                       # original 2023 crop analysis
│   │   │   ├── crop_mapping_feature_selection.ipynb
│   │   │   ├── crop_mapping_feature_selection.csv
│   │   │   ├── selected_features.txt
│   │   │   ├── well_crop_features_all.csv
│   │   │   ├── wells_nitrate.csv
│   │   │   └── figures
│   │   │       ├── boxplot_multicrop.png
│   │   │       ├── boxplot_veg_berry.png
│   │   │       ├── correlation_heatmap.png
│   │   │       ├── crop_history_lag.png
│   │   │       ├── distributions.png
│   │   │       ├── lag_scatter_1997.png
│   │   │       ├── map_crops_2023.png
│   │   │       ├── missingness.png
│   │   │       ├── scatter_veg_berry.png
│   │   │       ├── selected_features_corr.png
│   │   │       ├── target_boxplot.png
│   │   │       └── target_vs_features.png
│   │   ├── groundwater_depth
│   │   │   ├── groundwater_depth_eda.ipynb
│   │   │   ├── depth_groundwater.csv
│   │   │   ├── groundwater_depth_clean.csv
│   │   │   └── thiessen_polygons.gpkg
│   │   └── surface_water
│   │       ├── surfacewater_eda.ipynb
│   │       ├── surfacewater_clean.csv
│   │       └── thiessen_polygons.gpkg
│   ├── data_merge
│   │   ├── all_merge_final.ipynb                            # 5. builds the model dataset
│   │   └── all_merge_EDA_Final.ipynb
│   ├── feature_selection.ipynb                              # 6. selects model features
│   ├── regression_models.ipynb                              # 8. regression models
│   ├── classification_models.ipynb                          # 8. classification models (above 10 mg/L)
│   ├── data_splitting.ipynb                                 # 9. leakage analysis and choice of split
│   └── final_results.ipynb                                  # (to remove) empty
├── results
│   ├── feature_selection.csv                                # from 6.; keep/drop decision for every feature
│   ├── model_comparison.csv                                 # from 8.; mean and std of scores across folds
│   ├── splits                                               # from 7.; fold per well (.csv) and settings (.json)
│   │   ├── well_5_folds_10000m_exceedance_seed0.csv
│   │   ├── well_5_folds_10000m_exceedance_seed0.json
│   │   ├── well_5_folds_10000m_subbasin_x_exceedance_seed0.csv
│   │   ├── well_5_folds_10000m_subbasin_x_exceedance_seed0.json
│   │   ├── well_5_folds_7500m_subbasin_x_exceedance_seed0.csv
│   │   ├── well_5_folds_7500m_subbasin_x_exceedance_seed0.json
│   │   ├── well_5_folds_5000m_subbasin_x_exceedance_seed0.csv
│   │   └── well_5_folds_5000m_subbasin_x_exceedance_seed0.json
│   ├── model_runs                                           # from 8.; per-fold scores, author__model__task__split__subbasin__noelev.csv
│   │   ├── andrea__lasso__regression__10km__subbasin0__noelev0.csv
│   │   ├── andrea__lasso__regression__10km__subbasin1__noelev1.csv
│   │   ├── andrea__lasso__regression__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__lasso__regression__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__lasso__regression__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__linear-regression__regression__10km__subbasin0__noelev0.csv
│   │   ├── andrea__linear-regression__regression__10km__subbasin1__noelev1.csv
│   │   ├── andrea__linear-regression__regression__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__linear-regression__regression__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__linear-regression__regression__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__logistic__classification__10km__subbasin0__noelev0.csv
│   │   ├── andrea__logistic__classification__10km__subbasin1__noelev1.csv
│   │   ├── andrea__logistic__classification__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__logistic__classification__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__logistic__classification__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__logistic-l1__classification__10km__subbasin0__noelev0.csv
│   │   ├── andrea__logistic-l1__classification__10km__subbasin1__noelev1.csv
│   │   ├── andrea__logistic-l1__classification__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__logistic-l1__classification__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__logistic-l1__classification__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__logistic-l2__classification__10km__subbasin0__noelev0.csv
│   │   ├── andrea__logistic-l2__classification__10km__subbasin1__noelev1.csv
│   │   ├── andrea__logistic-l2__classification__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__logistic-l2__classification__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__logistic-l2__classification__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__ridge__regression__10km__subbasin0__noelev0.csv
│   │   ├── andrea__ridge__regression__10km__subbasin1__noelev1.csv
│   │   ├── andrea__ridge__regression__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__ridge__regression__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── andrea__ridge__regression__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── charlsey__random-forest__classification__10km__subbasin0__noelev0.csv
│   │   ├── charlsey__random-forest__classification__10km__subbasin1__noelev1.csv
│   │   ├── charlsey__random-forest__classification__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── charlsey__random-forest__classification__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── charlsey__random-forest__classification__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── charlsey__random-forest__regression__10km__subbasin0__noelev0.csv
│   │   ├── charlsey__random-forest__regression__10km__subbasin1__noelev1.csv
│   │   ├── charlsey__random-forest__regression__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── charlsey__random-forest__regression__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── charlsey__random-forest__regression__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── gavin__decision-tree__classification__10km__subbasin0__noelev0.csv
│   │   ├── gavin__decision-tree__classification__10km__subbasin1__noelev1.csv
│   │   ├── gavin__decision-tree__classification__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── gavin__decision-tree__classification__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── gavin__decision-tree__classification__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── gavin__decision-tree__regression__10km__subbasin0__noelev0.csv
│   │   ├── gavin__decision-tree__regression__10km__subbasin1__noelev1.csv
│   │   ├── gavin__decision-tree__regression__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── gavin__decision-tree__regression__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── gavin__decision-tree__regression__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── karuna__xgboost-brt__classification__10km__subbasin0__noelev0.csv
│   │   ├── karuna__xgboost-brt__classification__10km__subbasin1__noelev1.csv
│   │   ├── karuna__xgboost-brt__classification__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── karuna__xgboost-brt__classification__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── karuna__xgboost-brt__classification__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── karuna__xgboost-brt__regression__10km__subbasin0__noelev0.csv
│   │   ├── karuna__xgboost-brt__regression__10km__subbasin1__noelev1.csv
│   │   ├── karuna__xgboost-brt__regression__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── karuna__xgboost-brt__regression__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── karuna__xgboost-brt__regression__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── team__dummy-mean__regression__10km__subbasin0__noelev0.csv
│   │   ├── team__dummy-mean__regression__10km__subbasin1__noelev1.csv
│   │   ├── team__dummy-mean__regression__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── team__dummy-mean__regression__7-5km-subbasin__subbasin1__noelev1.csv
│   │   ├── team__dummy-mean__regression__5km-subbasin__subbasin1__noelev1.csv
│   │   ├── team__dummy-prior__classification__10km__subbasin0__noelev0.csv
│   │   ├── team__dummy-prior__classification__10km__subbasin1__noelev1.csv
│   │   ├── team__dummy-prior__classification__10km-subbasin__subbasin1__noelev1.csv
│   │   ├── team__dummy-prior__classification__7-5km-subbasin__subbasin1__noelev1.csv
│   │   └── team__dummy-prior__classification__5km-subbasin__subbasin1__noelev1.csv
│   ├── predictions                                          # from 8.; out-of-fold predictions, one column per model
│   │   ├── regression__10km__subbasin0.csv.gz
│   │   ├── regression__10km__subbasin1.csv.gz
│   │   ├── regression__10km_subbasin__subbasin1.csv.gz
│   │   ├── regression__7.5km_subbasin__subbasin1.csv.gz
│   │   ├── regression__5km_subbasin__subbasin1.csv.gz
│   │   ├── classification__10km__subbasin0.csv.gz
│   │   ├── classification__10km__subbasin1.csv.gz
│   │   ├── classification__10km_subbasin__subbasin1.csv.gz
│   │   ├── classification__7.5km_subbasin__subbasin1.csv.gz
│   │   └── classification__5km_subbasin__subbasin1.csv.gz
│   ├── evaluation                                           # from 9.
│   │   ├── estimated_leakage_by_split.png
│   │   ├── subbasin_mix_by_fold.png
│   │   ├── spatial_folds_maps_all_splits.png
│   │   ├── spatial_folds_map_10km.png
│   │   ├── spatial_folds_map_10km_subbasin.png
│   │   ├── spatial_folds_map_7.5km_subbasin.png
│   │   └── spatial_folds_map_5km_subbasin.png
│   ├── regression                                           # from 8.
│   │   ├── model_comparison_10km_subbasin.png
│   │   ├── fold_metrics_10km_subbasin_subbasin1.png
│   │   ├── observed_vs_predicted_10km_subbasin_subbasin1.png
│   │   └── residuals_10km_subbasin_subbasin1.png
│   └── classification                                       # from 8.
│       ├── model_comparison_10km_subbasin_subbasin1.png
│       ├── fold_metrics_probability_10km_subbasin_subbasin1.png
│       ├── fold_metrics_threshold_10km_subbasin_subbasin1.png
│       ├── pr_roc_calibration_10km_subbasin_subbasin1.png
│       ├── probability_distributions_10km_subbasin_subbasin1.png
│       ├── threshold_metrics_10km_subbasin_subbasin1.png
│       ├── confusion_matrices_by_threshold_10km_subbasin_subbasin1.png
│       ├── confusion_matrices_t0.25_10km_subbasin_subbasin1.png
│       ├── confusion_matrices_t0.3_10km_subbasin_subbasin1.png
│       ├── confusion_matrices_t0.35_10km_subbasin_subbasin1.png
│       ├── confusion_matrices_t0.4_10km_subbasin_subbasin1.png
│       ├── confusion_matrices_t0.45_10km_subbasin_subbasin1.png
│       └── confusion_matrices_t0.5_10km_subbasin_subbasin1.png
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
        ├── datetime.py                                      # date parsing for the groundwater data
        └── spatial.py                                       # (to remove) no longer used
```

Step 0: `python -m src.data.crop_mapping` downloads the DWR crop maps.
Step 7: `python -m src.evaluation.make_splits <split name>` for `10km`, `10km_subbasin`, `7.5km_subbasin` and `5km_subbasin`.
Steps 1–3 can run in any order; the two step-8 notebooks can run in either order.

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

**Shared folder:** [add link]. It has the PRISM export and the DEM, which cannot be recreated exactly
from public downloads.


# Selected scientific references and public data sources

[1] Ransom et al. (2017), A hybrid machine learning model to predict and visualize nitrate concentration throughout the Central Valley aquifer: https://pubs.usgs.gov/publication/70187880

[2] Nolan, Fienen & Lorenz (2015), A statistical learning framework for groundwater nitrate models of the Central Valley: https://pubs.usgs.gov/publication/70157562

[3] Belitz et al., 2010, Estimation of aquifer scale proportion using equal area grids: Assessment of regional scale groundwater quality: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010WR009321

[4] USDA NRCS Web Soil Survey/SSURGO: https://websoilsurvey.nrcs.usda.gov/

[5] USGS National Hydrography: https://www.usgs.gov/national-hydrography

[6] California DWR crop mapping: https://data.cnra.ca.gov/dataset/statewide-crop-mapping ; USDA Cropland Data Layer: https://data.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php

[7] PRISM Climate Group: https://prism.oregonstate.edu/

[8] GAMA - Irrigated Lands Regulatory Program: https://devodp.technology.ca.gov/dataset/ground-water-water-quality-results/resource/c97e1fcf-8ad0-4913-9234-ba682ea73bc4

[9] ILRP/INMP source: https://gispublic.waterboards.ca.gov/portalserver/rest/services/Hosted/Region_3_INMP_Data_Map/FeatureServer.
