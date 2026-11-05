# Predicting Areas of High-Risk Nitrate Pollution in Salinas Valley

Agricultural runoff carries pollutants into rivers and coastal waters, harming ecosystems and water quality. Vegetated Treatment Systems (VTS), such as bioreactors, constructed wetlands and vegetated ditches, can help reduce this pollution, but they are not equally effective everywhere. Their success depends on where they are placed, how they are designed, and how well they are maintained.

This project will predict areas with higher risk of nitrate pollution in the Salinas Valley and determine where VTS could be most suitable. We will train our model on nitrate pollutant measurements in groundwater wells throughout the Salinas Valley and use soil characteristics, precipitation, temperature, and types of cropland as predictor variables.  Additionally, we want to identify where additional nitrate monitoring is most needed in the Salinas Valley.

# Repo Structure
|  File/Directory   |      Description     | 
|----------|-------------|
| `evaluation_plan.md` |  Describes the data splitting strategy|
| `modeling_plan.md`   |  Describes the models used in the project|
| `README.md` |  Repo structure and some literature references |
| `summary.md` |  Summary of results and future work |
| | |
|`data/groundwaterdata.csv` | Large file, [accessible here](https://devodp.technology.ca.gov/dataset/ground-water-water-quality-results/resource/c97e1fcf-8ad0-4913-9234-ba682ea73bc4) |
|`data/NHD_H_18060005_HU8_GDB/NHD_H_18060005_HU8_GDB.gdb` | |
|`data/datasets/ssurgo_soildata/ (CA053, CA069, CA665, CA772, CA788)` | |
|`data/PRISM_ppt_tmean_stable_4km_201301_202603.csv` | |
|`data/elevation/dem_watershed_10m.tif` | |
| | |
| `notebooks/classification_models.ipynb` |  Score comparisions for classification models |
| `notebooks/data_splitting.ipynb` | Data splitting procedure and leakage analysis |
| `notebooks/feature_selection.ipynb` | Exploratory Data Analysis and Cleaning |
| `notebooks/final_results.ipynb` | Fitting best model on validation data |
| `notebooks/regression_results.ipynb` | Score comparisions for regression models |
| `notebooks/data/data_cleaning/` |  |
| `notebooks/data/data_merge/` |  |
| | |
| `results/classification/` | Folder containing plots for classification models |
| `results/evaluation/` | ? |
| `results/model_runs/` |  ? |
| `results/predictions/` |  ? |
| `results/regression/` | Folder containing plots for regression models  |
| `results/feature_selection.csv` |  ? |
| `results/model_comparison.csv` |  ? |
| | |
| `src/evalution/` | Python scripts for data splitting ?  |
| `src/features/` | Python scripts for EDA ?  |
| `src/inmp/` | **Remove/rename**  |
| `src/merge/` | Python scripts for data merging |
| `src/models/` | Python scripts for modeling |
| `src/shared/` | Python scripts for miscellanous tasks |

# Selected references 

[1] Ransom et al. (2017), A hybrid machine learning model to predict and visualize nitrate concentration throughout the Central Valley aquifer: https://pubs.usgs.gov/publication/70187880

[2] Nolan, Fienen & Lorenz (2015), A statistical learning framework for groundwater nitrate models of the Central Valley: https://pubs.usgs.gov/publication/70157562

[3] Belitz et al., 2010, Estimation of aquifer scale proportion using equal area grids: Assessment of regional scale groundwater quality: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010WR009321
