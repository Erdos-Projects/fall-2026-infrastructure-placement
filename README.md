# Predicting Areas of High-Risk Nitrate Pollution in Salinas Valley

Agricultural runoff carries pollutants into rivers and coastal waters, harming ecosystems and water quality. Vegetated Treatment Systems (VTS), such as bioreactors, constructed wetlands and vegetated ditches, can help reduce this pollution, but they are not equally effective everywhere. Their success depends on where they are placed, how they are designed, and how well they are maintained.

This project will predict areas with higher risk of nitrate pollution in the Salinas Valley and determine where VTS could be most suitable. We will train our model on nitrate pollutant measurements in groundwater wells throughout the Salinas Valley and use soil characteristics, precipitation, temperature, and types of cropland as predictor variables.  Additionally, we want to identify where additional nitrate monitoring is most needed in the Salinas Valley.

# Repo Structure

```
├── README.md
├── evaluation_plan.md
├── modeling_plan.md
├── notebooks
│   ├── data_cleaning
│   ├── data_merge
│       ├── XX
│   ├── data_splitting.ipynb
│   ├── feature_selection.ipynb
|   ├── regression_models.ipynb
|   ├── classification_models.ipynb
│   ├── final_results.ipynb
│   └── models
│       ├── 
├── results
│   ├── evaluation
│   │   ├── estimated_leakage_by_split.png
│   │   ├── spatial_folds_map_5km_subbasin.png
│   │   ├── spatial_folds_map_7.5km_subbasin.png
│   │   ├── spatial_folds_map_10km.png
│   │   ├── spatial_folds_map_10km_subbasin.png
│   │   ├── spatial_folds_maps_all_splits.png
│   │   └── subbasin_mix_by_fold.png
│   ├── regression
│   │   ├── fold_metrics_10km_subbasin_subbasin1.png
│   │   ├── residuals_10km_subbasin_subbasin1.png
│   │   ├── observed_vs_predicted_10km_subbasin_subbasin1.png
│   │   └── model_comparison_10km_subbasin.png
│   ├── classification
│   │   ├── threshold_metrics_10km_subbasin_subbasin1.png
│   │   ├── fold_metrics_10km_subbasin_subbasin1.png
│   │   ├── probability_distributions_10km_subbasin_subbasin1.png
│   │   ├── pr_roc_calibration_10km_subbasin_subbasin1.png
│   │   ├── confusion_matrices_10km_subbasin_subbasin1.png
│   │   └── model_comparison_10km_subbasin_subbasin1.png
│   ├── feature_selection.csv
│   ├── model_runs
│   │   ├── 
│   └── splits
│       ├── 
└── src
    ├── evaluation
    │   ├── leakage.py
    │   ├── make_splits.py
    │   ├── results_log.py
    │   └── splits.py
    ├── features
    │   ├── preprocessing.py
    │   └── transformers.py
    ├── inmp  *MOVE OR RENAME*
    │   ├── fetch.py
    │   ├── preprocessing.py
    │   └── transformers.py
    ├── merge
    │   ├── merging_functions.py
    │   └── soils.py
    ├── models *POSSIBLY ADD FUNCTIONS HERE*
    │   └── random_forest.py
    └── shared
        ├── arcgis.py
        ├── io.py
        └── spatial.py
```



# Selected scientific references 

[1] Ransom et al. (2017), A hybrid machine learning model to predict and visualize nitrate concentration throughout the Central Valley aquifer: https://pubs.usgs.gov/publication/70187880

[2] Nolan, Fienen & Lorenz (2015), A statistical learning framework for groundwater nitrate models of the Central Valley: https://pubs.usgs.gov/publication/70157562

[3] Belitz et al., 2010, Estimation of aquifer scale proportion using equal area grids: Assessment of regional scale groundwater quality: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010WR009321
