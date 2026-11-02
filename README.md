# Problem Statement - Predict Areas of High-Risk Nitrate Pollution in Salinas Valley 

Agricultural runoff carries nitrogen and other pollutants into rivers and coastal waters, harming ecosystems and water quality. Vegetated Treatment Systems (VTS), such as bioreactors, constructed wetlands and vegetated ditches, can help reduce this pollution, but they are not equally effective everywhere. Their success depends on where they are placed, how they are designed, and how well they are maintained.

This project will predict areas with higher risk of nitrate pollution in the Salinas Valley and determine where VTS could be most suitable. 

# Repo Structure

```
├── README.md
├── evaluation_plan.md
├── modeling_plan.md
├── notebooks
│   ├── data_cleaning
│   ├── data_merge
│       ├── XX
│   ├── leakage_test.ipynb
│   ├── feature_selection.ipynb
│   ├── final_results.ipynb
│   ├── modeling_summary.ipynb
│   └── models
│       ├── 
├── results
│   ├── evaluation
│   │   ├── estimated_leakage_by_split.png
│   │   └── spatial_folds_map.png
│   │   └── MORE FILES HERE
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
