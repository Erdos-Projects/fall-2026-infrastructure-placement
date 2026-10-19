# Evaluation Plan - Areas of High-Risk Nitrate Pollution in Salinas Valley 

## Unit of Analysis:
Data is arranged geographically and temporally (2022 - 2025). 

## Data split strategy:
Models without forecasting will take the average measurement for a given well over a specific time period.

### Splitting methods:

Splitting methods and leakage analysis shown in [`notebooks/evaluation/leakage_tests.ipynb`](./notebooks/evaluation/leakage_test.ipynb).

**Geospatial splitting:**
Split train and test sets via spatial declustering equal area grid cell approach (see [3]). Use a grid size of 10 km to avoid geospatial data leakage.

**Temporal splitting:**
Split train and test sets between early and late measurements. Given a set of test years, use prior years for train data.

**Geospatial and temporal splitting:**
Given a set of test years and a geospatial split, the training set will be comprised of data from prior years in grid cells that are not contained in the test set.




## Stress tests:

### Splitting methods:

Potential data leakage for the spatial declustering equal area grid cell approach was calculated by comparing the root mean squared error for a K-Nearest-Neighbors regressor to the root mean squared error for one that simply predicted the total mean of the training data. Estimated leakage was measured by computing $1 - \dfrac{\mathrm{RSME}_{\mathrm{KNN}}}{\mathrm{RSME}_{\mathrm{mean}}}$. A split that does not have significant data leakage should perform about as well under KNN as a model that predicts the mean. This is also compared to splitting methods that have no grouping and splitting methods that group by well ID only. Analysis of estimated leakage found that grid cells of 5 km and above perform approximately as well under KNN as predicting the mean. Thus, a cell size of 10 km is will likely suffice to prevent data leakage.