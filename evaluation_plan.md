# Evaluation Plan - Predicting Areas of High-Risk Nitrate Pollution in Salinas Valley

## Unit of Analysis:
Data is arranged geographically and temporally (2022 - 2025). 

## Splitting methods:

Potential data leakage for the spatial declustering equal area grid cell approach was calculated by comparing the root mean squared error (RMSE) for a k-Nearest-Neighbors (kNN) regressor to the RMSE for one that simply predicted the total mean of the training data. Estimated leakage was measured by computing $1 - \frac{\mathrm{RMSE}_{\mathrm{KNN}}}{\mathrm{RMSE}_{\mathrm{mean}}}$. A split that does not have significant data leakage should perform about as well under KNN as a model that predicts the mean. This is also compared to splitting methods that have no grouping and splitting methods that group by well ID only. Analysis of estimated leakage found that grid cells of 5 km and above perform approximately as well under KNN as predicting the mean. 

Further stratifying was required to equally split the exceedance data (values of nitrate concentration above 10 mg/L) among different folds. Similarly, we introduced a subbasin-stratefied split strategy that improves fold-to-fold score stability. The best performing data split strategy was chosen to be 10 km grid cells with the exceedance and subbasin stratification. See `notebooks/data_splitting.ipynb` for more details. 
