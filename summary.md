# Problem Statement

Agricultural runoff carries pollutants into rivers and coastal waters, harming ecosystems and water quality. Vegetated Treatment Systems (VTS), such as bioreactors, constructed wetlands and vegetated ditches, can help reduce this pollution, but they are not equally effective everywhere. Their success depends on where they are placed, how they are designed, and how well they are maintained.

This project will predict areas with higher risk of nitrate pollution in the Salinas Valley and determine where VTS could be most suitable. We will train our model on nitrate pollutant measurements in groundwater wells throughout the Salinas Valley and use soil characteristics, precipitation, temperature, and types of cropland as predictor variables.  Additionally, we want to identify where additional nitrate monitoring is most needed in the Salinas Valley.

# Data sources

This project currently uses data from several publicly accessible resources:

- United States Department of Agriculture (USDA) National Resources Conversation Service (NRCS) Web Soil Survey/SSURGO: https://websoilsurvey.nrcs.usda.gov/

- United States Geological Survey (USGS) National Hydrography: https://www.usgs.gov/national-hydrography

- California National Resources Agency, California Department of Water Resources crop mapping: https://data.cnra.ca.gov/dataset/statewide-crop-mapping

- USDA Cropland Data Layer: https://data.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php

- PRISM Climate Group: https://prism.oregonstate.edu/

- California Open Data, GAMA - Irrigated Lands Regulatory Program: https://devodp.technology.ca.gov/dataset/ground-water-water-quality-results/resource/c97e1fcf-8ad0-4913-9234-ba682ea73bc4


# Key Performance Indicators

The project is concerned with two factors: safety of public health, and cost of deploying and maintaining VTS. Nitrate pollution in water exceeding 10 mg/L is deemed harmful. We prioritize recall. If there is an actual exceedance above 10 mg/L, we need our model to capture it. We prioritize precision. We do not want to deploy VTS in areas where it is not needed.


# Results

We approached modeling from two angles: regression and classification. The regression task seeks to directly predict the nitrate pollutant concentration. We compared a dummy model and simple linear model to linear models with two regularization strategies (ridge and lasso), a decision tree model, a random forest model, and a boosted regression tree model. Per model, we tuned hyperparameters with extensive nested cross validation, using `GridSearchCV` or `RandomizedSearchCV` in each set of training folds, holding one whole fold out at the time, so the test fold is never used for tuning. Our base scoring metric was RMSE, where our lowest RMSE came from the random forest model, averaging 0.973 per fold, with only small improvement over the dummy, 1.208. Plots of predicted nitrate values versus observed nitrate values showed poor reproductions of real trends, with high residuals. We concluded that the features available did not encode enough information to produce accurate regression models. 

The classification task seeks to classify locations as exceedances (above 10 mg/L) or non-exceedances. We compared a dummy model and simple logistic model to logistic models with two regularization strategies (L1 and L2), a decision tree model, a random forest model, and a boosted regression tree model. Our basic scoring metric was AUC score, where again the random forest was the strongest performing, 0.291, compared to the weakest performing, the logistic L2 model, with an AUC of 0.206. Per our KPIs, the f1 score, balancing recall and precision, was highest for the random forest model with the slightly lower cutoff threshold of 0.37, with an f1 of 0.680. We found the random forest model to be the most stable across folds and perform the strongest overall. 


# Limitations

We needed to adopt a data splitting strategy that accounts for the fact that nearby wells have similar nitrate levels. We also made sure to divide the percentage of exceedances among each fold. We used the strategy from Ref. [3], see `README.md`, dividing our map into a square grid, performing a group k-fold split, and grouping points inside the grid cell. We evaluated the data leakage by comparing the RMSE of a k-nearest-neighbors (kNN) model to a model outputting the training mean. "Estimated leakage" is defined as $1 - \frac{\mathrm{RMSE}_{\mathrm{KNN}}}{\mathrm{RMSE}_\mathrm{mean}}$, where $\mathrm{RMSE}_{\mathrm{KNN}}$ is the root mean squared error of the kNN estimator and $\mathrm{RMSE}_{\mathrm{mean}}$ is the root mean squared error of predicting the mean. If the kNN estimator performs similarly to predicting the mean, then the estimated leakage should be approximately 0; positive values indicate that kNN is performing better. We found that 10 km equal-area cells had approximately no leakage. 

However, from preliminary modeling with this data split strategy, we found large variation in scoring across folds, with some folds performing much worse. We realized that grid cells inside the Salinas Valley subbasins were not being uniformly distributed among folds. Altering our data split strategy again, we created a subbasin-stratefied split, finding good fold-to-fold stability, with little leakage, with 5 km and above size subbasin-stratefied splits. In the end, we chose the 10 km subbasin split, since we did not perform extensive leakage testing, and we preferred to be conservative. The resulting limitation of our model is that it cannot easily generalize to new locations, and relies upon stratefication amongst elevation data. 

# Next Steps

Going forward, we plan to create an interactive website to display an interactive map showing the areas of high-risk nitrate pollution in Salinas Valley. We hope to improve our model with new data sources. For instance, the work by Ransom et al. (Ref. [1]) has shown that alternative features, like the nitrogen applied/nitrogen removed per acre, could improve performance. This particular set of data can be requested, but is not currently publicly available. 