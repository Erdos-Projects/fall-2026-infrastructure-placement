# Modeling Plan - Predicting Areas of High-Risk Nitrate Pollution in Salinas Valley

We approached modeling from two angles: regression and classification. The regression task seeks to directly predict the nitrate pollutant concentration. We compared a dummy model and simple linear model to linear models with two regularization strategies (ridge and lasso), a decision tree model, a random forest model, and a boosted regression tree model. Per model, we tuned hyperparameters with extensive nested cross validation, using `GridSearchCV` or `RandomizedSearchCV` in each set of training folds, holding one whole fold out at the time, so the test fold is never used for tuning.

The classification task seeks to classify locations as exceedances (above nitrate concentrations of 10 mg/L) or non-exceedances. We compared a dummy model and simple logistic model to logistic models with two regularization strategies (L1 and L2), a decision tree model, a random forest model, and a boosted regression tree model.

