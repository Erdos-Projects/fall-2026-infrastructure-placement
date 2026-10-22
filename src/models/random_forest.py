from src.evaluation import splits
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error


def make_inner_subfolds(data, FEATURES, N_SUBFOLDS = 4, THRESHOLD_MG_L = 10, SEED = 77):

    # Resets data to use consecutive integer indices: 0, 1, 2, ..., n-1. drop=True discards the old index
    data = data.reset_index(drop=True)

    #Selects only rows for which holdout is False.
    development = data.loc[~data["holdout"]].copy()
    X_dev = development[FEATURES].to_numpy()
    y_dev = development["log_nitrate"].to_numpy()

    # Actual-scale labels for stratification. nitrate is log scale
    exceeds_dev = np.expm1(y_dev) > THRESHOLD_MG_L

    # Adds a column called subfold and initializes every row to -1.
    # The value -1 is just a placeholder meaning that no subfold has been assigned yet. 
    # The subsequent loop replaces it with 0, 1, 2, or 3.
    development["subfold"] = -1


    #development["fold"].unique() finds the distinct original fold IDs.
    #sorted(...) sorts them in ascending order.
    #The loop processes each original development fold independently.

    for original_fold in sorted(development["fold"].unique()):
        #Creates a Boolean mask indicating which rows belong to the current original fold.
        mask = development["fold"].eq(original_fold)
        # select only those rows belong to original_fold = 0 etc.
        part = development.loc[mask]

        # Generate new spatial subfold assignments
        subfolds = splits.spatial_folds(
            cells=part["cell"].to_numpy(),
            exceeds=exceeds_dev[mask.to_numpy()],
            n_splits=N_SUBFOLDS,
            seed=SEED + int(original_fold),
        )
        #Store the new assignments
        development.loc[mask, "subfold"] = subfolds

    print(
        development.groupby(["fold", "subfold"])
        .size()
        .unstack(fill_value=0)
    )
    return development, X_dev, y_dev


def make_search(inner_cv, param_grid, SEED=77):
    return GridSearchCV(
        estimator=RandomForestRegressor(
            random_state=SEED,
            n_jobs=1,
        ),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
    )

def fit_model_with_nested_cv(development, X_dev, y_dev, N_SUBFOLDS = 4, THRESHOLD_MG_L = 10, SEED = 77):
    # Actual-scale labels for stratification. nitrate is log scale
    exceeds_dev = np.expm1(y_dev) > THRESHOLD_MG_L
    param_grid = {
    "n_estimators": [100, 300, 500],
    "max_features": [1.0, "sqrt", 0.5],
    }
    #"n_estimators": the number of decision trees in the random forest.
    #"max_features": how many predictor features a tree can consider when searching for the best split.
    #   1.0: consider all predictor features at each split.
    #   "sqrt": consider approximately the square root of the number of features.
    #   0.5: consider 50% of the available features.


    # resets the dataframe index to consecutive integers.
    development = development.reset_index(drop=True)
    fold_dev = development["fold"].to_numpy()
    subfold_dev = development["subfold"].to_numpy()

    #Initialize storage for predictions and results
    oof_pred = np.full(len(development), np.nan)
    outer_results = []
    best_params_by_fold = []

    # Finds the distinct original fold IDs and processes them in ascending order.
    for outer_fold in sorted(np.unique(fold_dev)):
        #Identify the outer training and test observations
        #np.flatnonzero() returns the integer positions where a condition is true.
        outer_test = np.flatnonzero(fold_dev == outer_fold)
        outer_train = np.flatnonzero(fold_dev != outer_fold)
        #outer_test: positions of all observations belonging to the current outer fold.
        #outer_train: positions of all observations belonging to every other development fold.

        # Selects the subfold labels corresponding only to the outer training observations.
        train_subfolds = subfold_dev[outer_train]

        #Initializes an empty list to hold the inner splits.
        inner_cv = []

        #With N_SUBFOLDS = 4, the loop processes subfolds 0, 1, 2, and 3.
        for inner_fold in range(N_SUBFOLDS):
            #Selects the positions within train_subfolds assigned to the current subfold.
            inner_valid = np.flatnonzero(
                train_subfolds == inner_fold
            )
            #Selects every other position, excluding the current validation subfold.
            inner_train = np.flatnonzero(
                train_subfolds != inner_fold
            )
            #Adds the training and validation positions as a tuple.
            inner_cv.append((inner_train, inner_valid))

        # After the loop, inner_cv contains four splits. 
        #
        # Each split uses three subfold groups for training and one for validation.
        # For example: 
        # Outer Fold 0 is set aside for evaluating the tuned model. 

        # 1. Validation = Outer Folds 1–3, Inner Subfold 0 ~ 1.0, 2.0, 3.0
        # 1. Training = Outer Folds 1–3, Inner Subfolds 1,2,3. ~ 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3

        # 2. Validation = Outer Folds 1–3, Inner Subfold 1 ~ 1.1, 2.1, 3.1
        # 2. Training = Outer Folds 1–3, Inner Subfolds 0,2,3. ~ 1.0, 1.2, 1.3, 2.0, 2.2, 2.3, 3.0, 3.2, 3.3
        # ... 

        # creates a GridSearchCV object using the inner splits for the current outer fold.
        #Because there are nine combinations and four inner splits, 
        # this requires 36 model fits per outer fold.
        # With four outer folds, that is 144 inner-CV fits, plus four refits of the selected models.
        search = make_search(inner_cv, param_grid, SEED)

        # evaluates the nine hyperparameter combinations using only the outer training data.
        search.fit(X_dev[outer_train], y_dev[outer_train])
        # Uses the best model selected by the inner CV to predict the observations in the outer test fold.
        oof_pred[outer_test] = search.predict(X_dev[outer_test])

        # Get the inner-fold RMSEs for the best parameter combination
        best_idx = search.best_index_

        inner_rmse = {
            f"inner_fold_{i}_rmse_log": -search.cv_results_[f"split{i}_test_score"][best_idx]
            for i in range(N_SUBFOLDS)
        }

        # Include them in this outer fold's results
        outer_results.append({
            "outer_fold": outer_fold,
            "rmse_log": root_mean_squared_error(
                y_dev[outer_test], oof_pred[outer_test]
            ),
            "best_inner_rmse_log": -search.best_score_,
            **inner_rmse,
        })
        
        best_params_by_fold.append({
        "outer_fold": outer_fold,
        **search.best_params_,
        })


    outer_results = pd.DataFrame(outer_results)
    best_params_by_fold = pd.DataFrame(best_params_by_fold)

    print("Nested CV results (outer and inner fold RMSE):")
    print(outer_results.round(3).to_string(index=False))

    print("\nBest parameters per outer fold:")
    print(best_params_by_fold.to_string(index=False))

    print(f"Overall RMSE (log): {root_mean_squared_error(y_dev, oof_pred):.3f}")
    print(f"Accuracy of the >10 mg/L call: {np.mean((np.expm1(oof_pred) > 10) == exceeds_dev):.3f}")