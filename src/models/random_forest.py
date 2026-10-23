from src.evaluation import splits
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.metrics import root_mean_squared_error # type: ignore


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


# Set PRINT = True to see a detailed report. 
# you need to create a dictionary param_grid = {"hyperparameter_name" : [1, 2, 3]}

def fit_model_with_nested_cv(development, X_dev, y_dev, param_grid, make_search_model, N_SUBFOLDS = 4, THRESHOLD_MG_L = 10, SEED = 77, PRINT=False):
    # Actual-scale labels for stratification. nitrate is log scale
    exceeds_dev = np.expm1(y_dev) > THRESHOLD_MG_L

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
        # Because there are 20 combinations and four inner splits, 
        # this requires 80 model fits per outer fold.
        # With four outer folds, that is 320 inner-CV fits, plus four refits of the selected models.
        search = make_search_model(inner_cv, param_grid, SEED)

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

    if (PRINT == True):
        print("Nested CV results (outer and inner fold RMSE):")
        print(outer_results.round(3).to_string(index=False))

        print("\nBest parameters per outer fold:")
        print(best_params_by_fold.to_string(index=False))

        print(f"Overall RMSE (log): {root_mean_squared_error(y_dev, oof_pred):.3f}")
        print(f"Accuracy of the >10 mg/L call: {np.mean((np.expm1(oof_pred) > 10) == exceeds_dev):.3f}")

    return outer_results, best_params_by_fold


# you need to set your model
# model = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEAT, random_state=SEED, n_jobs=1)

def fit_model_with_normal_cv(development, X_dev, y_dev, model, THRESHOLD_MG_L = 10):
    # Actual-scale labels for stratification. nitrate is log scale
    exceeds_dev = np.expm1(y_dev) > THRESHOLD_MG_L

    # resets the dataframe index to consecutive integers.
    development = development.reset_index(drop=True)
    fold_dev = development["fold"].to_numpy()

    #Initialize storage for predictions and results
    oof_pred = np.full(len(development), np.nan)
    outer_results = []

    # Finds the distinct original fold IDs and processes them in ascending order.
    for outer_fold in sorted(np.unique(fold_dev)):
        #Identify the outer training and test observations
        #np.flatnonzero() returns the integer positions where a condition is true.
        outer_test = np.flatnonzero(fold_dev == outer_fold)
        outer_train = np.flatnonzero(fold_dev != outer_fold)
        #outer_test: positions of all observations belonging to the current outer fold.
        #outer_train: positions of all observations belonging to every other development fold.

        model.fit(X_dev[outer_train], y_dev[outer_train])
        oof_pred[outer_test] = model.predict(X_dev[outer_test])

        outer_results.append({
            "outer_fold": outer_fold,
            "rmse_log": root_mean_squared_error(
                y_dev[outer_test], oof_pred[outer_test]
            )
        })

    outer_results = pd.DataFrame(outer_results)

    print("CV results (outer fold RMSE):")
    print(outer_results.round(3).to_string(index=False))

    print(f"Overall RMSE (log): {root_mean_squared_error(y_dev, oof_pred):.3f}")
    print(f"Accuracy of the >10 mg/L call: {np.mean((np.expm1(oof_pred) > 10) == exceeds_dev):.3f}")



def find_best_hyperparameter(all_best_params):
    # Ranks the combinations by how often they were selected across all seeds and outer folds.
    overall_frequency = (
        all_best_params
        .groupby(["max_features", "n_estimators"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    overall_frequency["frequency"] = (
        overall_frequency["count"] / len(all_best_params)
    )

    print(f"Overall frequency:")
    print(f"{overall_frequency}")
    #Find the most frequent combination for each outer fold
    fold_frequency = (
        all_best_params
        .groupby(["outer_fold", "max_features", "n_estimators"])
        .size()
        .reset_index(name="count")
    )

    # Normalize within each outer fold
    fold_frequency["frequency"] = (
        fold_frequency["count"]
        / fold_frequency.groupby("outer_fold")["count"].transform("sum")
    )

    fold_frequency = fold_frequency.sort_values(
        ["outer_fold", "count"],
        ascending=[True, False],
    )

    #print(f"The most frequent combination for each outer fold:")
    #print(f"{fold_frequency.to_string(index=False)}")

    most_frequent_per_fold = (
        fold_frequency
        .sort_values(
            ["outer_fold", "count", "n_estimators"],
            ascending=[True, False, True],
        )
        .drop_duplicates("outer_fold")
    )

    print(f"Extract only the most frequent combination for each outer fold:")
    print(f"{most_frequent_per_fold.to_string(index=False)}")
