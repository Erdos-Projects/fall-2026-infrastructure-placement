from src.evaluation import splits, leakage
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.base import clone

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    root_mean_squared_error, 
    r2_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

 

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

def fit_model_with_normal_cv(development, X_dev, y_dev, model, THRESHOLD_MG_L = 10, PRINT=False):
    """
    Fit a model using normal cross-validation over the predefined folds.

    Returns
    -------
    results : dict
        oof_pred: OOF predictions on the log scale.
        y_true: Actual target values on the log scale.
        fold: Fold assignment for each observation.
        outer_results: RMSE for each held-out fold.
        overall_rmse: Pooled RMSE on the log scale.
        threshold_mg_l: Threshold used for classification.
    """
    # Actual-scale labels for stratification. nitrate is log scale
    exceeds_dev = np.expm1(y_dev) > THRESHOLD_MG_L

    # resets the dataframe index to consecutive integers.
    development = development.reset_index(drop=True)
    # Convert to NumPy arrays to ensure positional indexing.
    y_true = np.asarray(y_dev)
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
            "fold": outer_fold,
            #"n_test": len(outer_test),
            "rmse_log": root_mean_squared_error(
                y_true[outer_test],
                oof_pred[outer_test]
                ),
            "r2" : r2_score(
                y_true[outer_test],
                oof_pred[outer_test]   
                )
        })

    outer_results = pd.DataFrame(outer_results)
    overall_rmse = root_mean_squared_error(y_dev, oof_pred)
    overall_r2 = r2_score(y_dev, oof_pred)
    if PRINT == True:
        print("CV results (outer fold RMSE):")
        print(outer_results.round(3).to_string(index=False))
        print(f"Overall RMSE (log): {overall_rmse:.3f}")
        print(f"R²: {overall_r2:.3f} ")
        print(f"Accuracy of the >10 mg/L call: {np.mean((np.expm1(oof_pred) > 10) == exceeds_dev):.3f}")

    return {
        "oof_pred": oof_pred,
        "y_true": y_true,
        "fold": fold_dev,
        "outer_results": outer_results,
        "overall_rmse": overall_rmse,
        "overall_r2" : overall_r2,
        "threshold_mg_l": THRESHOLD_MG_L
    }


def plot_actual_vs_predicted(results, LOG_SCALE=False):
    """
    Plot actual versus OOF-predicted nitrate concentrations.

    Parameters
    ----------
    results : dict
        Output from fit_model_with_normal_cv.
    LOG_SCALE : bool
        If True, plot the target on its original log scale.
        Otherwise, plot nitrate concentrations in mg/L.
    """

    y_true = results["y_true"]
    oof_pred = results["oof_pred"]
    fold = results["fold"]

    if LOG_SCALE:
        actual = y_true
        predicted = oof_pred
        axis_label = "log_nitrate"
    else:
        actual = np.expm1(y_true)
        predicted = np.expm1(oof_pred)
        axis_label = "Nitrate concentration (mg/L)"

    fig, ax = plt.subplots(figsize=(8, 6))

    for outer_fold in sorted(np.unique(fold)):
        mask = fold == outer_fold

        ax.scatter(
            actual[mask],
            predicted[mask],
            alpha=0.6,
            label=f"Fold {outer_fold}"
        )

    # Perfect-prediction reference line.
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    ax.plot(
        [lower, upper],
        [lower, upper],
        "k--",
        label="Perfect prediction"
    )

    ax.set_xlabel(f"Actual {axis_label}")
    ax.set_ylabel(f"Predicted {axis_label}")
    ax.set_title("Actual vs. Predicted Nitrate Concentrations")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_residuals(results, LOG_SCALE=False):
    """
    Plot OOF residuals against predicted nitrate concentrations.

    Parameters
    ----------
    results : dict
        Output from fit_model_with_normal_cv.
    LOG_SCALE : bool
        If True, calculate residuals on the log scale.
        Otherwise, use mg/L.
    """

    y_true = results["y_true"]
    oof_pred = results["oof_pred"]
    fold = results["fold"]

    if LOG_SCALE:
        actual = y_true
        predicted = oof_pred
        axis_label = "log_nitrate"
    else:
        actual = np.expm1(y_true)
        predicted = np.expm1(oof_pred)
        axis_label = "Nitrate concentration (mg/L)"

    residuals = actual - predicted

    fig, ax = plt.subplots(figsize=(8, 5))

    for outer_fold in sorted(np.unique(fold)):
        mask = fold == outer_fold

        ax.scatter(
            predicted[mask],
            residuals[mask],
            alpha=0.6,
            label=f"Fold {outer_fold}"
        )

    ax.axhline(0, color="black", linestyle="--")

    ax.set_xlabel(f"Predicted {axis_label}")
    ax.set_ylabel(f"Residual ({axis_label})")
    ax.set_title("Residuals vs. Predicted Values")
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_spatial_results(
    results,
    development,
    LAT_COL="gm_latitude",
    LON_COL="gm_longitude",
    LOG_SCALE=False
):
    """
    Map actual nitrate concentrations and OOF residuals.

    Parameters
    ----------
    results : dict
        Output from fit_model_with_normal_cv.
    development : DataFrame
        DataFrame corresponding row-for-row to the evaluation data.
    LAT_COL, LON_COL : str
        Latitude and longitude column names.
    LOG_SCALE : bool
        If True, map actual values and residuals on the log scale.
        Otherwise, use mg/L.
    """

    y_true = results["y_true"]
    oof_pred = results["oof_pred"]

    lat = development[LAT_COL].to_numpy()
    lon = development[LON_COL].to_numpy()

    if LOG_SCALE:
        actual = y_true
        predicted = oof_pred
        value_label = "log_nitrate"
        residual_label = "Residual (log scale)"
    else:
        actual = np.expm1(y_true)
        predicted = np.expm1(oof_pred)
        value_label = "Nitrate (mg/L)"
        residual_label = "Residual (mg/L)"

    residuals = actual - predicted

    # Use a shared, symmetric scale for positive and negative residuals.
    error_limit = np.max(np.abs(residuals))

    if error_limit == 0:
        error_limit = 1

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Map actual concentrations.
    sc1 = axes[0].scatter(
        lon,
        lat,
        c=actual,
        cmap="viridis",
        alpha=0.8
    )

    axes[0].set_title("Actual Nitrate Concentrations")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    fig.colorbar(sc1, ax=axes[0], label=value_label)

    # Map residuals.
    sc2 = axes[1].scatter(
        lon,
        lat,
        c=residuals,
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
        alpha=0.8
    )

    axes[1].set_title("Out-of-Fold Prediction Errors")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    fig.colorbar(sc2, ax=axes[1], label=residual_label)

    plt.tight_layout()
    plt.show()


def plot_rmse_by_fold(results):
    """
    Plot RMSE for each outer fold.

    The RMSE is calculated on the log scale, matching the
    evaluation performed by fit_model_with_normal_cv.
    """

    outer_results = results["outer_results"]
    overall_rmse = results["overall_rmse"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        outer_results["fold"].astype(str),
        outer_results["rmse_log"]
    )

    ax.axhline(
        overall_rmse,
        color="red",
        linestyle="--",
        label=f"Pooled RMSE = {overall_rmse:.3f}"
    )

    ax.set_xlabel("Outer fold")
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("Spatial Cross-Validation Performance")
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_rmse_by_fold_mg_l(results):
    """
    Plot RMSE in mg/L for each outer fold.
    """

    y_true = results["y_true"]
    oof_pred = results["oof_pred"]
    fold = results["fold"]

    actual = np.expm1(y_true)
    predicted = np.expm1(oof_pred)

    fold_results = []

    for outer_fold in sorted(np.unique(fold)):
        mask = fold == outer_fold

        rmse = root_mean_squared_error(
            actual[mask],
            predicted[mask]
        )

        fold_results.append({
            "outer_fold": outer_fold,
            "rmse_mg_l": rmse
        })

    fold_results = pd.DataFrame(fold_results)

    overall_rmse = root_mean_squared_error(actual, predicted)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        fold_results["outer_fold"].astype(str),
        fold_results["rmse_mg_l"]
    )

    ax.axhline(
        overall_rmse,
        color="red",
        linestyle="--",
        label=f"Pooled RMSE = {overall_rmse:.2f} mg/L"
    )

    ax.set_xlabel("Outer fold")
    ax.set_ylabel("RMSE (mg/L)")
    ax.set_title("Spatial Cross-Validation RMSE on Original Scale")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_threshold_results(
    results,
    THRESHOLD_MG_L=None,
    LOG_SCALE=False
):
    """
    Plot actual versus predicted nitrate concentrations
    with threshold lines.

    Parameters
    ----------
    results : dict
        Output from fit_model_with_normal_cv.
    THRESHOLD_MG_L : float, optional
        Threshold in mg/L. Defaults to the value in results.
    LOG_SCALE : bool
        If True, plot log_nitrate values and the log-transformed
        threshold. Otherwise, plot nitrate concentrations in mg/L.
    """

    if THRESHOLD_MG_L is None:
        THRESHOLD_MG_L = results["threshold_mg_l"]

    y_true = results["y_true"]
    oof_pred = results["oof_pred"]
    fold = results["fold"]

    if LOG_SCALE:
        actual = y_true
        predicted = oof_pred
        threshold = np.log1p(THRESHOLD_MG_L)
        axis_label = "log_nitrate"
    else:
        actual = np.expm1(y_true)
        predicted = np.expm1(oof_pred)
        threshold = THRESHOLD_MG_L
        axis_label = "Nitrate concentration (mg/L)"

    fig, ax = plt.subplots(figsize=(8, 6))

    for outer_fold in sorted(np.unique(fold)):
        mask = fold == outer_fold

        ax.scatter(
            actual[mask],
            predicted[mask],
            alpha=0.6,
            label=f"Fold {outer_fold}"
        )

    # Perfect-prediction reference line.
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    ax.plot(
        [lower, upper],
        [lower, upper],
        "k--",
        label="Perfect prediction"
    )

    # Threshold lines.
    ax.axvline(
        threshold,
        color="red",
        linestyle=":"
    )

    ax.axhline(
        threshold,
        color="red",
        linestyle=":",
        label=f"{THRESHOLD_MG_L} mg/L threshold"
    )

    ax.set_xlabel(f"Actual {axis_label}")
    ax.set_ylabel(f"Predicted {axis_label}")
    ax.set_title("Predictions Relative to the Nitrate Threshold")
    ax.legend()

    plt.tight_layout()
    plt.show()



def fit_classifier_with_nested_cv(development, X_dev, y_dev, param_grid, make_search_model, 
                                  N_SUBFOLDS = 4, SEED = 77, THRESHOLD_MG_L = 10,
                                  CLASSIFICATION_THRESHOLD=0.5, PRINT=False):

    y_class = (
        np.expm1(np.asarray(y_dev).ravel()) > THRESHOLD_MG_L
    ).astype(int)

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
        search.fit(X_dev[outer_train], y_class[outer_train])

        # Predict probabilities for the held-out outer fold.
        outer_prob = search.predict_proba(X_dev[outer_test])[:, 1]
        
        oof_pred[outer_test] = outer_prob

        # Convert probabilities to binary classifications.
        outer_class = (outer_prob >= CLASSIFICATION_THRESHOLD).astype(int)
        y_test = y_class[outer_test]

        # Metrics on the outer test fold.
        outer_ap = average_precision_score(
            y_test, outer_prob
        )

        # ROC-AUC is undefined if the test fold contains only one class.
        outer_auc = (
            roc_auc_score(y_test, outer_prob)
            if len(np.unique(y_test)) == 2
            else np.nan
        )


        outer_results.append({
            "outer_fold": outer_fold,

            "average_precision": outer_ap,
            "roc_auc": outer_auc,

            "precision": precision_score(
                y_test,
                outer_class,
                zero_division=0,
            ),

            "recall": recall_score(
                y_test,
                outer_class,
                zero_division=0,
            ),

            "f1": f1_score(
                y_test,
                outer_class,
                zero_division=0,
            ),

            "balanced_accuracy": balanced_accuracy_score(
                y_test,
                outer_class,
            ),

            "brier_score_loss" : brier_score_loss(
                y_test, 
                outer_prob 
            ),

            # Best inner-CV score, without negation.
            "best_inner_ap": search.best_score_,

            # Class balance in the outer test fold.
            "n_exceedances": int(y_test.sum()),
            "n_samples": len(y_test),
        })

        # Get the inner-fold RMSEs for the best parameter combination
        best_idx = search.best_index_

        inner_ap = {
            f"inner_fold_{i}_ap":
                search.cv_results_[
                    f"split{i}_test_score"
                ][best_idx]
            for i in range(N_SUBFOLDS)
        }

        # Add inner-fold scores to the outer-fold results.
        outer_results[-1].update(inner_ap)
        
        best_params_by_fold.append({
        "outer_fold": outer_fold,
        **search.best_params_,
        })
    
    outer_results = pd.DataFrame(outer_results)
    best_params_by_fold = pd.DataFrame(best_params_by_fold)

    # Overall out-of-fold metrics.
    overall_class = (
        oof_pred >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    overall_ap = average_precision_score(
        y_class, oof_pred
    )

    overall_auc = (
        roc_auc_score(y_class, oof_pred)
        if len(np.unique(y_class)) == 2
        else np.nan
    )

    if PRINT:

        print("Nested CV results:")
        print(outer_results.round(3).to_string(index=False))

        print("\nBest parameters per outer fold:")
        print(best_params_by_fold.to_string(index=False))

        print("\nOverall out-of-fold results:")
        print(f"Average precision: {overall_ap:.3f}")
        print(f"ROC-AUC:           {overall_auc:.3f}")

        print(
            f"Precision:         "
            f"{precision_score(y_class, overall_class, zero_division=0):.3f}"
        )

        print(
            f"Recall:            "
            f"{recall_score(y_class, overall_class, zero_division=0):.3f}"
        )

        print(
            f"F1:                "
            f"{f1_score(y_class, overall_class, zero_division=0):.3f}"
        )

        print(
            f"Balanced accuracy: "
            f"{balanced_accuracy_score(y_class, overall_class):.3f}"
        )

    return outer_results, best_params_by_fold, oof_pred


def fit_classifier_with_normal_cv(development, X_dev, y_dev, model, THRESHOLD_MG_L = 10,
                                  CLASSIFICATION_THRESHOLD=0.5, PRINT=False):

    y_class = (
        np.expm1(np.asarray(y_dev).ravel()) > THRESHOLD_MG_L
    ).astype(int)

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



        fold_model = clone(model)
        fold_model.fit(X_dev[outer_train], y_class[outer_train])
        outer_prob = fold_model.predict_proba(X_dev[outer_test])[:, 1]
        
        oof_pred[outer_test] = outer_prob

        # Convert probabilities to binary classifications.
        outer_class = (outer_prob >= CLASSIFICATION_THRESHOLD).astype(int)
        y_test = y_class[outer_test]

        # Metrics on the outer test fold.
        outer_ap = average_precision_score(
            y_test, outer_prob
        )

        # ROC-AUC is undefined if the test fold contains only one class.
        outer_auc = (
            roc_auc_score(y_test, outer_prob)
            if len(np.unique(y_test)) == 2
            else np.nan
        )

        outer_results.append({
            "outer_fold": outer_fold,

            "average_precision": outer_ap,
            "roc_auc": outer_auc,

            "precision": precision_score(
                y_test,
                outer_class,
                zero_division=0,
            ),

            "recall": recall_score(
                y_test,
                outer_class,
                zero_division=0,
            ),

            "f1": f1_score(
                y_test,
                outer_class,
                zero_division=0,
            ),

            "balanced_accuracy": balanced_accuracy_score(
                y_test,
                outer_class,
            ),

            "brier_score_loss" : brier_score_loss(
                y_test, 
                outer_prob 
            ),

            # Class balance in the outer test fold.
            "n_exceedances": int(y_test.sum()),
            "n_samples": len(y_test),
            "exceedance_rate": y_test.mean(),
        })
    
    outer_results = pd.DataFrame(outer_results)

    # Overall out-of-fold metrics.
    overall_class = (
        oof_pred >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    overall_ap = average_precision_score(
        y_class, oof_pred
    )

    overall_auc = (
        roc_auc_score(y_class, oof_pred)
        if len(np.unique(y_class)) == 2
        else np.nan
    )

    oof_results = pd.DataFrame({
        "actual": y_class,
        "probability": oof_pred,
        "predicted": (
            oof_pred >= CLASSIFICATION_THRESHOLD
        ).astype(int),
        "fold": fold_dev,
    })

    if PRINT:
        print("\nOverall out-of-fold results:")
        print(f"Average precision: {overall_ap:.3f}")
        print(f"ROC-AUC:           {overall_auc:.3f}")

        print(
            f"Precision:         "
            f"{precision_score(y_class, overall_class, zero_division=0):.3f}"
        )

        print(
            f"Recall:            "
            f"{recall_score(y_class, overall_class, zero_division=0):.3f}"
        )

        print(
            f"F1:                "
            f"{f1_score(y_class, overall_class, zero_division=0):.3f}"
        )

        print(
            f"Balanced accuracy: "
            f"{balanced_accuracy_score(y_class, overall_class):.3f}"
        )

    return outer_results, oof_results


def plot_confusion_matrix(oof_results, threshold=0.5):
    from sklearn.metrics import ConfusionMatrixDisplay

    actual = oof_results["actual"].to_numpy()

    predicted = (
        oof_results["probability"].to_numpy() >= threshold
    ).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        actual,
        predicted,
        labels=[0, 1],
        display_labels=["No exceedance", "Exceedance"],
        cmap="Blues",
        values_format="d",
        ax=ax,
    )

    ax.set_title(
        f"Out-of-fold confusion matrix (threshold = {threshold:.2f})"
    )

    plt.tight_layout()
    plt.show()


def plot_precision_recall(oof_results):
    from sklearn.metrics import PrecisionRecallDisplay

    actual = oof_results["actual"].to_numpy()
    probabilities = oof_results["probability"].to_numpy()

    ap = average_precision_score(actual, probabilities)
    prevalence = actual.mean()

    fig, ax = plt.subplots(figsize=(7, 5))

    PrecisionRecallDisplay.from_predictions(
        actual,
        probabilities,
        name=f"Random Forest (AP = {ap:.3f})",
        ax=ax,
    )

    ax.axhline(
        prevalence,
        linestyle="--",
        label=f"Prevalence baseline = {prevalence:.3f}",
    )

    ax.set_title("Out-of-fold precision-recall curve")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_roc_curve(oof_results):
    from sklearn.metrics import RocCurveDisplay

    actual = oof_results["actual"].to_numpy()
    probabilities = oof_results["probability"].to_numpy()

    if len(np.unique(actual)) < 2:
        raise ValueError(
            "ROC curve requires both classes in the dataset."
        )

    auc = roc_auc_score(actual, probabilities)

    fig, ax = plt.subplots(figsize=(7, 5))

    RocCurveDisplay.from_predictions(
        actual,
        probabilities,
        name=f"Random Forest (AUC = {auc:.3f})",
        ax=ax,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random ranking",
    )

    ax.set_title("Out-of-fold ROC curve")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_calibration_curve(oof_results, n_bins=10):
    actual = oof_results["actual"].to_numpy()
    probabilities = oof_results["probability"].to_numpy()

    observed_rate, mean_predicted = calibration_curve(
        actual,
        probabilities,
        n_bins=n_bins,
        strategy="uniform",
    )

    brier = brier_score_loss(actual, probabilities)

    prevalence = actual.mean()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        mean_predicted,
        observed_rate,
        marker="o",
        label="Random Forest",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    ax.axhline(
        prevalence,
        linestyle=":",
        label=f"Exceedance prevalence = {prevalence:.3f}",
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed exceedance frequency")
    ax.set_title(f"Probability calibration (Brier score = {brier:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_probability_distributions(oof_results):
    actual = oof_results["actual"].to_numpy()
    probabilities = oof_results["probability"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        probabilities[actual == 0],
        bins=20,
        alpha=0.6,
        density=True,
        label="Actual: no exceedance",
    )

    ax.hist(
        probabilities[actual == 1],
        bins=20,
        alpha=0.6,
        density=True,
        label="Actual: exceedance",
    )

    ax.set_xlabel("Predicted probability of exceedance")
    ax.set_ylabel("Density")
    ax.set_title("Predicted probabilities by actual nitrate status")

    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_outer_fold_metrics(outer_results):
    metrics = [
        "average_precision",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "balanced_accuracy",
    ]

    available_metrics = [
        metric for metric in metrics
        if metric in outer_results.columns
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(outer_results))
    width = 0.8 / len(available_metrics)

    for i, metric in enumerate(available_metrics):
        ax.bar(
            x + (i - (len(available_metrics) - 1) / 2) * width,
            outer_results[metric],
            width=width,
            label=metric.replace("_", " ").title(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Fold {fold}" for fold in outer_results["outer_fold"]]
    )

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Classifier performance across spatial outer folds")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_outer_fold_brier(outer_results):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        outer_results["outer_fold"].astype(str),
        outer_results["brier_score_loss"],
    )

    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Brier score")
    ax.set_title("Probability prediction error across spatial folds")

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_threshold_metrics(oof_results):
    actual = oof_results["actual"].to_numpy()
    probabilities = oof_results["probability"].to_numpy()

    thresholds = np.linspace(0.05, 0.95, 91)

    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)

        precisions.append(
            precision_score(actual, predicted, zero_division=0)
        )

        recalls.append(
            recall_score(actual, predicted, zero_division=0)
        )

        f1_scores.append(
            f1_score(actual, predicted, zero_division=0)
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.plot(thresholds, f1_scores, label="F1-score")

    ax.axvline(
        0.5,
        linestyle="--",
        label="Default threshold = 0.5",
    )

    ax.set_xlabel("Classification probability threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision, recall, and F1 versus threshold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()