"""Shared spatial cross-validation for the regression and classification notebooks."""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneGroupOut
from sklearn.metrics import root_mean_squared_error, r2_score, roc_auc_score, brier_score_loss

from src.features.preprocessing import load_model_data
from src.evaluation import splits
from src.evaluation.make_splits import load_split
from src.evaluation.results_log import save_run      # replaces the existing save_run import
from src.paths import ROOT, PREDICTIONS as PREDICTIONS_DIR


# Columns saved with the predictions: enough to identify each sample and make every plot
ID_COLS = ["gm_well_id", "gm_latitude", "gm_longitude", "gm_samp_collection_date", "subbasin", "cell", "fold", "log_nitrate", "exceeds"]


THRESHOLD_MG_L = 10

# (split, use subbasin features); subbasin features replace elevation
CONFIGS = [
    ("10km", False),
    ("10km", True),
    ("10km_subbasin", True),
    ("7.5km_subbasin", True),
    ("5km_subbasin", True),
]

# One column from each group that sums to 100%, dropped for linear models to avoid perfect collinearity
REFERENCE_COLS = ["soil_drainage_moderate_pct", "soil_hydgrpdcd_pct_b", "subbasin_FOREBAY AQUIFER"]


def load_development(split, use_subbasin, threshold=THRESHOLD_MG_L):
    """Folds 0-3 of one configuration (fold 4 stays untouched), with log nitrate and the 0/1 exceedance label."""
    X, y, meta = load_model_data(subbasin=use_subbasin, drop_elevation=use_subbasin)
    features = list(X.columns)
    data = load_split(pd.concat([meta, X], axis=1).assign(log_nitrate=y), split=split)
    dev = data[~data["holdout"]].reset_index(drop=True)
    dev["exceeds"] = (np.expm1(dev["log_nitrate"]) > threshold).astype(int)
    return dev, features


def all_features(dev, features):
    return features


def linear_features(dev, features):
    """Drop the reference columns, and columns with a single value (subbasins with no samples)."""
    return [f for f in features if f not in REFERENCE_COLS and dev[f].nunique() > 1]


def tuned(estimator, grid, scoring, n_iter=None, seed=0):
    """Hyperparameter search on the training folds only, holding out one whole fold at a time."""
    if n_iter is None:
        return GridSearchCV(estimator, grid, cv=LeaveOneGroupOut(), scoring=scoring, n_jobs=-1)
    return RandomizedSearchCV(estimator, grid, n_iter=n_iter, cv=LeaveOneGroupOut(), scoring=scoring,
                              random_state=seed, n_jobs=-1)


def regression_scores(y_true, pred):
    return {"rmse_log": root_mean_squared_error(y_true, pred), "r2": r2_score(y_true, pred)}


def classification_scores(y_true, prob, base_rate):
    """AUC, Brier score, and Brier skill against always predicting the training exceedance rate."""
    brier = brier_score_loss(y_true, prob)
    brier_base = brier_score_loss(y_true, np.full(len(y_true), base_rate))
    return {"auc": roc_auc_score(y_true, prob), "brier": brier, "brier_skill": 1 - brier / brier_base}


def spatial_cv(make_model, dev, features, task):
    """Fit a new model on three folds and score it on the fourth, for each fold in turn.
    If make_model returns a search, it is tuned inside the three training folds."""
    target = "log_nitrate" if task == "regression" else "exceeds"
    X, y = dev[features], dev[target]
    oof = np.full(len(dev), np.nan)
    rows = []
    for train, test in splits.iter_folds(dev["fold"]):
        model = make_model()
        is_search = isinstance(model, (GridSearchCV, RandomizedSearchCV))
        fit_kwargs = {"groups": dev.loc[train, "fold"]} if is_search else {}
        model.fit(X[train], y[train], **fit_kwargs)
        if task == "regression":
            oof[test] = model.predict(X[test])
            scores = regression_scores(y[test], oof[test])
        else:
            oof[test] = model.predict_proba(X[test])[:, 1]
            scores = classification_scores(y[test], oof[test], y[train].mean())
        row = {"fold": int(dev.loc[test, "fold"].iloc[0]), **scores}
        if is_search:
            row.update({k.split("__")[-1]: v for k, v in model.best_params_.items()})
        rows.append(row)
    return pd.DataFrame(rows), oof


def predictions_path(task, split, use_subbasin):
    return PREDICTIONS_DIR / f"{task}__{split}__subbasin{int(use_subbasin)}.csv.gz"


def save_predictions(dev, preds, task, split, use_subbasin):
    """Save one configuration's out-of-fold predictions: the ID columns plus one column per model."""
    table = dev[[c for c in ID_COLS if c in dev.columns]].copy()
    for name, pred in preds.items():
        table[name] = pred
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(predictions_path(task, split, use_subbasin), index=False)


def load_predictions(task, configs=CONFIGS):
    """Rebuild devs and oof from the saved files, in the same form run_models returns."""
    devs, oof = {}, {}
    for split, use_subbasin in configs:
        table = pd.read_csv(predictions_path(task, split, use_subbasin),
                            dtype={"gm_well_id": "string"}, float_precision="round_trip")
        id_cols = [c for c in ID_COLS if c in table.columns]
        devs[(split, use_subbasin)] = table[id_cols]
        for name in table.columns:
            if name not in id_cols:
                oof[(name, split, use_subbasin)] = table[name].to_numpy()
    return devs, oof


def run_models(models, task, configs=CONFIGS, save=True):
    """Run every model on every configuration; save each run's scores to results/model_runs/
    and its out-of-fold predictions to results/predictions/.
    models: {name: (author, make_model, feature_rule)}.
    Returns the development tables and out-of-fold predictions, keyed by configuration."""
    devs, oof = {}, {}
    for split, use_subbasin in configs:
        dev, features = load_development(split, use_subbasin)
        devs[(split, use_subbasin)] = dev
        preds = {}
        for name, (author, make_model, feature_rule) in models.items():
            feats = feature_rule(dev, features)
            start = time.time()
            fold_scores, preds[name] = spatial_cv(make_model, dev, feats, task)
            oof[(name, split, use_subbasin)] = preds[name]
            if save:
                save_run(fold_scores, model=name, task=task, author=author, split=split,
                         use_subbasin=use_subbasin, drop_elevation=use_subbasin, n_features=len(feats))
            print(f"{split:15} subbasin={use_subbasin!s:5} {name:20} {time.time() - start:6.1f} s")
        if save:
            save_predictions(dev, preds, task, split, use_subbasin)
    return devs, oof
