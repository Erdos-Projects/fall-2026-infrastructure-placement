"""Shared log of model results, so every notebook saves scores in the same format.

Each call to save_run() writes one small CSV to results/model_runs/ (one file per model
and configuration, so teammates never edit the same file and there are no Git conflicts).
load_runs() combines all of them, and summarize() builds the comparison table.

Example
-------
from src.evaluation.results_log import save_run, load_runs, summarize

save_run(ridge_scores, model="Ridge", task="regression", author="Andrea",
         split=SPLIT, use_subbasin=USE_SUBBASIN, drop_elevation=DROP_ELEVATION,
         n_features=len(LINEAR_FEATURES))

summarize(load_runs())
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.paths import ROOT, MODEL_RUNS as RUNS_DIR


METRICS = {
    "regression": ["rmse_log", "r2"],
    "classification": ["auc", "brier_skill"],
}
KEYS = ["task", "model", "author", "split", "use_subbasin", "drop_elevation"]


def _slug(text):
    """Make a string safe for a file name."""
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def save_run(fold_scores, model, task, author, split, use_subbasin, drop_elevation,
             n_features, notes=""):
    """Save one model's per-fold scores (folds 0-3) to results/model_runs/.

    fold_scores needs a "fold" column plus the metric columns for its task:
    regression -> rmse_log, r2 ; classification -> auc, brier_skill.
    Extra columns (e.g. the chosen alpha) are kept. Re-running overwrites the same file.
    """
    if task not in METRICS:
        raise ValueError(f"task must be one of {list(METRICS)}")
    missing = [c for c in ["fold"] + METRICS[task] if c not in fold_scores.columns]
    if missing:
        raise KeyError(f"fold_scores is missing columns: {missing}")

    run = fold_scores.copy()
    info = {"task": task, "model": model, "author": author, "split": split,
            "use_subbasin": bool(use_subbasin), "drop_elevation": bool(drop_elevation),
            "n_features": n_features, "notes": notes,
            "saved": datetime.now().strftime("%Y-%m-%d %H:%M")}
    for i, (col, value) in enumerate(info.items()):
        run.insert(i, col, value)

    name = "__".join(_slug(v) for v in [author, model, task, split,
                                         f"subbasin{int(use_subbasin)}",
                                         f"noelev{int(drop_elevation)}"])
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.csv"
    run.to_csv(path, index=False)
    print(f"Saved {len(run)} folds to {path.relative_to(ROOT)}")
    return path


def load_runs():
    """Combine every saved run into one table (one row per model, configuration and fold)."""
    files = sorted(RUNS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No runs saved yet in {RUNS_DIR.relative_to(ROOT)}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# Columns that are not hyperparameters: configuration keys, scores and bookkeeping
SCORES = ["rmse_log", "r2", "auc", "brier", "brier_skill"]
NOT_PARAMS = set(KEYS) | set(SCORES) | {"fold", "n_features", "notes", "saved"}


def _format_value(value):
    """15.0 -> '15', 31.622776601683796 -> '31.62', missing -> 'None' (the search chose None)."""
    if pd.isna(value):
        return "None"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _sort_key(text):
    """Order numbers numerically ('5' before '20'), then words such as 'None' or 'sqrt'."""
    try:
        return (0, float(text), "")
    except ValueError:
        return (1, 0.0, text)


def _most_common(values):
    """Most common value across folds, with how many folds chose it, e.g. '8 (3/4)' or '6 / 8 (2/4 each)'."""
    counts = values.map(_format_value).value_counts()
    top = counts[counts == counts.max()]
    each = " each" if len(top) > 1 else ""
    return f"{' / '.join(sorted(top.index, key=_sort_key))} ({top.iloc[0]}/{len(values)}{each})"


def summarize(runs, task=None):
    """Mean and standard deviation of the scores across folds for each model and configuration,
    plus the most commonly chosen value of each tuned hyperparameter."""
    if task is not None:
        runs = runs[runs["task"] == task]
    metrics = [m for t in METRICS.values() for m in t if m in runs.columns]
    table = runs.groupby(KEYS)[metrics].agg(["mean", "std"]).round(3)
    table = table.dropna(axis=1, how="all")

    # A hyperparameter belongs to a model if that model saved a value for it in any run;
    # within such a model, an empty value means the search chose None (e.g. max_depth=None)
    params = [c for c in runs.columns if c not in NOT_PARAMS]
    chosen = {}
    for param in params:
        models = runs.loc[runs[param].notna(), "model"].unique()
        rows = runs[runs["model"].isin(models)]
        if len(rows):
            chosen[(param, "most common")] = rows.groupby(KEYS)[param].apply(_most_common)
    if chosen:
        table = table.join(pd.DataFrame(chosen))
    return table
