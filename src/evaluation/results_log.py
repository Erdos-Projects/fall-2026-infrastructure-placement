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

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "results" / "model_runs"

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


def summarize(runs, task=None):
    """Mean and standard deviation across folds for each model and configuration."""
    if task is not None:
        runs = runs[runs["task"] == task]
    metrics = [m for t in METRICS.values() for m in t if m in runs.columns]
    table = runs.groupby(KEYS)[metrics].agg(["mean", "std"]).round(3)
    return table.dropna(axis=1, how="all")
