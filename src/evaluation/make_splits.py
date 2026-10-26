"""Shared train/test splits: one fold per well, from equal-area grid cells.

Each split is registered by name in SPLITS. Build a split once, from the repository root:
    python -m src.evaluation.make_splits 5km_subbasin
Load one in a notebook:
    from src.evaluation.make_splits import load_split
    data = load_split(df)                          # the original 10 km split
    data = load_split(df, split="5km_subbasin")    # 5 km cells, stratified by subbasin
"""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.evaluation import splits

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "results" / "splits"
SOURCE = ROOT / "notebooks" / "groundwater" / "groundwater_nitrate_clean.csv"
THRESHOLD_MG_L = 10

# ---- registered splits (never change an entry once its file exists; add a new one instead) ----

def _split(cell_size_m, stratify, n_splits=5, seed=0):
    """Settings of one split; the file name is derived from them, so it is always consistent."""
    return {
        "cell_size_m": cell_size_m, "stratify": stratify, "n_splits": n_splits, "seed": seed,
        "file": f"well_{n_splits}_folds_{cell_size_m}m_{stratify}_seed{seed}.csv",
    }


SPLITS = {
    "10km": _split(10_000, "exceedance"),
    "5km_subbasin": _split(5_000, "subbasin_x_exceedance"),
    "7.5km_subbasin": _split(7_500, "subbasin_x_exceedance"),
    "10km_subbasin": _split(10_000, "subbasin_x_exceedance"),
}


DEFAULT_SPLIT = "10km"


def split_config(split):
    """Settings of a registered split, with its holdout fold and file path."""
    if split not in SPLITS:
        raise KeyError(f"Unknown split {split!r}; registered splits: {list(SPLITS)}")
    config = dict(SPLITS[split])
    config["holdout_fold"] = config["n_splits"] - 1
    config["path"] = SPLIT_DIR / config["file"]
    return config


def stratify_labels(nitrate, stratify):
    """One label per sample for balancing the folds."""
    exceeds = nitrate["nitrate_mg_L"] > THRESHOLD_MG_L
    if stratify == "exceedance":
        return exceeds
    if stratify == "subbasin_x_exceedance":
        return nitrate["subbasin"].astype(str) + "_" + exceeds.astype(str)
    raise ValueError(f"Unknown stratify option {stratify!r}")


def build(split):
    """Compute the fold table and settings record for a registered split (writes nothing)."""
    config = split_config(split)
    nitrate = pd.read_csv(SOURCE, dtype={"gm_well_id": "string"})
    if "subbasin" in config["stratify"]:
        from src.features.transformers import add_subbasin, SUBBASIN_PATH
        nitrate = add_subbasin(nitrate)

    labels = stratify_labels(nitrate, config["stratify"])
    table = splits.well_fold_table(nitrate, labels, cell_size_m=config["cell_size_m"],
                                   n_splits=config["n_splits"], seed=config["seed"])

    settings = {
        "split": split, "source": str(SOURCE.relative_to(ROOT)), "source_rows": len(nitrate),
        "cell_size_m": config["cell_size_m"], "n_splits": config["n_splits"], "seed": config["seed"],
        "stratify": config["stratify"], "threshold_mg_L": THRESHOLD_MG_L,
        "holdout_fold": config["holdout_fold"], "created": date.today().isoformat(),
    }
    if "subbasin" in config["stratify"]:
        sample_folds = nitrate[["gm_well_id"]].merge(table, on="gm_well_id", how="left")["fold"]
        settings["subbasins"] = str(SUBBASIN_PATH.relative_to(ROOT))
        settings["balance"] = splits.fold_balance(sample_folds, nitrate["subbasin"].astype(str),
                                                  nitrate["nitrate_mg_L"] > THRESHOLD_MG_L)
    return table, settings


def write(split):
    """Build a registered split and save its fold table (.csv) and settings (.json)."""
    path = split_config(split)["path"]
    if path.exists():
        raise FileExistsError(f"{path.name} already exists; a shared split must not change. "
                              "Register a new split name instead.")
    table, settings = build(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(settings, indent=2))
    print(f"Wrote {path.relative_to(ROOT)}: {len(table)} wells")
    print(table["fold"].value_counts().sort_index().to_string())


def load_split(df, split=DEFAULT_SPLIT, well_col="gm_well_id"):
    """Attach `cell` and `fold` from a registered split, plus `holdout` (True for its final test fold)."""
    config = split_config(split)
    if not config["path"].exists():
        raise FileNotFoundError(f"Split {split!r} has not been built yet: "
                                f"python -m src.evaluation.make_splits {split}")
    out = splits.attach_folds(df, config["path"], well_col=well_col)
    out["holdout"] = out["fold"] == config["holdout_fold"]
    return out


def main():
    parser = argparse.ArgumentParser(description="Build a registered train/test split.")
    parser.add_argument("split", choices=list(SPLITS), help="name of the split to build")
    write(parser.parse_args().split)


if __name__ == "__main__":
    main()
