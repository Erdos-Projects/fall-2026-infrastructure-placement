"""Create the shared train/test split: one fold per well, from 10 km equal-area grid cells.

Run once from the repository root:
    python -m src.evaluation.make_splits
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.evaluation import splits

ROOT = Path(__file__).resolve().parents[2]

# ---- settings (changing any of these means a NEW split file) ----
CELL_SIZE_M = 10_000
N_SPLITS = 5
SEED = 0
THRESHOLD_MG_L = 10
HOLDOUT_FOLD = N_SPLITS-1                      # never used for training or tuning

SOURCE = ROOT / "notebooks" / "groundwater" / "groundwater_nitrate_clean.csv"
SPLIT_PATH = ROOT / "results" / "splits" / f"well_{N_SPLITS}_folds_{CELL_SIZE_M // 1000}km_seed{SEED}.csv"



def build():
    nitrate = pd.read_csv(SOURCE, dtype={"gm_well_id": "string"})
    table = splits.well_fold_table(nitrate, cell_size_m=CELL_SIZE_M, n_splits=N_SPLITS, seed=SEED, threshold=THRESHOLD_MG_L)
    settings = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_rows": len(nitrate),
        "cell_size_m": CELL_SIZE_M,
        "n_splits": N_SPLITS,
        "seed": SEED,
        "threshold_mg_L": THRESHOLD_MG_L,
        "holdout_fold": HOLDOUT_FOLD,
        "created": date.today().isoformat(),
    }
    return table, settings


def main():
    if SPLIT_PATH.exists():
        raise FileExistsError(f"{SPLIT_PATH.name} already exists; the shared split must not change. Change a setting to create a new split file instead.")
    table, settings = build()
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SPLIT_PATH, index=False)
    SPLIT_PATH.with_suffix(".json").write_text(json.dumps(settings, indent=2))
    print(f"Wrote {SPLIT_PATH.relative_to(ROOT)}: {len(table)} wells")
    print(table["fold"].value_counts().sort_index().to_string())


def load_split(df, well_col="gm_well_id"):
    """Attach the shared `cell` and `fold columns. Adds `holdout`(Truefor the final test fold)."""
    out = splits.attach_folds(df, SPLIT_PATH, well_col=well_col)
    out["holdout"] = out["fold"] == HOLDOUT_FOLD
    return out


if __name__ == "__main__":
    main()
