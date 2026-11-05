#!/usr/bin/env bash
# Regenerate the project from the raw data: data cleaning -> merge -> features -> splits -> models -> final results.
#
# Run with the project's conda environment active (from any folder):
#     ./run.sh                 # every step
#     ./run.sh --from 8        # start at step 8 (e.g. after a failure, or when the data steps are up to date)
#     ./run.sh --eda           # also run the exploratory notebooks at the end
#     ./run.sh --dry-run       # list the steps without running them
#
# The raw downloads must already be in data/ (see "Data not included in the repository" in the README).
# Each notebook runs from its own folder, and its outputs are saved back into it. The script stops at the first error; the failing notebook keeps the error message in its saved output.
#
# Settings inside the notebooks decide how much work some steps do:
#   regression_models / classification_models: RUN_MODELS = True refits every model (about 20 minutes each); False loads the saved predictions.
#   final_results: RUN_FINAL_EVALUATION = True fits the final models and scores the hold-out fold.

set -euo pipefail
cd "$(dirname "$0")"

FROM=0
EDA=""
DRY_RUN=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --eda) EDA=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1 (see ./run.sh --help)" >&2; exit 1 ;;
    esac
done

STEP=0          # number of the current step; steps before --from are skipped

step() {
    STEP=$1
    if [ "$STEP" -ge "$FROM" ]; then
        echo "== $STEP. $2"
    fi
}

run() {
    if [ "$STEP" -lt "$FROM" ]; then
        return
    fi
    echo "   $*"
    if [ -z "$DRY_RUN" ]; then
        "$@"
    fi
}

notebook() {
    [ -f "$1" ] || { echo "Notebook not found: $1" >&2; exit 1; }
        run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$1"
}

START=$(date +%s)

step 0 "Download the DWR crop maps (only the missing years)"
run python -m src.data.crop_mapping --keep-existing

step 1 "Groundwater nitrate and well buffers"
notebook notebooks/data_cleaning/groundwater/gwater_clean.ipynb

step 2 "SSURGO soil attributes"
notebook notebooks/data_cleaning/soils/Soil_data_cleanup_team.ipynb

step 3 "PRISM monthly climate"
notebook notebooks/data_cleaning/precipitation/prism_climate_cleaning_final.ipynb

step 4 "Annual crop features"
notebook notebooks/data_cleaning/crop_mapping/crop_mapping_annual_2021_2024.ipynb

step 5 "Merge into the model dataset"
notebook notebooks/data_merge/all_merge_final.ipynb

step 6 "Feature selection"
notebook notebooks/feature_selection.ipynb

step 7 "Train/test splits (existing splits are kept)"
run python -c "from src.evaluation.make_splits import SPLITS, split_config, write; [write(s) for s in SPLITS if not split_config(s)['path'].exists()]"

step 8 "Regression and classification models"
notebook notebooks/regression_models.ipynb
notebook notebooks/classification_models.ipynb

step 9 "Leakage analysis and choice of split"
notebook notebooks/data_splitting.ipynb

step 10 "Final models: hold-out evaluation and feature importance"
notebook notebooks/final_results.ipynb

step 11 "Robustness check on other hold-out folds"
notebook notebooks/robustness_subbasin_balanced_holdout.ipynb

step 12 "Risk map"
notebook notebooks/risk_map_random_forest.ipynb

if [ -n "$EDA" ]; then
    step 13 "Exploratory notebooks"
    notebook notebooks/data_cleaning/groundwater/groundwater_spatial_eda.ipynb
    notebook notebooks/data_cleaning/groundwater_depth/groundwater_depth_eda.ipynb
    notebook notebooks/data_cleaning/crop_mapping/crop_mapping_eda.ipynb
    notebook notebooks/data_cleaning/crop_mapping/crop_mapping_feature_selection.ipynb
    notebook notebooks/data_merge/all_merge_EDA_Final.ipynb
fi

if [ -z "$DRY_RUN" ]; then
    echo "Done in $(( ($(date +%s) - START) / 60 )) minutes."
fi
