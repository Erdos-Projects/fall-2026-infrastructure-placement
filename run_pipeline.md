## Running the full pipeline

**1. Get the code**
```bash
git clone https://github.com/Erdos-Projects/fall-2026-infrastructure-placement.git   # skip if you already have it
cd fall-2026-infrastructure-placement
git fetch origin
git switch <branch-name>
```

**2. Create the environment** (once; takes a while)
```bash
conda env create -f environment.yml
```

**3. Activate it** (in every new terminal before running anything)
```bash
conda activate fall_2026_environmental_infrastructure
```

**4. Put the raw data in place**
Copy these files into the repo, at exactly these paths (they are not in GitHub; the README's "Data not included in the repository" section says where each comes from):
- `data/groundwaterdata.csv`
- `data/PRISM_ppt_tmean_stable_4km_201301_202603.csv`
- `data/NHD_H_18060005_HU8_GDB/NHD_H_18060005_HU8_GDB.gdb`
- `data/datasets/ssurgo_soildata/` (folders CA053, CA069, CA665, CA772, CA788)
- `data/elevation/dem_watershed_10m.tif`

The DWR crop maps (`data/crops/`) are downloaded automatically in step 0.

**5. Check the steps, then run everything** (from the repo root)
```bash
./run.sh --dry-run      # lists the steps without running anything
./run.sh                # runs the whole pipeline (refitting the models takes about 40 minutes of it)
```
You need internet access for the crop download and the first run of the merge.
If a step fails, fix the problem and restart from that step, e.g. `./run.sh --from 8`.

**6. Report back**
```bash
git status
```
Every notebook will show as modified, because their outputs are saved again. Data and result files should be unchanged, apart from timestamps in `results/model_runs/` and the files in `results/predictions/`.
Please send:
- the output of `git status`;
- any error message;
- whether the crop step printed "Existing crop CSV matches" and the merge printed "Existing output matches".

