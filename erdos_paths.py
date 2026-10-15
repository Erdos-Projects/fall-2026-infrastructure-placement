"""Portable external data paths; no scientific transformations."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_env_file = PROJECT_ROOT / ".env"
_settings = {}
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            _settings[key.strip()] = value.strip().strip("\"'")
_value = os.environ.get("ERDOS_DATA_DIR") or _settings.get("ERDOS_DATA_DIR")
if not _value:
    raise RuntimeError("Copy .env.example to .env and set ERDOS_DATA_DIR to your local project data folder.")
DATA_ROOT = Path(_value).expanduser()
if not DATA_ROOT.is_absolute() or not DATA_ROOT.is_dir():
    raise RuntimeError("ERDOS_DATA_DIR must name an existing absolute directory.")
RAW = DATA_ROOT / "raw"
PROCESSED = DATA_ROOT / "processed"
OUTPUTS = DATA_ROOT / "outputs"
SHARED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
# This explicit allowlist stays empty until provenance and sharing are reviewed.
APPROVED_SHARED_TABLES = set()

def table_path(subject, filename):
    if filename in APPROVED_SHARED_TABLES and (SHARED / filename).is_file():
        return SHARED / filename
    return PROCESSED / subject / filename

def output_path(path):
    """Create output parents; require explicit consent before replacing results."""
    path = Path(path).resolve()
    roots = [PROCESSED.resolve(), OUTPUTS.resolve(), SHARED.resolve(), RESULTS.resolve()]
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError(f"Output must be under processed, outputs, shared data, or results: {path}")
    if path.exists() and os.environ.get("ERDOS_OVERWRITE_OUTPUTS") != "1":
        answer = input(f"Replace existing output {path.name}? Type yes to continue: ")
        if answer.strip().lower() != "yes":
            raise FileExistsError(f"Existing output preserved: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

# Only Andrea's notebooks use these subject-folder helpers.
# Legacy helpers above remain unchanged, including for INMP.
SUBJECT_FOLDERS = {'groundwater': 'groundwater', 'soils': 'soils',
                   'climate': 'precipitation', 'elevation': 'elevation'}

def notebook_table(subject, filename):
    return PROJECT_ROOT / 'notebooks' / SUBJECT_FOLDERS[subject] / filename

def notebook_output(path):
    path = Path(path).resolve()
    allowed = [(PROJECT_ROOT / 'notebooks' / folder).resolve()
               for folder in SUBJECT_FOLDERS.values()]
    if not any(path.is_relative_to(root) for root in allowed):
        raise ValueError(f'Output must stay in an approved notebook subject folder: {path}')
    if path.exists() and os.environ.get('ERDOS_OVERWRITE_OUTPUTS') != '1':
        answer = input(f'Replace existing output {path.name}? Type yes to continue: ')
        if answer.strip().lower() != 'yes':
            raise FileExistsError(f'Existing output preserved: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
