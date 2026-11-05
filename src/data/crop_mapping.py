"""Download the DWR statewide crop-mapping geodatabases used by crop_mapping_annual_2021_2024.ipynb.

Source: https://data.cnra.ca.gov/dataset/statewide-crop-mapping
Each year is saved as one unzipped .gdb folder in data/crops/ (paths.RAW_CROPS), with the year in
its folder name, which is what find_gdb() in the notebook looks for. A geodatabase already there for
the same year is replaced, after the new download has unzipped cleanly.

Run from therepository root:
    python -m src.data.crop_mapping                     # 2020-2024 (2020 is only used for the 2021 changes)
    python -m src.data.crop_mapping --years 2024        # one year
    python -m src.data.crop_mapping --keep-existing     # only download the years that are missing
"""
import argparse
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from src.paths import ROOT, RAW_CROPS


DOWNLOAD = "https://data.cnra.ca.gov/dataset/6c3d65e3-35bb-49e1-a51e-49d5a2cf09a9/resource"

# Geodatabase zips listed on the CNRA page: year -> (resource path, size in bytes)
GDB_ZIPS = {
    2020: ("44c1bde8-7ac4-4582-b8de-d09264e180fb/download/i15_crop_mapping_2020-gdb.zip", 88_308_707),
    2021: ("cd1ce211-ac75-44b4-9ea4-345ce2fd0548/download/i15_crop_mapping_2021_gdb.zip", 94_630_663),
    2022: ("e41f74d2-7ff9-4871-bc95-1e9673fc53cb/download/i15_crop_mapping_2022.gdb.zip", 98_690_638),
    2023: ("4e17ca38-268e-4bf5-bbc5-09636d44ed60/download/i15_crop_mapping_2023_final.gdb.zip", 100_756_773),
    2024: ("157956a5-507e-4c6c-b161-b21361488576/download/i15_crop_mapping_2024_provisional_20251208.gdb.zip", 242_354_228),
}


def gdb_folders(folder):
    """Every geodatabase folder under `folder` (a .gdb folder always contains a00000001.gdbtable)."""
    return [p.parent for p in folder.rglob("a00000001.gdbtable")]


def existing_gdb(dest, year):
    """The geodatabase already saved for this year, found the same way the notebook's find_gdb() does."""
    matches = [p for p in gdb_folders(dest) if str(year) in p.name.lower()]
    if len(matches) > 1:
        raise RuntimeError(f"More than one geodatabase for {year} in {dest}: {matches}")
    return matches[0] if matches else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def signed_url(url):
    """The CNRA link redirects to a signed S3 URL written as s3.amazonaws.com:443. Python would send the
    ":443" in the Host header, which breaks the signature (HTTP 403), so drop the default port."""
    try:
        urllib.request.build_opener(_NoRedirect).open(url)
    except urllib.error.HTTPError as error:
        if error.code in (301, 302, 303, 307, 308):
            return error.headers["Location"].replace(".amazonaws.com:443/", ".amazonaws.com/", 1)
        raise
    return url


def download(url, target, expected_size):
    """Download to a .part file first, so an interrupted download is never mistaken for a full one."""
    part = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(signed_url(url)) as response, open(part, "wb") as out:
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:6.0f} / {expected_size / 1e6:.0f} MB", end="", flush=True)
    print()
    if part.stat().st_size != expected_size:
        raise RuntimeError(f"{target.name}: got {part.stat().st_size:,} bytes, expected {expected_size:,}")
    part.rename(target)


def get_year(year, dest, keep_existing=False):
    found = existing_gdb(dest, year)
    if found is not None and keep_existing:
        print(f"{year}: already present ({found.name}), skipping")
        return

    resource, size = GDB_ZIPS[year]
    print(f"{year}: downloading {resource.rsplit('/', 1)[-1]}")
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "download.zip"
        download(f"{DOWNLOAD}/{resource}", zip_path, size)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp / "unzipped")

        gdbs = gdb_folders(tmp / "unzipped")
        if len(gdbs) != 1:
            raise RuntimeError(f"{year}: expected one geodatabase in the zip, found {gdbs}")
        # Keep DWR's folder name when it already contains the year; otherwise add the year
        name = gdbs[0].name if str(year) in gdbs[0].name else f"i15_Crop_Mapping_{year}.gdb"
        # Replace the existing copy only now, after the new one has downloaded and unzipped cleanly
        if found is not None:
            shutil.rmtree(found)
            print(f"{year}: replaced {found.name}")
        shutil.move(gdbs[0], dest / name)
    print(f"{year}: saved {(dest / name).relative_to(ROOT) if (dest / name).is_relative_to(ROOT) else dest / name}")


def main():
    parser = argparse.ArgumentParser(description="Download DWR statewide crop-mapping geodatabases.")
    parser.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024],
                        choices=sorted(GDB_ZIPS))
    parser.add_argument("--keep-existing", action="store_true",
                        help="skip years already in data/crops/ instead of replacing them")
    args = parser.parse_args()

    RAW_CROPS.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        get_year(year, RAW_CROPS, args.keep_existing)

    # Check the folder the way the notebook will read it: exactly one geodatabase per year
    for year in args.years:
        if existing_gdb(RAW_CROPS, year) is None:
            raise RuntimeError(f"No geodatabase for {year} in {RAW_CROPS.relative_to(ROOT)}")
    print("All requested years are in place.")



if __name__ == "__main__":
    main()
