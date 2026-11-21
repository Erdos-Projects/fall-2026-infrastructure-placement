from pathlib import Path
import sys

import pandas as pd
import geopandas as gpd

# Make src importable from any notebook folder (identical in every notebook)
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())))
from src import paths

PROJECT_ROOT = paths.ROOT


def clip_to_watershed(SOIL_FOLDER, HUC8_GDB):
    print("Find and combine all SSURGO soil polygon shapefiles.")
    soil_shps = sorted(SOIL_FOLDER.rglob("soilmu_a_*.shp"))
    soil_layers = [gpd.read_file(path) for path in soil_shps]

    soils = gpd.GeoDataFrame(
        pd.concat(soil_layers, ignore_index=True),
        crs=soil_layers[0].crs
    )
    print(f"{len(soil_shps)} survey areas in the raw data with {len(soils):,} polygons")
    print("")

    # Load HUC8 watershed boundary
    huc8 = gpd.read_file(
        HUC8_GDB,
        layer="WBDHU8"
    )

    # Match coordinate systems
    huc8 = huc8.to_crs(soils.crs)

    print("Clip soils to the watershed:")
    soils_huc8 = gpd.clip(soils, huc8)

    print("Dataframe reduced to", soils_huc8.shape, "out of", soils.shape)

    return soils_huc8


def column_names(soils_huc8, SOIL_FOLDER):
    # Find all SSURGO component tables 
    comp_files = sorted(SOIL_FOLDER.rglob("comp.txt"))

    # Read and combine all SSURGO component tables
    component_tables = [
        pd.read_csv(path, sep="|", header=None)
        for path in comp_files
    ]

    components = pd.concat(
        component_tables,
        ignore_index=True
    )

    #display(components.head())

    # Load SSURGO metadata
    meta = pd.read_csv(
        next(SOIL_FOLDER.rglob("mstabcol.txt")),
        sep="|",
        header=None
    )

    # Get the real column names for the component table
    component_cols = (
        meta[meta[0] == "component"]
        .sort_values(1)[2]
        .tolist()
    )

    # Assign the column names
    components.columns = component_cols
    print("Column names from metadata:")
    print(components.columns)
    print("")

    print("Get only the MUKEY values that are inside the HUC8 watershed.") 
    huc8_mukeys = soils_huc8["MUKEY"].astype(str).unique()

    # Keep only components belonging to soil map units inside the HUC8
    components_huc8 = components[
        components["mukey"].astype(str).isin(huc8_mukeys)
    ]

    print("Dataframe reduced to size:") 
    print(components_huc8.shape)
    print("")

    # Find, read and combine the map-unit aggregate tables (one per survey)
    muaggatt_files = sorted(SOIL_FOLDER.rglob("muaggatt.txt"))
    muaggatt_tables = [
        pd.read_csv(path, sep="|", header=None)
        for path in muaggatt_files
    ]

    muaggatt = pd.concat(
        muaggatt_tables,
        ignore_index=True
    )

    muaggatt_cols = (
        meta[meta[0] == "muaggatt"]
        .sort_values(1)[2]
        .tolist()
    )

    muaggatt.columns = muaggatt_cols

    muaggatt_huc8 = muaggatt[
        muaggatt["mukey"].astype(str).isin(huc8_mukeys)
    ]
    print("Find, read and combine the map-unit aggregate tables, creating dataframe of size:")
    print(muaggatt_huc8.shape)
    print("")

    return muaggatt_huc8 

def soil_attributes(muaggatt_huc8):
    # Select soil variables relevant to nitrate transport and hydrology
    soil_vars = [
        "mukey",
        "slopegradwta",
        "brockdepmin",
        "wtdepannmin",
        "aws0150wta",
        "drclassdcd",
        "hydgrpdcd",
        "flodfreqdcd"
    ]

    soils_attributes = muaggatt_huc8[soil_vars]

    print("Select soil variables relevant to nitrate transport and hydrology:")
    display(soils_attributes.head())
    print("")

    # Keep soil predictors with good data coverage
    soil_vars_final = [
        "mukey",# mukey = unique ID for each SSURGO soil map unit
        "slopegradwta",# slopegradwta = weighted average slope gradient for the map unit
        "brockdepmin", # brockdepmin = minimum depth to bedrock within the map unit
        "aws0150wta", # aws0150wta = weighted average available water storage in the top 150 cm of soil
        "drclassdcd",# drclassdcd = dominant drainage class of the map unit
        "hydgrpdcd"# hydgrpdcd = dominant hydrologic soil group of the map unit
    ]
    soils_attributes = muaggatt_huc8[soil_vars_final]

    print("Keep soil predictors with good data coverage:")
    display(soils_attributes.head())
    print("")

    print("mukey = unique ID for each SSURGO soil map unit")
    print("slopegradwta = weighted average slope gradient for the map unit")
    print("brockdepmin = minimum depth to bedrock within the map unit")
    print("aws0150wta = weighted average available water storage in the top 150 cm of soil")
    print("drclassdcd = dominant drainage class of the map unit")
    print("hydgrpdcd = dominant hydrologic soil group of the map unit")
    print("")

    return soils_attributes

def join_attributes_to_polygons(soils_huc8, soils_attributes):

    # Make MUKEY the same data type in both tables
    soils_huc8["MUKEY"] = soils_huc8["MUKEY"].astype(str)
    soils_attributes["mukey"] = soils_attributes["mukey"].astype(str)

    # Join soil attributes to the HUC8 soil polygons
    soils_model = soils_huc8.merge(
        soils_attributes,
        left_on="MUKEY",
        right_on="mukey",
        how="left"
    )

    #print(soils_model.shape)

    # Remove duplicate join key
    soils_model = soils_model.drop(columns=["mukey"])

    print("Check missing values at the unique soil map-unit level.")
    print("")
    soil_units = (
        soils_model
        .drop_duplicates("MUKEY")
    )

    print("Missing values:")
    display(soil_units[
        ["slopegradwta", "brockdepmin", "aws0150wta",
        "drclassdcd", "hydgrpdcd"]
    ].isna().sum())
    print("")

    print("Check for duplicated rows and invalid geometries (both should be 0)")
    print("Duplicated rows:", soils_model.duplicated().sum())
    print("Invalid geometries:", (~soils_model.geometry.is_valid).sum())
    print("")

    return soils_model 
