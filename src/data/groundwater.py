
# Imports
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Make src importable from any notebook folder (identical in every notebook)
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())))
from src import paths

ROOT = paths.ROOT



def groundwater_cleaning():
    print("Load the groundwater data.") 
    groundwater = pd.read_csv(paths.RAW_GROUNDWATER, dtype={"src_samp_collection_time": "string"}, low_memory=False)
    print("")

    print("Load Salinas Valley/HUC8 watershed boundary.")
    print("")
    huc8 = gpd.read_file(
        paths.HUC8_GDB,
        layer="WBDHU8"
    )

    #groundwater.head()

    print("Convert groundwater coordinates to geographic points using GeoPandas.")
    groundwater_gdf = gpd.GeoDataFrame(
        groundwater,
        geometry=gpd.points_from_xy(
            groundwater["gm_longitude"],
            groundwater["gm_latitude"]
        ),
        crs="EPSG:4326"
    )
    print("")

    print("Here we see the first five rows of the raw groundwater data set:")
    display(groundwater_gdf.head())
    print("")


    print("We check coordinate reference systems:")
    print("groundwater crs:", groundwater_gdf.crs)
    print("huc8 crs:", huc8.crs)
    print("")

    print("Reproject HUC8 to match groundwater CRS.")
    huc8 = huc8.to_crs(groundwater_gdf.crs)

    print("groundwater crs:", groundwater_gdf.crs)
    print("huc8 crs:", huc8.crs)
    print("")

    print("Keep only groundwater points inside the HUC8 watershed.")
    groundwater_huc8 = gpd.clip(
        groundwater_gdf,
        huc8
    )

    print("Dataframe shape after clipping:")
    print(groundwater_huc8.shape)
    print("")


    print("Inspect counties represented inside the HUC8:")
    display(groundwater_huc8["gm_county_name"].value_counts())

    groundwater_huc8 = groundwater_huc8.drop_duplicates()


    print("Flag censored measurements.")
    groundwater_huc8["is_censored"] = (
        groundwater_huc8["gm_result_modifier"] != "="
    )

    print("Number of censored measurements:")
    display(groundwater_huc8["is_censored"].value_counts())
    print("")

    groundwater_clean = groundwater_huc8[
        groundwater_huc8["is_censored"] == False
    ].copy()

    print("Dataframe shape after removing censored measurements:")
    print(groundwater_clean.shape)
    print("")

    print("Convert sample date to datetime.")
    groundwater_clean["gm_samp_collection_date"] = pd.to_datetime(
        groundwater_clean["gm_samp_collection_date"]
    )
    print("")

    # Standardize sampling time as four-character text
    groundwater_clean["src_samp_collection_time"] = (
        groundwater_clean["src_samp_collection_time"]
        .astype("string")
        .str.zfill(4)
    )

    #groundwater_clean[
    #    ["gm_samp_collection_date", "src_samp_collection_time"]
    #].head()

    print("Check available chemical measurements.")
    print(groundwater_clean["gm_chemical_name"].value_counts().head(30))
    print("")

    print("Keep only nitrate-related measurements.")
    print("")

    groundwater_nitrate = groundwater_clean[
        groundwater_clean["gm_chemical_name"].isin(
            ["Nitrate as N", "Nitrate+Nitrite"]
        )
    ].copy()

    #print(groundwater_nitrate.shape)

    # Check repeated nitrate measurements from the same sample
    duplicate_measurements = groundwater_nitrate.duplicated(
        subset=[
            "gm_well_id",
            "gm_samp_collection_date",
            "src_samp_collection_time",
            "gm_chemical_name"
        ],
        keep=False
    )


    print("Inspect repeated nitrate measurements:")
    display(groundwater_nitrate[
        duplicate_measurements
    ][
        [
            "gm_well_id",
            "gm_samp_collection_date",
            "src_samp_collection_time",
            "gm_chemical_name",
            "gm_result",
            "gm_reporting_limit"
        ]
    ].sort_values(
        [
            "gm_well_id",
            "gm_samp_collection_date",
            "src_samp_collection_time",
            "gm_chemical_name"
        ]
    ).head(5))
    print(" ")

    print("Number of duplicated measurements:")
    print(duplicate_measurements.sum())
    print(" ")

    # Average replicate nitrate measurements from the same sample
    groundwater_nitrate_clean = (
        groundwater_nitrate
        .groupby(
            [
                "gm_well_id",
                "gm_well_category",
                "gm_samp_collection_date",
                "src_samp_collection_time",
                "gm_chemical_name",
                "gm_latitude",
                "gm_longitude"
            ],
            as_index=False
        )
        .agg(
            gm_result=("gm_result", "mean"),
            gm_reporting_limit=("gm_reporting_limit", "min")
        )
    )

    print("Dataframe shape after removing duplicates.")
    print(groundwater_nitrate_clean.shape)
    print(" ")

    print("Ensure that no duplicated measurements remain.")
    print(groundwater_nitrate_clean.duplicated(
        subset=[
            "gm_well_id",
            "gm_samp_collection_date",
            "src_samp_collection_time",
            "gm_chemical_name"
        ]
    ).sum())
    print(" ")

    print("Create one nitrate concentration per groundwater sample.")
    groundwater_final = (
        groundwater_nitrate_clean
        .groupby(
            [
                "gm_well_id",
                "gm_well_category",
                "gm_samp_collection_date",
                "src_samp_collection_time",
                "gm_latitude",
                "gm_longitude"
            ],
            as_index=False
        )
        .agg(
            nitrate_mg_L=("gm_result", "mean")
        )
    )

    print("Dataframe shape after aggregating measurements:")
    print(groundwater_final.shape)
    print(" ")

    print("Perform final check of the target variable:")
    display(groundwater_final["nitrate_mg_L"].describe())
    print(" ")

    print("Check for missing values:")
    display(groundwater_final.isna().sum())

    return groundwater_final

def flag_outliers(df):
    q1 = df["nitrate_mg_L"].quantile(0.25)
    q3 = df["nitrate_mg_L"].quantile(0.75)
    iqr = q3 - q1

    #print(f"Interquartile range: {iqr:.3f}")
    lower = 0
    upper = q3 + 1.5 * iqr

    df = df.copy()
    df["is_outlier"] = (
        (df["nitrate_mg_L"] < lower) |
        (df["nitrate_mg_L"] > upper)
    )
    return df 

# Calculate statistics for each location
def summarize_location(df):
    outlier_values = df.loc[df["is_outlier"], "nitrate_mg_L"]
    normal_values = df.loc[~df["is_outlier"], "nitrate_mg_L"].dropna()

    return pd.Series({
        "outlier": outlier_values.max(),
        "mean_including_outlier": df["nitrate_mg_L"].mean(),
        "mean_excluding_outlier": normal_values.mean(),
        "num_measurements": df["nitrate_mg_L"].count(),
        "num_non_outliers": len(normal_values)
    })
    
def groundwater_outliers(groundwater_final):
    ## Globally using interquartile range method. 
    ## Anything above 75% percentile.

    outliers = flag_outliers(groundwater_final)

    print("We can assess the number of datapoints above the 75% percentile.")
    print(f"Number of outliers: {outliers['is_outlier'].sum()} out of {len(outliers)}.")

    #display(outliers.sort_values("nitrate_mg_L", ascending=False))

    # Identify outliers within spacial groupings
    outliers_by_location = outliers[outliers["is_outlier"]]

    # Get all measurements at the flagged locations
    locations = outliers.loc[
        outliers["is_outlier"],
        ["gm_latitude", "gm_longitude"]
    ].drop_duplicates()

    all_measurements = groundwater_final.merge(
        locations,
        on=["gm_latitude", "gm_longitude"],
        how="inner"
    )

    # Restore the global outlier flags by matching each measurement
    all_measurements = all_measurements.merge(
        outliers[
            ["gm_latitude", "gm_longitude",
            "gm_samp_collection_date", "nitrate_mg_L", "is_outlier"]
        ],
        on=["gm_latitude", "gm_longitude",
            "gm_samp_collection_date", "nitrate_mg_L"],
        how="left",
        suffixes=("", "_flag")
    )

    # Sort outliers first, then other measurements by descending nitrate value
    all_measurements = all_measurements.sort_values(
        ["gm_latitude", "gm_longitude", "is_outlier", "nitrate_mg_L"],
        ascending=[True, True, False, False]
    )


    comparison = (
        all_measurements.groupby(["gm_latitude", "gm_longitude"])
        .apply(summarize_location)
        .reset_index()
    )

    comparison_filtered = comparison[(comparison["num_measurements"] > 2) & (comparison["num_non_outliers"] > 2)]

    print("")
    print("We can assess how an outlier affects the measurement at a single location.")
    print("We could filter out all the points where the mean_including_outlier ")
    print("and the mean_excluding_outlier signficantly diverge given that num_measurements > 2 ")
    print("and num_non_outliers > 2. ")
    display(comparison_filtered.head())
    print("")

    print("This figure visually represents all the measurements above the 75th percentile as a function of time.")

    plt.figure(figsize=(12, 5))

    plt.scatter(
        groundwater_final["gm_samp_collection_date"],
        groundwater_final["nitrate_mg_L"],
        s=8,
        alpha=0.3,
        label="Other measurements"
    )

    plt.scatter(
        outliers["gm_samp_collection_date"].loc[outliers["is_outlier"]],
        outliers["nitrate_mg_L"].loc[outliers["is_outlier"]],
        s=25,
        color="red",
        label="Global outliers"
    )

    plt.xlabel("Date")
    plt.ylabel("Nitrate result")
    plt.title("Nitrate Measurements and Global Outliers")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print("")
    print("We decide to leave the groundwater data as is, no outliers removed based on this set of analysis.")

def create_datetime(df, date_col="Date", time_col=None):
    date_part = df[date_col].astype(str)

    if time_col is not None and time_col in df.columns:
        time_part = (
            df[time_col]
            .astype(str)
            .str.strip()
            .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
        )

        # Convert times like "930" to "09:30"
        time_part = time_part.where(
            time_part.str.contains(":", na=False),
            time_part.str.zfill(4).str[:2] + ":" + time_part.str[-2:]
        )

        # Default missing times to midnight
        time_part = time_part.fillna("00:00:00")
        

    else:
        # No time column supplied: default to midnight
        time_part = pd.Series("00:00:00", index=df.index)

    df[date_col] = pd.to_datetime(
        date_part + " " + time_part,
        errors="coerce"
    )

    return df


def temporal_analysis1(groundwater_final):

    ## creating datetime column

    groundwater = create_datetime(groundwater_final, date_col="gm_samp_collection_date", time_col="src_samp_collection_time")
    groundwater = groundwater.drop(columns=["src_samp_collection_time"])

    ## 3191 unique geographical points out of 10665
    print("We have", groundwater[["gm_latitude", "gm_longitude"]].drop_duplicates().shape[0], "number of unique geographical points out of", len(groundwater))
    print("")

    print("Now we want to count unique time datapoints for each coordinate pair.")
    temporal_counts = (
        groundwater.groupby(["gm_latitude", "gm_longitude"])["gm_samp_collection_date"]
        .nunique()
        .reset_index(name="num_timepoints")
        .sort_values("num_timepoints")
        .reset_index(drop=True)
    )

    mean_count = temporal_counts["num_timepoints"].mean()
    median_count = temporal_counts["num_timepoints"].median()
    max_count = temporal_counts["num_timepoints"].max()
    min_count = temporal_counts["num_timepoints"].min()

    print(f"Mean number of unique timepoints: {mean_count:.2f}")
    print(f"Median number of unique timepoints: {median_count:.2f}")
    print(f"Max number of unique timepoints: {max_count:.2f}")
    print(f"Min number of unique timepoints: {min_count:.2f}")
    print("")

    # Assign each coordinate pair its sorted index
    temporal_counts["pair_index"] = temporal_counts.index

    # Merge the sorted indices back into the original data
    groundwater_plot = groundwater.merge(
        temporal_counts[["gm_latitude", "gm_longitude", "pair_index"]],
        on=["gm_latitude", "gm_longitude"],
        how="left"
    )

    print("Here we plot the number of unique timepoints as a function of a geographic index.")
    plt.figure(figsize=(12, 5))
    plt.plot(temporal_counts.index, temporal_counts["num_timepoints"], ".")
    plt.xlabel("Latitude-Longitude Pair Index (sorted)")
    plt.ylabel("Number of Unique Time Datapoints")
    plt.title("Temporal Coverage Across Locations")
    plt.grid(True, alpha=0.3)
    plt.show()
    print("")


    # Calculate median and maximum result for each coordinate pair
    result_stats = (
        groundwater_plot.groupby("pair_index")["nitrate_mg_L"]
        .agg(["median", "max", "min", "mean", "std"])
    )

    # Difference between maximum and median
    result_stats["max_minus_min"] = (
        result_stats["max"] - result_stats["min"]
    )

    # Sort by increasing difference between maximum and median
    result_stats = result_stats.sort_values(
        "max_minus_min"
    ).reset_index(drop=True)

    print("Here we plot the difference between the maximum and minimum results as a function of a geographic index.")
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(
        result_stats.index,
        result_stats["max_minus_min"],
        ".",
        markersize=3
    )

    plt.xlabel("Latitude-Longitude Pair Index")
    plt.ylabel("Maximum Result − Minimum Result")
    plt.title("Difference Between Maximum and Minimum Result by Location")
    plt.grid(True, alpha=0.3)
    plt.show()

    print("There is a large variance between minimum and maximum measurements (~160 mg/L).")
    print("")

    # Total number of unique geographical points across the dataset
    total_points = groundwater[["gm_latitude", "gm_longitude"]].drop_duplicates().shape[0]

    # Extract year
    groundwater["year"] = groundwater["gm_samp_collection_date"].dt.year

    # Count unique geographical points measured per year
    yearly_coverage = (
        groundwater.groupby("year")
        .apply(
            lambda group: group[["gm_latitude", "gm_longitude"]]
            .drop_duplicates()
            .shape[0],
            include_groups=False
        )
        .reset_index(name="unique_geographical_points")
    )

    # Calculate percentage of all geographical points measured each year
    yearly_coverage["percentage_of_total"] = (
        yearly_coverage["unique_geographical_points"] / total_points * 100
    )

    #print(f"Total unique geographical points: {total_points}")
    print("Show percentage of points falling within each year of the dataset:")
    display(yearly_coverage)
    print("")

    print("Measurements within 2022-2025 have more consistently been performed.")

    return groundwater



def temporal_analysis(start_year, end_year, groundwater, groundwater_period):
    groundwater_filtered = groundwater[
        groundwater["gm_samp_collection_date"].dt.year.between(
            start_year, end_year
        )
    ].copy()

    # Count unique time datapoints for each coordinate pair
    temporal_counts = (
        groundwater_filtered
        .groupby(["gm_latitude", "gm_longitude"])["gm_samp_collection_date"]
        .nunique()
        .reset_index(name="num_timepoints")
        .sort_values("num_timepoints")
        .reset_index(drop=True)
    )

    mean_count = temporal_counts["num_timepoints"].mean()
    median_count = temporal_counts["num_timepoints"].median()
    max_count = temporal_counts["num_timepoints"].max()
    min_count = temporal_counts["num_timepoints"].min()

    print(f"Mean number of unique timepoints: {mean_count:.2f}")
    print(f"Median number of unique timepoints: {median_count:.2f}")
    print(f"Max number of unique timepoints: {max_count:.2f}")
    print(f"Min number of unique timepoints: {min_count:.2f}")

    # Assign each coordinate pair its sorted index
    temporal_counts["pair_index"] = temporal_counts.index

    ######## OUTLIER ANALYSIS ########

    # Merge sorted indices into the filtered data
    groundwater_plot = groundwater_filtered.merge(
        temporal_counts[
            ["gm_latitude", "gm_longitude", "pair_index"]
        ],
        on=["gm_latitude", "gm_longitude"],
        how="left"
    )

    # Calculate nitrate statistics for each coordinate pair
    result_stats = (
        groundwater_plot.groupby("pair_index")["nitrate_mg_L"]
        .agg(["median", "max", "min", "mean", "std"])
    )

    # Difference between maximum and minimum
    result_stats["max_minus_min"] = (
        result_stats["max"] - result_stats["min"]
    )

    threshold = 10

    # Identify coordinate pairs exceeding the variability threshold OVERALL
    high_variability = result_stats[
        result_stats["max_minus_min"] > threshold
    ]

    # Get coordinate pairs exceeding the threshold
    high_variability_locations = temporal_counts.loc[
        temporal_counts["pair_index"].isin(high_variability.index),
        ["gm_latitude", "gm_longitude"]
    ].drop_duplicates()

    # Retrieve all original groundwater columns for those locations
    selected_groundwater = groundwater.merge(
        high_variability_locations,
        on=["gm_latitude", "gm_longitude"],
        how="inner"
    )

    print(
        f"\nAll locations with overall nitrate concentration ranges "
        f"(max-min) exceeding {threshold} mg/L:"
    )

    print(
        len(selected_groundwater[
            selected_groundwater["gm_samp_collection_date"].dt.year.between(
                start_year, end_year
            )
        ])
    )

    ######## EXCLUDING DATA ########

    # Temporarily add year to the period dataframe to allow year-level grouping
    groundwater_period = groundwater_period.copy()
    groundwater_period["_year"] = groundwater_period["gm_samp_collection_date"].dt.year

    # 1. Condition A: Max-Min > threshold IN THAT SPECIFIC YEAR
    yearly_stats = groundwater_period.groupby(["gm_latitude", "gm_longitude", "_year"]).agg(
        max_nitrate=("nitrate_mg_L", "max"),
        min_nitrate=("nitrate_mg_L", "min")
    ).reset_index()
    yearly_stats["max_minus_min"] = yearly_stats["max_nitrate"] - yearly_stats["min_nitrate"]

    # 2. Condition B: Location has multiple well IDs overall
    overall_wells = groundwater_period.groupby(["gm_latitude", "gm_longitude"]).agg(
        unique_wells=("gm_well_id", "nunique")
    ).reset_index()

    # Combine both conditions
    exclusion_stats = yearly_stats.merge(overall_wells, on=["gm_latitude", "gm_longitude"])

    # Identify the exact (lat, lon, year) groups to drop
    locations_years_to_exclude = exclusion_stats[
        (exclusion_stats["max_minus_min"] > threshold) & 
        (exclusion_stats["unique_wells"] > 1)
    ][["gm_latitude", "gm_longitude", "_year"]]

    # Filter groundwater_period to exclude those groups
    groundwater_period = groundwater_period.merge(
        locations_years_to_exclude.assign(exclude=True),
        on=["gm_latitude", "gm_longitude", "_year"],
        how="left"
    )

    groundwater_period = (
        groundwater_period[groundwater_period["exclude"].isna()]
        .drop(columns=["exclude", "_year"])
        .copy()
    )

    print(f"\nLocation-years excluded: {len(locations_years_to_exclude)}")
    print(f"Remaining measurements: {len(groundwater_period)}")

    ######## PLOTTING ########

    plt.figure(figsize=(12, 5))
    plt.plot(
        temporal_counts.index,
        temporal_counts["num_timepoints"],
        "."
    )
    plt.xlabel("Latitude-Longitude Pair Index (sorted)")
    plt.ylabel("Number of Unique Time Datapoints")
    plt.title(
        f"[{start_year}-{end_year}] Temporal Coverage Across Locations"
    )
    plt.grid(True, alpha=0.3)
    plt.show()

    # Sort by increasing difference between maximum and minimum
    result_stats = result_stats.sort_values(
        "max_minus_min"
    ).reset_index(drop=True)

    plt.figure(figsize=(12, 5))
    plt.plot(
        result_stats.index,
        result_stats["max_minus_min"],
        ".",
        markersize=3
    )
    plt.xlabel("Latitude-Longitude Pair Index")
    plt.ylabel("Maximum Result − Minimum Result")
    plt.title(
        f"[{start_year}-{end_year}] Difference Between Maximum "
        "and Minimum Result by Location"
    )
    plt.grid(True, alpha=0.3)
    plt.show()

    return groundwater_period


def temporal_analysis2(groundwater):
    # Keep data from 2022–2025
    # this dataframe is important, because in the next few plots I will be excluding outliers from it
    years = [2022, 2023, 2024, 2025]
    groundwater_period = groundwater[
        groundwater["gm_samp_collection_date"].dt.year.isin(years)
    ].copy()
    start_year = 2022
    end_year = 2022
    print("Year =", start_year)
    groundwater_period = temporal_analysis(start_year, end_year, groundwater, groundwater_period)
    print("")

    start_year = 2023
    end_year = 2023
    print("Year =", start_year)
    groundwater_period = temporal_analysis(start_year, end_year, groundwater, groundwater_period)
    print("")

    start_year = 2024
    end_year = 2024
    print("Year =", start_year)
    groundwater_period = temporal_analysis(start_year, end_year, groundwater, groundwater_period)
    print("")

    start_year = 2025
    end_year = 2025
    print("Year =", start_year)
    groundwater_period = temporal_analysis(start_year, end_year, groundwater, groundwater_period)
    print("")

    ## original full dataset restricted by year

    groundwater_check = groundwater[
        groundwater["gm_samp_collection_date"].dt.year.isin(years)]

    print("Display the excluded outlier datapoints:")
    ## compare to groundwater_period
    comparison = groundwater_check.merge(
        groundwater_period.drop_duplicates(),
        how="outer",
        indicator=True
    )

    display(comparison[comparison["_merge"] == "left_only"].drop(columns="_merge").sort_values(by=["year", "gm_latitude", "gm_longitude"]))


    return groundwater_period


def change_over_time(groundwater_period):
    print("We want to see how the nitrate concentration changes over time.")
    print("")
    years = [2022, 2023, 2024, 2025]
    location_cols = ["gm_latitude", "gm_longitude"]


    groundwater_period["year"] = (
        groundwater_period["gm_samp_collection_date"].dt.year
    )

    # Find locations with at least one measurement in every year
    years_per_location = (
        groundwater_period.groupby(location_cols)["year"]
        .nunique()
    )

    complete_locations = years_per_location[
        years_per_location == len(years)
    ].index

    print(f"Number of wells present in all years 2022–2025: {len(complete_locations)}")
    print("")
    print("In the following analysis, keep only those wells.")
    print("")

    complete_data = groundwater_period.set_index(location_cols).loc[
        groundwater_period.set_index(location_cols).index.isin(complete_locations)
    ].reset_index()

    # Calculate max - min across all measurements from 2022–2025
    location_stats = (
        complete_data.groupby(location_cols)["nitrate_mg_L"]
        .agg(["mean", "min", "max", "median"])
    )

    annual_means = (
        complete_data.groupby(location_cols + ["year"])["nitrate_mg_L"]
        .mean()
        .unstack("year")
        .reindex(columns=years)
    )

    print("See how the mean nitrate concentration changes year by year:")
    display(annual_means.sort_values(by=[2022, 'gm_latitude', "gm_longitude"]))
    print("")

    location_stats["max_minus_min"] = (
        location_stats["max"] - location_stats["min"]
    )

    # Sort by increasing max - min and assign geographic index
    location_stats = location_stats.sort_values(
        "max_minus_min"
    ).reset_index(drop=True)

    print("We plot the range of nitrate concentrate over the time span per location.")
    plt.figure(figsize=(12, 5))
    plt.plot(
        location_stats.index,
        location_stats["max_minus_min"],
        ".",
        markersize=3
    )

    plt.xlabel("Geographic Location Index (sorted by max − min)")
    plt.ylabel("Nitrate Range (mg/L)")
    plt.title("Nitrate Measurement Range Across 2022–2025")
    plt.grid(True, alpha=0.3)
    plt.show()
    print("")
    print("Again, some locations have a wide variation in measurement.")

    #print("We plot the min, median and max nitrate per location.")

    #plt.figure(figsize=(12, 6))

    #plt.plot(location_stats.index, location_stats["min"], label="Minimum")
    #plt.plot(location_stats.index, location_stats["median"], label="Median")
    #plt.plot(location_stats.index, location_stats["max"], label="Maximum")

    #plt.xlabel("Geographic Location Index (sorted by max − min)")
    #plt.ylabel("Nitrate Concentration (mg/L)")
    #plt.title("Minimum, Median, and Maximum Nitrate by Location (2022–2025)")
    #plt.legend()
    #plt.grid(True, alpha=0.3)
    #plt.tight_layout()
    #plt.show()