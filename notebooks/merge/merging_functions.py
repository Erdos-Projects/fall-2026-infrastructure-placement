import pandas as pd
import geopandas as gpd

def merge_csv_with_polygons(
    csv_path,
    polygon_path,
    output_path,
    lat_col="latitude",
    lon_col="longitude"
):
    # Read original CSV
    df = pd.read_csv(csv_path)

    # Read polygon geometries and attributes
    polygons = gpd.read_file(polygon_path)

    if polygons.crs is None:
        raise ValueError("Polygon file has no CRS defined.")

    # Create point geometries from CSV coordinates
    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(
            df[lon_col],
            df[lat_col]
        ),
        crs="EPSG:4326"
    )

    # Transform points to the polygon CRS
    points = points.to_crs(polygons.crs)

    # Avoid duplicate column names
    polygon_attributes = polygons.drop(columns="geometry").copy()
    overlapping = set(df.columns) & set(polygon_attributes.columns)

    polygon_attributes = polygon_attributes.rename(
        columns={col: f"polygon_{col}" for col in overlapping}
    )

    polygons = polygons[["geometry"]].join(polygon_attributes)

    # Match each point to its containing polygon
    merged = gpd.sjoin(
        points,
        polygons,
        how="left",
        predicate="within"
    )

    # Drop geometry and join bookkeeping for CSV output
    merged = merged.drop(
        columns=["geometry", "index_right"],
        errors="ignore"
    )

    merged.to_csv(output_path, index=False)

    return merged
