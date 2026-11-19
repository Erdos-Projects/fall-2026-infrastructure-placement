#Notes: add ipyleaflet and ipywidgets to environment.yml

def create_interactive_risk_map():
        
    import geopandas as gpd
    import json
    import os

    import sys
    from pathlib import Path
    from ipyleaflet import (
        Map,
        GeoJSON,
        LayersControl,
        basemaps,
        FullScreenControl,
        ScaleControl,
        TileLayer,
        WidgetControl
    )
    from ipywidgets import HTML

    # Make src importable from any notebook folder (identical in every notebook)
    sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())))
    from src import paths

    ROOT = paths.ROOT

    CELL_SIZE_M = 500
    RUN = f"{CELL_SIZE_M}m"

    LOCAL_OUTPUT_DIR = paths.DATA / "outputs"

    # Load saved grid polygons
    polygons = gpd.read_file(
        LOCAL_OUTPUT_DIR / f"risk_map_{RUN}.gpkg"
    )

    from src.features.transformers import SUBBASIN_PATH
    # Load subbasins
    subbasins = gpd.read_file(SUBBASIN_PATH)

    # Convert to geographic coordinates
    polygons = polygons.to_crs("EPSG:4326")
    subbasins = subbasins.to_crs("EPSG:4326")

    BG, FG = "#0b1a4a", "white"

    RISK_COLORS = {
        "Low (<25%)": "#3fb37f",
        "Moderate (25-50%)": "#f2a541",
        "High (>50%)": "#e4572e",
    }


    def make_risk_layer(gdf, column, name):

        data = gdf.dropna(subset=[column]).copy()

        geojson = json.loads(data.to_json())

        def style(feature):
            risk = feature["properties"][column]

            return {
                "fillColor": RISK_COLORS.get(risk, "#808080"),
                "color": "#8fa3d9",
                "weight": 0.3,
                "fillOpacity": 0.6,
            }

        return GeoJSON(
            data=geojson,
            name=name,
            style_callback=style,
            hover_style={
                "color": "white",
                "weight": 1.5,
                "fillOpacity": 1,
            },
        )


    model_layer = make_risk_layer(
        polygons,
        "model_risk_level",
        "Random Forest Model",
    )

    hybrid_layer = make_risk_layer(
        polygons,
        "hybrid_risk_level",
        "Hybrid Model",
    )

    center = subbasins.geometry.union_all().centroid

    dark_basemap = TileLayer(
        url=(
            "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_5but_1_0272ad50eddb1b49b81f4f24"
        ),
        attribution=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">'
            'OpenStreetMap</a> contributors '
            '&copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
        max_zoom=20,
    )

    m = Map(
        center=(center.y, center.x),
        zoom=10,
        basemap=dark_basemap,
        scroll_wheel_zoom=True,
    )

    boundary_layer = GeoJSON(
        data=json.loads(subbasins.to_json()),
        name="Subbasin Boundaries",
        style={
            "color": "#8fa3d9",
            "weight": 1,
            "fillOpacity": 0,
        },
    )

    legend_html = """
    <div style="
        background-color: #0b1a4a;
        color: white;
        padding: 12px;
        border-radius: 6px;
        font-size: 13px;
        min-width: 150px;
    ">
        <b style="font-size: 14px;">Nitrate Groundwater Contamination Risk</b>
        <hr style="margin: 6px 0; border-color: white;">

        <div style="display: flex; align-items: center; margin: 5px 0;">
            <span style="background: #3fb37f; width: 15px; height: 15px;
                        display: inline-block; margin-right: 8px;"></span>
            Low Risk (0-25% likelihood of exceedance)
        </div>

        <div style="display: flex; align-items: center; margin: 5px 0;">
            <span style="background: #f2a541; width: 15px; height: 15px;
                        display: inline-block; margin-right: 8px;"></span>
            Moderate Risk (25–50% likelihood of exceedance) 
        </div>

        <div style="display: flex; align-items: center; margin: 5px 0;">
            <span style="background: #e4572e; width: 15px; height: 15px;
                        display: inline-block; margin-right: 8px;"></span>
            High Risk (50–100% likelihood of exceedance)
        </div>
    </div>
    """

    legend = HTML(value=legend_html)
    legend_control = WidgetControl(
        widget=legend,
        position="bottomright"
    )

    m.add(legend_control)

    #m.add_layer(boundary_layer)
    m.add_layer(model_layer)
    m.add_layer(hybrid_layer)

    m.add_control(LayersControl(position="topright"))
    m.add_control(FullScreenControl())
    m.add_control(ScaleControl(position="bottomleft"))

    return m
