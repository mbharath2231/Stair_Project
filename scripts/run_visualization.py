#!/usr/bin/env python3
"""
CLI Runner for Interactive Folium Maps and Local GeoTIFF Visualization
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import config
from stair.core import init_earth_engine, load_aoi, get_landsat_collection, get_modis_collection, calculate_ndvi
from stair.fusion import learn_spatial_rules, synthesize_missing_day
from stair.visualization import (
    create_true_color_map,
    create_band_physics_map,
    create_cloud_comparison_map,
    plot_local_geotiff,
)


def run_cloudy_true_color(aoi, gdf):
    print("\nGenerating Cloudy Day True Color Map...")
    cloudy_col = (
        ee.ImageCollection(config.LANDSAT8_COLLECTION)
        .filterBounds(aoi)
        .filterDate("2021-05-01", "2021-07-01")
        .filter(ee.Filter.gt("CLOUD_COVER", 60))
    )
    cloudy_img = cloudy_col.first()
    cloudy_date = ee.Date(cloudy_img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    print(f"Found cloudy Landsat image on: {cloudy_date}")

    out_path = config.MAPS_OUTPUT_DIR / "cloudy_day_map.html"
    create_true_color_map(cloudy_img, gdf, date_label=cloudy_date, output_path=out_path)
    print(f"Saved: {out_path}")


def run_band_physics(aoi, gdf, golden_date=config.DEFAULT_REFERENCE_DATE):
    print(f"\nGenerating Band Physics Map (Red Absorption vs NIR Reflection) for {golden_date}...")
    start = ee.Date(golden_date)
    landsat_img = get_landsat_collection(aoi, start, start.advance(1, "day"), apply_mask=False).first()

    out_path = config.MAPS_OUTPUT_DIR / "band_physics_map.html"
    create_band_physics_map(landsat_img, gdf, output_path=out_path)
    print(f"Saved: {out_path}")


def run_cloud_comparison(aoi, gdf, golden_date=config.DEFAULT_REFERENCE_DATE):
    print("\nGenerating Cloud Restoration Comparison Map (Real vs STAIR Synthetic NDVI)...")
    # 1. Spatial rules from golden day
    g_start = ee.Date(golden_date)
    golden_landsat = get_landsat_collection(aoi, g_start, g_start.advance(1, "day"), apply_mask=False).first()
    golden_modis = get_modis_collection(aoi, g_start, g_start.advance(1, "day")).first()
    rules = learn_spatial_rules(golden_landsat, golden_modis)

    # 2. Find a cloudy day
    cloudy_col = (
        ee.ImageCollection(config.LANDSAT8_COLLECTION)
        .filterBounds(aoi)
        .filterDate("2021-07-01", "2021-08-30")
        .filter(ee.Filter.gt("CLOUD_COVER", 40))
    )
    cloudy_landsat = cloudy_col.first()
    cloudy_date = ee.Date(cloudy_landsat.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    print(f"Testing cloud restoration on corrupted scene: {cloudy_date}")

    real_red = cloudy_landsat.select(config.LANDSAT_BANDS["red"])
    real_nir = cloudy_landsat.select(config.LANDSAT_BANDS["nir"])
    real_ndvi = calculate_ndvi(real_nir, real_red, name="real_ndvi").clip(aoi)

    # 3. Synthesize STAIR prediction
    c_start = ee.Date(cloudy_date)
    cloudy_modis = get_modis_collection(aoi, c_start, c_start.advance(1, "day")).first()
    synth_img = synthesize_missing_day(cloudy_modis, rules, aoi)
    synth_ndvi = synth_img.select("synthetic_ndvi")

    out_path = config.MAPS_OUTPUT_DIR / "cloud_validation_map.html"
    create_cloud_comparison_map(real_ndvi, synth_ndvi, gdf, date_str=cloudy_date, output_path=out_path)
    print(f"Saved: {out_path}")


def run_tiff_plot(tif_path=None):
    print("\nPlotting Local GeoTIFF...")
    target_tif = Path(tif_path) if tif_path else (config.DATA_DIR / "synthetic_ndvi_20210622.tif")
    if not target_tif.exists():
        print(f"GeoTIFF file not found at: {target_tif}. Skipping local plot.")
        return

    out_png = config.OUTPUT_DIR / f"{target_tif.stem}_plot.png"
    plot_local_geotiff(target_tif, output_image_path=out_png, title="STAIR Synthetic NDVI Matrix", show_plot=False)


def main():
    parser = argparse.ArgumentParser(description="STAIR Interactive Visualization and Plotting")
    parser.add_argument(
        "--mode",
        choices=["all", "folium_cloud", "folium_bands", "folium_cloudy_true_color", "plot_tiff"],
        default="plot_tiff",
        help="Visualization mode"
    )
    parser.add_argument("--tif-path", help="Path to local GeoTIFF to plot")
    parser.add_argument("--golden-date", default=config.DEFAULT_REFERENCE_DATE, help="Golden reference date")
    parser.add_argument("--project-id", default=config.EE_PROJECT_ID, help="EE project ID")
    args = parser.parse_args()

    if args.mode in ["all", "folium_cloud", "folium_bands", "folium_cloudy_true_color"]:
        init_earth_engine(project_id=args.project_id)
        aoi, gdf = load_aoi()

        if args.mode in ["all", "folium_cloudy_true_color"]:
            run_cloudy_true_color(aoi, gdf)
        if args.mode in ["all", "folium_bands"]:
            run_band_physics(aoi, gdf, golden_date=args.golden_date)
        if args.mode in ["all", "folium_cloud"]:
            run_cloud_comparison(aoi, gdf, golden_date=args.golden_date)

    if args.mode in ["all", "plot_tiff"]:
        run_tiff_plot(args.tif_path)

    print("\nVisualization generation complete!")


if __name__ == "__main__":
    main()
