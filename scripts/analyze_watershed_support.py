#!/usr/bin/env python3
"""
CLI Tool: Watershed-Wide Landsat Observation Statistics
======================================================
Analyzes the entire watershed to compute exactly how many clean Landsat 8
observations (pixels) were used to fit the linear regression line across the
entire landscape, including histograms, summary statistics, and before/after support.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import config
from stair.core import init_earth_engine, load_aoi, get_landsat_collection
from stair.temporal import compute_temporal_support


def analyze_watershed(target_date: str, start_date: str, end_date: str, band: str, scale: float = 30.0):
    aoi, gdf = load_aoi()

    print(f"\n{'='*75}")
    print(f"WATERSHED OBSERVATION DISTRIBUTION ANALYSIS")
    print(f"Target Date: {target_date} | Reference Window: {start_date} to {end_date}")
    print(f"Spectral Band: {band} | Resolution Scale: {scale}m")
    print(f"{'='*75}")
    print("Querying Earth Engine across ~4.2 million watershed pixels...")

    # 1. Image Collection
    col = get_landsat_collection(aoi, start_date, end_date, apply_mask=True)
    n_points_img = col.select(band).count().clip(aoi).rename("n_points")

    # 2. Histogram of total observations
    hist_raw = n_points_img.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e10
    ).getInfo().get("n_points", {})

    # 3. Min, Max, Mean stats
    stats = n_points_img.reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), "", True),
        geometry=aoi,
        scale=scale,
        maxPixels=1e10
    ).getInfo()

    # 4. Temporal Support Breakdown (Before vs After)
    support = compute_temporal_support(col, target_date, band, aoi)
    type_hist = support.select("Temporal_Type").reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e10
    ).getInfo().get("Temporal_Type", {})

    # Sort histogram by number of observations
    sorted_obs = sorted([(int(float(k)), v) for k, v in hist_raw.items()], key=lambda x: x[0])
    total_pixels = sum([v for _, v in sorted_obs])

    # Convert pixel count to square kilometers (each pixel is 30m x 30m = 900 m² = 0.0009 km²)
    pixel_area_km2 = (scale * scale) / 1e6
    total_area_km2 = total_pixels * pixel_area_km2

    print(f"\n1. WATERSHED SUMMARY STATISTICS:")
    print(f"  • Total Watershed Surface Area: {total_area_km2:,.1f} km² ({total_pixels:,.0f} pixels)")
    print(f"  • Average Observations per Pixel: {stats.get('n_points_mean', 0):.2f} passes")
    print(f"  • Minimum Observations:          {stats.get('n_points_min', 0):.0f} pass")
    print(f"  • Maximum Observations:          {stats.get('n_points_max', 0):.0f} passes (due to overlapping satellite orbits)")

    print(f"\n2. OBSERVATION COUNT DISTRIBUTION (How many pixels used X observations):")
    print(f"{'Observations (N)':<18} | {'Pixel Count':<14} | {'Area (km²)':<12} | {'Percentage':<10} | {'Distribution Bar'}")
    print("-" * 75)

    for n, count in sorted_obs:
        pct = (count / total_pixels) * 100.0 if total_pixels > 0 else 0
        area_km2 = count * pixel_area_km2
        bar_len = int(pct / 2.5)  # Max 40 chars
        bar = "█" * bar_len
        print(f"{n:<2} clear passes      | {count:>12,.0f} | {area_km2:>9.1f} km² | {pct:>7.2f}%  | {bar}")

    print("=" * 75)

    print(f"\n3. RECONSTRUCTION RELIABILITY BREAKDOWN (For Target Date {target_date}):")
    type_labels = {
        "2": "True Interpolation (Bracketed: data before AND after)",
        "1": "Forward Extrapolation (Only data before)",
        "3": "Backward Extrapolation (Only data after)",
        "0": "Insufficient Observations (< 2 points)"
    }

    type_total = sum(type_hist.values()) if type_hist else 1
    for k in ["2", "1", "3", "0"]:
        cnt = type_hist.get(k, 0)
        pct = (cnt / type_total) * 100.0
        label = type_labels.get(k, f"Class {k}")
        print(f"  • {label:<54}: {pct:5.1f}% ({cnt * pixel_area_km2:,.1f} km²)")

    print("\n" + "=" * 75)
    print("HOW TO VISUALIZE THIS IN QGIS:")
    print("  1. Open 'STAIR_Evaluation_Matrix_*.tif' or 'STAIR_Temporal_Support_*.tif'")
    print("  2. Go to Layer Properties > Symbology > Render type: 'Paletted/Unique values'")
    print("  3. Select Band: 'N_Points' (or 'N_Total') and click 'Classify'")
    print("     Every pixel will be colored by the exact number of Landsat observations used!")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Analyze watershed-wide Landsat observation count distribution")
    parser.add_argument("--target-date", default="2021-07-03", help="Target cloudy date (YYYY-MM-DD)")
    parser.add_argument("--start-date", default="2021-05-15", help="Reference window start (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2021-08-15", help="Reference window end (YYYY-MM-DD)")
    parser.add_argument("--band", default="SR_B5", help="Spectral band (default: SR_B5)")
    parser.add_argument("--scale", type=float, default=30.0, help="Analysis resolution scale (default: 30m)")
    parser.add_argument("--project-id", default=config.EE_PROJECT_ID, help="EE project ID")
    args = parser.parse_args()

    init_earth_engine(project_id=args.project_id)
    analyze_watershed(
        target_date=args.target_date,
        start_date=args.start_date,
        end_date=args.end_date,
        band=args.band,
        scale=args.scale
    )


if __name__ == "__main__":
    main()
