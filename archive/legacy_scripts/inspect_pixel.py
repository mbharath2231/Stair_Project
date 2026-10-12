#!/usr/bin/env python3
"""
CLI Tool: Inspect Single Pixel Temporal History
Given a longitude and latitude coordinate (e.g. from QGIS), prints the exact
dates, clear vs. cloudy status, and spectral reflectance values used before
and after the cloud cover day to reconstruct that specific pixel.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import config
from stair.core import init_earth_engine, load_aoi, get_landsat_collection, mask_landsat_clouds


def inspect_pixel(lon: float, lat: float, target_date: str, start_date: str, end_date: str, band: str):
    point = ee.Geometry.Point([lon, lat])
    t0 = ee.Date(target_date)

    print(f"\n{'='*75}")
    print(f"PIXEL TEMPORAL INSPECTOR: Lon={lon:.5f}, Lat={lat:.5f}")
    print(f"Target Date to Impute: {target_date} | Band: {band}")
    print(f"Window: {start_date} to {end_date}")
    print(f"{'='*75}")

    # 1. Fetch all scenes over this point
    raw_col = (
        ee.ImageCollection(config.LANDSAT8_COLLECTION)
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .sort("system:time_start")
    )

    n_scenes = raw_col.size().getInfo()
    if n_scenes == 0:
        print("No Landsat 8 scenes found over this coordinate in the specified window.")
        return

    img_list = raw_col.toList(n_scenes)
    scenes_info = []

    for i in range(n_scenes):
        img = ee.Image(img_list.get(i))
        t_millis = img.date().millis().getInfo()
        date_str = ee.Date(t_millis).format("YYYY-MM-dd").getInfo()

        # Check cloud mask at this exact coordinate
        qa_val = img.select("QA_PIXEL").reduceRegion(ee.Reducer.first(), point, scale=30).get("QA_PIXEL").getInfo()
        if qa_val is None:
            continue

        cloud_shadow = (qa_val & (1 << 3)) != 0
        cloud = (qa_val & (1 << 4)) != 0
        is_clear = (not cloud) and (not cloud_shadow)

        # Get reflectance value
        refl_val = img.select(band).reduceRegion(ee.Reducer.first(), point, scale=30).get(band).getInfo()
        
        # Determine position relative to target date
        is_target = (date_str == target_date)
        if is_target:
            position = "TARGET DAY"
        elif t_millis < t0.millis().getInfo():
            days_diff = (t0.millis().getInfo() - t_millis) / (1000 * 60 * 60 * 24)
            position = f"{int(days_diff)}d BEFORE"
        else:
            days_diff = (t_millis - t0.millis().getInfo()) / (1000 * 60 * 60 * 24)
            position = f"{int(days_diff)}d AFTER"

        scenes_info.append({
            "date": date_str,
            "position": position,
            "is_clear": is_clear,
            "refl": refl_val,
            "is_target": is_target,
            "t_days": t_millis / (1000 * 60 * 60 * 24)
        })

    # Summary tables
    print(f"{'Date':<12} | {'Timing':<15} | {'Pixel State':<18} | {'Observed Refl':<14} | {'Role'}")
    print("-" * 75)

    clear_before = []
    clear_after = []
    target_obs = None

    for s in scenes_info:
        refl_str = f"{s['refl']:.4f}" if s['refl'] is not None else "NoData"
        if s["is_target"]:
            target_obs = s
            state_str = "CLOUDY (Gapped)" if not s["is_clear"] else "CLEAR"
            role = "Target to restore"
        elif s["is_clear"]:
            state_str = "CLEAR (Valid)"
            role = "Used in Regression"
            if "BEFORE" in s["position"]:
                clear_before.append(s)
            else:
                clear_after.append(s)
        else:
            state_str = "CLOUDY (Masked)"
            role = "Excluded from math"

        print(f"{s['date']:<12} | {s['position']:<15} | {state_str:<18} | {refl_str:<14} | {role}")

    print("=" * 75)
    print(f"\nSUMMARY FOR PIXEL ({lon:.5f}, {lat:.5f}):")
    print(f"  • Clean Observations BEFORE: {len(clear_before)} dates -> {[s['date'] for s in clear_before]}")
    print(f"  • Clean Observations AFTER:  {len(clear_after)} dates -> {[s['date'] for s in clear_after]}")
    print(f"  • Total Clean Points:        {len(clear_before) + len(clear_after)}")

    if len(clear_before) > 0 and len(clear_after) > 0:
        print("  • Reconstruction Status:     [TRUE INTERPOLATION] (Pixel is bracketed on both sides)")
    elif len(clear_before) > 0:
        print("  • Reconstruction Status:     [FORWARD EXTRAPOLATION] (Only past observations available)")
    elif len(clear_after) > 0:
        print("  • Reconstruction Status:     [BACKWARD EXTRAPOLATION] (Only future observations available)")
    else:
        print("  • Reconstruction Status:     [INSUFFICIENT DATA] (No clear observations in window)")


def main():
    parser = argparse.ArgumentParser(description="Inspect single pixel temporal observations before and after target date")
    parser.add_argument("--lon", type=float, required=True, help="Longitude (e.g. -83.95)")
    parser.add_argument("--lat", type=float, required=True, help="Latitude (e.g. 41.85)")
    parser.add_argument("--target-date", default=config.DEFAULT_TARGET_DATE, help="Target cloudy date (YYYY-MM-DD)")
    parser.add_argument("--start-date", default="2021-05-15", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2021-08-15", help="End date (YYYY-MM-DD)")
    parser.add_argument("--band", default="SR_B5", help="Band name (default: SR_B5)")
    parser.add_argument("--project-id", default=config.EE_PROJECT_ID, help="EE project ID")
    args = parser.parse_args()

    init_earth_engine(project_id=args.project_id)
    inspect_pixel(
        lon=args.lon,
        lat=args.lat,
        target_date=args.target_date,
        start_date=args.start_date,
        end_date=args.end_date,
        band=args.band
    )


if __name__ == "__main__":
    main()
