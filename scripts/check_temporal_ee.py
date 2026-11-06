#!/usr/bin/env python3
"""Numerically compare actual EE execution with NumPy on tiny masked images."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import date, timedelta

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stair.temporal import interpolate_temporal_numpy, interpolate_temporal_ee


def check(project, output):
    import ee
    ee.Initialize(project=project)
    region = ee.Geometry.Rectangle([-.02, -.02, .02, .02], geodesic=False)
    east = ee.Image.pixelLonLat().select("longitude").gt(0)

    def image(row, day=None):
        values = np.nan_to_num(row, nan=0).tolist()
        valid = np.isfinite(row).astype(int).tolist()
        value = ee.Image.constant(values[0]).where(east, values[1]).rename("SR_B5").toDouble()
        mask = ee.Image.constant(valid[0]).where(east, valid[1])
        value = value.updateMask(mask).clip(region).setDefaultProjection("EPSG:4326", None, 100)
        return value.set("system:time_start", ee.Date(day).millis()) if day else value

    cases = [
        ("nearest_clear", [[.2, .2], [np.nan, .4], [.8, .8]], ["2022-07-01", "2022-07-05", "2022-07-11"], False),
        ("target_excluded", [[.2, .2], [99., 99.], [.8, .8]], ["2022-07-01", "2022-07-06", "2022-07-11"], False),
        ("prior_strict", [[.4, np.nan]], ["2022-07-01"], False),
        ("prior_carry", [[.4, np.nan]], ["2022-07-01"], True),
        ("future_carry", [[np.nan, .7]], ["2022-07-11"], True),
        ("empty_strict", [], [], False),
        ("empty_carry", [], [], True),
        ("stale", [[.3, .3], [.7, .7]], ["2022-05-01", "2022-09-01"], True),
    ]
    raw = np.array([np.nan, .123])
    combined, expected = [], {}
    for name, rows, days, carry in cases:
        values = np.asarray(rows, dtype=float).reshape(-1, 2)
        reference = interpolate_temporal_numpy(values, days, "2022-07-06", raw, allow_one_sided=carry)
        result = interpolate_temporal_ee(ee.ImageCollection([image(row, d) for row, d in zip(values, days)]),
                                         image(raw), "2022-07-06", allow_one_sided=carry)
        for key, actual in [("prediction", result.prediction), ("filled", result.filled),
                             ("before_age_days", result.diagnostics.select("before_age_days")),
                             ("after_age_days", result.diagnostics.select("after_age_days")),
                             ("support_count", result.diagnostics.select("support_count")),
                             ("status", result.diagnostics.select("prediction_status")),
                             ("filled_origin", result.diagnostics.select("filled_origin"))]:
            label = f"{name}_{key}"
            combined.append(actual.unmask(-9999).rename(label).toDouble())
            expected[label] = np.nan_to_num(getattr(reference, key), nan=-9999)
    points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([x, .005]), {"pixel": i})
                                    for i, x in enumerate([-.005, .005])])
    samples = ee.Image.cat(combined).sampleRegions(collection=points, properties=["pixel"], scale=100,
                                                   projection="EPSG:4326").getInfo()["features"]
    if len(samples) != 2:
        raise AssertionError("Expected both test pixels from EE")
    for feature in samples:
        props = feature["properties"]
        i = props["pixel"]
        for key, values in expected.items():
            np.testing.assert_allclose(props[key], values[i], atol=1e-10, rtol=1e-10, err_msg=f"{key}, pixel {i}")
    report = {"status": "passed", "execution": "Actual Earth Engine server evaluation vs NumPy",
              "cases": [c[0] for c in cases], "pixels": 2,
              "compared_fields_per_case": 7, "checked_values": len(expected) * 2,
              "absolute_tolerance": 1e-10, "relative_tolerance": 1e-10}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def check_real_chip(project, cache_path, output):
    """Check full-grid EE predictions, masks, ages and status against cached NumPy."""
    import ee
    import requests
    from rasterio.io import MemoryFile
    from stair.landsat import landsat_reflectance_collection
    ee.Initialize(project=project)
    with np.load(cache_path, allow_pickle=False) as cache:
        values, days = cache["observations"], cache["dates"].tolist()
        meta = json.loads(str(cache["manifest"]))
    t0, bands = meta["target_date"], meta["bands"]
    transform, width, height = meta["transform"], meta["width"], meta["height"]
    xmin, ymax, res = transform[2], transform[5], transform[0]
    region = ee.Geometry.Rectangle([xmin, ymax-height*res, xmin+width*res, ymax], proj=meta["crs"], geodesic=False)
    collection = landsat_reflectance_collection(region, meta["fetch_start"], meta["fetch_end_exclusive"], bands)
    daily = []
    for day in days:
        next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        image = collection.filterDate(day, next_day).sort("source_id").mosaic()
        daily.append(image.setDefaultProjection(ee.Projection(meta["crs"], transform))
                     .set("system:time_start", ee.Date(day).millis()))
    collection = ee.ImageCollection(daily)
    raw = daily[days.index(t0)]
    local = interpolate_temporal_numpy(values, days, t0)
    images, expected, labels = [], [], []
    for i, band in enumerate(bands):
        result = interpolate_temporal_ee(collection, raw, t0, band)
        for key, img in [("prediction", result.prediction),
                         ("before_age_days", result.diagnostics.select("before_age_days")),
                         ("after_age_days", result.diagnostics.select("after_age_days")),
                         ("support_count", result.diagnostics.select("support_count")),
                         ("status", result.diagnostics.select("prediction_status"))]:
            label = f"{band}_{key}"
            images.append(img.rename(label).unmask(-9999).toDouble())
            expected.append(np.nan_to_num(getattr(local, key)[i], nan=-9999))
            labels.append(label)
    url = ee.Image.cat(images).getDownloadURL({"crs":meta["crs"], "crs_transform":transform,
                                              "dimensions":[width,height], "format":"GEO_TIFF"})
    response = requests.get(url, timeout=(20, 120))
    response.raise_for_status()
    with MemoryFile(response.content) as memory, memory.open() as src:
        actual = src.read()
        if actual.shape != (len(expected), height, width):
            raise AssertionError("Real-chip EE parity download has wrong grid dimensions")
    results = {}
    for label, predicted, reference in zip(labels, actual, expected):
        np.testing.assert_allclose(predicted, reference, atol=1e-7, rtol=1e-7, err_msg=label)
        results[label] = {"max_absolute_difference": float(np.max(abs(predicted-reference)))}
    report = {"status":"passed", "execution":"Full real-chip EE execution vs NumPy", "target_date":t0,
              "pixels_per_field":width*height, "compared_fields":len(labels),
              "absolute_tolerance":1e-7, "relative_tolerance":1e-7, "results":results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Real-chip parity passed: {width*height} pixels x {len(labels)} fields")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.getenv("EE_PROJECT_ID", "stair-project-509517"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/temporal_validation/ee_parity.json")
    parser.add_argument("--real-chip", type=Path, help="Also compare a downloaded real observation cache")
    args = parser.parse_args()
    check(args.project, args.output)
    if args.real_chip:
        check_real_chip(args.project, args.real_chip, args.output.with_name("ee_real_chip_parity.json"))
