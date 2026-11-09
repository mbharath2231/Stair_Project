#!/usr/bin/env python3
"""Validate five aligned Earth Engine stages locally, then optionally export to Drive."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys
import time

import ee
import numpy as np
import rasterio
from affine import Affine
from rasterio.io import MemoryFile
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stair.landsat import DEFAULT_BANDS, OFFSET, SCALE, prepare_landsat
from stair.spatial import fit_spatial_offsets, segment_reference
from stair.temporal import interpolate_temporal_ee
from stair.validation import make_holdout, score_holdout

BANDS = list(DEFAULT_BANDS)
CRS = "EPSG:32616"
TRANSFORM = [30, 0, 727110, 0, -30, 4640580]
SIZE = 167
NODATA = -9999.0
STAGES = ("01_original", "02_cloud_masked", "03_temporal", "04_global", "05_adaptive")
WINDOW = 32


def download(image):
    """Materialize the EE image on the same grid used by the Drive export."""
    url = image.toFloat().unmask(NODATA, sameFootprint=False).getDownloadURL({
        "crs": CRS, "crs_transform": TRANSFORM, "dimensions": [SIZE, SIZE], "format": "GEO_TIFF",
    })
    response = requests.get(url, timeout=(20, 180))
    if not response.ok:
        raise RuntimeError(f"Earth Engine TIFF download failed with HTTP {response.status_code}")
    with MemoryFile(response.content) as memory, memory.open() as src:
        if src.crs.to_epsg() != 32616 or src.transform != Affine(*TRANSFORM) or src.shape != (SIZE, SIZE):
            raise ValueError("Earth Engine download did not match the validation grid")
        data = src.read().astype(float)
    data[data == NODATA] = np.nan
    return data


def write_local(path, array, target, stage):
    with rasterio.open(path, "w", driver="GTiff", dtype="float32", count=len(BANDS),
                       width=SIZE, height=SIZE, crs=CRS, transform=Affine(*TRANSFORM),
                       compress="deflate", nodata=NODATA) as dst:
        dst.write(np.where(np.isfinite(array), array, NODATA).astype(np.float32))
        dst.descriptions = tuple(BANDS)
        dst.update_tags(target_date=target, stage=stage, units="surface_reflectance")


def build(args):
    target = date.fromisoformat(args.target)
    reference = date.fromisoformat(args.reference)
    if not 0 < (target - reference).days <= 21:
        raise ValueError("Choose a clear reference from the preceding 21 days")
    _, _, x, _, _, y = TRANSFORM
    region = ee.Geometry.Rectangle([x, y - 30 * SIZE, x + 30 * SIZE, y], proj=CRS, geodesic=False)
    projection = ee.Projection(CRS, TRANSFORM)
    start, end = target - timedelta(days=WINDOW), target + timedelta(days=WINDOW + 1)
    raw = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
           .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
           .filterBounds(region).filterDate(start.isoformat(), end.isoformat())
           .sort("system:time_start"))
    rows = raw.reduceColumns(ee.Reducer.toList(2), ["system:index", "system:time_start"]).get("list").getInfo()
    from datetime import datetime, timezone
    scene_dates = [datetime.fromtimestamp(timestamp / 1000, timezone.utc).date().isoformat()
                   for _, timestamp in rows]
    days = sorted(set(scene_dates))
    if args.target not in days or args.reference not in days:
        raise ValueError(f"Target or reference acquisition missing. Available dates: {days}")

    def mosaic(collection, day, sort_key):
        day_start = ee.Date(day)
        image = collection.filterDate(day_start, day_start.advance(1, "day")).sort(sort_key).mosaic()
        return image.setDefaultProjection(projection).clip(region).set("system:time_start", day_start.millis())

    prepared = raw.map(prepare_landsat)
    original = mosaic(raw, args.target, "system:index").select(BANDS).multiply(SCALE).add(OFFSET).toFloat()
    clear_target = mosaic(prepared, args.target, "source_id")
    reference_image = mosaic(prepared, args.reference, "source_id")
    references = ee.ImageCollection([mosaic(prepared, day, "source_id") for day in days if day != args.target])
    observed = clear_target
    if args.holdout_cloud:
        holdout = make_holdout((SIZE, SIZE))
        # Paint exactly the same pixel runs used by the local temporal baseline.
        strips = []
        for row, pixels in enumerate(holdout):
            edges = np.flatnonzero(np.diff(np.r_[False, pixels, False]))
            for left, right in edges.reshape(-1, 2):
                bounds = [x + 30 * int(left), y - 30 * (row + 1), x + 30 * int(right), y - 30 * row]
                strips.append(ee.Feature(ee.Geometry.Rectangle(bounds, proj=CRS, geodesic=False)))
        cloud = ee.Image.constant(0).toByte().setDefaultProjection(projection).paint(ee.FeatureCollection(strips), 1)
        observed = observed.updateMask(cloud.Not())
    temporal_prediction = ee.Image.cat([
        interpolate_temporal_ee(references, observed, args.target, band, max_distance_days=WINDOW).prediction.rename(band)
        for band in BANDS
    ]).setDefaultProjection(projection)

    calibration = download(ee.Image.cat([observed, temporal_prediction, reference_image]))
    observed_values, prediction_values, reference_values = calibration[:3], calibration[3:6], calibration[6:9]
    if np.isfinite(reference_values).all(axis=0).mean() < .95:
        raise ValueError("Choose a reference with at least 95% QA-clear coverage")
    segment_values, centers, scale, gap_scores = segment_reference(reference_values)
    print(f"Reference gap statistic selected {len(centers)} clusters: {gap_scores}", flush=True)
    distances = [reference_image.toDouble().subtract(ee.Image.constant(center.tolist()).rename(BANDS))
                 .divide(ee.Image.constant(scale.tolist()).rename(BANDS)).pow(2)
                 .reduce(ee.Reducer.sum()).multiply(-1).rename(f"class{index}")
                 for index, center in enumerate(centers)]
    segments = ee.Image.cat(distances).toArray().arrayArgmax().arrayGet([0]).rename("cluster")
    global_offsets, class_offsets, support = fit_spatial_offsets(
        observed_values, prediction_values, reference_values, segment_values,
    )
    global_delta = ee.Image.constant(global_offsets).rename(BANDS).setDefaultProjection(projection)
    labels = list(class_offsets)
    adaptive_delta = ee.Image.cat([
        segments.remap(labels, [class_offsets[label][i] for label in labels], offset)
        .unmask(offset, sameFootprint=False).rename(band)
        for i, (band, offset) in enumerate(zip(BANDS, global_offsets))
    ])
    images = [original, observed, observed.unmask(temporal_prediction),
              observed.unmask(temporal_prediction.add(global_delta)),
              observed.unmask(temporal_prediction.add(adaptive_delta))]
    # Verify the actual server-evaluated rasters before starting Drive tasks.
    results = download(ee.Image.cat(images + [segments] + ([clear_target] if args.holdout_cloud else [])))
    arrays = [results[i * 3:(i + 1) * 3] for i in range(len(STAGES))]
    np.testing.assert_array_equal(results[len(STAGES) * 3], segment_values)
    clear = np.isfinite(arrays[1])
    if not np.array_equal(arrays[1], observed_values, equal_nan=True):
        raise AssertionError("Observed output differs from the calibration image")
    expected = [prediction_values, prediction_values + np.array(global_offsets)[:, None, None]]
    adaptive_expected = expected[1].copy()
    for label, offsets in class_offsets.items():
        match = segment_values == label
        adaptive_expected[:, match] = prediction_values[:, match] + np.array(offsets)[:, None]
    expected.append(adaptive_expected)
    for stage, expected_fill in zip(arrays[2:], expected):
        np.testing.assert_array_equal(stage[clear], arrays[1][clear])
        np.testing.assert_allclose(stage, np.where(clear, arrays[1], expected_fill),
                                   atol=1e-7, rtol=1e-7, equal_nan=True)
    metadata = {
        "target": args.target, "reference": args.reference, "bands": BANDS, "crs": CRS,
        "transform": TRANSFORM, "dimensions": [SIZE, SIZE], "source_scenes": list(zip(scene_dates, [r[0] for r in rows])),
        "window_days_per_side": WINDOW, "kmeans_clusters": len(centers),
        "gap_statistic_scores": gap_scores, "kmeans_centers": centers.tolist(),
        "segmentation_seed": 42, "gap_null_draws": 20, "reference_sample_limit": 2000,
        "mask": "QA and simulated holdout" if args.holdout_cloud else "Landsat QA_PIXEL and QA_RADSAT",
        "formula": "temporal + mean(clear target) + mean(gap reference) - mean(clear reference) - mean(gap temporal)",
        "global_offsets": global_offsets, "class_offsets": class_offsets, "calibration_support": support,
        "clear_target_pixels": int(clear.all(axis=0).sum()),
        "supported_gap_pixels": int((~clear.all(axis=0) & np.isfinite(arrays[2]).all(axis=0)).sum()),
        "unresolved_pixels": int((~np.isfinite(arrays[2]).all(axis=0)).sum()),
        "observed_pixels_preserved": True,
    }
    if args.holdout_cloud:
        truth = results[len(STAGES) * 3 + 1:]
        np.testing.assert_array_equal(arrays[1], np.where(holdout[None], np.nan, truth))
        metadata["holdout_metrics"] = {
            name: score_holdout(truth, array, holdout, BANDS) for name, array in zip(STAGES[2:], arrays[2:])
        }
    if not args.holdout_cloud or args.drive_folder:
        for name, array in zip(STAGES, arrays):
            write_local(args.output / f"{name}.tif", array, args.target, name)
    return images, region, metadata


def save_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def wait_for_exports(path):
    jobs = json.loads(path.read_text(encoding="utf-8"))
    while True:
        statuses = ee.data.getTaskStatus([job["id"] for job in jobs])
        for job, status in zip(jobs, statuses):
            job.update(state=status["state"], urls=status.get("destination_uris", []),
                       error=status.get("error_message"))
        save_json(path, jobs)
        print(" | ".join(f"{job['file']} {job['state']}" for job in jobs), flush=True)
        if not any(job["state"] in {"READY", "RUNNING", "UNSUBMITTED"} for job in jobs):
            if any(job["state"] != "COMPLETED" for job in jobs):
                raise RuntimeError(f"One or more Drive exports failed. See {path}")
            return
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="2022-07-29")
    parser.add_argument("--reference", default="2022-07-22")
    parser.add_argument("--project", default=os.getenv("EE_PROJECT_ID", "stair-project-509517"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--drive-folder", help="Optional Drive destination for the five TIFFs")
    parser.add_argument("--holdout-cloud", action="store_true", help="Score an artificial gap on a clear target")
    parser.add_argument("--wait-existing", action="store_true", help="Monitor tasks recorded in the output directory")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "outputs/spatial_validation" / (args.target + ("_holdout" if args.holdout_cloud else ""))
    args.output.mkdir(parents=True, exist_ok=True)
    tasks_path = args.output / "drive_tasks.json"
    ee.Initialize(project=args.project)
    if args.wait_existing:
        wait_for_exports(tasks_path)
        return
    if args.drive_folder and tasks_path.exists():
        parser.error("This output already has Drive tasks. Use --wait-existing to check them.")
    images, region, metadata = build(args)
    save_json(args.output / "run.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    if args.drive_folder:
        jobs = []
        for name, image in zip(STAGES, images):
            prefix = f"STAIR_{args.target.replace('-', '')}_{name}"
            task = ee.batch.Export.image.toDrive(
                image=image.toFloat().unmask(NODATA, sameFootprint=False), description=prefix,
                folder=args.drive_folder, fileNamePrefix=prefix, region=region,
                crs=CRS, crsTransform=TRANSFORM, fileFormat="GeoTIFF",
                formatOptions={"noData": NODATA}, maxPixels=SIZE * SIZE * 2,
            )
            task.start()
            jobs.append({"id": task.id, "file": prefix + ".tif", "folder": args.drive_folder})
            save_json(tasks_path, jobs)
            if len(jobs) == 1:
                # Let Earth Engine create a missing folder before submitting more exports.
                wait_for_exports(tasks_path)
        wait_for_exports(tasks_path)


if __name__ == "__main__":
    main()
