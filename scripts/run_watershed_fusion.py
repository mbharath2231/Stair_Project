#!/usr/bin/env python3
"""Generate and validate a five-day STAIR sequence over the River Raisin watershed."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import date, timedelta
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from affine import Affine
import ee
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fusion_week import ndvi
from scripts.watershed_inputs import (
    BANDS, LANDSAT_PRODUCTS, MODIS_PRODUCT, Grid, NODATA, cache_root, day_filter,
    landsat_collection, landsat_day, read_modis_inputs, read_raster, tiled_cache, write_raster,
)
from stair.spatial import fit_spatial_offsets, segment_reference
from stair.temporal import interpolate_temporal_numpy
from stair.validation import score_holdout

FINE_CRS = "EPSG:32616"
FINE_BASE = Affine(30, 0, 727110, 0, -30, 4640580)
# Earth Engine's SR-ORG:6974 alias is not in the local PROJ registry.
MODIS_SINUSOIDAL = "+proj=sinu +lon_0=0 +R=6371007.181 +units=m +no_defs +type=crs"
LANDSAT_WINDOW = 32
MODIS_WINDOW = 8
SPATIAL_WINDOW = 21
DEFAULT_ANCHORS = ("2022-07-14", "2022-07-21", "2022-07-29", "2022-07-30", "2022-08-15")


def calendar(start, end):
    return [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]


def distance(a, b):
    return (date.fromisoformat(a) - date.fromisoformat(b)).days


def nearby(mapping, day, limit):
    return {key: value for key, value in sorted(mapping.items()) if 0 < abs(distance(key, day)) <= limit}


def validity(values):
    return np.isfinite(values).all(axis=0)


def temporal_from_files(paths, day, grid, *, observed=None, ages=False):
    """Use the maintained interpolator on bounded windows of the shared grid."""
    result = np.full((len(BANDS), *grid.shape), np.nan, dtype=np.float32)
    prior = np.full(grid.shape, np.nan, dtype=np.float32) if ages else None
    later = np.full(grid.shape, np.nan, dtype=np.float32) if ages else None
    paths = nearby(paths, day, LANDSAT_WINDOW)
    if not paths:
        return (result if observed is None else observed.copy()), prior, later
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in paths.values()]
        for window in grid.windows(384):
            ys, xs = window.toslices()
            observations = np.stack([src.read(window=window) for src in sources])
            observations[observations == NODATA] = np.nan
            raw = observed[:, ys, xs] if observed is not None else None
            estimate = interpolate_temporal_numpy(observations, list(paths), day, raw,
                                                  max_distance_days=LANDSAT_WINDOW)
            result[:, ys, xs] = estimate.filled
            if ages:
                prior[ys, xs] = estimate.before_age_days[0]
                later[ys, xs] = estimate.after_age_days[0]
    return result, prior, later


def restore(observed, temporal, reference, *, segmentation=None):
    """Correct supported gaps and keep allowed observations unchanged."""
    clear = validity(observed)
    supported = ~clear & validity(temporal)
    if reference is None or min(int(clear.sum()), int(supported.sum())) < 30:
        return temporal, temporal, {"fallback": "temporal", "classes": 0}
    segments, centers, _, scores = segment_reference(reference) if segmentation is None else segmentation
    offset, class_offsets, support = fit_spatial_offsets(observed, temporal, reference, segments)
    global_result = temporal.copy()
    adaptive = temporal.copy()
    global_result[:, supported] += np.asarray(offset)[:, None]
    adaptive[:, supported] = global_result[:, supported]
    for label, correction in class_offsets.items():
        use = supported & (segments == label)
        adaptive[:, use] = temporal[:, use] + np.asarray(correction)[:, None]
    if not (np.array_equal(global_result[:, clear], observed[:, clear])
            and np.array_equal(adaptive[:, clear], observed[:, clear])):
        raise RuntimeError("Spatial correction changed QA-clear observations")
    return global_result, adaptive, {"classes": len(centers), "gap_scores": scores, "support": support,
                                     "global_offset": offset, "class_offsets": class_offsets}


def choose_reference(paths, day, coverage):
    options = [d for d in nearby(paths, day, SPATIAL_WINDOW) if coverage[d] >= 0.60]
    if not options:
        return None
    return min(options, key=lambda d: (-coverage[d], abs(distance(d, day)), d))


def resample_modis(values, native, fine, inside):
    aligned = np.full((len(BANDS), *fine.shape), np.nan, dtype=np.float32)
    reproject(values, aligned, src_transform=native.transform, src_crs=native.crs,
              src_nodata=np.nan, dst_transform=fine.transform, dst_crs=fine.crs,
              dst_nodata=np.nan, resampling=Resampling.bilinear, num_threads=2)
    aligned[:, ~inside] = np.nan
    return aligned


def add_layer(output, catalog, group, sensor, stage, date_string, grid, values, bands=BANDS):
    relative = Path(group) / f"{sensor}_{stage}.tif"
    (output / relative).parent.mkdir(parents=True, exist_ok=True)
    units = "surface_reflectance"
    if bands == ("NDVI",):
        units = "index"
    elif bands == ("prior_age", "later_age"):
        units = "days"
    elif bands == ("fusion_support",):
        units = "category"
    elif bands == ("modis_temporal_source",):
        units = "category"
    write_raster(output / relative, grid, values, bands, date=date_string, stage=stage, units=units)
    catalog.append({"group": group, "sensor": sensor, "stage": stage, "file": relative.as_posix(),
                    "bands": list(bands)})


def fuse_benchmark(truth, methods, bands, inside):
    eligible = inside & validity(truth)
    common = eligible.copy()
    for values in methods.values():
        common &= validity(values)
    result = {}
    for name, values in methods.items():
        covered = int((eligible & validity(values)).sum())
        result[name] = {
            "covered_truth_pixels": covered,
            "coverage": covered / int(eligible.sum()) if eligible.any() else None,
            "common_support": score_holdout(truth, values, common, bands),
        }
    return result


def run(args, output):
    days = calendar(args.start, args.start + timedelta(days=4))
    anchors = sorted(str(d) for d in args.anchors)
    if len(set(anchors)) != len(anchors) or any(day in days for day in anchors):
        raise ValueError("Anchor dates must be unique and outside all five prediction dates")
    if (not any(0 < distance(days[0], a) and distance(days[-1], a) <= LANDSAT_WINDOW for a in anchors)
            or not any(0 < distance(a, days[-1]) and distance(a, days[0]) <= LANDSAT_WINDOW for a in anchors)):
        raise ValueError("At least one anchor on each side must be within 32 days of the entire pilot")
    review_days = sorted(set(anchors + days))
    start = date.fromisoformat(review_days[0]) - timedelta(days=LANDSAT_WINDOW)
    end = date.fromisoformat(review_days[-1]) + timedelta(days=LANDSAT_WINDOW + 1)

    shape = gpd.read_file(args.boundary).to_crs(4326).geometry.union_all()
    boundary = gpd.GeoSeries([shape], crs=4326)
    projected = boundary.to_crs(FINE_CRS).iloc[0]
    fine = Grid.covering(FINE_CRS, FINE_BASE, projected.bounds)
    inside = fine.polygon_mask(projected)
    pixel_count = int(inside.sum())
    boundary.to_file(output / "watershed.gpkg", layer="watershed")

    ee.Initialize(project=args.project)
    collection, scenes = landsat_collection(shape.__geo_interface__, str(start), str(end))
    scene_days = sorted({day for day, _ in scenes})
    if missing := set(anchors) - set(scene_days):
        raise ValueError(f"No Landsat acquisition on anchor date(s) {sorted(missing)}")
    cache = cache_root(args.cache, shape, fine)
    paths, coverage = {}, {}
    for day in scene_days:
        if day not in review_days and not any(0 < abs(distance(day, target)) <= LANDSAT_WINDOW for target in review_days):
            continue
        path = cache / f"Landsat_QA_{day}.tif"
        print(f"Landsat {day}: {'cache' if path.exists() else 'downloading'}", flush=True)
        paths[day] = tiled_cache(path, landsat_day(collection, day, True), fine, inside)
        coverage[day] = int(validity(read_raster(path)).sum()) / pixel_count
    allowed = {day: paths[day] for day in paths if day not in days and coverage[day] > 0}
    if set(anchors) - allowed.keys():
        raise ValueError("Each anchor requires at least one QA-clear Landsat pixel")

    modis_window = args.modis_days_per_side
    modis_start = date.fromisoformat(review_days[0]) - timedelta(days=modis_window)
    modis_end = date.fromisoformat(review_days[-1]) + timedelta(days=modis_window)
    product = ee.ImageCollection(MODIS_PRODUCT).filterDate(str(modis_start), str(modis_end + timedelta(days=1)))
    available = sorted(s.replace("_", "-") for s in product.aggregate_array("system:index").getInfo())
    if not available:
        raise ValueError("No MODIS retrievals in the restoration window")
    projection = ee.Image(day_filter(product, available[0]).first()).select("Nadir_Reflectance_Band2").projection().getInfo()
    ee_crs = projection["crs"]
    native_crs = MODIS_SINUSOIDAL if ee_crs == "SR-ORG:6974" else ee_crs
    native_shape = boundary.to_crs(native_crs).iloc[0]
    native = Grid.covering(native_crs, Affine(*projection["transform"]), native_shape.bounds,
                           padding=2, ee_crs=ee_crs)
    native_inside = native.polygon_mask(native_shape)
    # Include a small border when computing/resampling so the watershed boundary does not create gaps.
    buffer = native.polygon_mask(native_shape.buffer(2 * abs(native.transform.a)))
    print("Downloading native MODIS and mandatory QA", flush=True)
    modis, modis_originals = read_modis_inputs(native, product, available, review_days, buffer)
    modis_coverage = {day: float(validity(values)[native_inside].mean()) for day, values in modis.items()}
    catalog, stage_info = [], {}
    residuals, anchor_paths = {}, {}
    empty_modis = np.full((len(BANDS), *native.shape), np.nan, dtype=np.float32)
    for day in review_days:
        print(f"Stages {day}", flush=True)
        group = f"{day}_{'Anchor' if day in anchors else 'Prediction'}"
        stage_info[day] = {"role": "anchor" if day in anchors else "prediction",
                           "landsat_acquired": day in paths,
                           "landsat_qa_coverage": coverage.get(day)}
        # Only Landsat is withheld. MODIS on any other day, including within
        # the prediction window, is available for retrospective restoration.
        modis_refs = nearby(modis, day, modis_window)
        observed_modis = modis.get(day, empty_modis)
        stack = np.stack(list(modis_refs.values())) if modis_refs else empty_modis[None][:0]
        estimate = interpolate_temporal_numpy(stack, list(modis_refs), day, observed_modis,
                                              max_distance_days=modis_window)
        temporal_modis = estimate.filled
        ref_day = choose_reference(modis, day, modis_coverage)
        global_modis, adaptive_modis, details = restore(
            np.where(native_inside, observed_modis, np.nan),
            np.where(native_inside, temporal_modis, np.nan),
            np.where(native_inside, modis[ref_day], np.nan) if ref_day else None,
        )
        for stage, values in (("01_original", modis_originals.get(day, empty_modis)), ("02_QA_masked", observed_modis),
                              ("03_temporal", temporal_modis), ("04_global", global_modis),
                              ("05_adaptive", adaptive_modis)):
            add_layer(output, catalog, group, "MODIS", stage, day, native,
                      np.where(native_inside, values, np.nan))
        coarse = resample_modis(np.where(native_inside, adaptive_modis, temporal_modis), native, fine, inside)
        add_layer(output, catalog, group, "MODIS", "06_adaptive_aligned_30m", day, fine, coarse)
        source = estimate.filled_origin[0].copy()
        extended = (estimate.before_age_days[0] > MODIS_WINDOW) | (estimate.after_age_days[0] > MODIS_WINDOW)
        source[(source == 2) & extended] = 3
        add_layer(output, catalog, group, "MODIS", "07_temporal_source", day, native,
                  np.where(native_inside, source, np.nan)[None], ("modis_temporal_source",))
        stage_info[day]["modis"] = {"reference": ref_day, "retrieval_available": day in modis,
                                    "qa_coverage": modis_coverage.get(day, 0.0),
                                    "adaptive_coverage_native": float(validity(adaptive_modis)[native_inside].mean()),
                                    "aligned_coverage": int(validity(coarse).sum()) / pixel_count,
                                    "native_cells_by_temporal_source": {
                                        code: int((native_inside & (source == code)).sum()) for code in range(4)
                                    },
                                    "correction": details}
        del coarse

        if day not in paths:
            continue
        raw_path = tiled_cache(cache / f"Landsat_original_{day}.tif", landsat_day(collection, day, False),
                               fine, inside)
        for stage, file in (("01_original", raw_path), ("02_QA_masked", paths[day])):
            add_layer(output, catalog, group, "Landsat", stage, day, fine, read_raster(file))
        observed = read_raster(paths[day])
        temporal, _, _ = temporal_from_files(allowed, day, fine, observed=observed)
        ref_day = choose_reference(allowed, day, coverage)
        global_landsat, adaptive_landsat, details = restore(
            observed, temporal, read_raster(allowed[ref_day]) if ref_day else None)
        for stage, values in (("03_temporal", temporal), ("04_global", global_landsat), ("05_adaptive", adaptive_landsat)):
            # Prediction-day Landsat stages are separate gap-fill diagnostics, never used in fusion.
            add_layer(output, catalog, group, "Landsat", stage, day, fine, values)
        stage_info[day]["landsat"] = {
            "reference": ref_day, "correction": details,
            "temporal_coverage": int(validity(temporal).sum()) / pixel_count,
            "adaptive_coverage": int(validity(adaptive_landsat).sum()) / pixel_count,
        }
        if day in anchors:
            anchor_paths[day] = output / group / "Landsat_05_adaptive.tif"
            coarse = read_raster(output / group / "MODIS_06_adaptive_aligned_30m.tif")
            residuals[day] = output / group / "STAIR_06_anchor_residual.tif"
            add_layer(output, catalog, group, "STAIR", "06_anchor_residual", day, fine, adaptive_landsat - coarse)
        del observed, temporal, global_landsat, adaptive_landsat

    for day in days:
        print(f"Fusion {day}", flush=True)
        group = f"{day}_Prediction"
        coarse = read_raster(output / group / "MODIS_06_adaptive_aligned_30m.tif")
        residual, prior, later = temporal_from_files(residuals, day, fine, ages=True)
        supported = validity(residual) & validity(coarse) & inside
        fused = residual + coarse
        fused[:, ~supported] = np.nan
        if not supported.any():
            raise RuntimeError(f"No supported fusion pixels for {day}")
        support = np.where(inside, supported.astype(float), np.nan)[None]
        for stage, values, bands in (
            ("07_fusion_30m", fused, BANDS), ("08_NDVI", ndvi(fused), ("NDVI",)),
            ("09_fusion_support", support, ("fusion_support",)),
            ("10_anchor_ages", np.where(supported, np.stack((prior, later)), np.nan), ("prior_age", "later_age")),
        ):
            add_layer(output, catalog, group, "STAIR", stage, day, fine, values, bands)
        stage_info[day]["fusion"] = {"valid_pixels": int(supported.sum()),
                                     "coverage": int(supported.sum()) / pixel_count,
                                     "unresolved_pixels": pixel_count - int(supported.sum())}
        if day in paths and coverage[day] > 0:
            truth = read_raster(paths[day])
            landsat_only, _, _ = temporal_from_files(allowed, day, fine)
            same_pair = np.zeros_like(fused)
            span = prior + later
            for anchor, path in anchor_paths.items():
                age = distance(day, anchor)
                use = supported & ((prior == age) if age > 0 else (later == -age))
                if use.any():
                    weight = 1 - abs(age) / span[use]
                    same_pair[:, use] += read_raster(path)[:, use] * weight
            same_pair[:, ~supported] = np.nan
            methods = {"STAIR": fused, "Landsat_only": landsat_only, "same_anchor_Landsat": same_pair, "MODIS_only": coarse}
            stage_info[day]["held_out"] = {
                "truth_pixels": int(validity(truth).sum()),
                "reflectance": fuse_benchmark(truth, methods, list(BANDS), inside),
                "ndvi": fuse_benchmark(ndvi(truth), {k: ndvi(v) for k, v in methods.items()}, ["NDVI"], inside),
            }
            for stage, values in (("11_Landsat_only", landsat_only), ("12_same_anchor_Landsat", same_pair),
                                  ("13_NDVI_error", ndvi(fused) - ndvi(truth))):
                add_layer(output, catalog, group, "Validation", stage, day, fine, values,
                          ("NDVI",) if values.shape[0] == 1 else BANDS)

    report = {
        "watershed": "River Raisin", "boundary_source": args.boundary.name,
        "days": days, "anchors": anchors, "grid": fine.metadata(), "modis_grid": native.metadata(),
        "watershed_30m_pixels": pixel_count, "landsat_sources": scenes, "landsat_products": LANDSAT_PRODUCTS,
        "modis_product": MODIS_PRODUCT, "modis_qa": "full inversion in NIR, Red, and Green",
        "landsat_qa": "QA_PIXEL bits 0-5 clear and QA_RADSAT zero",
        "holdout": "All five prediction dates excluded from every Landsat fusion input and restoration reference.",
        "temporal": {"landsat_days_per_side": LANDSAT_WINDOW, "modis_days_per_side": modis_window,
                     "one_sided": False},
        "modis_temporal_source_codes": {"0": "unresolved", "1": "observed and QA approved",
                                        "2": "interpolated with both sides within 8 days",
                                        "3": "interpolated with an observation more than 8 days away"},
        "spatial_reference": "Select highest clear coverage (at least 60%) within 21 days of an independent date.",
        "registration": "Common projected grid, bilinear MODIS resampling, no empirical shift applied.",
        "support_codes": {"0": "unresolved", "1": "two-sided residual and valid target MODIS"},
        "dates": stage_info, "layers": catalog,
    }
    (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 7, 22))
    parser.add_argument("--anchors", nargs="+", type=date.fromisoformat, default=list(map(date.fromisoformat, DEFAULT_ANCHORS)))
    parser.add_argument("--boundary", type=Path, default=ROOT / "data/raisin_outline.shp")
    parser.add_argument("--project", default=os.getenv("EE_PROJECT_ID", "stair-project-509517"))
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/fusion_validation/_watershed_cache")
    parser.add_argument("--modis-days-per-side", type=int, default=MODIS_WINDOW,
                        help="Maximum age of each MODIS observation used in temporal restoration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.modis_days_per_side < 1:
        parser.error("--modis-days-per-side must be positive")
    output = args.output or ROOT / "outputs/fusion_validation" / f"raisin_{args.start}_five_days"
    output = output.resolve()
    if output.exists():
        parser.error(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}_", dir=output.parent) as temp:
        working = Path(temp) / "run"
        working.mkdir()
        run(args, working)
        if output.exists():
            raise FileExistsError(output)
        shutil.move(str(working), output)
    print(output, flush=True)


if __name__ == "__main__":
    main()
