#!/usr/bin/env python3
"""Run a seven-day retrospective Landsat–MODIS fusion pilot on the Clayton chip."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

import ee
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_stair_stages import BANDS, CRS, SIZE, TRANSFORM, WINDOW, download, write_local
from stair.fusion import stair_dual_fusion_numpy
from stair.landsat import landsat_reflectance_collection
from stair.spatial import fit_spatial_offsets, segment_reference
from stair.temporal import interpolate_temporal_numpy
from stair.validation import score_holdout

MODIS_PRODUCT = "MODIS/061/MCD43A4"
MODIS_BANDS = (2, 1, 4)  # Correspond to Landsat NIR, Red, Green.
FCC_LOW = np.array([0.08, 0.005, 0.01])
FCC_HIGH = np.array([0.60, 0.165, 0.15])


def read_inputs(prior, later, days):
    x, y = TRANSFORM[2], TRANSFORM[5]
    region = ee.Geometry.Rectangle([x, y - 30 * SIZE, x + 30 * SIZE, y], proj=CRS, geodesic=False)
    start = (prior - timedelta(days=WINDOW)).isoformat()
    end = (later + timedelta(days=WINDOW + 1)).isoformat()
    collection = landsat_reflectance_collection(region, start, end)
    rows = collection.reduceColumns(ee.Reducer.toList(2), ["source_id", "system:time_start"]).get("list").getInfo()
    scenes = [(scene, datetime.fromtimestamp(ms / 1000, timezone.utc).date().isoformat())
              for scene, ms in rows]
    landsat_days = sorted({day for _, day in scenes})
    for anchor in (prior, later):
        if anchor.isoformat() not in landsat_days:
            raise ValueError(f"No Landsat acquisition on {anchor}. Available dates: {landsat_days}")
    landsat_images = [
        collection.filterDate(day, ee.Date(day).advance(1, "day")).sort("source_id").mosaic()
        for day in landsat_days
    ]
    landsat = dict(zip(landsat_days, np.split(download(ee.Image.cat(landsat_images)), len(landsat_days))))

    modis_days = [prior.isoformat(), *days, later.isoformat()]
    coarse = (ee.ImageCollection(MODIS_PRODUCT).filterBounds(region)
              .filterDate(modis_days[0], ee.Date(modis_days[-1]).advance(1, "day")))
    available = set(coarse.aggregate_array("system:index").getInfo())
    images = []
    for day in modis_days:
        if day.replace("-", "_") not in available:
            raise ValueError(f"No MCD43A4 retrieval for {day}")
        image = ee.Image(coarse.filterDate(day, ee.Date(day).advance(1, "day")).first())
        refl = image.select([f"Nadir_Reflectance_Band{band}" for band in MODIS_BANDS], BANDS)
        qa = image.select([f"BRDF_Albedo_Band_Mandatory_Quality_Band{band}" for band in MODIS_BANDS])
        good = qa.eq(0).And(refl.gte(0)).And(refl.lte(32766)).reduce(ee.Reducer.min())
        images.append(refl.updateMask(good).multiply(0.0001).toFloat().resample("bilinear"))
    modis = dict(zip(modis_days, np.split(download(ee.Image.cat(images)), len(images))))
    return landsat, modis, scenes


def restore_anchor(allowed, anchor):
    """Fill an anchor using observations from outside the prediction week."""
    observed = allowed[anchor]
    clear = np.isfinite(observed).all(axis=0)
    info = {"observed_pixels": int(clear.sum()), "spatial_reference": None, "unresolved_pixels": 0}
    if clear.all():
        return observed, info
    if not clear.any():
        raise ValueError(f"Anchor {anchor} has no QA-clear Landsat pixels")
    predicted = interpolate_temporal_numpy(np.stack(list(allowed.values())), list(allowed), anchor,
                                            observed, max_distance_days=WINDOW).filled
    earlier_clear = [day for day, image in allowed.items()
                     if 0 < (date.fromisoformat(anchor) - date.fromisoformat(day)).days <= 21
                     and np.isfinite(image).all(axis=0).mean() >= 0.95]
    if earlier_clear:
        reference_day = max(earlier_clear)
        reference = allowed[reference_day]
        segments, centers, _, _ = segment_reference(reference)
        global_offset, class_offsets, support = fit_spatial_offsets(observed, predicted, reference, segments)
        corrected = predicted + np.array(global_offset)[:, None, None]
        for label, offset in class_offsets.items():
            in_class = segments == label
            corrected[:, in_class] = predicted[:, in_class] + np.array(offset)[:, None]
        predicted = np.where(clear[None], observed, corrected)
        info.update(spatial_reference=reference_day, classes=len(centers), calibration=support)
    info["unresolved_pixels"] = int((~np.isfinite(predicted).all(axis=0)).sum())
    np.testing.assert_array_equal(predicted[:, clear], observed[:, clear])
    return predicted, info


def ndvi(reflectance):
    nir, red = reflectance[:2]
    total = nir + red
    valid = np.isfinite(total) & (total > 0) & (nir >= 0) & (red >= 0)
    return np.divide(nir - red, total, out=np.full_like(total, np.nan), where=valid)[None]


def compare(truth, methods, bands):
    """Keep per-method coverage and compare errors on identical pixel support."""
    eligible = np.isfinite(truth).all(axis=0)
    valid = {name: np.isfinite(image).all(axis=0) for name, image in methods.items()}
    common = np.logical_and.reduce(list(valid.values()))
    count = int(eligible.sum())
    return {
        name: {
            "coverage": int((eligible & valid[name]).sum()) / count if count else None,
            "common_support": score_holdout(truth, image, common, bands),
        }
        for name, image in methods.items()
    }


def reflectance_change(first, second):
    common = (np.isfinite(first) & np.isfinite(second)).all(axis=0)
    return {
        "common_pixels": int(common.sum()),
        "median_absolute_change": dict(zip(
            BANDS, np.median(np.abs(second[:, common] - first[:, common]), axis=1).tolist()
            if common.any() else [None] * len(BANDS),
        )),
    }


def fuse_week(landsat, modis, days, prior, later):
    """Synthesize the whole week before using its Landsat scenes for scoring."""
    allowed = {day: image for day, image in landsat.items() if day not in days}
    left, left_info = restore_anchor(allowed, prior.isoformat())
    right, right_info = restore_anchor(allowed, later.isoformat())
    left_modis, right_modis = modis[prior.isoformat()], modis[later.isoformat()]
    forecasts, daily = {}, {}
    for day in days:
        weight = (date.fromisoformat(day) - prior).days / (later - prior).days
        coarse = modis[day]
        forecast = stair_dual_fusion_numpy(left, left_modis, right, right_modis, coarse,
                                          1 - weight, weight, clip_range=None)
        forecasts[day] = forecast
        daily[day] = {
            "prior_weight": 1 - weight,
            "later_weight": weight,
            "modis_valid_pixels": int(np.isfinite(coarse).all(axis=0).sum()),
            "fused_valid_pixels": int(np.isfinite(forecast).all(axis=0).sum()),
        }
        if not daily[day]["fused_valid_pixels"]:
            raise ValueError(f"No supported fusion pixels for {day}")

    allowed_stack = np.stack(list(allowed.values()))
    for day in days:
        if day not in landsat or not np.isfinite(landsat[day]).all(axis=0).any():
            continue
        coarse = modis[day]
        methods = {
            "fusion": forecasts[day],
            "landsat_only": interpolate_temporal_numpy(allowed_stack, list(allowed), day,
                                                       max_distance_days=WINDOW).prediction,
            "prior_pair": coarse + (left - left_modis),
            "modis_only": coarse,
        }
        truth = landsat[day]
        daily[day]["reflectance_validation"] = compare(truth, methods, BANDS)
        daily[day]["ndvi_validation"] = compare(ndvi(truth), {k: ndvi(v) for k, v in methods.items()}, ["NDVI"])
    return forecasts, daily, {prior.isoformat(): left_info, later.isoformat(): right_info}


def write_fcc(path, forecasts, modis, truth):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 10), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle(f"Clayton Township | {min(forecasts)} to {max(forecasts)}",
                    fontsize=16, fontweight="bold")
    remaining = [(day, image) for day, image in forecasts.items() if day not in truth]
    rows = (len(remaining) + 2) // 3
    benchmark, daily = figure.subfigures(2, 1, height_ratios=[len(truth), max(rows, 1)])

    def panel(ax, image, label):
        rgb = (np.moveaxis(image, 0, -1) - FCC_LOW) / (FCC_HIGH - FCC_LOW)
        rgb = np.where(np.isfinite(rgb).all(axis=-1, keepdims=True), np.clip(rgb, 0, 1), 1)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()

    benchmark.suptitle("Benchmark against withheld Landsat", fontweight="bold")
    for axes, (day, observed) in zip(benchmark.subplots(len(truth), 3, squeeze=False), truth.items()):
        panel(axes[0], modis[day], f"{day}  MODIS")
        panel(axes[1], forecasts[day], "STAIR fusion")
        panel(axes[2], observed, "QA-clear Landsat truth")
    daily.suptitle("Daily STAIR fusion on dates without usable Landsat", fontweight="bold")
    if remaining:
        axes = daily.subplots(rows, 3, squeeze=False).ravel()
        for ax in axes:
            ax.set_axis_off()
        for ax, (day, image) in zip(axes, remaining):
            panel(ax, image, day)
    figure.supxlabel("Shared FCC stretch   |   NIR 0.08–0.60   /   Red 0.005–0.165   /   Green 0.01–0.15",
                     fontsize=10)
    figure.savefig(path, dpi=190)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 7, 22))
    parser.add_argument("--prior", type=date.fromisoformat, default=date(2022, 7, 21))
    parser.add_argument("--later", type=date.fromisoformat, default=date(2022, 7, 30))
    parser.add_argument("--project", default=os.getenv("EE_PROJECT_ID", "stair-project-509517"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    end = args.start + timedelta(days=6)
    if not args.prior < args.start <= end < args.later:
        parser.error("The two anchor dates must bracket the entire seven-day window")
    if max((end - args.prior).days, (args.later - args.start).days) > WINDOW:
        parser.error(f"Both anchor dates must be within {WINDOW} days of every prediction")
    days = [(args.start + timedelta(days=i)).isoformat() for i in range(7)]
    output = args.output or ROOT / "outputs" / "fusion_validation" / f"{days[0]}_to_{days[-1]}"
    if output.exists():
        parser.error(f"Choose a new output directory: {output}")

    ee.Initialize(project=args.project)
    landsat, modis, scenes = read_inputs(args.prior, args.later, days)
    truth = {day: landsat[day] for day in days if day in landsat and np.isfinite(landsat[day]).all(axis=0).any()}
    if not truth:
        raise ValueError("Choose a week with at least one QA-clear Landsat pixel for independent validation")
    forecasts, daily, anchors = fuse_week(landsat, modis, days, args.prior, args.later)
    metadata = {
        "days": days, "grid": {"crs": CRS, "transform": TRANSFORM, "dimensions": [SIZE, SIZE]},
        "bands": BANDS, "modis_product": MODIS_PRODUCT, "modis_bands": MODIS_BANDS,
        "modis_qa": "full inversion (0) in all three bands before bilinear resampling",
        "registration": "same projected grid; no estimated cross-sensor offset applied",
        "anchors": anchors, "landsat_scenes": scenes,
        "holdout_policy": "all Landsat dates in the seven-day window withheld from all fitting and prediction",
        "benchmark_dates": list(truth),
        "daily": daily,
        "first_to_last_change": {
            "fusion": reflectance_change(forecasts[days[0]], forecasts[days[-1]]),
            "modis": reflectance_change(modis[days[0]], modis[days[-1]]),
        },
        "consecutive_change": {
            day: {
                "fusion": reflectance_change(forecasts[previous], forecasts[day]),
                "modis": reflectance_change(modis[previous], modis[day]),
            }
            for previous, day in zip(days, days[1:])
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for day, reflectance in forecasts.items():
        write_local(output / f"STAIR_fusion_{day}.tif", reflectance, day, "fusion")
        write_local(output / f"MODIS_aligned_{day}.tif", modis[day], day, "MODIS_aligned")
    for day, observed in truth.items():
        write_local(output / f"Landsat_truth_{day}.tif", observed, day, "Landsat_truth")
    write_fcc(output / "fusion_week_fcc.png", forecasts, modis, truth)
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
