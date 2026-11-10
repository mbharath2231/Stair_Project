#!/usr/bin/env python3
"""Compare two Clayton fusion weeks against QA-clear Landsat for spring NDVI."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import ee
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_stair_stages import CRS, NODATA, SIZE, TRANSFORM, download
from scripts.run_fusion_week import ndvi
from stair.landsat import landsat_reflectance_collection

BENCHMARKS = ("2024-03-13", "2024-04-14")
NDVI_LIMITS = (0.05, 0.90)
PATCH_SIZE = 7


def landsat_context(project):
    ee.Initialize(project=project)
    x, y = TRANSFORM[2], TRANSFORM[5]
    region = ee.Geometry.Rectangle([x, y - 30 * SIZE, x + 30 * SIZE, y], proj=CRS, geodesic=False)
    collection = landsat_reflectance_collection(region, BENCHMARKS[0], "2024-05-01")
    rows = collection.reduceColumns(ee.Reducer.toList(2), ["source_id", "system:time_start"]).get("list").getInfo()
    scenes = [(scene, datetime.fromtimestamp(ms / 1000, timezone.utc).date().isoformat())
              for scene, ms in rows]
    days = sorted({day for _, day in scenes})
    images = [collection.filterDate(day, ee.Date(day).advance(1, "day")).sort("source_id").mosaic()
              for day in days]
    arrays = np.split(download(ee.Image.cat(images)), len(images))
    observations = {day: ndvi(image)[0] for day, image in zip(days, arrays)
                    if np.isfinite(image).all(axis=0).mean() >= 0.60}
    crop = download(ee.Image("USDA/NASS/CDL/2024").select("cropland"))[0]
    return observations, crop, scenes


def read_ndvi(path):
    with rasterio.open(path) as src:
        if src.crs.to_string() != CRS or tuple(src.transform)[:6] != tuple(TRANSFORM) or src.shape != (SIZE, SIZE):
            raise ValueError(f"Raster is outside the shared Clayton grid: {path}")
        return ndvi(src.read(masked=True).filled(np.nan))[0], src.profile


def change_stats(actual, predicted, area):
    valid = area & np.isfinite(actual).all(axis=0) & np.isfinite(predicted).all(axis=0)
    observed_delta = (actual[1] - actual[0])[valid]
    predicted_delta = (predicted[1] - predicted[0])[valid]
    substantial = abs(observed_delta) >= 0.05
    return {
        "common_pixels": int(valid.sum()),
        "landsat_median_change": float(np.median(observed_delta)),
        "fusion_median_change": float(np.median(predicted_delta)),
        "change_rmse": float(np.sqrt(np.mean((predicted_delta - observed_delta) ** 2))),
        "change_correlation": float(np.corrcoef(observed_delta, predicted_delta)[0, 1]),
        "observed_change_at_least_0_05_pixels": int(substantial.sum()),
        "direction_agreement_on_those_pixels": float(np.mean(
            np.sign(observed_delta[substantial]) == np.sign(predicted_delta[substantial])
        )),
    }


def select_examples(actual, fused, crop):
    """Choose illustrative patches by observed change only, with common support."""
    candidates = []
    common = np.isfinite(np.stack([*actual, *fused.values()])).all(axis=0)
    for row in range(3, SIZE - PATCH_SIZE, PATCH_SIZE):
        for col in range(3, SIZE - PATCH_SIZE, PATCH_SIZE):
            window = np.s_[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            code = crop[row, col]
            if code not in (1, 5) or not (crop[window] == code).all() or not common[window].all():
                continue
            change = float(np.median(actual[1][window] - actual[0][window]))
            candidates.append((change, row, col, int(code)))
    selections = []
    for label, order in (
        ("A", lambda item: -item[0]),
        ("B", lambda item: abs(item[0])),
        ("C", lambda item: item[0]),
    ):
        change, row, col, code = min(
            (item for item in candidates if all(
                max(abs(item[1] - s["row"]), abs(item[2] - s["col"])) >= 3 * PATCH_SIZE
                for s in selections
            )), key=order,
        )
        selections.append({"site": label, "row": row, "col": col, "cdl_crop": "Corn" if code == 1 else "Soybeans",
                           "landsat_median_pixel_change": change})
    return selections


def save_samples(output, samples, sources):
    step, _, x, _, _, y = TRANSFORM
    geometry, records = [], []
    for sample in samples:
        row, col = sample["row"], sample["col"]
        geometry.append(box(x + col * step, y - (row + PATCH_SIZE) * step,
                            x + (col + PATCH_SIZE) * step, y - row * step))
        window = np.s_[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
        for source, days in sources.items():
            for day, image in sorted(days.items()):
                pixels = image[window]
                if np.isfinite(pixels).all():
                    records.append({"site": sample["site"], "cdl_crop": sample["cdl_crop"], "date": day,
                                    "source": source, "pixels": pixels.size, "median_ndvi": float(np.median(pixels))})
    gpd.GeoDataFrame(samples, geometry=geometry, crs=CRS).to_file(output / "sample_patches.gpkg", layer="patches")
    with (output / "sample_ndvi.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def render(output, actual, fused, samples, records, stats):
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    fig = Figure(figsize=(11, 8), layout="constrained")
    FigureCanvasAgg(fig)
    fig.suptitle("Clayton Township  |  Spring 2024 NDVI", fontsize=20, fontweight="bold")
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1.75))
    for row, (source, images) in enumerate((("Withheld Landsat", actual), ("STAIR fusion", fused))):
        for col, (day, image) in enumerate(zip(BENCHMARKS, images)):
            ax = fig.add_subplot(grid[row, col])
            panel = ax.imshow(image, cmap="viridis", vmin=NDVI_LIMITS[0], vmax=NDVI_LIMITS[1],
                              interpolation="nearest")
            ax.set_title(f"{day[5:]}  |  {source}", fontsize=10)
            ax.set_axis_off()
            for s in samples:
                ax.add_patch(Rectangle((s["col"] - 0.5, s["row"] - 0.5), PATCH_SIZE, PATCH_SIZE,
                                       facecolor="none", edgecolor="white", linewidth=1))
                ax.text(s["col"] + PATCH_SIZE + 1, s["row"] + 4, s["site"], color="white", weight="bold",
                        bbox={"facecolor": "#172133", "edgecolor": "none", "pad": 2}, fontsize=9)
    fig.colorbar(panel, ax=fig.axes[:4], orientation="horizontal", fraction=0.08, pad=0.03,
                 label="Shared NDVI display scale")

    ax = fig.add_subplot(grid[:, 2])
    colors = {"A": "#007f6d", "B": "#405dac", "C": "#b75729"}
    for sample in samples:
        site = sample["site"]
        obs = [r for r in records if r["site"] == site and r["source"] == "Landsat"]
        ax.scatter([datetime.fromisoformat(r["date"]) for r in obs], [r["median_ndvi"] for r in obs],
                   facecolors="white", edgecolors=colors[site], linewidths=1.5, zorder=5)
        for start in BENCHMARKS:
            selected = [r for r in records if r["site"] == site and r["source"] == "Fusion"
                        and r["date"][:7] == start[:7]]
            ax.plot([datetime.fromisoformat(r["date"]) for r in selected],
                    [r["median_ndvi"] for r in selected], marker=".", color=colors[site],
                    label=f'{site} · {sample["cdl_crop"]}' if start == BENCHMARKS[0] else None)
    for day in BENCHMARKS:
        ax.axvline(datetime.fromisoformat(day), color="#777777", linestyle=":", linewidth=1)
    ax.set_title("Same 210 m patches across dates\nOpen circles = Landsat  ·  Solid lines = fusion", fontsize=11)
    ax.set_ylabel("Median NDVI")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", color="#dddddd")
    ax.legend(loc="best", frameon=False, title="2024 CDL cash crop")
    fig.supxlabel(
        f"Median March–April change on {stats['common_pixels']:,} corn/soybean pixels  |  "
        f"Landsat {stats['landsat_median_change']:+.3f}  ·  Fusion {stats['fusion_median_change']:+.3f}",
        fontsize=11,
    )
    fig.savefig(output / "spring_ndvi_review.png", dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.getenv("EE_PROJECT_ID", "stair-project-509517"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion_validation/spring_2024_phenology")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Choose a new output directory: {args.output}")
    root = ROOT / "outputs/fusion_validation"
    runs = [root / name for name in ("2024-03-13_to_2024-03-19", "2024-04-14_to_2024-04-20")]
    observed, crop, scenes = landsat_context(args.project)
    fused, profile = {}, None
    for run in runs:
        for path in sorted(run.glob("STAIR_fusion_*.tif")):
            fused[path.stem[-10:]], profile = read_ndvi(path)
    for day, run in zip(BENCHMARKS, runs):
        np.testing.assert_allclose(observed[day], read_ndvi(run / f"Landsat_truth_{day}.tif")[0],
                                   atol=2e-7, rtol=0, equal_nan=True)
    actual = np.stack([observed[day] for day in BENCHMARKS])
    estimated = np.stack([fused[day] for day in BENCHMARKS])
    stats = change_stats(actual, estimated, np.isin(crop, (1, 5)))
    samples = select_examples(actual, fused, crop)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    profile.update(count=1, dtype="float32", nodata=NODATA, compress="deflate")
    for source, dates in (("Landsat", observed), ("Fusion", fused)):
        for day, image in dates.items():
            with rasterio.open(output / f"{source}_NDVI_{day}.tif", "w", **profile) as dst:
                dst.write(np.where(np.isfinite(image), image, NODATA).astype(np.float32), 1)
                dst.set_band_description(1, "NDVI")
    for day, run in zip(BENCHMARKS, runs):
        for stem in ("Landsat_truth", "STAIR_fusion"):
            path = run / f"{stem}_{day}.tif"
            shutil.copyfile(path, output / path.name)
    records = save_samples(output, samples, {"Landsat": observed, "Fusion": fused})
    render(output, actual, estimated, samples, records, stats)
    report = {"benchmark_dates": BENCHMARKS, "observed_dates": sorted(observed), "fused_dates": sorted(fused),
              "ndvi_display_limits": NDVI_LIMITS,
              "cdl": "USDA/NASS/CDL/2024 corn and soybean classes", "landsat_scenes": scenes,
              "sample_design": "Illustrative 7 x 7 patches selected for high, near-zero, and low observed change",
              "samples": samples, "corn_soy_change": stats}
    (output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output.resolve(), stats, flush=True)


if __name__ == "__main__":
    main()
