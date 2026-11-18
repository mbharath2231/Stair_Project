#!/usr/bin/env python3
"""Evaluate Landsat cloud filling on withheld pixels using density and paired error."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import ee
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_stair_stages import BANDS, CRS, SIZE, TRANSFORM, WINDOW, download, write_local
from scripts.run_watershed_fusion import restore, validity
from stair.landsat import landsat_reflectance_collection
from stair.ndvi import ndvi
from stair.temporal import interpolate_temporal_numpy
from stair.validation import make_holdout, score_holdout

TARGETS = ("2024-03-13", "2024-04-14", "2022-07-22")
STAGE_TIFFS = (
    ("01_original", "Original Landsat"),
    ("02_cloud_masked", "Simulated cloud mask"),
    ("03_temporal", "Temporal interpolation"),
    ("04_global", "Global correction"),
    ("05_adaptive", "Adaptive correction"),
)
FEATURES = ("NIR", "Red", "Green", "NDVI")
STYLES = {
    "Observed": ("#20262d", "-", 2.25),
    "Temporal": ("#3671b0", "--", 1.5),
    "Global": ("#a64f0f", "-.", 1.5),
    "Adaptive": ("#007c66", ":", 2.0),
}


def load_scenes(day, cache_dir, project):
    cache = cache_dir / f"clayton_{day}.npz"
    if not cache.is_file():
        ee.Initialize(project=project)
        resolution, _, x, _, _, y = TRANSFORM
        region = ee.Geometry.Rectangle([x, y - SIZE * resolution, x + SIZE * resolution, y],
                                       proj=CRS, geodesic=False)
        target = date.fromisoformat(day)
        start = str(target - timedelta(days=WINDOW))
        end = str(target + timedelta(days=WINDOW + 1))
        collection = landsat_reflectance_collection(region, start, end)
        rows = collection.reduceColumns(
            ee.Reducer.toList(2), ["system:time_start", "source_id"]).get("list").getInfo()
        scenes = [(datetime.fromtimestamp(ms / 1000, timezone.utc).date().isoformat(), source)
                  for ms, source in rows]
        dates = sorted({row[0] for row in scenes})
        if day not in dates:
            raise ValueError(f"No Landsat acquisition on {day}")
        images = [collection.filterDate(d, ee.Date(d).advance(1, "day")).sort("source_id").mosaic()
                  for d in dates]
        observations = np.stack(np.split(download(ee.Image.cat(images)), len(dates)))
        cache_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".clayton_", dir=cache_dir) as tmp:
            staging = Path(tmp) / cache.name
            np.savez_compressed(staging, dates=dates, observations=observations, scenes=json.dumps(scenes))
            staging.replace(cache)
    with np.load(cache, allow_pickle=False) as data:
        dates, images, scenes = data["dates"].tolist(), data["observations"], json.loads(str(data["scenes"]))
    return dict(zip(dates, images, strict=True)), scenes


def estimate_stages(images, day, holdout, reference):
    """The target enters only as the observed array after masking the holdout."""
    inputs = {d: image for d, image in images.items() if d != day}
    truth = images[day]
    observed = np.where(holdout[None], np.nan, truth)
    temporal = interpolate_temporal_numpy(np.stack(list(inputs.values())), list(inputs), day,
                                          observed, max_distance_days=WINDOW).filled
    global_result, adaptive, details = restore(observed, temporal, inputs[reference])
    return truth, {"Temporal": temporal, "Global": global_result, "Adaptive": adaptive}, details


def compare(truth, stages, holdout, names):
    """Use the same finite withheld pixels in the PDFs and all paired comparisons."""
    eligible = holdout & validity(truth)
    common = eligible.copy()
    coverage = {}
    for name, values in stages.items():
        covered = eligible & validity(values)
        coverage[name] = int(covered.sum())
        common &= covered
    if not common.any():
        raise ValueError(f"No shared holdout support for {names}")
    samples = {"Observed": truth[:, common]}
    scores = {}
    for name, values in stages.items():
        samples[name] = values[:, common]
        band_scores = score_holdout(truth, values, common, names)["bands"]
        for i, band in enumerate(names):
            # For equal-size unweighted samples, match order statistics to obtain W1.
            band_scores[band]["wasserstein_1"] = float(np.mean(
                np.abs(np.sort(samples["Observed"][i]) - np.sort(samples[name][i]))
            ))
        scores[name] = band_scores
    return samples, {"eligible": int(eligible.sum()), "common": int(common.sum()),
                     "covered_by_stage": coverage, "scores": scores}


def evaluate(images, day, pattern, reference, *, stage_directory=None):
    holdout = make_holdout((SIZE, SIZE), pattern)
    truth, stages, details = estimate_stages(images, day, holdout, reference)
    clear = ~holdout & validity(truth)
    for image in stages.values():
        if not np.array_equal(image[:, clear], truth[:, clear]):
            raise RuntimeError(f"Restoration changed an observed pixel on {day}")
    reflection, score = compare(truth, stages, holdout, list(FEATURES[:3]))
    ndvi_samples, ndvi_score = compare(ndvi(truth), {k: ndvi(v) for k, v in stages.items()},
                                       holdout, ["NDVI"])
    if stage_directory is not None:
        stage_directory.mkdir(parents=True, exist_ok=False)
        masked = np.where(holdout[None], np.nan, truth)
        rasters = (truth, masked, stages["Temporal"], stages["Global"], stages["Adaptive"])
        for (stem, _), data in zip(STAGE_TIFFS, rasters, strict=True):
            write_local(stage_directory / f"{stem}.tif", data, day, stem)
    return {"reflectance": reflection, "ndvi": ndvi_samples}, {
        "reflectance": score, "ndvi": ndvi_score, "spatial_correction": details,
    }


def choose_reference(images, day):
    target = date.fromisoformat(day)
    candidates = []
    for d, image in images.items():
        offset = (date.fromisoformat(d) - target).days
        clear = float(validity(image).mean())
        if 0 < abs(offset) <= 21 and clear >= 0.95:
            candidates.append((abs(offset), offset > 0, d, clear))
    if not candidates:
        raise ValueError(f"No independent Landsat reference with 95% clear coverage within 21 days of {day}")
    return min(candidates)[2:]


def bins_for(arrays, width):
    minimum = min(values.min() for values in arrays)
    maximum = max(values.max() for values in arrays)
    start = int(np.floor(minimum / width))
    stop = max(start + 1, int(np.ceil(maximum / width)))
    edges = width * np.arange(start, stop + 1)
    edges[0] = min(edges[0], minimum)
    edges[-1] = max(edges[-1], maximum)
    return edges


def plot(output, result, scores, *, pattern, errors=False):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    fig = Figure(figsize=(11, 8.6), facecolor="white")
    FigureCanvasAgg(fig)
    fig.subplots_adjust(left=.10, right=.98, top=.83, bottom=.12, hspace=.44, wspace=.19)
    axes = fig.subplots(len(result), 2, squeeze=False, sharex="col", sharey="col")
    title = "Pixel error distributions after Landsat cloud filling" if errors else "Observed and restored distributions after Landsat cloud filling"
    fig.text(.10, .97, title, fontsize=17, weight="bold", va="top", color="#1b242d")
    subtitle = ("Errors are restored minus withheld values at the same locations."
                if errors else "Density of observed and restored values at the same withheld locations.")
    subtitle += f" Clayton Township · {pattern} holdout"
    fig.text(.10, .917, subtitle, fontsize=10, color="#485767", va="top")
    names = list(STYLES)[1:] if errors else list(STYLES)
    handles = [Line2D([], [], color=STYLES[k][0], linestyle=STYLES[k][1], lw=STYLES[k][2], label=k)
               for k in names]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .892),
               frameon=False, ncol=len(names), fontsize=10)
    for col, (component, index, label, width) in enumerate(
            [("reflectance", 0, "NIR reflectance", 0.02), ("ndvi", 0, "NDVI", 0.05)]):
        arrays = {day: {name: (values[index] - sample[component]["Observed"][index] if errors else values[index])
                        for name, values in sample[component].items() if name in names}
                  for day, sample in result.items()}
        bins = bins_for([values for panel in arrays.values() for values in panel.values()], width)
        for row, (day, panel) in enumerate(arrays.items()):
            ax = axes[row, col]
            if errors:
                ax.axvline(0, color="#909ba5", lw=1, ls="--")
            for name, values in panel.items():
                counts = np.histogram(values, bins=bins)[0]
                if counts.sum() != len(values):
                    raise RuntimeError("Density bins dropped one or more scored pixels")
                estimates = counts / (len(values) * np.diff(bins))
                color, dash, lw = STYLES[name]
                ax.stairs(estimates, bins, color=color, linestyle=dash, linewidth=lw, baseline=None)
            count = scores[day][pattern][component]["common"]
            ax.set_title(f"{day}  •  {label}  •  n = {count:,}", loc="left", fontsize=10.5, pad=8)
            ax.set_xlim(bins[0], bins[-1])
            ax.grid(axis="y", color="#e8edf1", linewidth=.65)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#8c99a4")
            ax.tick_params(labelsize=9, colors="#44505d")
        axes[-1, col].set_xlabel(label + (" error" if errors else ""), fontsize=10)
    fig.supylabel("Probability density", x=.025, fontsize=11)
    fig.text(.10, .037, "Area-normalized histograms with fixed bin widths by variable. All stages share the same valid pixels.",
             fontsize=9, color="#485767")
    fig.savefig(output, dpi=260)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", default=list(TARGETS))
    parser.add_argument("--patterns", nargs="+", choices=["cloud", "stripes"], default=["cloud", "stripes"])
    parser.add_argument("--project", default=os.getenv("EE_PROJECT_ID", "stair-project-509517"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion_validation/cloud_density_clayton")
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/fusion_validation/_cloud_density_cache")
    parser.add_argument("--export-stages", action="store_true", help="Write five aligned GeoTIFFs per date and mask")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Choose a new output directory: {args.output}")
    if len(set(args.dates)) != len(args.dates) or len(set(args.patterns)) != len(args.patterns):
        parser.error("Dates and patterns must be unique")
    results = {pattern: {} for pattern in args.patterns}
    scores = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cloud_density_", dir=args.output.parent) as tmp:
        staging = Path(tmp) / args.output.name
        staging.mkdir()
        for day in args.dates:
            if date.fromisoformat(day).isoformat() != day:
                parser.error(f"Use YYYY-MM-DD for {day}")
            images, scenes = load_scenes(day, args.cache, args.project)
            reference, clear = choose_reference(images, day)
            print(f"{day}: spatial reference {reference}, {clear:.1%} clear", flush=True)
            scores[day] = {"reference": reference, "reference_qa_coverage": clear,
                           "source_scenes": scenes}
            for pattern in args.patterns:
                stage_directory = staging / pattern / day if args.export_stages else None
                results[pattern][day], scores[day][pattern] = evaluate(
                    images, day, pattern, reference, stage_directory=stage_directory)
        for pattern in results:
            plot(staging / f"{pattern}_reflectance_density.png", results[pattern], scores, pattern=pattern)
            plot(staging / f"{pattern}_error_density.png", results[pattern], scores, pattern=pattern, errors=True)
        report = {
            "grid": {"crs": CRS, "transform": TRANSFORM, "dimensions": [SIZE, SIZE]},
            "temporal_days_per_side": WINDOW, "reference_search_days": 21, "reference_min_qa_coverage": 0.95,
            "reflectance_source_bands": BANDS, "patterns": args.patterns,
            "density_bin_widths": {"NIR": 0.02, "NDVI": 0.05},
            "density_measure": "Wasserstein-1 on equally weighted empirical samples of common valid pixels",
            "score_policy": "Restore after masking the target; score only originally QA-clear held-out pixels.",
            "dates": scores,
        }
        if args.export_stages:
            report["stage_tiffs"] = dict(STAGE_TIFFS)
        (staging / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
