"""Comparable scientific plots and analytical rasters for temporal validation."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np


def write_temporal_report(truth, observed, result, holdout, metadata, output_dir, *,
                          observations=None, observation_dates=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import rasterio
    from affine import Affine
    from .validation import score_holdout

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bands = metadata["bands"]
    metrics = score_holdout(truth, result.prediction, holdout, bands)
    known = np.isfinite(observed)
    if not np.array_equal(observed[known], result.filled[known]):
        raise AssertionError("Temporal filling changed observed clear pixels")
    metrics["observed_values_preserved"] = True
    metadata = dict(metadata, metrics=metrics)
    (out / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

    def save_tif(name, array, descriptions, categorical=False):
        array = np.asarray(array)
        if array.ndim == 2:
            array = array[None]
        nodata = 255 if categorical else -9999.0
        dtype = "uint8" if categorical else "float32"
        profile = dict(driver="GTiff", height=array.shape[1], width=array.shape[2],
                       count=array.shape[0], dtype=dtype, crs=metadata["crs"],
                       transform=Affine(*metadata["transform"]), nodata=nodata, compress="deflate")
        with rasterio.open(out / name, "w", **profile) as dst:
            dst.write(np.where(np.isfinite(array), array, nodata).astype(dtype))
            dst.descriptions = tuple(descriptions)
            units = "count" if "support_count" in name else ("category" if categorical else ("days" if "age" in name else "surface reflectance"))
            dst.update_tags(stage="temporal", target_date=metadata["target_date"], units=units)

    eligible = holdout & np.all(np.isfinite(truth), axis=0)
    error = np.where(eligible[None], result.prediction - truth, np.nan)
    save_tif("00_qa_clear_truth.tif", truth, bands)
    save_tif("01_allowed_observations.tif", observed, bands)
    save_tif("02_temporal_prediction.tif", result.prediction, bands)
    save_tif("03_temporal_filled.tif", result.filled, bands)
    save_tif("04_signed_error_heldout.tif", error, bands)
    save_tif("05_holdout_mask.tif", holdout, ["holdout"], True)
    save_tif("06_support_count.tif", result.support_count, bands, True)
    save_tif("07_prediction_status.tif", result.status, bands, True)
    save_tif("08_before_age_days.tif", result.before_age_days, bands)
    save_tif("09_after_age_days.tif", result.after_age_days, bands)
    save_tif("10_filled_origin.tif", result.filled_origin, bands, True)

    # Fixed stretches throughout. FCC display is never reused for computation.
    stretches = {"SR_B5": (.10, .65), "SR_B4": (.0, .25), "SR_B3": (.0, .25)}
    def fcc(array):
        rgb = np.stack([np.clip((array[i] - stretches[b][0]) /
                               (stretches[b][1] - stretches[b][0]), 0, 1)
                        for i, b in enumerate(bands)], axis=-1)
        return np.where(np.all(np.isfinite(array), axis=0)[..., None], rgb, .85)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    for ax, array, title in zip(axes[0], [truth, observed, result.filled],
                               ["QA-clear target (withheld truth)", "Allowed target pixels + artificial gaps", "Temporal fill (clear pixels preserved)"]):
        ax.imshow(fcc(array), interpolation="nearest")
        ax.set_title(title)
    nir = bands.index("SR_B5")
    im = axes[1, 0].imshow(np.ma.masked_invalid(error[nir]), cmap="RdBu_r", vmin=-.10, vmax=.10, interpolation="nearest")
    axes[1, 0].set_title("NIR signed error inside held-out gap\nGray = not scored / unresolved")
    fig.colorbar(im, ax=axes[1, 0], shrink=.7, label="Prediction - truth (reflectance)")
    status_colors = ListedColormap(["#bbbbbb", "#287bb5", "#e79732", "#b1509c"])
    im = axes[1, 1].imshow(result.status[nir], cmap=status_colors,
                           norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), interpolation="nearest")
    axes[1, 1].set_title("NIR prediction support (whole image)")
    cb = fig.colorbar(im, ax=axes[1, 1], shrink=.7, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["Unresolved", "Interpolated", "Prior carry", "Future carry"])
    span = result.before_age_days[nir] + result.after_age_days[nir]
    im = axes[1, 2].imshow(np.ma.masked_invalid(span), vmin=0,
                           vmax=2 * metadata["max_distance_days"], cmap="viridis", interpolation="nearest")
    axes[1, 2].set_title("NIR supporting interval\nBefore age + after age")
    fig.colorbar(im, ax=axes[1, 2], shrink=.7, label="Days")
    for ax in axes.flat:
        ax.set_facecolor("#dddddd")
        ax.set_xticks([])
        ax.set_yticks([])
    coverage = metrics["coverage_fraction"]
    coverage_text = f"{coverage:.1%}" if coverage is not None else "no clear holdout pixels"
    fig.suptitle(f"Temporal interpolation | {metadata['target_date']} | {metadata['label']}\n"
                 f"Held-out coverage: {coverage_text} | Fixed FCC stretch | No spatial correction", fontsize=14)
    fig.savefig(out / "temporal_overview.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(bands), 3, figsize=(13, 4 * len(bands)), layout="constrained", squeeze=False)
    for i, band in enumerate(bands):
        vmin, vmax = stretches[band]
        for ax, array, title in zip(axes[i, :2], [truth[i], result.filled[i]], ["QA-clear truth", "Temporal reconstruction"]):
            im = ax.imshow(np.ma.masked_invalid(array), cmap="YlGn", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{band}: {title}")
            fig.colorbar(im, ax=ax, shrink=.7, label="Reflectance")
        im = axes[i, 2].imshow(np.ma.masked_invalid(abs(error[i])), cmap="magma", vmin=0, vmax=.10, interpolation="nearest")
        score = metrics["bands"][band]["rmse"]
        axes[i, 2].set_title(f"Absolute error in held-out pixels\nRMSE: {score:.5f}" if score is not None else "No supported holdout predictions")
        fig.colorbar(im, ax=axes[i, 2], shrink=.7, label="Absolute reflectance error")
    for ax in axes.flat:
        ax.set_facecolor("#dddddd")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Identical truth/reconstruction scales per band; identical error scale across bands", fontsize=13)
    fig.savefig(out / "temporal_bands.png", dpi=150)
    plt.close(fig)
    trace_html = ""
    if observations is not None and observation_dates is not None:
        import matplotlib.dates as mdates
        dates = np.array(observation_dates, dtype="datetime64[D]")
        target_date = np.datetime64(metadata["target_date"], "D")
        row, col = truth.shape[1] // 2, truth.shape[2] // 2
        fig, axes = plt.subplots(len(bands), 1, figsize=(12, 8), layout="constrained")
        for i, (ax, band) in enumerate(zip(axes, bands)):
            series = observations[:, i, row, col]
            valid = np.isfinite(series) & (dates != target_date)
            ax.plot(dates[valid], series[valid], "o", color="#667786", label="Allowed clear references")
            ages = [result.before_age_days[i, row, col], result.after_age_days[i, row, col]]
            if np.all(np.isfinite(ages)):
                chosen_dates = [target_date - np.timedelta64(int(ages[0]), "D"),
                                target_date + np.timedelta64(int(ages[1]), "D")]
                chosen_values = [series[np.flatnonzero(dates == d)[0]] for d in chosen_dates]
                ax.plot(chosen_dates, chosen_values, "-", color="#287bb5", label="Selected interpolation interval")
            ax.plot(target_date, truth[i, row, col], "x", color="#b52938", markersize=10,
                    markeredgewidth=2, label="Withheld target truth")
            ax.plot(target_date, result.prediction[i, row, col], "D", color="#287bb5", label="Temporal prediction")
            ax.axvline(target_date, color="#bbbbbb", linestyle=":")
            ax.set_ylabel(f"{band}\nReflectance")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.grid(alpha=.2)
        axes[0].legend(loc="best", ncol=2)
        fig.suptitle(f"How one fixed center pixel is reconstructed (row {row}, column {col})\n"
                     "Target truth is displayed for evaluation only; it is excluded from interpolation", fontsize=13)
        fig.savefig(out / "temporal_trace.png", dpi=150)
        plt.close(fig)
        trace_html = '<img src="temporal_trace.png" alt="Clear reference dates and interpolation at one fixed center pixel">'
    rows = "".join(f"<tr><td>{html.escape(b)}</td>" + "".join(
        f"<td>{v:.5f}</td>" if isinstance(v, (int, float)) else "<td>Unavailable</td>"
        for v in (m["rmse"], m["mae"], m["bias"], m["pearson_r"], m["r2"])) + "</tr>"
        for b, m in metrics["bands"].items())
    page = f"""<!doctype html><meta charset="utf-8"><title>STAIR temporal validation</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#192b37}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ccc}}code{{background:#eee}}a{{color:#176ba0}}</style>
<h1>STAIR: temporal interpolation</h1><p>{html.escape(metadata['label'])} | Target {metadata['target_date']}</p>
<p>Evaluation: {html.escape(metadata['evaluation_kind'])}. Target-day observations are excluded from all temporal references. Only allowed clear target values enter the fill. Ground truth is used for scoring and plots.</p>
<p>Coverage: {metrics['scored_pixels']} / {metrics['eligible_pixels']} clear held-out pixels. Unresolved: {metrics['unresolved_pixels']}. Observed values preserved exactly.</p>
<img src="temporal_overview.png" alt="Observed target, gaps, temporal reconstruction, error and temporal support">
<table><tr><th>Band</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>Pearson r</th><th>Prediction R²</th></tr>{rows}</table>
<p>Gray pixels are missing or outside the scored region. Visual stretches are fixed and documented in the plotting code. Values are not clipped or smoothed. A two-sided prediction requires an observation within {metadata['max_distance_days']} days on each side. One-sided carry enabled: {metadata['allow_one_sided']}.</p>
<img src="temporal_bands.png" alt="Truth, reconstruction and held-out absolute error for each spectral band">
{trace_html}
<p>Global correction and adaptive correction will be evaluated as separate stages using this same target, grid, holdout, and scales. Differences or improvements will not be artificially amplified.</p>
<p><a href="run.json">Run parameters, source provenance and metrics</a></p>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    return metrics
