"""Density estimates and paired agreement for a simulated cloud holdout."""
from __future__ import annotations

import numpy as np

from stair.validation import score_holdout


def matched_samples(truth, predictions, mask, bands):
    """Keep one common set of pixels across all bands and methods."""
    eligible = mask & np.isfinite(truth).all(axis=0)
    common = eligible.copy()
    for image in predictions.values():
        common &= np.isfinite(image).all(axis=0)
    metrics = {
        "eligible_pixels": int(eligible.sum()),
        "common_pixels": int(common.sum()),
        "methods": {},
    }
    if not common.any():
        return common, metrics

    sorted_truth = np.sort(truth[:, common], axis=1)
    for name, image in predictions.items():
        scores = score_holdout(truth, image, common, bands)["bands"]
        delta = (image[:, common] - truth[:, common])
        # For equal-length samples, the mean distance between sorted values
        # is the empirical 1-Wasserstein distance between the distributions.
        distance = np.mean(np.abs(np.sort(image[:, common], axis=1) - sorted_truth), axis=1)
        limits = np.quantile(delta, [0.05, 0.95], axis=1)
        for index, band in enumerate(bands):
            scores[band].update(wasserstein=float(distance[index]),
                                paired_error_p05=float(limits[0, index]),
                                paired_error_p95=float(limits[1, index]))
        metrics["methods"][name] = scores
    return common, metrics


def gaussian_density(samples, grid, bandwidth):
    """Evaluate a normalized Gaussian kernel mixture without a SciPy dependency."""
    density = np.zeros_like(grid)
    for start in range(0, len(samples), 2048):
        offsets = (grid[:, None] - samples[None, start:start + 2048]) / bandwidth
        density += np.exp(-0.5 * offsets**2).sum(axis=1)
    return density / (len(samples) * bandwidth * np.sqrt(2 * np.pi))


def density_panel(ax, series, styles, *, centered=False):
    combined = np.concatenate(list(series.values()))
    # Use one bandwidth on a common grid in each panel. The lower bound is
    # one quantization step of Landsat Collection 2 surface reflectance.
    bandwidth = max(1.06 * combined.std() * len(next(iter(series.values()))) ** -0.2, 0.0000275)
    low, high = combined.min() - 4 * bandwidth, combined.max() + 4 * bandwidth
    if centered:
        extent = max(abs(low), abs(high))
        low, high = -extent, extent
        ax.axvline(0, color="#667384", linewidth=1, linestyle=":")
    grid = np.linspace(low, high, 600)
    for name, values in series.items():
        ax.plot(grid, gaussian_density(values, grid, bandwidth), **styles[name])
    ax.set_xlim(low, high)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e2e7ed", linewidth=0.7)
    ax.spines[["right", "top"]].set_visible(False)
    return float(bandwidth)


def write_holdout_density(path, truth, predictions, mask, bands, target):
    """Write a figure and report on the hidden, originally valid Landsat pixels."""
    from matplotlib.figure import Figure

    common, report = matched_samples(truth, predictions, mask, bands)
    if not common.any():
        return report

    labels = {
        "truth": ("Withheld Landsat", "#172435", "-"),
        "03_temporal": ("Temporal", "#aa6037", (0, (5, 2))),
        "04_global": ("Global", "#5069aa", (0, (3, 2))),
        "05_adaptive": ("Adaptive", "#00816f", "-"),
    }
    styles = {name: {"label": label, "color": color, "linestyle": pattern, "linewidth": 1.8}
              for name, (label, color, pattern) in labels.items()}
    display_bands = {"SR_B5": "Near-infrared", "SR_B4": "Red", "SR_B3": "Green"}
    fig = Figure(figsize=(12.0, 7.3))
    axes = fig.subplots(2, len(bands), squeeze=False)
    fig.subplots_adjust(left=0.078, right=0.973, bottom=0.10, top=0.79, wspace=0.34, hspace=0.57)
    fig.text(0.078, 0.96, f"Cloud-fill validation  |  Clayton Township, {target}",
             fontsize=19, fontweight="bold", color="#172435", va="top")
    fig.text(0.078, 0.902,
             f"Simulated cloud holdout  ·  {report['common_pixels']:,} of {report['eligible_pixels']:,} "
             "hidden clear pixels shared across methods",
             fontsize=11, color="#455267", va="top")
    truth_values = truth[:, common]
    filled_values = {name: image[:, common] for name, image in predictions.items()}
    bandwidths = {}
    for index, band in enumerate(bands):
        distribution, residual = axes[:, index]
        series = {"truth": truth_values[index],
                  **{name: values[index] for name, values in filled_values.items()}}
        paired = {name: values[index] - truth_values[index] for name, values in filled_values.items()}
        bandwidths[band] = {
            "reflectance": density_panel(distribution, series, styles),
            "paired_error": density_panel(residual, paired, styles, centered=True),
        }
        name = display_bands.get(band, band)
        distribution.set_title(name, loc="left", fontweight="bold", fontsize=12)
        distribution.set_xlabel("Surface reflectance")
        residual.set_title(f"{name} paired error", loc="left", fontweight="bold", fontsize=12)
        residual.set_xlabel("Filled − withheld reflectance")
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability density")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.072, 0.88), ncols=4, frameon=False)
    fig.text(0.078, 0.024, "The lower curves compare each prediction with its Landsat value at the same pixel.",
             fontsize=10, color="#455267")
    report["gaussian_kde_bandwidth"] = bandwidths
    fig.savefig(path, dpi=180, facecolor="white")
    return report
