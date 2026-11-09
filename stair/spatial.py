"""Fit global and class-specific offsets on co-registered reflectance grids."""
from __future__ import annotations

import numpy as np


def _assign(points, centers):
    distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)
    return labels, distances[np.arange(len(points)), labels]


def _kmeans(points, count, rng):
    best_centers = None
    best_dispersion = np.inf
    for _ in range(4):
        centers = [points[rng.integers(len(points))]]
        while len(centers) < count:
            _, distance = _assign(points, np.asarray(centers[-1:]))
            nearest = distance if len(centers) == 1 else np.minimum(nearest, distance)
            centers.append(points[rng.choice(len(points), p=nearest / nearest.sum())])
        centers = np.asarray(centers)
        for _ in range(75):
            labels, distance = _assign(points, centers)
            updated = centers.copy()
            for index in range(count):
                members = points[labels == index]
                if len(members):
                    updated[index] = members.mean(axis=0)
                else:
                    updated[index] = points[distance.argmax()]
            if np.array_equal(updated, centers):
                break
            centers = updated
        _, distance = _assign(points, centers)
        dispersion = distance.sum()
        if dispersion < best_dispersion:
            best_centers, best_dispersion = centers, dispersion
    return best_centers, best_dispersion


def segment_reference(reference, *, seed=42, sample_limit=2000, null_draws=20):
    """Select the highest gap statistic over k=2..8 on clear reference pixels.

    Bands are scaled by their sampled ranges. The null distribution is uniform
    in that bounding box. The same sampled pixels and null draws are used at
    each k. Centers are returned in physical reflectance units.
    """
    valid = np.isfinite(reference).all(axis=0)
    pixels = reference[:, valid].T
    if len(pixels) < 8:
        raise ValueError("Gap statistic needs at least eight clear reference pixels")
    rng = np.random.default_rng(seed)
    sample = pixels[rng.choice(len(pixels), min(len(pixels), sample_limit), replace=False)]
    lower = sample.min(axis=0)
    span = np.ptp(sample, axis=0)
    scale = np.where(span > 0, span, 1)
    normalized = (sample - lower) / scale
    maximum = min(8, len(np.unique(normalized, axis=0)) - 1)
    if maximum < 2:
        segments = np.full(valid.shape, np.nan)
        segments[valid] = 0
        return segments, sample.mean(axis=0, keepdims=True), scale, {}
    null_samples = rng.random((null_draws, len(sample), sample.shape[1])) * (span / scale)
    scores, centers_by_count = {}, {}
    for count in range(2, maximum + 1):
        centers, dispersion = _kmeans(normalized, count, rng)
        null_dispersion = [_kmeans(null, count, rng)[1] for null in null_samples]
        scores[count] = float(np.log(null_dispersion).mean() - np.log(dispersion))
        centers_by_count[count] = centers
    centers = centers_by_count[max(scores, key=scores.get)]
    physical_centers = centers * scale + lower
    labels, _ = _assign(pixels / scale, physical_centers / scale)
    segments = np.full(valid.shape, np.nan)
    segments[valid] = labels
    return segments, physical_centers, scale, scores


def fit_spatial_offsets(observed, prediction, reference, segments, *, min_pixels=30):
    """Center temporal gap values on the reference-based expected gap mean.

    For a community, the expected gap mean is the observed clear mean plus
    the reference gap-minus-clear contrast. Subtract the temporal gap mean
    before applying that level. Each pair of means uses common finite pixels.
    A class with insufficient support inherits the global offset. If global
    support is also insufficient, the temporal estimate is retained.
    """
    if observed.ndim != 3 or observed.shape != prediction.shape or observed.shape != reference.shape:
        raise ValueError("Expected aligned band-by-row-by-column arrays")
    if segments.shape != observed.shape[1:] or min_pixels < 1:
        raise ValueError("Expected matching segments and a positive support threshold")
    clear = np.isfinite(observed).all(axis=0)
    valid_reference = np.isfinite(reference).all(axis=0)
    clear_support = clear & valid_reference
    gap_support = ~clear & valid_reference & np.isfinite(prediction).all(axis=0)

    def fit(area):
        free = area & clear_support
        gap = area & gap_support
        stats = {"clear_pixels": int(free.sum()), "gap_pixels": int(gap.sum())}
        if min(stats.values()) < min_pixels:
            return None, stats
        observed_mean = observed[:, free].mean(axis=1)
        reference_contrast = reference[:, gap].mean(axis=1) - reference[:, free].mean(axis=1)
        offset = observed_mean + reference_contrast - prediction[:, gap].mean(axis=1)
        return offset, stats

    global_offset, global_stats = fit(np.ones(clear.shape, dtype=bool))
    global_stats["fallback_to_temporal"] = global_offset is None
    if global_offset is None:
        global_offset = np.zeros(observed.shape[0])
    class_offsets = {}
    class_stats = {}
    for label in np.unique(segments[np.isfinite(segments)]).astype(int):
        offset, stats = fit(segments == label)
        stats["fallback_to_global"] = offset is None
        if offset is None:
            offset = global_offset
        class_offsets[int(label)] = offset.tolist()
        class_stats[int(label)] = stats
    return global_offset.tolist(), class_offsets, {"global": global_stats, "classes": class_stats}
