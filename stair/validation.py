"""Holdout definitions and scoring, kept outside the interpolation engine."""
from __future__ import annotations
import numpy as np


def make_holdout(shape: tuple[int, int], pattern: str = "cloud") -> np.ndarray:
    """A fixed spatial mask independent of spectral values or algorithm errors."""
    height, width = shape
    y, x = np.mgrid[:height, :width]
    if pattern == "cloud":
        return (((x / width - .49) / .23)**2 + ((y / height - .51) / .28)**2 < 1) | (
            ((x / width - .65) / .14)**2 + ((y / height - .40) / .17)**2 < 1)
    if pattern == "stripes":
        return ((x + y // 2) % max(10, width // 7)) < max(3, width // 22)
    if pattern == "scene":
        return np.ones(shape, dtype=bool)
    raise ValueError("pattern must be cloud, stripes, or scene")


def score_holdout(truth: np.ndarray, prediction: np.ndarray, holdout: np.ndarray,
                  bands: list[str]) -> dict:
    """Score only held-out, originally clear pixels, using common band support.

    A method with missing predictions reports its coverage; missing scores are
    None rather than zero. Finite negative reflectance is not silently removed.
    """
    if truth.shape != prediction.shape or truth.ndim != 3 or len(bands) != truth.shape[0]:
        raise ValueError("truth/prediction must have matching (band, y, x) shapes")
    if holdout.shape != truth.shape[1:] or holdout.dtype != bool:
        raise ValueError("holdout must be a matching boolean spatial mask")
    eligible = holdout & np.all(np.isfinite(truth), axis=0)
    valid = eligible & np.all(np.isfinite(prediction), axis=0)
    n, total = int(valid.sum()), int(eligible.sum())
    metrics = {"eligible_pixels": total, "scored_pixels": n,
               "unresolved_pixels": total - n,
               "coverage_fraction": n / total if total else None, "bands": {}}
    for i, name in enumerate(bands):
        if not n:
            metrics["bands"][name] = {key: None for key in ("rmse", "mae", "bias", "pearson_r", "r2")}
            continue
        actual, pred = truth[i][valid].astype(float), prediction[i][valid].astype(float)
        error = pred - actual
        sst = float(np.sum((actual - actual.mean())**2))
        r = float(np.corrcoef(actual, pred)[0, 1]) if n > 1 and np.std(actual) > 0 and np.std(pred) > 0 else None
        metrics["bands"][name] = {
            "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(abs(error))),
            "bias": float(error.mean()), "pearson_r": r,
            "r2": float(1 - np.sum(error**2) / sst) if sst > 0 else None,
        }
    return metrics
