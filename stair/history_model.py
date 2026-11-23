"""Date-balanced ridge correction of retrospective NIR/red estimates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

FEATURES = ("intercept", "fused_nir", "fused_red", "modis_nir", "modis_red",
            "season_sin", "season_cos")
REFLECTANCE_SCALES = np.array([0.5, 0.25, 0.5, 0.25])
FEATURES_BY_SOURCE = {
    "fusion": FEATURES,
    "landsat_only": ("intercept", "interpolated_nir", "interpolated_red", *FEATURES[-2:]),
    "modis_only": ("intercept", "modis_nir", "modis_red", *FEATURES[-2:]),
    "spatial": ("intercept", "adaptive_nir", "adaptive_red", "reference_nir", "reference_red", *FEATURES[-2:]),
}


def features(baseline, coarse, day):
    """Fixed physical scaling keeps both training histories on one feature scale."""
    baseline = np.asarray(baseline)
    if baseline.ndim != 3 or baseline.shape[0] != 2:
        raise ValueError("Expected a (2, height, width) NIR/Red baseline")
    if coarse is not None and baseline.shape != np.shape(coarse):
        raise ValueError("Expected matching (2, height, width) NIR/Red arrays")
    calendar_day = date.fromisoformat(day)
    start = date(calendar_day.year, 1, 1)
    year_days = (date(calendar_day.year + 1, 1, 1) - start).days
    phase = 2 * np.pi * (calendar_day - start).days / year_days
    reflectance = baseline if coarse is None else np.concatenate((baseline, coarse))
    scaled = reflectance.astype(float) / REFLECTANCE_SCALES[:len(reflectance), None, None]
    constants = np.broadcast_to(np.array([1.0, np.sin(phase), np.cos(phase)])[:, None, None],
                               (3, *baseline.shape[1:]))
    return np.concatenate((constants[:1], scaled, constants[1:]))


@dataclass(frozen=True)
class HistoryCorrection:
    coefficients: np.ndarray
    ridge: float
    training_dates: tuple[str, ...]
    training_pixel_dates: int
    input_source: str = "fusion"

    def predict(self, fused, coarse, day):
        check_inputs(self.input_source, coarse)
        predictors = features(fused, coarse, day)
        valid = np.isfinite(predictors).all(axis=0)
        result = np.full(fused.shape, np.nan, dtype=float)
        result[:, valid] = fused[:, valid] + self.coefficients.T @ predictors[:, valid]
        return result

    def save(self, path):
        np.savez_compressed(path, coefficients=self.coefficients, ridge=self.ridge,
                            features=FEATURES_BY_SOURCE[self.input_source], input_source=self.input_source,
                            training_dates=self.training_dates,
                            training_pixel_dates=self.training_pixel_dates)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            coefficients = saved["coefficients"]
            ridge = float(saved["ridge"])
            dates = tuple(saved["training_dates"].tolist())
            count = int(saved["training_pixel_dates"])
            source = saved["input_source"].item() if "input_source" in saved else "fusion"
            names = FEATURES_BY_SOURCE.get(source)
            if (names is None or saved["features"].tolist() != list(names)
                    or coefficients.shape != (len(names), 2)
                    or not np.isfinite(coefficients).all() or not np.isfinite(ridge) or ridge <= 0
                    or not dates or len(set(dates)) != len(dates) or count < len(dates)):
                raise ValueError("Invalid historical correction model")
        return cls(coefficients, ridge, dates, count, source)


def check_inputs(source, coarse):
    if coarse is None and source in ("fusion", "spatial"):
        raise ValueError(f"The {source} model requires a second reflectance input")
    if coarse is not None and source in ("landsat_only", "modis_only"):
        raise ValueError(f"The {source} model accepts no second reflectance input")


class RidgeFit:
    """Accumulate date-weighted sufficient statistics without retaining pixel arrays."""

    def __init__(self, *, ridge=0.001, input_source="fusion"):
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        self.ridge = ridge
        self.input_source = input_source
        columns = len(FEATURES_BY_SOURCE[input_source])
        self.gram = np.zeros((columns, columns))
        self.cross = np.zeros((columns, 2))
        self.dates, self.seen, self.count = [], set(), 0

    def add(self, day, baseline, secondary, truth):
        if day in self.seen:
            raise ValueError(f"Duplicate training date {day}")
        check_inputs(self.input_source, secondary)
        predictors = features(baseline, secondary, day)
        if truth.shape != baseline.shape:
            raise ValueError("Training truth must match the baseline grid")
        self.seen.add(day)
        valid = np.isfinite(predictors).all(axis=0) & np.isfinite(truth).all(axis=0)
        n = int(valid.sum())
        if not n:
            return
        x = predictors[:, valid].T
        errors = (truth[:, valid] - baseline[:, valid]).T
        self.gram += x.T @ x / n
        self.cross += x.T @ errors / n
        self.dates.append(day)
        self.count += n

    def finish(self):
        count = len(self.dates)
        if not count:
            raise ValueError("No supported training dates")
        penalty = np.eye(len(self.gram)) * self.ridge
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(self.gram / count + penalty, self.cross / count)
        return HistoryCorrection(coefficients, self.ridge, tuple(self.dates), self.count, self.input_source)


def fit_correction(samples, *, ridge=0.001, input_source="fusion"):
    """Fit truth minus baseline from (date, baseline, optional secondary input, truth)."""
    fit = RidgeFit(ridge=ridge, input_source=input_source)
    for sample in samples:
        fit.add(*sample)
    return fit.finish()
