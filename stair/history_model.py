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
    if (coarse is not None) != (source == "fusion"):
        raise ValueError("Only the fused model accepts a second reflectance input")


def fit_correction(samples, *, ridge=0.001, input_source="fusion"):
    """Fit Landsat truth minus baseline, weighting each acquisition equally.

    samples yields (date, baseline, optional extra MODIS, QA-clear Landsat).
    The caller constructs features without the target acquisition and restricts
    labels and all training context to years preceding the test year.
    """
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    columns = len(FEATURES_BY_SOURCE[input_source])
    gram = np.zeros((columns, columns))
    cross = np.zeros((columns, 2))
    dates, seen, count = [], set(), 0
    for day, baseline, coarse, truth in samples:
        if day in seen:
            raise ValueError(f"Duplicate training date {day}")
        seen.add(day)
        check_inputs(input_source, coarse)
        predictors = features(baseline, coarse, day)
        if truth.shape != baseline.shape:
            raise ValueError("Training truth must match the baseline grid")
        valid = np.isfinite(predictors).all(axis=0) & np.isfinite(truth).all(axis=0)
        n = int(valid.sum())
        if not n:
            continue
        x = predictors[:, valid].T
        errors = (truth[:, valid] - baseline[:, valid]).T
        gram += x.T @ x / n
        cross += x.T @ errors / n
        dates.append(day)
        count += n
    if not dates:
        raise ValueError("No supported training dates")
    penalty = np.eye(columns) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(gram / len(dates) + penalty, cross / len(dates))
    return HistoryCorrection(coefficients, ridge, tuple(dates), count, input_source)
