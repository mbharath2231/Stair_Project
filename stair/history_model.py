"""Date-balanced ridge correction of retrospective NIR/Red residual fusion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

FEATURES = ("intercept", "fused_nir", "fused_red", "modis_nir", "modis_red",
            "season_sin", "season_cos")
REFLECTANCE_SCALES = np.array([0.5, 0.25, 0.5, 0.25])


def features(fused, coarse, day):
    """Fixed physical scaling keeps both training histories on one feature scale."""
    fused, coarse = np.asarray(fused), np.asarray(coarse)
    if fused.shape != coarse.shape or fused.ndim != 3 or fused.shape[0] != 2:
        raise ValueError("Expected matching (2, height, width) NIR/Red arrays")
    calendar_day = date.fromisoformat(day)
    start = date(calendar_day.year, 1, 1)
    year_days = (date(calendar_day.year + 1, 1, 1) - start).days
    phase = 2 * np.pi * (calendar_day - start).days / year_days
    scaled = np.concatenate((fused, coarse)).astype(float) / REFLECTANCE_SCALES[:, None, None]
    constants = np.broadcast_to(np.array([1.0, np.sin(phase), np.cos(phase)])[:, None, None],
                               (3, *fused.shape[1:]))
    return np.concatenate((constants[:1], scaled, constants[1:]))


@dataclass(frozen=True)
class HistoryCorrection:
    coefficients: np.ndarray
    ridge: float
    training_dates: tuple[str, ...]
    training_pixel_dates: int

    def predict(self, fused, coarse, day):
        predictors = features(fused, coarse, day)
        valid = np.isfinite(predictors).all(axis=0)
        result = np.full(fused.shape, np.nan, dtype=float)
        result[:, valid] = fused[:, valid] + self.coefficients.T @ predictors[:, valid]
        return result

    def save(self, path):
        np.savez_compressed(path, coefficients=self.coefficients, ridge=self.ridge,
                            features=FEATURES, training_dates=self.training_dates,
                            training_pixel_dates=self.training_pixel_dates)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            coefficients = saved["coefficients"]
            ridge = float(saved["ridge"])
            dates = tuple(saved["training_dates"].tolist())
            count = int(saved["training_pixel_dates"])
            if (saved["features"].tolist() != list(FEATURES)
                    or coefficients.shape != (len(FEATURES), 2)
                    or not np.isfinite(coefficients).all() or not np.isfinite(ridge) or ridge <= 0
                    or not dates or len(set(dates)) != len(dates) or count < len(dates)):
                raise ValueError("Invalid historical correction model")
        return cls(coefficients, ridge, dates, count)


def fit_correction(samples, *, ridge=0.001):
    """Fit truth minus fusion, giving every supported acquisition total weight one.

    samples yields (date, withheld-day fusion, daily MODIS, QA-clear Landsat).
    The caller constructs features without the target acquisition and restricts
    labels and all training context to years preceding the test year.
    """
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    gram = np.zeros((len(FEATURES), len(FEATURES)))
    cross = np.zeros((len(FEATURES), 2))
    dates, seen, count = [], set(), 0
    for day, fused, coarse, truth in samples:
        if day in seen:
            raise ValueError(f"Duplicate training date {day}")
        seen.add(day)
        predictors = features(fused, coarse, day)
        if truth.shape != fused.shape:
            raise ValueError("Training truth must match the fusion grid")
        valid = np.isfinite(predictors).all(axis=0) & np.isfinite(truth).all(axis=0)
        n = int(valid.sum())
        if not n:
            continue
        x = predictors[:, valid].T
        errors = (truth[:, valid] - fused[:, valid]).T
        gram += x.T @ x / n
        cross += x.T @ errors / n
        dates.append(day)
        count += n
    if not dates:
        raise ValueError("No supported training dates")
    penalty = np.eye(len(FEATURES)) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(gram / len(dates) + penalty, cross / len(dates))
    return HistoryCorrection(coefficients, ridge, tuple(dates), count)
