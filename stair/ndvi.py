"""NDVI and date-bracketed Landsat–MODIS residual fusion."""
from __future__ import annotations

import numpy as np

from .temporal import interpolate_temporal_numpy


def ndvi(reflectance):
    """Convert reflectance ordered NIR, Red to a single-band NDVI array."""
    nir, red = reflectance[:2]
    total = nir + red
    valid = np.isfinite(total) & (total > 0) & (nir >= 0) & (red >= 0)
    return np.divide(nir - red, total, out=np.full_like(total, np.nan), where=valid)[None]


class DailyNDVI:
    """Match clear reflectance pairs and interpolate each pixel's residual."""

    def __init__(self, landsat, modis, *, max_age=32, min_clear_fraction=0.60):
        self.landsat = landsat
        self.modis = modis
        self.max_age = max_age
        self.shape = next(iter(landsat.values())).shape
        self.missing = np.full(self.shape, np.nan, dtype=np.float32)
        self.landsat_days = sorted(landsat)
        self.landsat_values = np.stack([landsat[d] for d in self.landsat_days])
        self.pair_days = []
        pairs = []
        for day in self.landsat_days:
            if day not in modis:
                continue
            fine = landsat[day]
            clear = np.isfinite(ndvi(fine)[0])
            if clear.mean() < min_clear_fraction:
                continue
            coarse = modis[day]
            valid = clear & np.isfinite(coarse).all(axis=0)
            if not valid.any():
                continue
            self.pair_days.append(day)
            pairs.append(np.where(valid[None], np.concatenate((fine - coarse, fine)), np.nan))
        self.pairs = np.stack(pairs) if pairs else np.empty((0, 2 * self.shape[0], *self.shape[1:]))
        self.pair_ordinals = np.asarray(self.pair_days, dtype="datetime64[D]")
        self.landsat_ordinals = np.asarray(self.landsat_days, dtype="datetime64[D]")

    def _interpolate(self, values, dates, ordinals, day, *, allow_one_sided=False):
        target = np.datetime64(day)
        age = np.timedelta64(self.max_age, "D")
        start = ordinals.searchsorted(target - age)
        end = ordinals.searchsorted(target + age, side="right")
        return interpolate_temporal_numpy(values[start:end], dates[start:end], day,
                                          max_distance_days=self.max_age, allow_one_sided=allow_one_sided)

    def predict(self, day, *, observed=True, fallback=False):
        """Return NDVI and its source: missing, observed, two-pair, one-pair, or Landsat interpolation."""
        actual = ndvi(self.landsat.get(day, self.missing))[0] if observed else self.missing[0]
        use_observed = np.isfinite(actual)
        if use_observed.all():
            return actual, np.ones(actual.shape, dtype=np.uint8)

        coarse = self.modis.get(day, self.missing)
        fit = self._interpolate(self.pairs[:, :2], self.pair_days, self.pair_ordinals, day,
                                allow_one_sided=fallback)
        fused = ndvi(coarse + fit.prediction)[0]
        flags = np.where(np.isfinite(fused) & (fit.status[0] == 1), 2, 0).astype(np.uint8)
        values = np.where(flags, fused, np.nan)
        if fallback:
            recent = (fit.before_age_days[0] <= 16) | (fit.after_age_days[0] <= 16)
            use_one_pair = (fit.status[0] >= 2) & recent & np.isfinite(fused)
            values[use_one_pair], flags[use_one_pair] = fused[use_one_pair], 3
            if ((flags == 0) & ~use_observed).any():
                interpolated = self._interpolate(self.landsat_values, self.landsat_days, self.landsat_ordinals, day)
                landsat_only = ndvi(interpolated.prediction)[0]
                use_landsat = (flags == 0) & np.isfinite(landsat_only)
                values[use_landsat], flags[use_landsat] = landsat_only[use_landsat], 4
        return np.where(use_observed, actual, values), np.where(use_observed, 1, flags).astype(np.uint8)

    def holdout_reflectance(self, day):
        """Return NIR/Red controls with the entire target Landsat day excluded."""
        fitted = self._interpolate(self.pairs, self.pair_days, self.pair_ordinals, day).prediction
        landsat_only = self._interpolate(self.landsat_values, self.landsat_days, self.landsat_ordinals, day).prediction
        coarse = self.modis.get(day, self.missing)
        return {
            "fusion": coarse + fitted[:2],
            "same_pairs_landsat": fitted[2:],
            "landsat_only": landsat_only,
            "modis_only": coarse,
        }

    def holdout(self, day):
        """Predict NDVI with the entire target Landsat day excluded."""
        return {name: ndvi(image) for name, image in self.holdout_reflectance(day).items()}
