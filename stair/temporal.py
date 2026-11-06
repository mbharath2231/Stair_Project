"""Nearest-clear calendar-day interpolation, without spatial adjustment.

Unsupported estimates stay missing. One-sided carries are opt-in and flagged.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

STATUS_NAMES = {0: "unresolved", 1: "interpolated", 2: "prior_only", 3: "future_only"}


def _day(value: str) -> int:
    """Require a calendar date, not an acquisition timestamp."""
    if date.fromisoformat(str(value)).isoformat() != str(value):
        raise ValueError("Dates must use YYYY-MM-DD")
    return int(np.datetime64(value, "D").astype(int))


def _check_window(max_distance_days: int) -> None:
    if isinstance(max_distance_days, bool) or not isinstance(max_distance_days, int) or max_distance_days < 1:
        raise ValueError("max_distance_days must be a positive integer")


@dataclass(frozen=True)
class TemporalResult:
    """Arrays match the target shape; ages are days, missing values are NaN.

    status describes prediction support even on observed pixels.
    filled_origin: 0 unresolved, 1 observed, 2 interpolated, 3 prior, 4 future.
    """
    prediction: np.ndarray
    filled: np.ndarray
    before_age_days: np.ndarray
    after_age_days: np.ndarray
    support_count: np.ndarray
    status: np.ndarray
    filled_origin: np.ndarray


def interpolate_temporal_numpy(
    observations: np.ndarray,
    observation_dates: list[str],
    target_date: str,
    raw_target: np.ndarray | None = None,
    *,
    valid_mask: np.ndarray | None = None,
    max_distance_days: int = 32,
    allow_one_sided: bool = False,
) -> TemporalResult:
    """Interpolate elements from their closest valid before/after observations.

    observations: (time, ...), in consistent physical units on one grid. Dates
    must be unique YYYY-MM-DD (any order); mosaic daily inputs first. The entire
    target day is excluded. valid_mask must match observations exactly; masked
    or nonfinite values are always excluded. The inclusive distance limit applies
    to EACH side. raw_target contains allowed observed values and NaN/masked gaps.
    """
    _check_window(max_distance_days)
    t0 = _day(target_date)
    values = np.asarray(np.ma.asarray(observations, dtype=float).filled(np.nan))
    if values.ndim < 2 or len(observation_dates) != values.shape[0]:
        raise ValueError("observations must have shape (time, ...) matching dates")
    days = np.array([_day(d) for d in observation_dates], dtype=np.int64)
    if len(np.unique(days)) != len(days):
        raise ValueError("observation dates must be unique; mosaic each day first")
    valid = np.isfinite(values)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask)
        if supplied.shape != values.shape or supplied.dtype != bool:
            raise ValueError("valid_mask must be boolean and match observations")
        valid &= supplied
    shape = values.shape[1:]
    before, after = np.full(shape, np.nan), np.full(shape, np.nan)
    before_age, after_age = np.full(shape, np.nan), np.full(shape, np.nan)
    for i in np.argsort(days):
        age = t0 - days[i]
        if 0 < age <= max_distance_days:
            before[valid[i]], before_age[valid[i]] = values[i][valid[i]], age
    for i in np.argsort(days)[::-1]:
        age = days[i] - t0
        if 0 < age <= max_distance_days:
            after[valid[i]], after_age[valid[i]] = values[i][valid[i]], age
    has_before, has_after = np.isfinite(before), np.isfinite(after)
    both = has_before & has_after
    prediction = np.full(shape, np.nan)
    w = np.divide(before_age, before_age + after_age, out=np.zeros(shape), where=both)
    prediction[both] = before[both] + w[both] * (after[both] - before[both])
    status = np.zeros(shape, dtype=np.uint8)
    status[both] = 1
    if allow_one_sided:
        prior, future = has_before & ~has_after, has_after & ~has_before
        prediction[prior], prediction[future] = before[prior], after[future]
        status[prior], status[future] = 2, 3
    if raw_target is None:
        target = np.full(shape, np.nan)
    else:
        target = np.asarray(np.ma.asarray(raw_target, dtype=float).filled(np.nan))
        if target.shape != shape:
            raise ValueError("raw_target must match the observation spatial/band shape")
    observed = np.isfinite(target)
    filled = np.where(observed, target, prediction)
    origin = np.where(status > 0, status + 1, 0).astype(np.uint8)
    origin[observed] = 1
    return TemporalResult(prediction, filled, before_age, after_age,
                          has_before.astype(np.uint8) + has_after.astype(np.uint8),
                          status, origin)


@dataclass(frozen=True)
class TemporalEEResult:
    prediction: Any
    filled: Any
    diagnostics: Any


def interpolate_temporal_ee(
    collection, raw_image, target_date: str, band_name: str = "SR_B5", *,
    max_distance_days: int = 32, allow_one_sided: bool = False,
) -> TemporalEEResult:
    """EE graph with the same selection/fallback policy as the NumPy engine.

    Caller supplies QA-masked, consistently scaled inputs. Time quality shares
    the reflectance mask; masked sentinels protect empty collections. No remote
    evaluation or authentication occurs here. UTC calendar days define distances.
    """
    import ee
    _check_window(max_distance_days)
    _day(target_date)
    t0 = ee.Date(target_date)
    target = raw_image.select(band_name)
    projection = target.projection()
    # Mosaics can have an unbounded geometry; do not clip to image.geometry().
    # Anchor diagnostics to the target's grid/footprint without changing values.
    template = target.unmask(0).multiply(0)
    empty = ee.Image.constant([0, 0, 0]).rename([band_name, "temporal_day", "temporal_quality"]).toDouble()
    empty = empty.setDefaultProjection(projection).updateMask(ee.Image.constant(0))

    def prepare(image, reverse=False):
        value = image.select(band_name).toDouble()
        calendar_day = ee.Date(image.date().format("YYYY-MM-dd"))
        offset = calendar_day.difference(t0, "day")
        day = ee.Image.constant(offset).rename("temporal_day").toDouble().updateMask(value.mask())
        quality = day.multiply(-1 if reverse else 1).rename("temporal_quality")
        return value.addBands(day).addBands(quality)

    before_col = collection.filterDate(t0.advance(-max_distance_days, "day"), t0)
    after_col = collection.filterDate(t0.advance(1, "day"), t0.advance(max_distance_days + 1, "day"))
    before = before_col.map(lambda img: prepare(img)).merge(ee.ImageCollection([empty])).qualityMosaic("temporal_quality")
    after = after_col.map(lambda img: prepare(img, True)).merge(ee.ImageCollection([empty])).qualityMosaic("temporal_quality")
    vb, va = before.select(band_name), after.select(band_name)
    db, da = before.select("temporal_day"), after.select("temporal_day")
    hb, ha = vb.mask().unmask(0), va.mask().unmask(0)
    both = hb.And(ha)
    span = da.subtract(db)
    weight = db.multiply(-1).divide(span)
    prediction = vb.add(weight.multiply(va.subtract(vb))).updateMask(both.And(span.gt(0)))
    status = template.where(both, 1)
    if allow_one_sided:
        prior, future = hb.And(ha.Not()), ha.And(hb.Not())
        prediction = prediction.unmask(vb.updateMask(prior)).unmask(va.updateMask(future))
        status = status.where(prior, 2).where(future, 3)
    prediction = prediction.rename(f"{band_name}_linear").set("system:time_start", t0.millis())
    filled = target.unmask(prediction).rename(f"{band_name}_temporal_patched")
    origin = template.where(status.gt(0), status.add(1)).where(target.mask(), 1)
    diagnostics = ee.Image.cat([
        db.multiply(-1).rename("before_age_days"), da.rename("after_age_days"),
        hb.add(ha).rename("support_count").toByte(),
        status.rename("prediction_status").toByte(), origin.rename("filled_origin").toByte(),
    ])
    return TemporalEEResult(prediction, filled, diagnostics)


def impute_temporal(collection, raw_image, target_date: str, band_name: str = "SR_B5", *,
                    max_distance_days: int = 32, allow_one_sided: bool = False):
    """Compatibility wrapper returning (prediction, filled); strict by default.

    Older callers no longer receive valid zero fallbacks or unlimited-distance
    references. Use interpolate_temporal_ee for support diagnostics.
    """
    result = interpolate_temporal_ee(collection, raw_image, target_date, band_name,
                                     max_distance_days=max_distance_days,
                                     allow_one_sided=allow_one_sided)
    return result.prediction, result.filled
