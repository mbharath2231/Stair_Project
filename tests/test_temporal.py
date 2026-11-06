import numpy as np
import pytest
from stair.temporal import interpolate_temporal_numpy as interpolate
from stair.validation import make_holdout, score_holdout


def test_selects_older_clear_pixel_when_nearest_scene_cloudy():
    values = np.array([[.2, .2], [np.nan, .4], [.8, .8]])
    result = interpolate(values, ["2022-07-01", "2022-07-05", "2022-07-11"], "2022-07-06")
    np.testing.assert_allclose(result.prediction, [.5, .4 + (.8-.4)/6])
    np.testing.assert_equal(result.before_age_days, [5, 1])
    np.testing.assert_equal(result.after_age_days, [5, 5])


def test_excludes_entire_target_day_and_handles_unsorted_cross_year_dates():
    result = interpolate(np.array([[.9], [.3], [999.]]),
                         ["2023-01-04", "2022-12-29", "2023-01-01"], "2023-01-01")
    np.testing.assert_allclose(result.prediction, [.6])


@pytest.mark.parametrize("values,dates,strict,carry,status", [
    ([[.4]], ["2022-07-01"], np.nan, .4, 2),
    ([[.7]], ["2022-07-11"], np.nan, .7, 3),
    ([[np.nan]], ["2022-07-01"], np.nan, np.nan, 0),
])
def test_one_sided_policies(values, dates, strict, carry, status):
    np.testing.assert_allclose(interpolate(np.array(values), dates, "2022-07-06").prediction, [strict])
    result = interpolate(np.array(values), dates, "2022-07-06", allow_one_sided=True)
    np.testing.assert_allclose(result.prediction, [carry])
    np.testing.assert_equal(result.status, [status])


def test_empty_series_preserves_observed_but_never_fabricates_zero():
    result = interpolate(np.empty((0, 2)), [], "2022-07-06", np.array([0., np.nan]))
    np.testing.assert_equal(result.filled, [0., np.nan])
    np.testing.assert_equal(result.status, [0, 0])
    np.testing.assert_equal(result.filled_origin, [1, 0])


def test_inclusive_window_rejects_stale_observations():
    result = interpolate(np.array([[.2], [.8]]), ["2022-07-01", "2022-07-11"],
                         "2022-07-06", max_distance_days=5)
    np.testing.assert_allclose(result.prediction, [.5])
    assert np.isnan(interpolate(np.array([[.2], [.8]]), ["2022-07-01", "2022-07-11"],
                               "2022-07-06", max_distance_days=4).prediction).all()


def test_masked_arrays_explicit_validity_and_infinite_values():
    values = np.ma.array([[.2, .2, np.inf], [.3, .3, .4], [.8, .8, .8]],
                         mask=[[False, False, False], [True, False, False], [False, False, False]])
    mask = np.ones(values.shape, dtype=bool)
    mask[1, 1] = False
    result = interpolate(values, ["2022-07-01", "2022-07-04", "2022-07-11"], "2022-07-06", valid_mask=mask)
    np.testing.assert_allclose(result.prediction, [.5, .5, .4 + 2/7*.4])


def test_clear_pixels_unchanged_and_linear_trajectory_exact():
    rng = np.random.default_rng(42)
    intercept, slope = rng.uniform(.1, .4, (2, 3, 5, 6))
    slope *= .01
    values = np.stack([intercept + d*slope for d in [0, 4, 12]])
    truth = intercept + 7*slope
    observed = truth.copy()
    observed[:, 1:4, 2:5] = np.nan
    result = interpolate(values, ["2022-07-01", "2022-07-05", "2022-07-13"], "2022-07-08", observed)
    np.testing.assert_allclose(result.prediction, truth, atol=1e-14)
    np.testing.assert_array_equal(result.filled[np.isfinite(observed)], observed[np.isfinite(observed)])


@pytest.mark.parametrize("kwargs", [{"max_distance_days": 0}, {"max_distance_days": True},
                                   {"valid_mask": np.ones((2, 1))}, {"raw_target": np.zeros((2,))}])
def test_reject_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        interpolate(np.array([[.1], [.2]]), ["2022-07-01", "2022-07-11"], "2022-07-06", **kwargs)


def test_reject_duplicate_dates_and_bad_dates():
    with pytest.raises(ValueError):
        interpolate(np.zeros((2, 1)), ["2022-07-01"]*2, "2022-07-06")
    with pytest.raises(ValueError):
        interpolate(np.zeros((1, 1)), ["bad"], "2022-07-06")


def test_scoring_never_includes_known_clear_pixels_and_reports_missing():
    truth = np.ones((3, 3, 3))
    pred = truth.copy()
    gap = np.zeros((3, 3), dtype=bool)
    gap[1, :] = True
    pred[:, 0, :] = 99
    pred[:, 1, 0] = 1.2
    pred[:, 1, 1] = np.nan
    scores = score_holdout(truth, pred, gap, ["NIR", "Red", "Green"])
    assert scores["scored_pixels"] == 2
    assert scores["eligible_pixels"] == 3
    assert scores["bands"]["NIR"]["rmse"] == pytest.approx(np.sqrt(.02))
    assert scores["bands"]["NIR"]["r2"] is None
    missing = score_holdout(truth, pred*np.nan, gap, ["NIR", "Red", "Green"])
    assert missing["bands"]["NIR"]["rmse"] is None


@pytest.mark.parametrize("pattern", ["cloud", "stripes", "scene"])
def test_holdout_is_deterministic(pattern):
    np.testing.assert_array_equal(make_holdout((31, 32), pattern), make_holdout((31, 32), pattern))
