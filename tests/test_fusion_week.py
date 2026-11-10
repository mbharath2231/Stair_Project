from datetime import date, timedelta

import numpy as np

from scripts.run_fusion_week import fuse_week


def test_week_uses_bracketed_residuals_and_withheld_landsat_only_for_scoring():
    prior, later = date(2022, 7, 21), date(2022, 7, 29)
    days = [(prior + timedelta(days=i)).isoformat() for i in range(1, 8)]
    left, right = prior.isoformat(), later.isoformat()
    shape = (3, 4, 4)
    landsat = {
        left: np.full(shape, 0.4),
        days[0]: np.full(shape, 0.45),
        right: np.full(shape, 0.6),
        "2022-07-30": np.full(shape, 0.65),
    }
    landsat[right][:, 0, 0] = np.nan  # Force reconstruction of part of the later anchor.
    modis = {day: np.full(shape, 0.35) for day in days}
    modis[left], modis[right] = np.full(shape, 0.3), np.full(shape, 0.4)
    modis[days[-1]][:, 0, 1] = np.nan
    predictions, report, anchors = fuse_week(landsat, modis, days, prior, later)
    assert list(predictions) == days
    np.testing.assert_allclose(predictions[days[0]][:, 1, 1], 0.4625)
    assert np.isfinite(predictions[days[0]][:, 0, 0]).all()
    assert np.isnan(predictions[days[-1]][:, 0, 1]).all()
    assert anchors[right]["unresolved_pixels"] == 0
    assert "reflectance_validation" not in report[days[-1]]

    landsat[days[0]][:] = 0.8
    other_predictions, other_report, other_anchors = fuse_week(landsat, modis, days, prior, later)
    for day in days:
        np.testing.assert_array_equal(predictions[day], other_predictions[day])
    assert anchors == other_anchors
    assert report[days[0]]["reflectance_validation"] != other_report[days[0]]["reflectance_validation"]
