import numpy as np

from stair.ndvi import DailyNDVI


def test_ndvi_uses_bracketing_residuals_and_labels_fallbacks_without_target_leakage():
    def bands(nir, red):
        return np.array([[nir], [red]], dtype=float)

    earlier, target, later = "2024-03-01", "2024-03-09", "2024-03-17"
    landsat = {
        earlier: bands([0.4] * 4, [0.2] * 4),
        target: bands([0.9, np.nan, np.nan, np.nan], [0.1, np.nan, np.nan, np.nan]),
        later: bands([0.6] * 4, [0.2] * 4),
    }
    modis = {
        earlier: bands([0.3] * 4, [0.15] * 4),
        target: bands([0.4, 0.4, 0.4, np.nan], [0.15, 0.15, 0.15, np.nan]),
        later: bands([0.4, 0.4, np.nan, 0.4], [0.15, 0.15, np.nan, 0.15]),
    }
    model = DailyNDVI(landsat, modis, min_clear_fraction=0.25)
    prediction, flags = model.predict(target, fallback=True)
    np.testing.assert_array_equal(flags, [[1, 2, 3, 4]])
    np.testing.assert_allclose(prediction, [[0.8, 0.35 / 0.75, 0.3 / 0.7, 0.3 / 0.7]])
    _, strict = model.predict(target)
    np.testing.assert_array_equal(strict, [[1, 2, 0, 0]])

    held_out, flags = model.predict(target, observed=False, fallback=True)
    np.testing.assert_array_equal(flags, [[2, 2, 3, 4]])
    landsat[target] = bands([0.2] * 4, [0.7] * 4)
    changed = DailyNDVI(landsat, modis).predict(target, observed=False, fallback=True)[0]
    np.testing.assert_array_equal(held_out, changed)
