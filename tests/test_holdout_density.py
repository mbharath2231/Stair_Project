import numpy as np

from stair.holdout_density import gaussian_density, matched_samples


def test_identical_distributions_can_hide_wrong_pixels_and_use_same_support():
    truth = np.array([[[0.1, 0.2, 0.3, 0.7, 0.9]]])
    mask = np.array([[True, True, True, True, False]])
    predictions = {
        "matched": truth.copy(),
        "shuffled": np.array([[[0.3, 0.2, 0.1, np.nan, 0.4]]]),
    }
    common, metrics = matched_samples(truth, predictions, mask, ["NIR"])
    np.testing.assert_array_equal(common, [[True, True, True, False, False]])
    assert metrics["eligible_pixels"] == 4 and metrics["common_pixels"] == 3
    assert metrics["methods"]["matched"]["NIR"]["rmse"] == 0
    assert metrics["methods"]["shuffled"]["NIR"]["wasserstein"] == 0
    assert metrics["methods"]["shuffled"]["NIR"]["rmse"] > 0


def test_gaussian_density_normalizes_even_for_a_constant_sample():
    grid = np.linspace(-0.1, 0.1, 1001)
    density = gaussian_density(np.zeros(20), grid, bandwidth=0.01)
    integral = np.sum((density[1:] + density[:-1]) * np.diff(grid) / 2)
    np.testing.assert_allclose(integral, 1.0, atol=1e-8)
