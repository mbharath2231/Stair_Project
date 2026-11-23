import numpy as np
import pytest

from scripts.compare_nonlinear_spatial import development_splits
from stair.nonlinear_spatial import design


def test_development_blocks_keep_2025_and_nearby_2024_training_dates_out():
    days = np.arange(np.datetime64("2022-01-01"), np.datetime64("2026-01-01"))
    for _, train, validation in development_splits("one_season", days):
        assert (days[train].astype("datetime64[Y]") == np.datetime64("2024", "Y")).all()
        distances = np.abs(days[train, None] - days[None, validation])
        assert (distances > np.timedelta64(32, "D")).all()
    _, train, validation = next(development_splits("twenty_full", days))
    assert days[train].max() == np.datetime64("2022-12-31")
    assert days[validation].max() == np.datetime64("2023-12-31")


def test_neighborhood_inputs_do_not_remove_valid_central_pixels():
    pytest.importorskip("scipy")
    adaptive = np.full((2, 12, 12), np.nan)
    reference = adaptive.copy()
    adaptive[:, 0, 0] = [0.4, 0.15]
    reference[:, 0, 0] = [0.3, 0.1]
    basic = design(adaptive, reference, "2024-06-10", spatial=False)
    contextual = design(adaptive, reference, "2024-06-10", spatial=True)
    np.testing.assert_array_equal(contextual[:len(basic)], basic)
    assert np.isfinite(contextual[:, 0, 0]).all()
    np.testing.assert_allclose(contextual[len(basic):len(basic) + 4, 0, 0], [0.4, 0.15, 0, 0], atol=1e-8)
