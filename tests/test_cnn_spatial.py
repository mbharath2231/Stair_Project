import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.compare_spatial_cnn import TrainingScenes, epoch_models
from scripts.tune_spatial_cnn import forward_folds
from stair.cnn_spatial import SpatialCNN, cnn_features


def test_missing_reflectance_has_explicit_input_masks():
    adaptive = np.full((2, 9, 11), 0.2, dtype=np.float32)
    reference = adaptive.copy()
    adaptive[:, 4, 4] = np.nan
    reference[:, 6, 6] = np.nan
    features, valid = cnn_features(adaptive, reference, "2024-06-12")

    assert features.shape == (8, 9, 11)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features[-2:, 4, 4], [0, 1])
    np.testing.assert_array_equal(features[-2:, 6, 6], [1, 0])
    prediction = SpatialCNN().predict(adaptive, reference, "2024-06-12")
    np.testing.assert_array_equal(prediction[:, valid], adaptive[:, valid])
    assert np.isnan(prediction[:, ~valid]).all()


def test_masked_cnn_loss_ignores_unscored_targets():
    adaptive = np.full((1, 2, 9, 9), 0.2, dtype=np.float32)
    predictors, _ = cnn_features(adaptive[0], adaptive[0], "2024-06-12")
    mask = np.zeros((1, 9, 9), dtype=bool)
    mask[:, 3:6, 3:6] = True
    truth = np.where(mask[:, None], adaptive + 0.05, np.nan)
    scenes = TrainingScenes(np.array(["2024-06-12"]), predictors[None], adaptive, truth, mask)

    _, trained = next(epoch_models(scenes, np.array([0]), 1, torch.device("cpu"), 1))
    prediction = trained.corrected(scenes.predictors, adaptive)
    assert np.isfinite(prediction).all()
    assert np.all(prediction[:, :, 4, 4] > adaptive[:, :, 4, 4])


def test_forward_folds_separate_acquisitions_by_year():
    days = np.array(["2021-04-03", "2022-04-10", "2022-10-05", "2023-03-15",
                     "2024-04-16", "2025-03-15"], dtype="datetime64[D]")
    folds = list(forward_folds(days))
    assert [year for year, _, _ in folds] == [2022, 2023, 2024]
    for year, train, validation in folds:
        assert days[train].max() < days[validation].min()
        np.testing.assert_array_equal(days[validation].astype("datetime64[Y]"), np.datetime64(str(year)))
        assert 5 not in train and 5 not in validation
