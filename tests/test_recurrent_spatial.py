import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.compare_recurrent_spatial import consecutive_blocks
from scripts.compare_spatial_history import scenarios_for_images
from stair.cnn_spatial import cnn_features
from stair.recurrent_spatial import KERNELS, RecurrentCorrection, sequence_frames


def test_sequence_uses_signed_day_offsets_and_excludes_hidden_dates():
    image = np.full((2, 5, 7), 0.2, dtype=np.float32)
    images = {day: image.copy() for day in ("2025-06-01", "2025-06-09", "2025-06-12", "2025-06-24")}
    images["2025-06-24"][:, 2, 3] = np.nan
    first, second = consecutive_blocks(images, 2)
    assert first == ("2025-06-01", "2025-06-09")
    frames, dates = sequence_frames(images, "2025-06-12", window=32, excluded=first)
    assert second == ("2025-06-12", "2025-06-24")
    assert dates == (None, None, "2025-06-24", None)
    np.testing.assert_array_equal(frames[0], 0)
    np.testing.assert_array_equal(frames[2, :3, 2, 3], 0)
    assert frames[2, 3, 2, 3] == 12 / 32


def test_consecutive_holdout_excludes_neighbors_from_stair_inputs():
    shape = (40, 40)
    gradient = np.linspace(0, 0.08, shape[0] * shape[1]).reshape(shape)
    reflectance = np.stack((0.4 + gradient, 0.15 - gradient / 2))
    target, hidden = "2025-06-12", "2025-06-14"
    images = {"2025-06-02": reflectance.copy(), target: reflectance.copy(),
              hidden: reflectance.copy(), "2025-06-21": reflectance.copy()}
    exclusions = {target: (target, hidden)}
    first = next(scenarios_for_images(images, ("cloud",), [], excluded_by_target=exclusions))
    images[hidden] += 0.4
    second = next(scenarios_for_images(images, ("cloud",), [], excluded_by_target=exclusions))
    assert first[6] not in exclusions[target]
    np.testing.assert_array_equal(first[-1]["adaptive"], second[-1]["adaptive"])


@pytest.mark.parametrize("kind", KERNELS)
def test_recurrent_residual_starts_from_adaptive(kind):
    image = np.full((2, 7, 9), 0.2, dtype=np.float32)
    image[:, 1, 1] = np.nan
    frames, _ = sequence_frames({"2025-05-15": image}, "2025-06-01", window=32)
    static, _ = cnn_features(image, image, "2025-06-01")
    predicted = RecurrentCorrection(kind).corrected(frames[None], static[None], image[None])[0]
    np.testing.assert_array_equal(predicted, image)
