import numpy as np

from scripts import compare_spatial_history as experiment
from scripts.evaluate_recurrent_quality import METHODS as QUALITY_METHODS, change_signs, edge_counts, edge_metrics
from scripts.evaluate_spatial_history import gap_depth, ssim_windows
from stair.history_model import RidgeFit
from stair.landsat_harmonization import harmonize_observations, to_oli


def test_hidden_pixels_do_not_change_temporal_or_spatial_predictions(monkeypatch, tmp_path):
    shape = (40, 40)
    variation = np.linspace(0, 0.08, np.prod(shape)).reshape(shape)
    reflectance = np.stack([0.4 + variation, 0.15 - variation / 2])
    day = "2025-06-12"
    images = {"2025-06-01": reflectance - 0.01, day: reflectance.copy(), "2025-06-22": reflectance + 0.01}
    monkeypatch.setattr(experiment, "read_cache", lambda *_: ({}, images, {}))
    args = (tmp_path, 2025, ("cloud",), [])
    first = next(row for row in experiment.scenarios(*args) if row[0] == day)
    images[day][:, first[2]] += 0.25
    second = next(row for row in experiment.scenarios(*args) if row[0] == day)
    assert first[6] != day
    for name in first[-1]:
        np.testing.assert_array_equal(first[-1][name], second[-1][name])


def test_spatial_calibration_changes_only_the_supported_holdout():
    base = np.broadcast_to(np.array([0.45, 0.20])[:, None, None], (2, 2, 2)).copy()
    reference = base.copy()
    reference[:, 1, 1] = np.nan
    fit = RidgeFit(input_source="spatial")
    fit.add("2024-06-01", base, reference, base + 0.03)
    holdout = np.array([[True, False], [False, True]])
    adjusted, count = experiment.calibrate(fit.finish(), base, reference, "2025-06-01", holdout)
    assert count == 1
    np.testing.assert_allclose(adjusted[:, 0, 0], base[:, 0, 0] + 0.03)
    np.testing.assert_array_equal(adjusted[:, 1, 1], base[:, 1, 1])
    np.testing.assert_array_equal(adjusted[:, ~holdout], base[:, ~holdout])


def test_gap_depth_uses_euclidean_distance_to_outside_cell_centers():
    mask = np.ones((3, 4), dtype=bool)
    mask[0, 0] = False
    mask[2, 3] = False
    expected = 30 * np.sqrt([[0, 1, 4, 4], [1, 2, 2, 1], [4, 4, 1, 0]])
    np.testing.assert_allclose(gap_depth(mask, 30), expected)


def test_ssim_distinguishes_a_spatial_reversal_with_identical_histogram():
    reference = np.array([[0.2, 0.3, 0.6, 0.8]])
    np.testing.assert_allclose(ssim_windows(reference, reference), [1.0])
    assert ssim_windows(reference, reference[:, ::-1])[0] < 0


def test_transition_scores_count_matches_extras_and_misses():
    observed = np.array([0.15, 0.04, 0.15, 0.03])
    predicted = np.array([0.17, 0.12, 0.06, 0.03])
    summary = edge_metrics(edge_counts(observed, predicted, 0.10))
    assert summary == {"true_positive": 1, "false_positive": 1, "false_negative": 1,
                       "precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_change_direction_requires_shared_support_and_reference_change():
    observed = np.zeros((5, 10))
    previous = dict.fromkeys(("truth", *QUALITY_METHODS), observed)
    later_truth = np.full_like(observed, 0.08)
    later_truth[:, 5:] = 0.03
    current = dict.fromkeys(("truth", *QUALITY_METHODS), later_truth)
    current["pixel_lstm"] = -later_truth
    common = np.ones_like(observed, dtype=bool)
    total, correct = change_signs(previous, current, common)
    assert total == 1
    assert correct["adaptive"] == 1 and correct["pixel_lstm"] == 0
    common[:3, :5] = False
    assert change_signs(previous, current, common)[0] == 0


def test_mixed_mosaic_harmonization_preserves_modern_pixels_and_source_mask():
    date = "2022-05-10"
    manifest = {"landsat_scenes": [("2_LE07_020031_20220510", date), ("2_LC09_020031_20220510", date)]}
    image = np.array([[[0.3, 0.4, np.nan]], [[0.1, 0.2, np.nan]]], dtype=np.float32)
    mask = np.array([[True, False, False]])
    harmonized = harmonize_observations(manifest, {date: image}, {date: (mask, image.copy())})[date]
    np.testing.assert_allclose(harmonized[:, 0, 0], [0.0264 + 0.9407 * 0.3, 0.0013 + 0.9387 * 0.1])
    np.testing.assert_array_equal(harmonized[:, :, 1:], image[:, :, 1:])
    np.testing.assert_array_equal(np.isnan(to_oli(image, True)), np.isnan(image))
