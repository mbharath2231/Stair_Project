import numpy as np
import pytest

from scripts.compare_training_history import compare_variables, monthly_bootstrap
from scripts import compare_training_history as experiment
from stair.history_model import HistoryCorrection, fit_correction
from stair.ndvi import DailyNDVI


def test_ridge_correction_recovers_constant_error_and_preserves_missing_support(tmp_path):
    fused = np.array([[[0.4, 0.5], [0.3, np.nan]], [[0.1, 0.2], [0.15, np.nan]]])
    coarse = fused * 0.9
    offset = np.array([0.025, -0.01])[:, None, None]
    dates = ["2024-03-10", "2024-05-10", "2024-08-10"]
    model = fit_correction((day, fused, coarse, fused + offset) for day in dates)
    assert model.training_dates == tuple(dates)
    assert model.training_pixel_dates == 9
    prediction = model.predict(fused, coarse, "2025-04-10")
    np.testing.assert_allclose(prediction, fused + offset, atol=1e-10)
    path = tmp_path / "model.npz"
    model.save(path)
    reopened = HistoryCorrection.load(path)
    np.testing.assert_array_equal(reopened.predict(fused, coarse, "2025-04-10"), prediction)


def test_fit_is_date_balanced_and_rejects_duplicate_labels():
    fused = np.array([[[0.4]], [[0.1]]])
    error = np.array([[[0.04]], [[0.02]]])
    dates = ["2024-03-10", "2024-05-10"]
    equal = fit_correction([(dates[0], fused, fused, fused + error),
                            (dates[1], fused, fused, fused - error)])
    repeated = np.repeat(fused, 100, axis=2)
    unequal = fit_correction([(dates[0], repeated, repeated, repeated + error),
                              (dates[1], fused, fused, fused - error)])
    np.testing.assert_allclose(equal.coefficients, unequal.coefficients, atol=1e-10)
    with pytest.raises(ValueError, match="Duplicate"):
        fit_correction([(dates[0], fused, fused, fused), (dates[0], fused, fused, fused)])
    with pytest.raises(ValueError, match="No supported"):
        fit_correction([])


def test_reflectance_features_cannot_use_the_withheld_target():
    def image(nir, red):
        return np.broadcast_to(np.array([nir, red])[:, None, None], (2, 2, 2)).copy()

    landsat = {"2025-03-01": image(0.3, 0.1), "2025-03-10": image(0.9, 0.8),
               "2025-03-18": image(0.5, 0.15)}
    modis = {day: image(0.2, 0.08) for day in landsat}
    original = DailyNDVI(landsat, modis).holdout_reflectance("2025-03-10")
    landsat["2025-03-10"][:] = np.nan
    changed = DailyNDVI(landsat, modis).holdout_reflectance("2025-03-10")
    for name in original:
        np.testing.assert_array_equal(original[name], changed[name])


def test_metrics_use_shared_support_and_report_coverage():
    truth = np.array([[[0.4, 0.5]], [[0.1, 0.2]]])
    partial = truth.copy()
    partial[:, :, 1] = np.nan
    scores, samples = compare_variables(truth, {"complete": truth, "partial": partial})
    for variable in scores:
        assert scores[variable]["eligible_pixels"] == 2
        assert scores[variable]["common_pixels"] == 1
        assert scores[variable]["methods"]["complete"]["coverage"] == 1
        assert scores[variable]["methods"]["partial"]["coverage"] == 0.5
        np.testing.assert_array_equal(samples[variable]["complete"], samples[variable]["partial"])


def test_month_block_bootstrap_keeps_methods_paired():
    rows = [{"date": f"2025-{month:02d}-15", "scores": {"NDVI": {
        "common_pixels": 100, "methods": {name: {"rmse": rmse} for name, rmse in (
            ("fusion", 0.1), ("six_month", 0.08), ("twenty_year", 0.06))}}}}
        for month in (3, 4, 5)]
    report = monthly_bootstrap(rows, repeats=100)
    delta = report["comparisons"]["twenty_year_minus_six_month"]
    assert report["supported_month_blocks"] == 3
    np.testing.assert_allclose(delta["percentile_95_interval"], [-0.02, -0.02])


def test_training_generator_limits_label_period_and_rejects_test_year_context(monkeypatch, tmp_path):
    dates = ("2024-03-01", "2024-03-10", "2024-03-18", "2024-09-01")
    image = np.broadcast_to(np.array([0.4, 0.1])[:, None, None], (2, 2, 2)).copy()
    landsat = {day: image.copy() for day in dates}
    modis = {day: image * 0.9 for day in dates}
    manifest = {"end_exclusive": "2024-12-03"}
    monkeypatch.setattr(experiment, "read_cache", lambda *_: (manifest, landsat, modis))
    samples = list(experiment.training_samples(tmp_path, [2024], "2024-03-01", "2024-09-01", 0.6, []))
    assert [sample[0] for sample in samples] == list(dates[:3])
    model = fit_correction(samples)
    assert model.training_dates == ("2024-03-10",)
    manifest["end_exclusive"] = "2025-02-01"
    with pytest.raises(ValueError, match="Training context reaches the test year"):
        list(experiment.training_samples(tmp_path, [2024], "2024-03-01", "2024-09-01", 0.6, []))
