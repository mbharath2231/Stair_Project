"""Nonlinear learners for residual corrections to adaptive Landsat gap filling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .history_model import features

ALGORITHMS = {
    "spline": "Cubic spline regression", "boosting": "Histogram gradient boosting",
    "forest": "Random forest", "extra_trees": "Extra Trees",
    "spatial_boosting": "Gradient boosting with neighborhood inputs",
}
BASE_COLUMNS = 6
TARGET_SCALE = np.array([0.1, 0.05], dtype=np.float32)


def neighborhood_features(image):
    """Local means and standard deviations from available values on the grid."""
    from scipy.ndimage import uniform_filter

    valid = np.isfinite(image)
    clean = np.where(valid, image, 0.0).astype(float)
    moments = []
    for side in (5, 11):
        size = (1, side, side)
        count = uniform_filter(valid.astype(float), size=size, mode="constant")
        mean = np.divide(uniform_filter(clean, size=size, mode="constant"), count,
                         out=np.full_like(clean, np.nan), where=count > 0)
        second = np.divide(uniform_filter(clean ** 2, size=size, mode="constant"), count,
                           out=np.full_like(clean, np.nan), where=count > 0)
        moments.extend((mean, np.sqrt(np.maximum(0, second - mean ** 2))))
    return np.concatenate(moments)


def design(adaptive, reference, day, spatial):
    """Base features match ridge, with optional 150 m and 330 m neighborhood context."""
    base = features(adaptive, reference, day)[1:]
    if spatial:
        base = np.concatenate((base, neighborhood_features(adaptive), neighborhood_features(reference)))
    return base.astype(np.float32)


@dataclass
class NonlinearCorrection:
    algorithm: str
    estimator: object

    def predict_rows(self, predictors, baseline):
        columns = predictors if self.algorithm == "spatial_boosting" else predictors[:, :BASE_COLUMNS]
        return baseline + self.estimator.predict(columns) * TARGET_SCALE

    def predict(self, adaptive, reference, day):
        predictors = design(adaptive, reference, day, self.algorithm == "spatial_boosting")
        valid = np.isfinite(predictors).all(axis=0)
        result = np.full(adaptive.shape, np.nan, dtype=float)
        if valid.any():
            result[:, valid] = self.predict_rows(predictors[:, valid].T, adaptive[:, valid].T).T
        return result


def fit_model(algorithm, x, residual, weights, workers):
    """Fit a reproducible candidate on date-balanced reflectance residuals."""
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown nonlinear algorithm {algorithm}")
    predictors = x if algorithm == "spatial_boosting" else x[:, :BASE_COLUMNS]
    sample_weight = weights * (len(weights) / weights.sum())
    target = residual / TARGET_SCALE
    if algorithm == "spline":
        estimator = make_pipeline(SplineTransformer(n_knots=5, degree=3, include_bias=False,
                                                    extrapolation="constant", sparse_output=True),
                                  Ridge(alpha=0.001 * len(x), solver="lsqr"))
        estimator.fit(predictors, target, ridge__sample_weight=sample_weight)
    elif algorithm in ("boosting", "spatial_boosting"):
        leaves = 23 if algorithm == "spatial_boosting" else 15
        estimator = MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=75, max_leaf_nodes=leaves, learning_rate=0.08, min_samples_leaf=120,
            max_bins=63, l2_regularization=2, early_stopping=False, random_state=2024))
        estimator.fit(predictors, target, sample_weight=sample_weight)
    else:
        tree = RandomForestRegressor if algorithm == "forest" else ExtraTreesRegressor
        estimator = tree(n_estimators=40, max_depth=12, min_samples_leaf=96, max_features=0.8,
                         bootstrap=True, max_samples=min(len(x), 100_000), random_state=2024, n_jobs=workers)
        estimator.fit(predictors, target, sample_weight=sample_weight)
    return NonlinearCorrection(algorithm, estimator)
