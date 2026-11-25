"""A compact convolutional residual correction for adaptive Landsat gap filling."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .history_model import features
from .nonlinear_spatial import BASE_COLUMNS, TARGET_SCALE

WIDTH = 16
DILATIONS = (1, 1, 3)


def cnn_features(adaptive, reference, day):
    """Keep the reflectance grid and mark missing adaptive and reference values."""
    base = features(adaptive, reference, day)[1:].astype(np.float32)
    adaptive_valid = np.isfinite(base[:2]).all(axis=0)
    reference_valid = np.isfinite(base[2:4]).all(axis=0)
    predictors = np.concatenate((np.where(np.isfinite(base), base, 0), [adaptive_valid, reference_valid]))
    return predictors, adaptive_valid & reference_valid


class SpatialCNN(nn.Module):
    """Predict scaled NIR and red residuals with an 11 × 11 receptive field."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = BASE_COLUMNS + 2
        for dilation in DILATIONS:
            layers.extend((nn.Conv2d(channels, WIDTH, 3, dilation=dilation, padding=dilation), nn.ReLU()))
            channels = WIDTH
        self.residual = nn.Sequential(*layers, nn.Conv2d(WIDTH, 2, 1))
        # The network starts from the adaptive estimate.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, predictors):
        return self.residual(predictors)

    @torch.inference_mode()
    def corrected(self, predictors, adaptive):
        """Return physical reflectance for a single image or a batch."""
        device = next(self.parameters()).device
        residual = self(torch.as_tensor(predictors, device=device)).cpu().numpy()
        return adaptive + residual * TARGET_SCALE[:, None, None]

    def predict(self, adaptive, reference, day):
        predictors, valid = cnn_features(adaptive, reference, day)
        corrected = self.corrected(predictors[None], adaptive[None])[0]
        return np.where(valid, corrected, np.nan)
