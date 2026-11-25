"""Pixel LSTM and ConvLSTM residual models for Landsat gap filling."""

from __future__ import annotations

from datetime import date

import numpy as np
import torch
from torch import nn

from .nonlinear_spatial import TARGET_SCALE

STEPS_PER_SIDE = 2
FRAME_CHANNELS = 4
HIDDEN_CHANNELS = 8
KERNELS = {"pixel_lstm": 1, "conv_lstm": 3}


def sequence_frames(images, day, *, window, excluded=()):
    """Select two available acquisition dates per side and encode gaps explicitly."""
    target = date.fromisoformat(day)
    candidates = [(date.fromisoformat(key) - target, key) for key in sorted(images) if key not in excluded and key != day]
    offsets = [(delta.days, key) for delta, key in candidates if abs(delta.days) <= window]
    before = [(age, key) for age, key in offsets if age < 0][-STEPS_PER_SIDE:]
    after = [(age, key) for age, key in offsets if age > 0][:STEPS_PER_SIDE]
    slots = [(0, None)] * (STEPS_PER_SIDE - len(before)) + before + after + [(0, None)] * (STEPS_PER_SIDE - len(after))
    shape = next(iter(images.values())).shape[1:]
    frames = np.zeros((len(slots), FRAME_CHANNELS, *shape), dtype=np.float32)
    for i, (age, key) in enumerate(slots):
        if key is None:
            continue
        reflectance = images[key]
        valid = np.isfinite(reflectance).all(axis=0)
        frames[i, :2] = np.where(valid, reflectance, 0) / np.array([0.5, 0.25], dtype=np.float32)[:, None, None]
        frames[i, 2] = valid
        frames[i, 3] = age / window
    return frames, tuple(key for _, key in slots)


class RecurrentCorrection(nn.Module):
    """Use identical LSTM gates with pixelwise or spatial state updates."""

    def __init__(self, kind):
        super().__init__()
        kernel = KERNELS[kind]
        self.gates = nn.Conv2d(FRAME_CHANNELS + HIDDEN_CHANNELS, 4 * HIDDEN_CHANNELS, kernel, padding=kernel // 2)
        self.residual = nn.Conv2d(8 + HIDDEN_CHANNELS, 2, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, frames, static):
        hidden = static.new_zeros((static.shape[0], HIDDEN_CHANNELS, *static.shape[-2:]))
        memory = torch.zeros_like(hidden)
        for frame in frames.unbind(dim=1):
            incoming, forget, output, candidate = self.gates(torch.cat((frame, hidden), dim=1)).chunk(4, dim=1)
            memory = forget.sigmoid() * memory + incoming.sigmoid() * candidate.tanh()
            hidden = output.sigmoid() * memory.tanh()
        return self.residual(torch.cat((static, hidden), dim=1))

    @torch.inference_mode()
    def corrected(self, frames, static, adaptive):
        residual = self(torch.from_numpy(frames), torch.from_numpy(static)).numpy()
        return adaptive + residual * TARGET_SCALE[:, None, None]
