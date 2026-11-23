"""Map Landsat Collection 2 red and NIR reflectance to an OLI reference scale."""
from __future__ import annotations

from collections import defaultdict
import re

import numpy as np

MISSION_PATTERN = re.compile(r"(?:^|[_/])(LT05|LE07|LC08|LC09)(?=_)")
LEGACY = frozenset(("LT05", "LE07"))
COEFFICIENT_SOURCE = "https://landsat-harmonization.earth/#coefficients"
# CONUS Collection 2 L2 ETM+ to OLI band-level linear fit, NIR then red.
# The ETM+ transform is used as a proxy for TM, as in the source application.
INTERCEPTS = (0.0264, 0.0013)
SLOPES = (0.9407, 0.9387)


def mission(scene_id):
    match = MISSION_PATTERN.search(scene_id)
    if match is None:
        raise ValueError(f"Unknown Landsat scene identifier {scene_id}")
    return match[1]


def scene_sources(manifest):
    """Group the exact original mosaic sort keys by UTC acquisition date."""
    scenes = defaultdict(list)
    for scene_id, day in manifest["landsat_scenes"]:
        scenes[day].append(scene_id)
    return dict(scenes)


def sensor_families(scenes):
    """Return whether each source is in the TM/ETM+ family."""
    return {mission(scene_id) in LEGACY for scene_id in scenes}


def to_oli(reflectance, legacy):
    """Apply the transform only to legacy observations, retaining the source mask."""
    if reflectance.ndim != 3 or reflectance.shape[0] != 2:
        raise ValueError("Expected NIR and red reflectance on one grid")
    if np.ndim(legacy) and (legacy.shape != reflectance.shape[1:] or legacy.dtype != bool):
        raise ValueError("Legacy pixel provenance must be a boolean mask on the reflectance grid")
    if not np.any(legacy):
        return reflectance
    corrected = (reflectance * np.asarray(SLOPES)[:, None, None]
                 + np.asarray(INTERCEPTS)[:, None, None])
    return np.where(legacy, corrected, reflectance).astype(reflectance.dtype)


def harmonize_observations(manifest, images, provenance):
    """Apply sensor-tagged harmonization to each archived daily mosaic."""
    sources = scene_sources(manifest)
    result = {}
    for day, values in images.items():
        families = sensor_families(sources[day])
        if len(families) == 1:
            legacy = next(iter(families))
        else:
            legacy, native = provenance[day]
            if not np.array_equal(values, native, equal_nan=True):
                raise ValueError(f"Source reflectance and provenance do not match for {day}")
        result[day] = to_oli(values, legacy)
    return result
