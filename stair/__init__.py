"""
STAIR: Spatio-Temporal Adaptive Image Restoration & Satellite Fusion
=====================================================================
A modular Python framework for Google Earth Engine implementation of the STAIR
algorithm (Luo et al., 2018) and spatiotemporal data fusion (Landsat 8 & MODIS).
"""

__version__ = "1.0.0"

from .core import (
    init_earth_engine,
    load_aoi,
    mask_landsat_clouds,
    get_landsat_collection,
    get_modis_collection,
    calculate_ndvi,
)

from .temporal import (
    compute_temporal_regression,
    compute_regression_metrics,
    compute_temporal_support,
    impute_temporal,
)

from .adaptive import (
    compute_global_correction,
    compute_adaptive_kmeans_correction,
    apply_edge_blending,
    stair_adaptive_pipeline,
)

from .fusion import (
    learn_spatial_rules,
    synthesize_missing_day,
    fuse_date,
)

from .diagnostics import (
    create_staircase_tracker,
    sample_agricultural_pixel,
    create_chip_bounds,
)

from .visualization import (
    create_true_color_map,
    create_band_physics_map,
    create_cloud_comparison_map,
    plot_local_geotiff,
)

__all__ = [
    "init_earth_engine",
    "load_aoi",
    "mask_landsat_clouds",
    "get_landsat_collection",
    "get_modis_collection",
    "calculate_ndvi",
    "compute_temporal_regression",
    "compute_regression_metrics",
    "compute_temporal_support",
    "impute_temporal",
    "compute_global_correction",
    "compute_adaptive_kmeans_correction",
    "apply_edge_blending",
    "stair_adaptive_pipeline",
    "learn_spatial_rules",
    "synthesize_missing_day",
    "fuse_date",
    "create_staircase_tracker",
    "sample_agricultural_pixel",
    "create_chip_bounds",
    "create_true_color_map",
    "create_band_physics_map",
    "create_cloud_comparison_map",
    "plot_local_geotiff",
]
