"""STAIR tools; optional Earth Engine and plotting dependencies load on demand."""
from importlib import import_module

__version__ = "1.1.0"
_EXPORTS = {
    "core": ["init_earth_engine", "load_aoi", "mask_landsat_clouds", "get_landsat_collection",
             "get_modis_collection", "calculate_ndvi"],
    "temporal": ["impute_temporal", "interpolate_temporal_numpy", "interpolate_temporal_ee"],
    "adaptive": ["compute_global_correction", "compute_adaptive_kmeans_correction",
                 "apply_edge_blending", "stair_adaptive_pipeline"],
    "fusion": ["learn_spatial_rules", "synthesize_missing_day", "fuse_date",
               "stair_dual_fusion_numpy", "stair_dual_fusion_ee"],
    "diagnostics": ["create_staircase_tracker", "sample_agricultural_pixel", "create_chip_bounds"],
    "visualization": ["create_true_color_map", "create_band_physics_map",
                      "create_cloud_comparison_map", "plot_local_geotiff"],
}
__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name):
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f".{module}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
