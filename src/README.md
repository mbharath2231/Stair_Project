# Legacy Scripts Reference & Migration Map

The flat experimental scripts in this directory have been refactored into the modular `stair/` library and unified CLI tools in `scripts/`.

### Migration Map

| Legacy Script in `src/` | New CLI Equivalent in `scripts/` | Underlying Library Module |
| :--- | :--- | :--- |
| `temporal.py`, `Spatial_implementation.py` | `python scripts/run_temporal.py` | [`stair.temporal`](file:///Users/bharathmikkilineni/Stair_Project/stair/temporal.py) |
| `adaptive_5km.py` | `python scripts/run_adaptive.py --mode 5km` | [`stair.adaptive`](file:///Users/bharathmikkilineni/Stair_Project/stair/adaptive.py) |
| `adaptive_correction.py` | `python scripts/run_adaptive.py --mode full_watershed` | [`stair.adaptive`](file:///Users/bharathmikkilineni/Stair_Project/stair/adaptive.py) |
| `gee_fusion_pipeline.py` | `python scripts/run_fusion.py` | [`stair.fusion`](file:///Users/bharathmikkilineni/Stair_Project/stair/fusion.py) |
| `red_band_validation.py` | `python scripts/run_fusion.py --validate-red` | [`stair.fusion`](file:///Users/bharathmikkilineni/Stair_Project/stair/fusion.py) |
| `generate_6month_dataset.py` | `python scripts/run_time_series.py --mode fusion_season` | [`stair.time_series`](file:///Users/bharathmikkilineni/Stair_Project/stair/diagnostics.py) |
| `pixel_time_series.py` | `python scripts/run_time_series.py --mode crop_chip` | [`stair.diagnostics`](file:///Users/bharathmikkilineni/Stair_Project/stair/diagnostics.py) |
| `watershed_staircase.py` | `stair.diagnostics.create_staircase_tracker()` | [`stair.diagnostics`](file:///Users/bharathmikkilineni/Stair_Project/stair/diagnostics.py) |
| `visualise_data.py` | `python scripts/run_visualization.py --mode folium_cloudy_true_color` | [`stair.visualization`](file:///Users/bharathmikkilineni/Stair_Project/stair/visualization.py) |
| `visualize_bands.py` | `python scripts/run_visualization.py --mode folium_bands` | [`stair.visualization`](file:///Users/bharathmikkilineni/Stair_Project/stair/visualization.py) |
| `visualize_cloud_fix.py` | `python scripts/run_visualization.py --mode folium_cloud` | [`stair.visualization`](file:///Users/bharathmikkilineni/Stair_Project/stair/visualization.py) |
| `visualize_local_tiff.py` | `python scripts/run_visualization.py --mode plot_tiff` | [`stair.visualization`](file:///Users/bharathmikkilineni/Stair_Project/stair/visualization.py) |
