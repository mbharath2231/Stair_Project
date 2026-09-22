"""
Visualization Tools for STAIR
=============================
Provides utilities for:
  - Interactive Folium HTML mapping with Google Earth Engine tile layers
  - True-color Landsat rendering
  - Red vs NIR band physics visualizer
  - Before/after cloud restoration comparison maps
  - Local GeoTIFF inspection and high-resolution matplotlib plotting
"""

from pathlib import Path
from typing import Optional, Dict, Any
import folium
import geopandas as gpd
import ee
import config


def add_ee_layer(
    folium_map: folium.Map,
    ee_image: ee.Image,
    vis_params: Dict[str, Any],
    name: str
) -> None:
    """
    Add an Earth Engine image layer to a Folium map using GEE tile fetcher.
    
    Args:
        folium_map: Folium Map object.
        ee_image: ee.Image to render.
        vis_params: Visualization parameters (min, max, bands, palette).
        name: Name to display in Folium LayerControl.
    """
    map_id_dict = ee.Image(ee_image).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Map Data &copy; Google Earth Engine",
        name=name,
        overlay=True,
        control=True
    ).add_to(folium_map)


def create_base_map(gdf: gpd.GeoDataFrame, zoom_start: int = 11) -> folium.Map:
    """
    Create a base Folium map centered on the boundary GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame with geometry in EPSG:4326.
        zoom_start: Initial zoom level.
        
    Returns:
        folium.Map instance.
    """
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2.0
    center_lon = (bounds[0] + bounds[2]) / 2.0
    return folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)


def create_true_color_map(
    image: ee.Image,
    gdf: gpd.GeoDataFrame,
    date_label: str,
    output_path: Optional[Path] = None
) -> Path:
    """
    Generate interactive HTML map displaying Landsat true-color (B4/B3/B2).
    
    Args:
        image: Landsat 8 image.
        gdf: Boundary GeoDataFrame.
        date_label: Date string for layer title.
        output_path: Target HTML file path.
        
    Returns:
        Path to saved HTML file.
    """
    out = Path(output_path or (config.MAPS_OUTPUT_DIR / "true_color_map.html"))
    m = create_base_map(gdf)
    
    vis = {"bands": ["SR_B4", "SR_B3", "SR_B2"], "min": 7000, "max": 12000}
    add_ee_layer(m, image, vis, f"Landsat True Color ({date_label})")
    
    folium.GeoJson(
        gdf,
        name="Watershed Boundary",
        style_function=lambda x: {"fillColor": "#00000000", "color": "red", "weight": 2.5}
    ).add_to(m)
    
    m.add_child(folium.LayerControl())
    m.save(str(out))
    return out


def create_band_physics_map(
    landsat_image: ee.Image,
    gdf: gpd.GeoDataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """
    Generate interactive HTML map comparing Red (absorption) vs NIR (reflection) bands.
    
    Args:
        landsat_image: Clear Landsat 8 image.
        gdf: Boundary GeoDataFrame.
        output_path: Target HTML file path.
        
    Returns:
        Path to saved HTML file.
    """
    out = Path(output_path or (config.MAPS_OUTPUT_DIR / "band_physics_map.html"))
    m = create_base_map(gdf)
    
    red_band = landsat_image.select(config.LANDSAT_BANDS["red"])
    nir_band = landsat_image.select(config.LANDSAT_BANDS["nir"])
    
    vis = {"min": 7000, "max": 15000, "palette": ["black", "gray", "white"]}
    add_ee_layer(m, red_band, vis, "Landsat RED (Chlorophyll Absorption)")
    add_ee_layer(m, nir_band, vis, "Landsat NIR (Cellular Reflection)")
    
    folium.GeoJson(
        gdf,
        name="Watershed Boundary",
        style_function=lambda x: {"fillColor": "#00000000", "color": "blue", "weight": 2.5}
    ).add_to(m)
    
    m.add_child(folium.LayerControl())
    m.save(str(out))
    return out


def create_cloud_comparison_map(
    real_ndvi: ee.Image,
    synth_ndvi: ee.Image,
    gdf: gpd.GeoDataFrame,
    date_str: str,
    output_path: Optional[Path] = None
) -> Path:
    """
    Generate interactive HTML map comparing cloudy raw NDVI vs STAIR restored NDVI.
    
    Args:
        real_ndvi: Corrupted Landsat NDVI on cloudy day.
        synth_ndvi: Restored synthetic STAIR NDVI on same day.
        gdf: Boundary GeoDataFrame.
        date_str: Date string for layer title.
        output_path: Target HTML file path.
        
    Returns:
        Path to saved HTML file.
    """
    out = Path(output_path or (config.MAPS_OUTPUT_DIR / "cloud_validation_map.html"))
    m = create_base_map(gdf)
    
    ndvi_vis = {"min": 0.0, "max": 0.9, "palette": ["red", "yellow", "green"]}
    add_ee_layer(m, real_ndvi, ndvi_vis, f"1. REAL Landsat (Cloud Corrupted) - {date_str}")
    add_ee_layer(m, synth_ndvi, ndvi_vis, f"2. STAIR Restored (Cloud-Free) - {date_str}")
    
    folium.GeoJson(
        gdf,
        name="Watershed Boundary",
        style_function=lambda x: {"fillColor": "#00000000", "color": "blue", "weight": 2.0}
    ).add_to(m)
    
    m.add_child(folium.LayerControl())
    m.save(str(out))
    return out


def plot_local_geotiff(
    tif_path: Path,
    output_image_path: Optional[Path] = None,
    title: str = "STAIR Synthetic NDVI Matrix",
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 0.9,
    show_plot: bool = False
) -> Path:
    """
    Open and plot a local GeoTIFF file using rasterio and matplotlib.
    Masks out empty/NoData pixels and applies custom colormap.
    
    Args:
        tif_path: Path to .tif file.
        output_image_path: Path to save output plot. Defaults to outputs/local_plot.png.
        title: Plot title.
        cmap: Matplotlib colormap.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        show_plot: Whether to call plt.show() (set False in headless environments).
        
    Returns:
        Path to saved PNG plot.
    """
    import rasterio
    import matplotlib.pyplot as plt
    import numpy as np
    
    tif = Path(tif_path)
    if not tif.exists():
        raise FileNotFoundError(f"GeoTIFF not found at {tif}")
        
    out_png = Path(output_image_path or (config.OUTPUT_DIR / f"{tif.stem}_plot.png"))
    
    with rasterio.open(tif) as src:
        matrix = src.read(1)
        # Mask out NoData (negative padding or nodata value)
        nodata = src.nodata if src.nodata is not None else -1.0
        matrix_masked = np.ma.masked_where((matrix < -1.0) | (matrix == nodata), matrix)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        img = ax.imshow(matrix_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        
        cbar = plt.colorbar(img, ax=ax, shrink=0.75, pad=0.04)
        cbar.set_label("NDVI Value", fontsize=12)
        
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.axis("off")
        
        fig.tight_layout()
        fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
        
        if show_plot:
            plt.show()
        plt.close(fig)
        
    print(f"Plot successfully saved to: {out_png}")
    return out_png
