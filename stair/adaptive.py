"""
STAIR Section 2.2.2: Spatial and Adaptive Correction
====================================================
Refines temporal interpolation using spatial information from a clear reference
image (t1). Implements:
  - Stage 3a: Global correction (watershed/patch-wide offset)
  - Stage 3b: Adaptive correction using K-Means clustering (class-specific offsets)
  - Stage 3c: Edge blending (boundary smoothing)
"""

from typing import Dict, Any, Optional
import ee
import config


def compute_global_correction(
    image_t1: ee.Image,
    mask_t0: ee.Image,
    region: ee.Geometry,
    linear_baseline: ee.Image,
    band_name: str = "SR_B5",
    scale: float = 30.0,
    tile_scale: int = 1
) -> ee.Image:
    """
    Compute global spatial offset delta = mean(gaps) - mean(valid) on clear day t1.
    
    Args:
        image_t1: Clear reference image on t1.
        mask_t0: Binary mask of target date t0 (1 = valid, 0 = cloud/gap).
        region: Geographic region / bounding geometry.
        linear_baseline: Temporal prediction (L_linear) to apply offset to.
        band_name: Spectral band name.
        scale: Resolution scale in meters.
        tile_scale: Earth Engine tileScale for large memory regions.
        
    Returns:
        ee.Image representing the globally corrected prediction.
    """
    ref_band = image_t1.select(band_name)
    
    gap_dict = ref_band.updateMask(mask_t0.eq(0)).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        bestEffort=True,
        maxPixels=1e10,
        tileScale=tile_scale
    )
    valid_dict = ref_band.updateMask(mask_t0.eq(1)).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        bestEffort=True,
        maxPixels=1e10,
        tileScale=tile_scale
    )
    
    mean_gap = ee.Number(gap_dict.get(band_name, 0))
    mean_valid = ee.Number(valid_dict.get(band_name, 0))
    delta_global = mean_gap.subtract(mean_valid)
    
    return linear_baseline.add(ee.Image.constant(delta_global)).rename(f"{band_name}_global")


def compute_adaptive_kmeans_correction(
    image_t1: ee.Image,
    mask_t0: ee.Image,
    region: ee.Geometry,
    linear_baseline: ee.Image,
    n_clusters: int = 4,
    band_name: str = "SR_B5",
    scale: float = 30.0,
    tile_scale: int = 1
) -> Any:
    """
    Compute land-cover adaptive offset using Weka K-Means clustering on clear day t1.
    Calculates class-specific offsets: delta_k = mean(gaps_k) - mean(valid_k).
    
    Args:
        image_t1: Clear reference image on t1.
        mask_t0: Binary mask of target date t0.
        region: Geographic region / bounding geometry.
        linear_baseline: Safe temporal prediction.
        n_clusters: Number of spectral clusters.
        band_name: Spectral band name.
        scale: Resolution scale in meters.
        tile_scale: Earth Engine tileScale.
        
    Returns:
        Tuple of (ee.Image adaptive prediction, ee.Image segments)
    """
    ref_band = image_t1.select(band_name)
    
    # Sample pixels to train unsupervised clusterer
    training_data = ref_band.sample(
        region=region,
        scale=scale,
        numPixels=5000,
        tileScale=tile_scale
    )
    clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training_data)
    segments = ref_band.cluster(clusterer).rename("seg")
    
    def get_segment_means(value_img: ee.Image, mask_img: ee.Image) -> ee.Image:
        stack = value_img.rename("val").updateMask(mask_img).addBands(segments)
        reduction = stack.reduceRegion(
            reducer=ee.Reducer.mean().group(groupField=1, groupName="seg"),
            geometry=region,
            scale=scale,
            bestEffort=True,
            maxPixels=1e10,
            tileScale=tile_scale
        )
        groups = ee.List(ee.Dictionary(reduction).get("groups", ee.List([])))
        
        def get_key(g): return ee.Dictionary(g).getNumber("seg")
        def get_val(g): return ee.Dictionary(g).getNumber("mean")
        
        keys = ee.List([-1]).cat(groups.map(get_key))
        vals = ee.List([0.0]).cat(groups.map(get_val))
        return segments.remap(keys, vals, 0).rename("mean_val")
    
    mean_gap_seg = get_segment_means(ref_band, mask_t0.eq(0))
    mean_valid_seg = get_segment_means(ref_band, mask_t0.eq(1))
    
    delta_adap = mean_gap_seg.subtract(mean_valid_seg)
    adaptive_prediction = linear_baseline.add(delta_adap).rename(f"{band_name}_adaptive")
    
    return adaptive_prediction, segments


def apply_edge_blending(
    patched_image: ee.Image,
    raw_image: ee.Image,
    radius: float = 1.5,
    band_name: str = "SR_B5"
) -> ee.Image:
    """
    Smooth boundaries between valid raw pixels and patched gap fills using focal mean.
    
    Args:
        patched_image: Adaptive or globally patched image.
        raw_image: Original cloud-masked target image.
        radius: Kernel radius for smoothing in pixels.
        band_name: Band name.
        
    Returns:
        ee.Image with seamless edge blending.
    """
    smoothed_patch = patched_image.focal_mean(radius=radius, kernelType="circle", units="pixels")
    return raw_image.select(band_name).unmask(smoothed_patch).rename(f"{band_name}_blended")


def stair_adaptive_pipeline(
    raw_image_t0: ee.Image,
    image_t1: ee.Image,
    linear_prediction: ee.Image,
    region: ee.Geometry,
    n_clusters: int = 4,
    band_name: str = "SR_B5",
    scale: float = 30.0,
    tile_scale: int = 1
) -> Dict[str, ee.Image]:
    """
    Execute full STAIR multi-stage restoration pipeline:
      - Stage 1: Raw broken target (with cloud holes)
      - Stage 2: Temporal interpolation (L_linear)
      - Stage 3a: Global spatial correction
      - Stage 3b: Adaptive K-Means spatial correction
      - Stage 3c: Flawless edge-blended output
      
    Args:
        raw_image_t0: Target image masked for clouds.
        image_t1: Clear reference image.
        linear_prediction: Output from Section 2.2.1 temporal regression.
        region: Geographic bounding geometry.
        n_clusters: Number of K-Means clusters.
        band_name: Band name to process.
        scale: Scale in meters.
        tile_scale: Tile scale for EE compute.
        
    Returns:
        Dictionary mapping stage names to ee.Image objects.
    """
    stage1_broken = raw_image_t0.select(band_name)
    mask_t0 = stage1_broken.mask()
    
    ref_band = image_t1.select(band_name)
    # Failsafe baseline: if linear prediction has holes due to persistent clouds, fall back to t1
    linear_safe = linear_prediction.unmask(ref_band)
    
    # Stage 2: Temporal
    stage2_temporal = stage1_broken.unmask(linear_safe)
    
    # Stage 3a: Global Correction
    pred_global = compute_global_correction(
        image_t1=image_t1,
        mask_t0=mask_t0,
        region=region,
        linear_baseline=linear_safe,
        band_name=band_name,
        scale=scale,
        tile_scale=tile_scale
    )
    stage3a_global = stage1_broken.unmask(pred_global)
    
    # Stage 3b: Adaptive K-Means
    pred_adaptive, segments = compute_adaptive_kmeans_correction(
        image_t1=image_t1,
        mask_t0=mask_t0,
        region=region,
        linear_baseline=linear_safe,
        n_clusters=n_clusters,
        band_name=band_name,
        scale=scale,
        tile_scale=tile_scale
    )
    stage3b_adaptive = stage1_broken.unmask(pred_adaptive)
    
    # Stage 3c: Edge Blending
    stage3c_blended = apply_edge_blending(
        patched_image=stage3b_adaptive,
        raw_image=stage1_broken,
        radius=1.5,
        band_name=band_name
    )
    
    return {
        "stage1_broken": stage1_broken.clip(region),
        "stage2_temporal": stage2_temporal.clip(region),
        "stage3a_global": stage3a_global.clip(region),
        "stage3b_adaptive": stage3b_adaptive.clip(region),
        "stage3c_blended": stage3c_blended.clip(region),
        "segments": segments.clip(region),
    }
