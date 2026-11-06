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
    image_t0: ee.Image,
    mask_t0: ee.Image,
    region: ee.Geometry,
    linear_baseline: ee.Image,
    band_name: str = "SR_B5",
    scale: float = 30.0,
    tile_scale: int = 1,
    image_t1: Optional[ee.Image] = None
) -> ee.Image:
    """
    Compute global spatial correction delta according to Luo et al. (2018):
      delta_global = Delta_temporal_bias + Delta_spatial_gap
      Delta_temporal_bias = mean(L(C_f, t0)) - mean(L_linear(C_f, t0))
      Delta_spatial_gap = mean(L(C_g, t1)) - mean(L(C_f, t1)) (if t1 provided)
    
    Args:
        image_t0: Actual target image with clouds (true values where valid).
        mask_t0: Binary mask of target date t0 (1 = valid / C_f, 0 = cloud gap / C_g).
        region: Geographic region / bounding geometry.
        linear_baseline: Temporal prediction (L_linear) for target day t0.
        band_name: Spectral band name.
        scale: Resolution scale in meters.
        tile_scale: Earth Engine tileScale for large memory regions.
        image_t1: Optional clear reference image (t1) for spatial gap offset.
        
    Returns:
        ee.Image representing the globally corrected prediction.
    """
    true_band = image_t0.select(band_name)
    pred_band = linear_baseline.select([0]).rename(band_name)
    
    clear_mask = mask_t0.eq(1)
    gap_mask = mask_t0.eq(0)
    
    # 1. Temporal calibration: difference on valid clear pixels (C_f) at date t0
    stats_t0 = true_band.updateMask(clear_mask).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, bestEffort=True, maxPixels=1e10, tileScale=tile_scale
    )
    stats_pred = pred_band.updateMask(clear_mask).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, bestEffort=True, maxPixels=1e10, tileScale=tile_scale
    )
    
    mean_t0 = ee.Number(ee.Algorithms.If(stats_t0.contains(band_name), stats_t0.get(band_name), 0))
    mean_pred = ee.Number(ee.Algorithms.If(stats_pred.contains(band_name), stats_pred.get(band_name), 0))
    delta_temp = mean_t0.subtract(mean_pred)
    
    # 2. Spatial community offset: difference between gap (C_g) and free (C_f) on clear day t1
    if image_t1 is not None:
        ref_band = image_t1.select(band_name)
        stats_ref_g = ref_band.updateMask(gap_mask).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=scale, bestEffort=True, maxPixels=1e10, tileScale=tile_scale
        )
        stats_ref_f = ref_band.updateMask(clear_mask).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=scale, bestEffort=True, maxPixels=1e10, tileScale=tile_scale
        )
        has_g = stats_ref_g.contains(band_name)
        has_f = stats_ref_f.contains(band_name)
        mean_ref_g = ee.Number(ee.Algorithms.If(has_g, stats_ref_g.get(band_name), 0))
        mean_ref_f = ee.Number(ee.Algorithms.If(has_f, stats_ref_f.get(band_name), 0))
        delta_spatial = ee.Number(ee.Algorithms.If(
            has_g.And(has_f),
            mean_ref_g.subtract(mean_ref_f),
            0.0
        ))
    else:
        delta_spatial = ee.Number(0.0)
        
    delta_global = delta_temp.add(delta_spatial)
    return linear_baseline.add(ee.Image.constant(delta_global)).rename(f"{band_name}_global")


def compute_adaptive_kmeans_correction(
    image_t0: ee.Image,
    image_t1: ee.Image,
    mask_t0: ee.Image,
    region: ee.Geometry,
    linear_baseline: ee.Image,
    n_clusters: int = 4,
    band_name: str = "SR_B5",
    scale: float = 30.0,
    tile_scale: int = 1,
    precomputed_segments: Optional[ee.Image] = None
) -> Any:
    """
    Compute land-cover adaptive offset using Weka K-Means clustering on clear day t1 (Luo et al., 2018):
      delta_c = Delta_temporal_bias(c) + Delta_spatial_gap(c)
      Delta_temporal_bias(c) = mean(L(C_{f,c}, t0)) - mean(L_linear(C_{f,c}, t0))
      Delta_spatial_gap(c) = mean(L(C_{g,c}, t1)) - mean(L(C_{f,c}, t1))
    
    Args:
        image_t0: Actual target image with clouds (true values where valid).
        image_t1: Clear reference image on t1 (used for clustering & spatial difference).
        mask_t0: Binary mask of target date t0 (1 = valid, 0 = cloud/gap).
        region: Geographic region / bounding geometry.
        linear_baseline: Temporal prediction.
        n_clusters: Number of spectral clusters.
        band_name: Spectral band name.
        scale: Resolution scale in meters.
        tile_scale: Earth Engine tileScale.
        precomputed_segments: Optional pre-clustered image to ensure multi-band consistency.
        
    Returns:
        Tuple of (ee.Image adaptive prediction, ee.Image segments)
    """
    ref_band = image_t1.select(band_name)
    true_band = image_t0.select(band_name)
    pred_band = linear_baseline.select([0]).rename(band_name)
    
    clear_mask = mask_t0.eq(1)
    gap_mask = mask_t0.eq(0)
    
    if precomputed_segments is not None:
        segments = precomputed_segments
    else:
        training_data = ref_band.sample(
            region=region,
            scale=scale,
            numPixels=5000,
            tileScale=tile_scale
        )
        clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training_data)
        segments = ref_band.cluster(clusterer).rename("seg")
    
    # 1. Global fallback deltas
    stats_t0 = true_band.updateMask(clear_mask).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, bestEffort=True, maxPixels=1e10, tileScale=tile_scale
    )
    stats_pred = pred_band.updateMask(clear_mask).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, bestEffort=True, maxPixels=1e10, tileScale=tile_scale
    )
    mean_t0 = ee.Number(ee.Algorithms.If(stats_t0.contains(band_name), stats_t0.get(band_name), 0))
    mean_pred = ee.Number(ee.Algorithms.If(stats_pred.contains(band_name), stats_pred.get(band_name), 0))
    delta_temp_global = mean_t0.subtract(mean_pred)
    
    stats_ref_g = ref_band.updateMask(gap_mask).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, bestEffort=True, maxPixels=1e10, tileScale=tile_scale
    )
    stats_ref_f = ref_band.updateMask(clear_mask).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, bestEffort=True, maxPixels=1e10, tileScale=tile_scale
    )
    mean_ref_g = ee.Number(ee.Algorithms.If(stats_ref_g.contains(band_name), stats_ref_g.get(band_name), 0))
    mean_ref_f = ee.Number(ee.Algorithms.If(stats_ref_f.contains(band_name), stats_ref_f.get(band_name), 0))
    delta_spatial_global = mean_ref_g.subtract(mean_ref_f)
    delta_global = delta_temp_global.add(delta_spatial_global)
    
    # 2. Cluster grouped reductions
    def get_grouped_means(img: ee.Image, mask_img: ee.Image):
        stack = img.rename("val").updateMask(mask_img).addBands(segments)
        reduction = stack.reduceRegion(
            reducer=ee.Reducer.mean().group(groupField=1, groupName="seg"),
            geometry=region, scale=scale, bestEffort=True, maxPixels=1e10, tileScale=tile_scale
        )
        return ee.List(ee.Dictionary(reduction).get("groups", ee.List([])))

    groups_t0_f = get_grouped_means(true_band, clear_mask)
    groups_pred_f = get_grouped_means(pred_band, clear_mask)
    groups_ref_g = get_grouped_means(ref_band, gap_mask)
    groups_ref_f = get_grouped_means(ref_band, clear_mask)
    
    # Also get counts on valid pixels to safeguard small clusters
    stack_count = segments.rename("seg").updateMask(clear_mask).addBands(segments)
    red_count = stack_count.reduceRegion(
        reducer=ee.Reducer.count().group(groupField=1, groupName="seg"),
        geometry=region, scale=scale, bestEffort=True, maxPixels=1e10, tileScale=tile_scale
    )
    groups_count = ee.List(ee.Dictionary(red_count).get("groups", ee.List([])))

    def to_dict(groups_list, key_name="mean"):
        keys = groups_list.map(lambda g: ee.Number(ee.Dictionary(g).get("seg")).format())
        vals = groups_list.map(lambda g: ee.Dictionary(g).get(key_name))
        return ee.Dictionary.fromLists(keys, vals)

    dict_t0_f = to_dict(groups_t0_f, "mean")
    dict_pred_f = to_dict(groups_pred_f, "mean")
    dict_ref_g = to_dict(groups_ref_g, "mean")
    dict_ref_f = to_dict(groups_ref_f, "mean")
    dict_count = to_dict(groups_count, "count")

    cluster_indices = ee.List.sequence(0, n_clusters - 1)
    
    def compute_cluster_delta(c_num):
        c_num = ee.Number(c_num)
        c_str = c_num.format()
        
        # Check count of valid pixels for cluster c
        has_cnt = dict_count.contains(c_str)
        cnt = ee.Number(ee.Algorithms.If(has_cnt, dict_count.get(c_str), 0))
        
        # Temporal bias for cluster c (with fallback to global if cnt < 30)
        has_t0 = dict_t0_f.contains(c_str)
        has_pred = dict_pred_f.contains(c_str)
        t0_val = ee.Number(ee.Algorithms.If(has_t0, dict_t0_f.get(c_str), 0))
        pred_val = ee.Number(ee.Algorithms.If(has_pred, dict_pred_f.get(c_str), 0))
        tb_cluster = t0_val.subtract(pred_val)
        
        use_cluster_tb = has_t0.And(has_pred).And(cnt.gte(30))
        tb_final = ee.Number(ee.Algorithms.If(use_cluster_tb, tb_cluster, delta_temp_global))
        
        # Spatial gap diff for cluster c
        has_rg = dict_ref_g.contains(c_str)
        has_rf = dict_ref_f.contains(c_str)
        rg_val = ee.Number(ee.Algorithms.If(has_rg, dict_ref_g.get(c_str), 0))
        rf_val = ee.Number(ee.Algorithms.If(has_rf, dict_ref_f.get(c_str), 0))
        sb_cluster = rg_val.subtract(rf_val)
        
        use_cluster_sb = has_rg.And(has_rf)
        sb_final = ee.Number(ee.Algorithms.If(use_cluster_sb, sb_cluster, delta_spatial_global))
        
        return tb_final.add(sb_final)

    deltas_list = cluster_indices.map(compute_cluster_delta)
    
    # Prepend key -1 -> delta_global as fallback
    remap_keys = ee.List([-1]).cat(cluster_indices)
    remap_vals = ee.List([delta_global]).cat(deltas_list)
    
    delta_img = segments.remap(remap_keys, remap_vals, delta_global).rename("delta")
    adaptive_prediction = linear_baseline.add(delta_img).rename(f"{band_name}_adaptive")
    
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
    tile_scale: int = 1,
    precomputed_segments: Optional[ee.Image] = None
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
        precomputed_segments: Optional pre-clustered image to ensure multi-band consistency.
        
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
        image_t0=raw_image_t0,
        mask_t0=mask_t0,
        region=region,
        linear_baseline=linear_safe,
        band_name=band_name,
        scale=scale,
        tile_scale=tile_scale,
        image_t1=image_t1
    )
    stage3a_global = stage1_broken.unmask(pred_global)
    
    # Stage 3b: Adaptive K-Means
    pred_adaptive, segments = compute_adaptive_kmeans_correction(
        image_t0=raw_image_t0,
        image_t1=image_t1,
        mask_t0=mask_t0,
        region=region,
        linear_baseline=linear_safe,
        n_clusters=n_clusters,
        band_name=band_name,
        scale=scale,
        tile_scale=tile_scale,
        precomputed_segments=precomputed_segments
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
