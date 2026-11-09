# STAIR work checkpoint

Updated 2026-11-09. This is the checkpoint for continuing on the other
device. See [CONTINUATION.md](CONTINUATION.md) for setup and exact commands.

## Start here

- Reopen the July 29 **false-color composite (FCC)** in QGIS and reproduce
  the July 22 cloud-shaped holdout on the new device.
- Then validate temporal, global, and adaptive stages on additional
  independent dates or chips. Use identical held-out pixels for all three
  stages, report coverage and per-band RMSE, MAE, and bias, and compare the
  same field areas using a shared FCC stretch. Check clear-pixel preservation
  and fallback behavior before extending to Landsat–MODIS fusion.
- The QGIS deliverable is five GeoTIFF stages for one date and grid: original,
  QA-masked, temporal, global, and adaptive. Keep that sequence and omit
  difference rasters.

## Maintained code

| Path | Responsibility |
| --- | --- |
| `stair/landsat.py`, `stair/temporal.py` | Landsat Collection 2 Level 2 reflectance and QA. Pixelwise two-sided temporal interpolation within 32 days per side, with the target excluded from references. |
| `stair/spatial.py` | Scene-wide and class-specific spatial offsets, reproducible reference segmentation, and sparse-support fallbacks. |
| `scripts/export_stair_stages.py` | Materialize and check the five stages on Earth Engine, write local TIFFs and a run manifest, and optionally export to Drive. The current grid is fixed to Clayton. |
| `scripts/build_stair_qgis.py --fcc` | Build the QGIS project and full-chip and field-detail panels from existing TIFFs. Run with QGIS Python. |
| `scripts/run_temporal_validation.py` | Separate cached temporal experiment with cloud, stripe, and full-scene holdout options. See [TEMPORAL_STAGE.md](TEMPORAL_STAGE.md). |

The handoff includes the reviewed spatial path, based on Git commit
`31964f8`. All 24 tests passed on Windows with Python 3.12.14. The older
`stair/adaptive.py` and fusion runners are historical paths. The Clayton
results below use `stair/spatial.py`.

## July 29 Clayton run

- Target Landsat 8 scene `LC08_021031_20220729`. Independent clear reference
  `LC08_020031_20220722`. The full source list is in the generated `run.json`.
- A 167 × 167 grid at 30 m, EPSG:32616, affine transform
  `[30, 0, 727110, 0, -30, 4640580]`. The three float32 bands are
  `SR_B5`, `SR_B4`, and `SR_B3`, with NoData `-9999`.
- All 16,375 QA-clear pixels were preserved exactly. All 11,514 QA gaps
  had temporal support and were filled. The updated adaptive stage selected
  seven reference classes. The earlier Drive folder used four fixed classes.
- The [reviewed Drive folder](https://drive.google.com/drive/folders/1pIxeL7LvhbqrsVK77INyJaIUbZGw43P4)
  contains the five updated TIFFs, verified against the local results.
- **FCC display RGB is TIFF bands 1/2/3**, or Landsat NIR/Red/Green (5/4/3).
  Shared surface-reflectance limits are 0.08–0.60, 0.005–0.165, and
  0.01–0.15 respectively, from rounded July 29 QA-clear percentiles.
  In the west-central cloud gap, global correction reduces the cyan cast
  and adaptive correction makes smaller local adjustments.

## Method choice and validation

The spatial correction uses the reference gap-to-clear contrast from Luo
et al. (2018), Section 2.2.2. The printed formula leaves the centering of
the temporal term ambiguous. Here the correction is
`temporal + mean(clear target) + mean(gap reference) - mean(clear reference) - mean(gap temporal)`
on paired finite support. Subtracting the gap mean preserves an unchanged
scene. The adaptive stage applies this calculation within each spectral
class of the independent reference. It selects the maximum gap statistic
over K = 2 through 8 using seed 42, up to 2,000 sample pixels and 20 uniform
null draws. Each offset requires at least 30 supported clear and 30
supported gap pixels. A sparse class inherits the global offset, and
insufficient global support retains the temporal estimate.

July 29 has no same-day clear truth in its natural gaps. The accuracy
experiment uses a separate July 22 target and July 21 reference, with
withheld clear observations used only for scoring. The Earth Engine
cloud holdout rerun selected five classes and scored all 6,385 withheld
pixels. The stripe holdout scored all 8,489 pixels using the cached NumPy
workflow. The current Earth Engine CLI exposes the cloud holdout only.

| July 22 holdout and band | Temporal RMSE | Global RMSE | Adaptive RMSE |
| --- | ---: | ---: | ---: |
| Cloud, NIR | 0.03377 | 0.02821 | 0.02808 |
| Cloud, Red | 0.01912 | 0.01776 | 0.01763 |
| Cloud, Green | 0.01548 | 0.01344 | 0.01334 |
| Stripe, Red | 0.01715 | 0.01607 | 0.01595 |

Global correction accounts for most of the RMSE gain on this chip.
Adaptive correction gives a smaller additional gain. Its cloud-holdout
NIR MAE rises slightly from 0.01992 to 0.01996. Treat the FCC inspection
and the controlled holdout as separate evidence.
