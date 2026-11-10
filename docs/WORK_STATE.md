# STAIR work checkpoint

Updated 2026-11-10. This is the checkpoint for continuing on the other
device. See [CONTINUATION.md](CONTINUATION.md) for setup and exact commands.

## March 13 fusion audit

Re-downloaded the Earth Engine inputs and independently evaluated
`M(target) + (1 - w) * (L(before) - M(before)) + w * (L(after) - M(after))`.
All seven March GeoTIFF masks matched and the maximum absolute value
difference was 2.98e-8. The user's downloaded March 13 TIFFs matched the
reviewed files byte-for-byte. The 367 NoData pixels match the strict
MODIS mask exactly. A rigid shift check of the fused image against
Landsat, on a common interior mask and within three pixels per axis,
gave the lowest error at zero shift in all three bands.

On March 13, holding the February 26 and April 13 anchors fixed, adding
the MODIS adjustment changed NIR RMSE from 0.02645 to 0.03675 and NDVI
RMSE from 0.03381 to 0.04759. It lowered mean predicted NIR by 0.01197
on a prediction already biased low. The same-anchor control differs
from the runner's standard Landsat-only baseline, which can use closer
valid Landsat observations.

Next development task is to assess independent Landsat–MODIS registration
on the anchor pairs, then compare shorter quality-screened brackets and
the same-anchor Landsat control across multiple withheld dates. The
zero-shift check compared the final raster with Landsat and does not
replace cross-sensor registration. Keep the withheld target out of any
offset or anchor selection.

## Spring NDVI change check

Compared March 13 and April 14 on the same 15,983 valid corn/soybean
pixels, selected with the 2024 USDA Cropland Data Layer. Both Landsat
dates were withheld from their respective fusion runs. Median pixelwise
NDVI change was +0.0233 in Landsat and +0.0406 in fusion. Pixelwise change
RMSE was 0.0440, with Pearson r = 0.731. Of 5,083 pixels with absolute
observed change of at least 0.05, fusion matched the direction on 96.7%.
The 0.05 cutoff defines this descriptive check.

The [QGIS package](https://drive.google.com/file/d/1ORWUorbYP8pp2FZrRQ8Df8Yw_da2SbKu/view)
contains a wider shared FCC stretch, matched NDVI maps, three illustrative
210 m sample patches, and their observed and fused NDVI series. The
samples were chosen from observed Landsat change to illustrate a strong
increase, near stability, and a slight decline. CDL labels indicate the
annual cash-crop class. Spring crop identity and management are unverified.
The chart leaves the gap between the two fusion weeks open and shows
available QA-clear Landsat points through April 30.

To regenerate after both spring fusion runs exist, execute
`scripts/spring_ndvi_review.py` with the project Python and Earth Engine
authentication. Then run `scripts/build_spring_review_qgis.py
outputs/fusion_validation/spring_2024_phenology` using QGIS Python.

## Spring 2024 QGIS examples

These seven-day runs use the same Clayton grid and full-inversion MODIS
quality rule as July. The dates and anchors were screened for input
coverage before evaluating the withheld Landsat. Run with the project
Python and Earth Engine authentication:

```sh
python scripts/run_fusion_week.py --start 2024-03-13 --prior 2024-02-26 --later 2024-04-13
python scripts/run_fusion_week.py --start 2024-04-14 --prior 2024-04-13 --later 2024-04-22
```

Run `scripts/build_fusion_qgis.py <run-directory>` with QGIS Python to
build `fusion_week.qgz`. Open **Project → Layouts → FCC week comparison**
for aligned MODIS, fusion, and withheld Landsat panels. The
[spring Drive folder](https://drive.google.com/drive/folders/1wpdxaci5AXc_LljR3KscUgpXQo3YUm8_)
has the two weekly ZIP packages and the combined NDVI review. Each weekly
package contains a relative-path QGIS project,
15 GeoTIFFs, an FCC preview, and a run report. Extract the full ZIP before
opening the project.

| Week | Withheld Landsat | Daily fused coverage | First-to-last median absolute NIR / Red change |
| --- | --- | ---: | ---: |
| March 13–19 | March 13 | 98.68–99.98% | 0.00833 / 0.00373 |
| April 14–20 | April 14 | 100% | 0.03563 / 0.01196 |
| July 22–28, 2022 | July 22 | 100% | 0.03878 / 0.00531 |

Each change uses the common valid pixels in the first and last images.
The spring run reports also record the same comparison for adjacent days
and for MODIS. March keeps missing inputs as NoData. Against withheld
Landsat, fused NDVI RMSE was 0.04759 in March and 0.06086 in April, versus
0.03222 and 0.06522 for Landsat-only interpolation. Both runs beat
MODIS-only on the held-out day. These benchmarks apply to their individual
dates and common valid support.

## Seven-day fusion pilot

Run `python scripts/run_fusion_week.py` with the project Python and an
authenticated Earth Engine account. Use `--output` to choose a fresh
directory. The default experiment covers July 22–28, 2022 on the same
Clayton grid as the gap-filling experiment. The seven dates each use their
own MODIS MCD43A4.061 retrieval, screened for full inversion in NIR, Red,
and Green before bilinear sampling on the common grid.

The July 21 and July 30 Landsat–MODIS pairs bracket the entire week.
July 30 is restored from outside-week observations and a July 21 spatial
reference. No Landsat acquired within the week contributes to a prediction
or to any restored anchor. The two residuals `Landsat - MODIS` are linearly
interpolated by date and added to the corresponding daily MODIS. July 22,
the only Landsat acquisition in the week, is used after prediction for
independent validation. The seven daily fused maps each have 27,889
valid pixels. The remaining six dates have no same-day Landsat score.

| July 22 method | NIR RMSE | Red RMSE | Green RMSE | NDVI RMSE |
| --- | ---: | ---: | ---: | ---: |
| MODIS only | 0.05908 | 0.03181 | 0.02247 | 0.13981 |
| Prior-pair fusion | 0.03875 | 0.01838 | 0.01533 | 0.07330 |
| Two-pair fusion, July 29 anchor | 0.03672 | 0.01814 | 0.01506 | 0.07132 |
| Two-pair fusion, July 30 anchor | 0.03412 | 0.01686 | 0.01388 | 0.06696 |
| Landsat-only interpolation | 0.03663 | 0.01724 | 0.01401 | 0.06752 |

The first trial used July 29 as the later anchor (`--later 2022-07-29`).
A dark patch in its QA-clear pixels propagated into the July 28 prediction,
even though local MODIS NIR increased. The July 30 anchor has 23,341
observed clear pixels compared with 16,375 on July 29. At a 7 × 7 patch
centered at 728205 E, 4638945 N (EPSG:32616), median July 28 fused NIR was
0.2221 with the July 29 anchor and 0.5769 with July 30. July 22 was used to
score both trials. Confirm performance on a new Landsat date before
generalizing the anchor choice.

The runner writes seven three-band fused GeoTIFFs, seven aligned MODIS
GeoTIFFs, one QA-screened Landsat truth TIFF, a shared-stretch FCC figure,
and `run.json` with coverage, dates, and scores. These are generated
outputs under `outputs/fusion_validation/`. TIFF display RGB uses bands
1/2/3 with the same channel limits as the July 29 comparison below.
MCD43A4 uses a 16-day retrieval window, so adjacent daily MODIS inputs
share observations. This pilot aligns the grids and leaves cross-sensor
offset estimation for a separate check. Extend validation to another
independent Landsat day and assess registration before expanding the window.

## Gap-filling workflow

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
