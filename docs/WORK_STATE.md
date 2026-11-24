# STAIR work checkpoint

Updated 2026-11-24. This is the checkpoint for continuing on either device.
See [CONTINUATION.md](CONTINUATION.md) for setup and exact commands.

## Clayton 2024 acquisition time series and map sheets

The current user task follows every Landsat acquisition from March through
October 2024 on the established 167 × 167, 30 m Clayton study grid. This
5.01 km square contains 55 acquisition dates. QA-screened original NDVI is
available on 47 dates. Temporal filling followed by global and adaptive
spatial correction provides NDVI on all 55, with 99.792–100% area coverage.
Clear reflectance and original valid NDVI are preserved exactly. Spatial
offsets apply on 28 dates. Eight fully masked targets use flagged temporal
interpolation because clear target support is absent.

Run `scripts/compare_landsat_season.py --year 2024` from the cached
observations. It writes original, temporal, global and adaptive reflectance
and NDVI TIFFs for every acquisition, source flags, anchor ages, seasonal
curves, and `all_acquisition_ndvi.csv`. Its independent reference is at least
95% clear within 21 days, and temporal brackets extend up to 32 days per
side. Segmentation uses the existing matched-season settings of 320 sample
pixels, four null draws and seed 42.

Build `landsat_restoration_time_series.qgz` with QGIS Python using
`scripts/build_landsat_restoration_qgis.py`. The project has 168 layers and
16 layouts. Original and restored overviews each show all 55 dates. Five
larger sheets per series show up to 12 dates each with identical extents
and NDVI scales. The project also includes the two individual seasonal
curves, their overlay and illustrative acquisition comparisons.

The curves use available-pixel medians and equally spaced acquisition-date
labels. Their spatial support changes when clouds hide parts of the
original scene. On June 8, the original median was 0.72963 over 1.925% of
the grid, while the restored median was 0.40509 over the whole grid. This
comparison describes coverage and seasonal behavior. Accuracy in natural
gaps requires withholding known clear observations. Crop records support
interpretation of crop identity and growth stages.

Earlier outputs from today include the observed-only March–October review
and an initial full-calendar daily Landsat–MODIS NDVI review. The observed
curve uses the same 19,679 clear pixels across 27 high-coverage acquisitions.
The initial daily product has 80.019% pixel-day coverage and 59 dates with
no supported estimates. Its independently withheld-date fusion NDVI RMSE
is 0.07584 on 37 dates, compared with 0.07308 for nearest-clear Landsat
interpolation on the same common support.

All 43 tests pass. Integration checks verified all 55 restoration sequences,
source flags, temporal exclusion and preservation of clear observations.
QGIS checks verified all 220 map-sheet panels and reopened the transferred
project with all 168 layers and 16 layouts.

Start with the [November 24 study guide](handoffs/2026-11-24/STAIR_Study_Guide_2026-11-24.docx)
and the [time-series download index](../outputs/README.md#clayton-2024-time-series-and-map-sheets-november-24).
Review the 2024 original and restored maps on the same fields, then select
field polygons for curves over fixed locations and validate fills with
controlled withholding. Extend to another year after reviewing this season.

## Clayton spatial correction benchmark

The matched ridge comparison trains on March–October 2024 and on the same
months in 2005–2024. The models were nearly tied on withheld 2025 Landsat
pixels. A paired refit with older Landsat reflectance harmonized to the
OLI scale slightly increased NDVI error.

Five nonlinear candidates were fitted per training history. Gradient
boosting with neighborhood inputs led the long-history comparison. It
reduced NDVI RMSE from 0.06913 to 0.06790 on clouds and from 0.06658 to
0.06478 on stripes. The cloud uncertainty interval includes zero. Ridge
led both gap tests among the one-season models. The tests cover Landsat
spatial correction before daily fusion.

The [study guide](handoffs/2026-11-23/STAIR_Clayton_Study_Guide_2026-11-23.docx)
explains the experiment and the next work. Compare boosting with and
without neighborhood inputs at the same tree size, then verify gains
on additional withheld years or sites. See the
[validation guide](CLOUD_FILLING_VALIDATION.md) for reproducible commands and
the [download index](../outputs/README.md#clayton-spatial-correction-and-nonlinear-models-november-23)
for the QGIS projects, model files, and evaluation archives.

## Twenty-year versus six-month fusion training

Implemented and ran the date-balanced ridge correction described in
[TRAINING_HISTORY_COMPARISON.md](TRAINING_HISTORY_COMPARISON.md).
Long-history training uses March–October 2005–2024. The user selected
March–August 2024 for the short history. Both are evaluated on withheld
2025 acquisitions with the existing fusion and Landsat/MODIS controls.
The fitted models and fixed-scale QGIS comparison are in
`outputs/fusion_validation/history_comparison_2005_2025/`.
NIR RMSE improves with long-history training. NDVI gains are small and
the nearest-clear Landsat control has the lowest NDVI RMSE in this test.
Coverage, sensor-era sensitivity and additional withheld years remain
part of the next evaluation. The guide records results and limitations.
The [November 18 handoff](https://github.com/mbharath2231/Stair_Project/releases/tag/untagged-b2806ae9169ed4d2b92d)
draft release provides both trained models, QGIS projects, GeoTIFFs and
evaluation reports for transfer. See [outputs/README.md](../outputs/README.md)
for archive contents and extraction paths. The historical input cache is
available from the earlier historical handoff for offline reproduction.

## Clayton false-color imagery, March–October 2005–2025

Run `python scripts/collect_historical_fcc.py` with Earth Engine access
to build the local archive under `data/clayton_fcc_march_october/`.
Each year has a `Landsat` folder with individual Collection 2 Tier 1
Level 2 scenes and a `MODIS` folder with one MCD43A4.061 image per
calendar date. Each TIFF stores float surface reflectance in
NIR, Red, Green order for display as RGB. The source bands are 4/3/2
for Landsat 5/7, 5/4/3 for Landsat 8/9, and 2/1/4 for MODIS.

Landsat files use the 30 m Clayton grid and retain clouds and shadows
from the source scenes. MODIS files use the product's native nominal
500 m sinusoidal cells and retain available retrievals of both quality
classes. Missing source pixels remain NoData. Each annual `index.json`
records the Earth Engine asset ID and valid pixel count for every
TIFF. Open the TIFFs directly in QGIS with RGB bands 1/2/3.
Load `Clayton_FCC_shared.qml` from the archive root to apply the
same display stretch across images.

## Clayton historical archive, March–October 2005–2025

Download the two surface-reflectance bands used for NDVI from Earth Engine
and run the Clayton model from the local cache:

```sh
python scripts/collect_ndvi_inputs.py --start-year 2005 --end-year 2025 --march-october
python scripts/run_daily_ndvi.py --start-year 2005 --end-year 2025 --march-october --fallback
```

The archive uses Landsat Collection 2 Tier 1 Level 2 and MODIS
MCD43A4.061. Landsat 5/7 use `SR_B4` for NIR and `SR_B3` for Red.
Landsat 8/9 use `SR_B5` and `SR_B4`. MODIS uses
`Nadir_Reflectance_Band2` and `Nadir_Reflectance_Band1`.
The annual NPZ files in `data/ndvi_inputs/march_october/` store aligned
NIR and Red in that order, separate Landsat and MODIS date lists, and
Landsat scene IDs. The arrays use the Clayton 30 m grid. MODIS comes
from the nominal 500 m product and is bilinearly resampled.
Input caches include 32 days on either side for interpolation at the
season boundaries. Landsat 7 scan gaps and unsupported inputs stay masked.

Run `python scripts/export_ndvi_reflectance.py` to export the cached
surface reflectance to `outputs/daily_ndvi/march_october/reflectance/<year>/`.
Each year has separate Landsat Red, Landsat NIR, MODIS Red, and MODIS
NIR TIFFs with 245 date-labeled bands and the same grid as the NDVI stacks.
Open `outputs/daily_ndvi/march_october/Clayton_2005_review.qgz` for
preselected dates and fixed symbology. Rebuild it using QGIS Python with
`scripts/build_historical_qgis.py outputs/daily_ndvi/march_october`.
When opening an annual TIFF directly, use singleband symbology and select
one date. QGIS can assign the first three dates as RGB channels by default.

The output index is `outputs/daily_ndvi/march_october/with_fallback/dataset.json`.
Each annual folder contains matching 245-band Landsat, MODIS, and modeled NDVI
GeoTIFFs, plus the annual source raster and `run.json`. Band 1 is March 1
and band 245 is October 31 in every year. The Landsat stack shows NoData
on dates without a valid observation. In the modeled output, the source
raster distinguishes Landsat observations, fusion, and Landsat-only
interpolation. The annual report gives the date-level holdout comparison
and records the SHA-256 of the input cache.

The Clayton download contains 1,010 Landsat acquisition dates and 5,145
MODIS product dates inside the 21 seasons. The modeled product has 92.6%
pixel-day coverage with the flagged fallbacks enabled and 85 dates with
no supported pixels. On 379 held-out Landsat dates with common valid
pixels, NDVI RMSE was 0.0850 for two-pair fusion, 0.0856 for Landsat
interpolation with the same pairs, 0.0826 for nearest-clear Landsat
interpolation, and 0.1513 for MODIS alone.

The Clayton runs use native mission reflectance. Cross-mission spectral
harmonization and Landsat–MODIS co-registration remain open before analyzing
long-term NDVI trends. Keep watershed expansion contingent on the Clayton
evaluation.

## Watershed MODIS restoration

The watershed runner now uses neighboring QA-approved MODIS observations
from inside the prediction period and attempts temporal interpolation when
a whole MODIS retrieval is absent. The Landsat holdout stays in place.
A separate run with 16 days per side in
`outputs/fusion_validation/raisin_2022-07-22_modis_fill/` reached
99.46–99.53% fusion coverage on the same five days. The new QGIS project's
MODIS temporal fill source layer identifies cells using more distant input.
On the July 22 withheld benchmark, shared-support NDVI RMSE went from
0.06207 to 0.06221, and newly supported pixels with valid NDVI scored
0.12105. See [the watershed pilot results](WATERSHED_PILOT.md).

## River Raisin watershed pilot

The [watershed pilot](WATERSHED_PILOT.md) processed five consecutive dates
from July 22 to July 26, 2022. The portable QGIS project and all restoration
and fusion layers are under `outputs/fusion_validation/raisin_2022-07-22_five_days/`.
The watershed contains 3,130,003 pixels at 30 m. Valid fusion coverage was
99.27–99.45%. July 22 was the independently withheld Landsat benchmark.
There was no Landsat acquisition over the polygon on July 23–26.

July 22 NDVI RMSE was 0.06201 for STAIR, 0.06238 for nearest-clear Landsat
interpolation, and 0.06154 for Landsat interpolation using STAIR's anchors.
The same-anchor result motivates checking registration and anchor selection
across more withheld acquisitions before increasing the run length.

## Multi-date spring fusion review regenerated on this device

Regenerated the March 13–19 and April 14–20, 2024 validation runs with the
existing weekly runner. Each week withheld its Landsat target from prediction
and used it only for scoring. The combined spring NDVI review uses the same
15,983 corn/soy pixels on both benchmark dates. Results reproduce the prior
checkpoint: March NDVI RMSE 0.04759 for fusion, 0.03222 for Landsat-only, and
0.12463 for MODIS-only; April RMSE 0.06086, 0.06522, and 0.14997 respectively.
March-to-April NDVI change RMSE is 0.04399, correlation 0.731, with 96.7%
direction agreement among 5,083 pixels whose observed change magnitude is at
least 0.05. These are two benchmark dates from one chip, not broad accuracy
evidence.

Artifacts are under `outputs/fusion_validation/` (ignored by Git). The combined
project is `spring_2024_phenology/spring_review.qgz`. PyQGIS built and reopened
the project successfully. The QGIS desktop could not be launched in this macOS
session because Apple pasteboard services returned an error; open the project
manually in QGIS. Cross-sensor registration analysis and additional withheld
dates remain next. Keep targets out of all anchor and offset selection.

## Daily NDVI for Clayton Township, 2016–2025

NDVI is `(NIR - Red) / (NIR + Red)` from scaled surface reflectance.
Use Landsat 8/9 `SR_B5` and `SR_B4` and MODIS MCD43A4.061
`Nadir_Reflectance_Band2` and `Nadir_Reflectance_Band1` in that order.
The collection cached 464 Landsat acquisition dates and 3,653 MODIS product
dates inside the decade, plus boundary padding for interpolation. The archive
uses the 167 × 167, 30 m Clayton grid in EPSG:32616.

```sh
python scripts/collect_ndvi_inputs.py
python scripts/run_daily_ndvi.py
python scripts/run_daily_ndvi.py --fallback
```

The first command requires Earth Engine authentication and writes
`data/ndvi_inputs/clayton_<year>.npz`. The model then runs locally. It
interpolates the nearest QA-clear paired Landsat–MODIS residuals up to 32 days
on either side, per pixel. It adds the residual to daily MODIS reflectance
before calculating NDVI. Clear same-day Landsat takes priority. Anchors use
observed pixels from scenes with at least 60% clear coverage. The model
leaves unresolved pixels masked and aligns both sensors on one grid.
Independent cross-sensor registration is pending.

Output is under `outputs/daily_ndvi/strict/` and
`outputs/daily_ndvi/with_fallback/`. Each year has an NDVI GeoTIFF and a
matching source-code GeoTIFF, both with one ISO-date-labelled band per day.
The source codes are 0 NoData, 1 Landsat observation, 2 two-pair fusion,
3 single-pair fusion within 16 days, and 4 Landsat-only interpolation.
Codes 3 and 4 are enabled by `--fallback`. Each `dataset.json` lists the
year files and validation summary. The annual `run.json` records daily
coverage, source scenes, input checksum, and leave-one-date-out scores.

The strict output has NDVI on 40.5% of pixel-days. The fallback output has
NDVI on 76.8%, including 84.5% from March through May. It has 496 calendar
days with no supported pixels, predominantly in winter. Across 148 withheld
dates with shared valid pixels, NDVI RMSE was 0.0830 for two-pair fusion,
0.0865 for Landsat-only interpolation using the same pairs, 0.0836 for
nearest-clear Landsat interpolation, and 0.1485 for MODIS. These scores use
the same 3,248,364 pixel-date comparisons across methods.

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

### Cloud holdout probability densities

Run the cloud-shaped holdout with `--density-report` to save
`cloud_holdout_density.png` and add the matched-pixel statistics to
`run.json`. For example:

```sh
python scripts/export_stair_stages.py --target 2022-07-22 --reference 2022-07-21 --holdout-cloud --density-report --output outputs/spatial_validation/cloud_density/2022-07-22
```

The upper plots compare Gaussian kernel density estimates of reflectance
for withheld Landsat and the three fill stages. The lower plots show the
distribution of fill minus withheld value at the same pixel. All methods
use the same eligible pixels within a date. The report records empirical
Wasserstein distance, paired RMSE and bias, and paired error percentiles.

The July 22, 2022 run scored 6,385 pixels. Two additional runs used
April 6, 2024 with March 29 as spatial reference (6,089 pixels) and
April 14, 2024 with April 13 as reference (6,385 pixels). Red RMSE for
temporal, global, and adaptive filling was 0.01912, 0.01776, and 0.01763
on July 22, 0.01181, 0.01243, and 0.01203 on April 6, and 0.03107,
0.02540, and 0.02651 on April 14. For the April 14 Red band, adaptive
filling reduced the distribution distance from 0.01408 to 0.01083
relative to global correction, while increasing paired RMSE. Results
are descriptive for these dates and this chip. Equivalence would require
a pre-specified error tolerance and validation across additional scenes
and masks.

### Spatial correction

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
