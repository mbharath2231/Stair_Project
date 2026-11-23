# Clayton cloud filling validation with probability densities

The experiment hides known Landsat pixels on the existing 167 × 167
Clayton grid. It uses March 13 and April 14, 2024, the spring fusion
benchmark dates, and July 22, 2022, the previous restoration benchmark.
This experiment evaluates Landsat gap filling before fusion.

For each date, run the temporal, global, and adaptive stages on the same
cloud-shaped mask, then repeat with diagonal stripes. Fit spatial
corrections using the visible target pixels and a different date. The
target is excluded from temporal references. The closest reference
within 21 days with at least 95% QA-clear coverage is selected by date and
coverage only. The selected references are February 26, April 13, and July
21, respectively. Each was 100% QA-clear on this grid.

Score only originally QA-clear withheld pixels that have finite values
in all methods. All three methods filled all 6,385 withheld reflectance
pixels per cloud test and all 8,489 per stripe test. April has 6,383 and
8,488 eligible NDVI pixels in the cloud and stripe masks, respectively,
because some withheld reference reflectance values fail the NDVI validity
rule. No stage changed any observed reflectance outside the holdout.

## Nonlinear spatial correction benchmark

The fitted models estimate the NIR and red residuals between QA-clear
Landsat and the adaptive STAIR estimate at deliberately hidden pixels.
The predicted residuals are added back to the adaptive estimate for those
pixels. NDVI is calculated after correcting both reflectance bands.
The inputs are adaptive NIR and red, reference NIR and red, and two
seasonal features derived from the day of year. The neighborhood
candidate also uses local means and standard deviations at 150 m and
330 m in the adaptive estimate and reference. These features are
computed with the target gaps already hidden.

Cubic splines model smooth curved responses. Random forest averages
trees, and Extra Trees randomizes the thresholds within those trees.
Histogram gradient boosting fits a sequence of trees to improve earlier
predictions. The neighborhood version adds local features and uses
23 leaves per tree versus 15 in the basic version, so the comparison
captures both differences. Both forests bootstrap up to 100,000 examples
per tree from the eligible training pool.

The 2024 fit used 194,392 training pixel-dates from 33 acquisitions.
The 2005–2024 fit used 1,702,287 from 325 acquisitions, with usable
samples in 19 years. Both periods use March–October and give each date
equal total training weight. The study used the native sensor archive.
The existing ridge fits and their scoring support were reproduced
on the same cached scenes.

Model selection used 2023 for the long history and separate May and
August blocks of 2024 for the short history, with a 32-day buffer
around each 2024 validation block. Random forest had the lowest
development NDVI RMSE among nonlinear candidates for 2024.
Neighborhood boosting won for 2005–2024. Every candidate was refit on
its full training period and tested on the same 32 acquisitions
from March–October 2025. Each score uses common valid pixels inside
the artificial gaps where QA-clear Landsat provides a reference.

| Model | 2024 cloud | 2024 stripes | 2005–2024 cloud | 2005–2024 stripes |
| --- | ---: | ---: | ---: | ---: |
| Ridge regression | **0.06883** | **0.06681** | 0.06913 | 0.06658 |
| Cubic splines | 0.06928 | 0.06800 | 0.06900 | 0.06677 |
| Random forest | 0.07082 | 0.06751 | 0.06947 | 0.06678 |
| Extra Trees | 0.07146 | 0.06817 | 0.06996 | 0.06672 |
| Histogram gradient boosting | 0.06990 | 0.06833 | 0.06930 | 0.06670 |
| Gradient boosting with neighborhood inputs | 0.07017 | 0.06856 | **0.06790** | **0.06478** |

The table reports NDVI RMSE on 190,672 cloud and 249,039 stripe
pixel-dates. Lower values indicate smaller errors. Relative to
the long-history ridge, neighborhood boosting lowered NDVI RMSE by
1.8% on cloud gaps and 2.7% on stripes. It lowered NDVI RMSE on
26 of 32 cloud dates and 28 of 32 stripe dates. The paired 95%
month-block interval for the RMSE change was −0.00224 to +0.00001
on clouds and −0.00271 to −0.00106 on stripes. NIR and red RMSE
and the error in NDVI changes between adjacent pixels also improved.
The one-season ridge had the lowest errors among the one-season fits
on both masks. This experiment measures 2025 gap filling in Clayton.

```powershell
python -m pip install -r requirements-nonlinear.txt
python scripts/compare_nonlinear_spatial.py
$env:QT_QPA_PLATFORM = 'windows'
& 'C:\Program Files\QGIS 3.44.15\bin\python-qgis-ltr.bat' scripts/build_history_comparison_qgis.py outputs/spatial_validation/clayton_nonlinear_2005_2025 --spatial-pattern cloud
```

Repeat the QGIS command with `--spatial-pattern stripes`. Use
**Project → Layouts → NDVI | Selected date comparisons** for examples
with the largest improvement and the largest increase in error under
the long-history nonlinear model. All candidate GeoTIFFs share the
same grid. The run directory also contains smooth probability density
curves, per-date scores and a manifest with development scores, model
settings and paired results. The density curves compare distributions
of held-out pixels. Pixel-matched errors measure reconstruction accuracy.

## Matched Landsat harmonization experiment

The experiment compares two March–October 2005–2024 spatial
calibrations using identical dates, target pixels, temporal windows,
and reference dates. One uses the mission-native archive. The other
adjusts legacy NIR and red to the Landsat 8/9 OLI scale before
recomputing the training labels, temporal estimates, and adaptive
corrections. The archive uses Collection 2 Level-2 surface reflectance in
physical units. The published CONUS band-level linear ETM+ to OLI fit
uses `NIR_OLI = 0.0264 + 0.9407 * NIR` and
`Red_OLI = 0.0013 + 0.9387 * Red`. The same transform is a proxy for
Landsat 5 TM. The test therefore evaluates this specific band-level
harmonization, with no separate local adjustment for Landsat 5 or
angle-based correction. [Published CONUS coefficients](https://landsat-harmonization.earth/) · [Application to TM and ETM+](https://www.climateengine.org/datasets/remotesensing/landsat5789-harmonization/)

For 14 daily mosaics listing both legacy and modern missions, Earth
Engine replayed the original QA masks and scene order. The reproduced
reflectance matched each original cached mosaic exactly. The derived
sensor mask restricted the transform to pixels contributed by the older
family. Both fits used the same 325 dates and 1,702,287 training
pixel-dates. `run.json` records the matched mask checksum for each date.
The native refit reproduced the saved coefficients exactly. Both models
used the original 2025 Landsat 8/9 reference images and their original
test masks, with identical common scoring pixels verified from the TIFFs.

| 2025 holdout | NDVI pixel-dates | 2024 season | Native long fit | Harmonized long fit | Harmonized minus native RMSE, 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Cloud | 190,672 | 0.06883 | 0.06913 | 0.06925 | +0.000029 to +0.000199 |
| Stripes | 249,039 | 0.06681 | 0.06658 | 0.06666 | −0.000018 to +0.000168 |

The harmonized fit had higher pooled NDVI RMSE on both masks, by 0.00012
on cloud gaps and 0.00008 on stripes. It lowered per-date NDVI RMSE on
10 of 32 dates under each mask. The month-block interval was positive
on clouds and crossed zero on stripes. NIR and red RMSE changes were
also small. The median absolute change between the two long-fit NDVI
maps was about 0.0009. Coverage was about 97.4% for both fits. March 15
shows the greatest improvement under harmonization, while July 6
shows its largest increase in error on cloud gaps. These estimates
describe the Clayton chip in 2025.

```powershell
python scripts/collect_landsat_provenance.py
python scripts/compare_spatial_harmonization.py
$env:QT_QPA_PLATFORM = 'windows'
& 'C:\Program Files\QGIS 3.44.15\bin\python-qgis-ltr.bat' scripts/build_history_comparison_qgis.py outputs/spatial_validation/clayton_harmonization_2005_2025 --spatial-pattern cloud
```

The first command creates `landsat_mixed_provenance.npz` beside the
annual input caches and needs Earth Engine credentials. The training
command uses that file and the existing matched March–October run
offline. Repeat the QGIS command for `--spatial-pattern stripes`.
Use **Project → Layouts → NDVI | Selected date comparisons** to view
the best and worst relative dates on the shared scale. The run also
includes smooth density curves, per-date metrics, three model files,
and the aligned GeoTIFFs for both masks.

## Matched March–October spatial calibration

The one-season fit uses March–October 2024. The long fit uses the same
months from 2005–2024, including the 2024 observations. Both fit the same
linear correction to adaptive Landsat gap filling. The long fit is
identical to the March–October long fit in the earlier experiment.
The one-season fit used 33 acquisitions and the long fit used 325.
Nineteen years supplied usable long-fit samples under the target and
reference eligibility rules described below.

Both models were tested on 32 March–October 2025 Landsat acquisitions.
Scores use the same originally clear withheld pixels for every method.
Prediction coverage for both fitted models was about 97.4% on each mask.

| NDVI test | Common pixel-dates | Adaptive alone | 2024 season | 2005–2024 seasons | Long minus short 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Cloud | 190,672 | 0.07189 | 0.06883 | 0.06913 | −0.00033 to +0.00098 |
| Stripes | 249,039 | 0.06791 | 0.06681 | 0.06658 | −0.00070 to +0.00020 |

The table reports RMSE and paired month-block intervals. Both fitted
models lowered pooled NDVI RMSE relative to the adaptive baseline.
The short fit had slightly lower cloud RMSE and the long fit had
slightly lower stripe RMSE. Both NDVI intervals span zero. With the
same calendar months, the data provide no clear overall NDVI benefit
from the extra training years in this chip and test year. The long fit
lowered red RMSE on stripes by about 0.000081, and the short fit had
slightly lower NIR RMSE under both masks. The report also records MAE,
bias, Wasserstein-1 and neighboring-pixel contrast error. The main
comparison controls the calendar months. The training sets still differ
in sensor composition and the conditions represented by the added years.

On July 29, 2025, the longer fit lowered cloud-gap NDVI RMSE from 0.06129
to 0.05872. On July 6, it increased RMSE from 0.07158 to 0.07654. In QGIS,
**Project → Layouts → NDVI | Selected date comparisons** shows the best
and worst relative dates on a common scale. The calibrated spatial
correction layout shows reference, adaptive baseline, and the two fits
for dates selected by coverage.

```powershell
python scripts/compare_spatial_history.py --matched-season
$env:QT_QPA_PLATFORM = 'windows'
& 'C:\Program Files\QGIS 3.44.15\bin\python-qgis-ltr.bat' scripts/build_history_comparison_qgis.py outputs/spatial_validation/clayton_full_season_history_2005_2025 --spatial-pattern cloud
```

Repeat the last command with `--spatial-pattern stripes` for the stripe
project. Both projects, their `rasters/` folder, the smooth March–October
density figures, per-date table, and report are in
`outputs/spatial_validation/clayton_full_season_history_2005_2025/`.

## Additional evaluation of the matched-season maps

The additional evaluation reads the saved NDVI GeoTIFFs for the 32
March–October 2025 acquisitions. It checks the original common pixel
counts before scoring the adaptive and two trained maps. All scores are
for originally clear Landsat pixels inside the artificial holdouts.

| Check | Mask | Adaptive | 2024 season | 2005–2024 seasons |
| --- | --- | ---: | ---: | ---: |
| Local structure, SSIM ↑ | Cloud | 0.82624 | **0.83574** | 0.83389 |
| Local structure, SSIM ↑ | Stripes | 0.83795 | **0.84461** | 0.84355 |
| High-NDVI detection, F1 ↑ | Cloud | 0.90193 | 0.90456 | **0.90487** |
| High-NDVI detection, F1 ↑ | Stripes | 0.92504 | 0.92501 | **0.92537** |
| Date-to-date NDVI change RMSE ↓ | Cloud | 0.11906 | **0.11414** | 0.11441 |
| Date-to-date NDVI change RMSE ↓ | Stripes | 0.11432 | **0.10990** | 0.10997 |
| Direction of substantial change ↑ | Cloud | **59.8%** | 58.9% | 59.0% |
| Direction of substantial change ↑ | Stripes | **64.9%** | 63.7% | 63.8% |

- **Local structure.** SSIM compares local brightness, contrast, and
  pattern. This implementation uses uniform 5 × 5 windows whose pixels
  all qualify for scoring, with NDVI data range 2. The windows overlap.
  Their count describes support and the results are descriptive for this
  chip and year. [Wang et al., 2004](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)
- **Gap depth.** Group the shared pixels by distance to the nearest cell
  outside the artificial mask. In clouds over 300 m from that edge, MAE
  was 0.04888 for adaptive correction, 0.04725 for one season, and 0.04749
  for the long history. One season had slightly lower MAE at all cloud
  depths. In stripes, adaptive correction had the lowest MAE in both
  populated distance groups, and the longer fit had the lowest 90th
  percentile absolute error in both groups.
- **High-NDVI detection.** Compare a threshold of NDVI ≥ 0.60 against
  the same threshold in withheld Landsat. F1 balances precision and
  recall. The full report records confusion counts and balanced
  accuracy. The threshold illustrates performance for high NDVI.
  Cover-crop labeling would require separate reference data.
- **Change between dates.** Compare predicted and observed NDVI changes
  for 31 consecutive observed-date pairs, on pixels available in both
  dates. Measure change RMSE across all shared pixels and agreement on
  the sign of change where observed |ΔNDVI| is at least 0.05. Each map
  comes from a separate holdout, so the paired date may serve as a
  reference for either prediction. This evaluates change in the saved
  products. A test where both images are missing needs simultaneous
  withholding. The June 3–11 pair favors the longer history on both
  mask shapes for change RMSE and can be inspected in the QGIS projects.

Run `python scripts/evaluate_spatial_history.py` after the matched-season
run. It writes two comparison figures and `run.json` to the run's
`additional_evaluation/` folder. The report includes support counts,
per-date scores, and date-pair results.

## Six-month and 20-year spatial calibration

This experiment applies the same Landsat temporal, global, and adaptive
stages to both training histories. It learns a separate linear correction
to the adaptive NIR and red predictions using the adaptive estimate,
independent reference reflectance, and time of year. During testing,
the correction applies inside the simulated gap. The validation uses
Landsat gap filling before the Landsat–MODIS fusion step. The learned
calibration is an extension of the spatial filling workflow.
[STAIR paper](https://doi.org/10.1016/j.rse.2018.04.042)

The short fit used 25 dates from March–August 2024. The main long fit used
230 dates from March–August 2005–2024, keeping the calendar months matched.
The additional long fit used 325 dates from March–October 2005–2024.
The two long fits had usable samples in 19 years. No 2012 acquisition met
the target and reference criteria. Targets required at least 60% valid
reflectance, and the nearest independent reference within 21 days required
95% valid reflectance. Segmentation used up to 320 reference pixels and
four null draws for each candidate class count, with the same settings
throughout. All training used the cloud-shaped mask. Stripes provide a
second test geometry. One chip and one held-out year limit generalization.

Each RMSE compares the same originally clear withheld pixels across all
methods. Twenty-two 2025 dates contribute in March–August and 32 in
March–October. The date groups are resampled together by calendar month
for paired uncertainty estimates.

| 2025 NDVI test | Adaptive | Six months | 20 years, Mar–Aug | 20 years, Mar–Oct |
| --- | ---: | ---: | ---: | ---: |
| Mar–Aug, cloud | 0.07557 | 0.07141 | 0.07155 | 0.07186 |
| Mar–Aug, stripes | 0.06952 | 0.06815 | 0.06785 | 0.06783 |
| Mar–Oct, cloud | 0.07189 | 0.07097 | 0.06900 | 0.06913 |
| Mar–Oct, stripes | 0.06791 | 0.06929 | 0.06672 | 0.06658 |

The primary cloud test uses 126,838 shared NDVI pixel-dates. Coverage
was about 96.1% of eligible pixels for both histories. The long minus
short RMSE difference was +0.00014, with a month-block 95% interval
of −0.00046 to +0.00087. Stripes also gave an interval spanning zero
in March–August. On the whole season, the matched long fit reduced
RMSE by about 2.8% on cloud masks and 3.7% on stripes relative to the
short fit. September and October are outside the short fit's training
season, so the two periods should be reported together.

For the primary cloud test, six months had slightly lower NIR RMSE,
NDVI MAE, and NDVI neighbor-contrast error. Red RMSE was effectively
tied. Adaptive correction had a closer NDVI distribution than either
historical fit by Wasserstein-1 on cloud and stripe masks. Distribution
agreement and paired pixel accuracy measure different aspects of filling.
The smooth Gaussian density plots compare the same sample locations with
the same bandwidth per panel. Their top row shows values and their bottom
row shows prediction minus Landsat. RMSE and other metrics use all
available common pixels. [Gaussian kernel density estimates](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)

On cloud masks, the median absolute NDVI difference between the matched
histories was 0.0034 in March–August and 0.0230 in September–October.
October 2 illustrates a late-season improvement in the long fit, from
0.07323 to 0.06104 RMSE. July 6 illustrates a setback, from 0.07111 to
0.07489. The QGIS example layout highlights both types of outcome.

Run the experiment offline from the repository root with the annual
Clayton input cache. The QGIS layout command uses the QGIS Python
installation on this Windows device.

```powershell
python scripts/compare_spatial_history.py
$env:QT_QPA_PLATFORM = 'windows'
& 'C:\Program Files\QGIS 3.44.15\bin\python-qgis-ltr.bat' scripts/build_history_comparison_qgis.py outputs/spatial_validation/clayton_history_2005_2025 --spatial-pattern cloud
```

Repeat the last command with `--spatial-pattern stripes` for the stripe
review. The output folder contains both QGIS projects, 512 GeoTIFFs,
smooth density curves as PNG and PDF, three fitted models, `run.json`,
and per-date metrics in the `cloud/` and `stripes/` folders. The QGIS
projects open with their `rasters/` subfolder beside them. Use **Project
→ Layouts** to view spatial correction stages and the calibrated output.

## How to read the density plots

The curves are area-normalized histograms with identical bins for all
methods and dates shown in a figure. Compare the curves for restored and
observed values to assess the brightness distribution and its spread.
Wasserstein-1 quantifies the difference between the two empirical
distributions in the original reflectance or NDVI units. Pixel locations
do not enter the Wasserstein calculation. Paired RMSE, MAE, bias, and
error density use the original pixel locations. Report coverage with
both comparisons.

## Cloud mask results

Each entry lists Wasserstein-1 followed by paired RMSE. Smaller values
indicate closer agreement for the respective measure.

| Target | Variable | Temporal | Global | Adaptive |
| --- | --- | ---: | ---: | ---: |
| March 13, 2024 | NIR | 0.00905 / 0.02889 | 0.01195 / 0.02993 | 0.01342 / 0.03066 |
| March 13, 2024 | NDVI | 0.00365 / 0.03106 | 0.01037 / 0.03286 | 0.01113 / 0.03356 |
| April 14, 2024 | NIR | 0.02787 / 0.04397 | 0.00869 / 0.03507 | 0.00899 / 0.03478 |
| April 14, 2024 | NDVI | 0.02030 / 0.06802 | 0.02592 / 0.06749 | 0.01260 / 0.06548 |
| July 22, 2022 | NIR | 0.01872 / 0.03377 | 0.00264 / 0.02821 | 0.00378 / 0.02808 |
| July 22, 2022 | NDVI | 0.01530 / 0.07496 | 0.01519 / 0.07267 | 0.01337 / 0.07244 |

On July 22, global correction brought the NIR distribution closest to
the withheld values. Adaptive correction produced the lowest NIR RMSE on
those pixels. On March 13, temporal interpolation had the lowest NIR
RMSE under the cloud mask and adaptive correction had the lowest NIR
RMSE under the stripe mask (0.03045; temporal was 0.03217). The ranking
depends on the target date, gap geometry, and evaluation measure.

This is a controlled test on one chip and three dates. For inference
across dates or areas, use independent dates and spatial blocks as
replicates. A simulated mask allows comparison with known values.
Naturally cloudy locations lack the same-day QA-clear reference needed
for this experiment.

## Reproduce

Run `python scripts/run_cloud_density_validation.py` with the project
environment. Earth Engine authentication is required for the first run.
A rerun can reuse the observation cache with `--cache` and write to a
fresh directory with `--output`. By default, figures and `run.json` are
in `outputs/fusion_validation/cloud_density_clayton/`. The manifest
records reference dates, scene IDs, coverage, and all metrics for
NIR, red, green, and NDVI.

## Review the five stages in QGIS

Export the same cloud-shaped holdout as five GeoTIFFs per date, then
build the QGIS project with the Python bundled with QGIS.

```sh
python scripts/run_cloud_density_validation.py --patterns cloud --export-stages --output outputs/fusion_validation/cloud_fill_qgis_clayton
QT_QPA_PLATFORM=offscreen /Applications/QGIS.app/Contents/MacOS/python scripts/build_cloud_fill_qgis.py outputs/fusion_validation/cloud_fill_qgis_clayton
```

Open `clayton_cloud_fill.qgz` from the output folder. Use **Project →
Layouts → All dates | five stages** for the overview and the dated layouts
for a closer view of the mask. The original GeoTIFF stores the QA-clear
observations used as validation truth. The five stages have the same
30 m grid and a shared NIR/red/green display stretch. The QGIS project
uses relative paths, so keep it with the `cloud/` folder when transferring.

## Review NDVI in QGIS

Derive NDVI from the NIR and red surface reflectance of each exported
stage, using the Landsat formula `(NIR − red) / (NIR + red)`. The export
applies the same validity rule as the density evaluation. It also writes
three signed error maps per date. Each shows restored minus original NDVI
only where all three fills and the reference have valid NDVI inside the
simulated gap.

```sh
python scripts/export_cloud_fill_ndvi.py outputs/fusion_validation/cloud_fill_qgis_clayton outputs/fusion_validation/cloud_fill_ndvi_clayton
QT_QPA_PLATFORM=offscreen /Applications/QGIS.app/Contents/MacOS/python scripts/build_cloud_fill_qgis.py outputs/fusion_validation/cloud_fill_ndvi_clayton
```

Open `ndvi_clayton_cloud_fill.qgz` and choose **All dates | five stages**
or **All dates | NDVI errors** in **Project → Layouts**. The first uses a
shared NDVI scale from −0.2 to 1.0. In the error comparison, blue indicates
underestimation, orange indicates overestimation, and light gray marks
pixels outside the common scoring area. The fixed display limits do not
change the stored GeoTIFF values.

Method references:

- [NIST handbook on probability density normalization](https://www.itl.nist.gov/div898/handbook/eda/section3/eda33e.htm)
- [Wasserstein-1 distance definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)
- [Luo et al. (2018) STAIR](https://doi.org/10.1016/j.rse.2018.04.042)
- [USGS Landsat NDVI formula](https://www.usgs.gov/landsat-missions/landsat-normalized-difference-vegetation-index)
