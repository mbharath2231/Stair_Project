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

## Landsat recurrent spatial correction

The pixel LSTM and ConvLSTM predict NIR and red residuals after adaptive
STAIR. Each receives up to two Landsat acquisitions per side within
32 days, using the aligned reflectance, a quality mask, and the signed
day offset. Missing sequence slots carry a zero validity flag. The head
also receives the same adaptive and reference features as the CNN.
The pixel LSTM updates its state independently at each location with
1 × 1 kernels. The ConvLSTM uses 3 × 3 kernels across the same input
channels. It has more trainable parameters.

Both models trained on the same 325 cloud holdouts from 2005–2024 as
the long-history ridge and CNN. Each selected among two, four, and six
epochs by mean validation-year NDVI MAE over forward-chaining 2022,
2023, and 2024 folds. The regular 2025 cloud and stripe comparison keeps
the same test dates, reference images, scoring pixels, and replayed
baselines as the CNN experiment.

The stress test takes disjoint blocks of two or three consecutive
acquisitions. For each target, it masks the usual cloud-shaped area
and excludes the entire block from temporal interpolation, reference
selection, and recurrent inputs. Clear pixels outside the target's
artificial gap remain available. Report coverage across all eligible
targets, including the dates skipped when an independent reference
is unavailable. Compare models within each block length, since
different references and scoring support can make errors across block
lengths incomparable.

```powershell
python scripts/compare_recurrent_spatial.py
python scripts/evaluate_recurrent_changes.py outputs/spatial_validation/clayton_recurrent_2005_2025
```

This run uses the historical input cache and the preceding nonlinear
and CNN checkpoints. The output folder contains the recurrent weights,
cross-validation and test metrics, and aligned GeoTIFFs for the regular
holdouts. The tile analysis measures changes in mean NDVI across
consecutive scored dates using the same pixels on both dates. A 150 m
tile requires at least 13 common withheld pixels. These fixed tiles
serve as a field-scale proxy. The tile scores use the regular single
holdouts, and the blocked-reference experiments have separate scores.

Both recurrent models selected four epochs in development. The table
reports 2025 NDVI MAE scored on common pixels within each column.

| Model | Single cloud gap | Single stripe gap | Two-acquisition cloud gap | Three-acquisition cloud gap |
| --- | ---: | ---: | ---: | ---: |
| Adaptive STAIR | 0.04886 | 0.04370 | 0.04781 | 0.04585 |
| Long-history ridge | 0.04753 | 0.04388 | **0.04673** | **0.04480** |
| Neighborhood boosting | **0.04692** | **0.04269** | 0.04676 | 0.04515 |
| Tuned CNN | 0.04854 | 0.04389 | 0.04756 | 0.04571 |
| Pixel LSTM | 0.04869 | 0.04382 | 0.04769 | 0.04567 |
| ConvLSTM | 0.04878 | 0.04377 | 0.04774 | 0.04567 |

The single-gap tests scored 190,672 cloud and 249,039 stripe
pixel-dates across 32 acquisitions. On the consecutive gaps, 30
two-acquisition targets and 28 three-acquisition targets contributed
common scores. The 95% month-block RMSE intervals favored boosting over
both recurrent models in the regular tests. Both blocked-reference
comparisons against boosting had intervals spanning zero. Overall
coverage including skipped targets was about 96.6% and 90.6% on the
two- and three-acquisition tests. The residual models share the adaptive
prediction support and use the matching ridge fallback for invalid NDVI.

On 150 m tiles, boosting had the lowest change MAE on both masks
(0.04317 for cloud and 0.04139 for stripes). Pixel LSTM reached 0.04485
and 0.04278, while ConvLSTM reached 0.04510 and 0.04287. These values
describe changes between independently reconstructed acquisition maps.
Both recurrent architectures used four sequence slots and one seed.

Use `scripts/build_history_comparison_qgis.py` with the run directory
and `--spatial-pattern cloud` or `--spatial-pattern stripes` to build
the regular holdout projects. **Project → Layouts → NDVI | Selected date
comparisons** compares ConvLSTM with neighborhood boosting on a shared
scale.

### Structure and change direction on the recurrent maps

Run `python scripts/evaluate_recurrent_quality.py` with both the recurrent
and original CNN benchmark folders available. It reconstructs the
original common NDVI mask per date using all archived CNN comparison
methods and the recurrent outputs. It verifies all 32 dates per mask,
their selected references, and the shared scoring counts. Clear NIR
and red match the masked input exactly for all six selected methods.
The largest difference in derived NDVI at observed pixels is
0.0000001192 from floating-point rounding.

Each cell gives cloud gaps first and stripe gaps second. MAE and RMSE
come from `run.json` on the same scoring support, with 190,672 cloud
and 249,039 stripe pixel-dates. Lower error is better. Higher SSIM,
transition F1, and direction agreement are better.

| Model | NDVI MAE ↓ | NDVI RMSE ↓ | SSIM ↑ | Transition F1 ↑ | Change direction ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adaptive STAIR | 0.04886 / 0.04370 | 0.07189 / 0.06791 | 0.8262 / 0.8379 | 0.5300 / **0.5244** | 79.83% / 82.58% |
| Long-history ridge | 0.04753 / 0.04388 | 0.06913 / 0.06658 | 0.8339 / 0.8436 | **0.5329** / 0.5243 | 78.51% / 81.19% |
| Neighborhood boosting | **0.04692 / 0.04269** | **0.06790 / 0.06478** | **0.8341 / 0.8448** | 0.5027 / 0.4966 | 80.10% / 81.61% |
| Tuned CNN | 0.04854 / 0.04389 | 0.07084 / 0.06727 | 0.8271 / 0.8377 | 0.5288 / 0.5226 | **80.33% / 82.70%** |
| Pixel LSTM | 0.04869 / 0.04382 | 0.07122 / 0.06735 | 0.8271 / 0.8389 | 0.5289 / 0.5228 | 79.87% / 82.40% |
| ConvLSTM | 0.04878 / 0.04377 | 0.07140 / 0.06742 | 0.8270 / 0.8387 | 0.5293 / 0.5230 | 79.91% / 82.49% |

SSIM compares local pattern and contrast in fully supported 5 × 5
windows. Transition F1 combines precision and recall for an absolute
NDVI jump of at least 0.10 in the same horizontal or vertical pixel
pair. It evaluates local NDVI transitions. Identifying actual field
boundaries requires independent field geometry. Boosting made fewer
false alarms relative to its predicted transitions and recovered
fewer observed transitions. The relative reduction in transition F1
held at 0.05, 0.10, and 0.15 NDVI thresholds.

Direction agreement uses 150 m tiles on shared withheld pixels between
consecutive scored dates, with observed absolute change of at least
0.05 NDVI. Paired bootstrap resampling groups values by calendar month
and assigns each change pair to its later date. The difference between
the tuned CNN and boosting in direction agreement spans zero on both
masks. The complete metrics, thresholds, support counts, and intervals
are in `additional_quality.json` beside the recurrent run manifest.

### Percent bias and efficiency for all benchmark methods

P-bias, NSE, and the original 2009 KGE use the same withheld 2025 NDVI
pixels as MAE and RMSE. Each cell gives cloud results first and stripe
results second. P-bias uses prediction minus observation divided by
the observed mean. Positive values indicate overprediction. The ideal
P-bias is zero, and the ideal NSE and KGE are one.

| Method | P-bias | NSE ↑ | KGE ↑ |
| --- | ---: | ---: | ---: |
| Temporal interpolation | **−0.0331%** / −0.0337% | 0.88761 / 0.89741 | 0.93827 / 0.94185 |
| Global correction | +0.1595% / +0.2270% | 0.88742 / 0.90394 | 0.94266 / 0.94938 |
| Adaptive STAIR | +0.6290% / +0.0886% | 0.89674 / 0.91419 | 0.94762 / 0.95305 |
| One-season ridge | +0.1709% / −0.3171% | 0.90536 / 0.91696 | 0.93872 / 0.93314 |
| 20-year ridge | +0.6753% / +0.1372% | 0.90451 / 0.91751 | 0.93974 / 0.93514 |
| One-season cubic splines | +0.3604% / −0.0214% | 0.90410 / 0.91396 | 0.93121 / 0.92265 |
| 20-year cubic splines | +0.4638% / −0.0354% | 0.90488 / 0.91705 | 0.93708 / 0.93077 |
| One-season random forest | +0.4777% / **−0.0060%** | 0.89981 / 0.91520 | 0.93644 / 0.93246 |
| 20-year random forest | +0.6417% / +0.1149% | 0.90359 / 0.91702 | 0.93915 / 0.93215 |
| One-season Extra Trees | +0.7144% / +0.2936% | 0.89798 / 0.91353 | 0.93553 / 0.93097 |
| 20-year Extra Trees | +0.6882% / +0.1657% | 0.90222 / 0.91717 | 0.93946 / 0.93395 |
| One-season histogram boosting | −0.3260% / −0.7112% | 0.90238 / 0.91311 | 0.93533 / 0.92928 |
| 20-year histogram boosting | +0.6528% / +0.1842% | 0.90404 / 0.91722 | 0.93735 / 0.93111 |
| One-season neighborhood boosting | −0.4082% / −0.8309% | 0.90162 / 0.91253 | 0.93639 / 0.93597 |
| 20-year neighborhood boosting | +0.3134% / +0.0063% | **0.90790** / **0.92191** | 0.94068 / 0.94006 |
| One-season CNN | +0.4914% / −0.0427% | 0.89663 / 0.91414 | **0.94790** / **0.95375** |
| 20-year CNN | +0.4713% / +0.1015% | 0.89893 / 0.91517 | 0.94414 / 0.94706 |
| Tuned 20-year CNN | +0.2826% / −0.0641% | 0.89974 / 0.91580 | 0.94552 / 0.94844 |
| Pixel LSTM | +0.7685% / +0.1996% | 0.89867 / 0.91560 | 0.94330 / 0.94359 |
| ConvLSTM | +0.8787% / +0.3292% | 0.89815 / 0.91543 | 0.94477 / 0.94626 |

Neighborhood boosting leads NSE on both masks. On fixed reference
pixels, NSE gives the same ranking as RMSE. The one-season CNN has
the highest KGE on both masks, slightly ahead of adaptive STAIR. Its
spread is close to the observed spread. Temporal interpolation has the
smallest absolute cloud P-bias, while the one-season random forest and
long-history neighborhood boosting are nearly tied for stripes.
P-bias can cancel opposing errors across pixels and dates, so read
it with the error and spatial metrics above. The `ndvi_diagnostics`
entries in the recurrent `run.json` include each model's correlation,
spread ratio, and mean ratio.

## CNN spatial correction benchmark

The CNN learns NIR and red corrections to the adaptive STAIR estimate.
Its inputs are the same adaptive and reference reflectance and calendar
features used by ridge, arranged on the 30 m grid, plus validity flags
for the two reflectance inputs. Its three convolution layers give each
prediction an 11 × 11 neighborhood, or 330 m across. The model starts
from the adaptive estimate and updates only the pixels inside each
artificial gap.

Training uses the same held-out pixels and acquisition dates as the
previous nonlinear run. The loss averages the squared band residual
errors within each acquisition before combining acquisitions. We
selected the number of training epochs with the existing pre-2025
development blocks. The test replays the saved ridge and nonlinear fits
on the same two masks and verifies the preceding benchmark's support and
per-date baseline scores before reporting any comparison.

The development blocks selected one epoch for the 2024 CNN and 13 for
the 2005–2024 CNN out of 16 candidates. The two final fits used 33 and
325 acquisitions, respectively. The table reports pooled NDVI RMSE on
190,672 common cloud and 249,039 common stripe pixel-dates.

| Model | Cloud | Stripes |
| --- | ---: | ---: |
| Adaptive STAIR | 0.07189 | 0.06791 |
| One-season ridge | 0.06883 | 0.06681 |
| One-season CNN | 0.07193 | 0.06793 |
| Long-history ridge | 0.06913 | 0.06658 |
| Long-history neighborhood boosting | **0.06790** | **0.06478** |
| Long-history CNN | 0.07113 | 0.06752 |

The long CNN lowered RMSE relative to adaptive STAIR. Both ridge fits
beat their corresponding CNN on both masks, and long-history
neighborhood boosting had the lowest RMSE. The paired 95% month-block
interval for long CNN minus
neighborhood boosting was +0.00129 to +0.00557 on clouds and +0.00134
to +0.00470 on stripes. The long CNN's NDVI standard deviation was
97.9% of the observed spread on clouds and 97.0% on stripes. Boosting
retained 96.4% and 95.5%, respectively. This explains the long CNN's
higher KGE compared with boosting even though its pixel error was
larger. Long CNN predictions used the matching ridge fallback at
7 cloud pixels and 14 stripe pixels across the benchmark.

The CNN beat long-history boosting on 9 of 32 cloud dates and 5 of 32
stripe dates. The best relative examples were June 11 for clouds and
October 18 for stripes. March 15 had the largest increase in error for
both masks and had incomplete scoring coverage. These comparisons
describe a single compact CNN on the Clayton 2025 test.

Use the historical input cache and extract the previous nonlinear
benchmark archive into the repository root. Both archives are listed in
the [output download index](../outputs/README.md).

```powershell
python -m pip install -r requirements-cnn.txt
python scripts/compare_spatial_cnn.py
$env:QT_QPA_PLATFORM = 'windows'
& 'C:\Program Files\QGIS 3.44.15\bin\python-qgis-ltr.bat' scripts/build_history_comparison_qgis.py outputs/spatial_validation/clayton_cnn_2005_2025 --spatial-pattern cloud
```

Repeat the QGIS command with `--spatial-pattern stripes`. Both projects
use a common display stretch. Open **Project → Layouts → NDVI | Selected
date comparisons** for dates with the lowest and highest NDVI RMSE
difference between long-history CNN and neighborhood boosting. The run
records the fitted CNN checkpoints, per-date scores, paired monthly
intervals, smooth density curves and pooled NDVI diagnostics.

### Time ordered cross validation and CNN tuning

The long-history CNN can be tuned on three forward-chaining folds.
Use March–October acquisitions in 2005–2021 to validate on 2022, in
2005–2022 to validate on 2023, and in 2005–2023 to validate on 2024.
Each held-out year uses whole acquisitions. Train the same architecture
at two learning rates, 0.001 and 0.002, and compare checkpoints after
4, 8, and 13 epochs. Choose the lowest mean validation-fold NDVI MAE.
Each validation year contributes the same weight to the selection.

The script writes the selected settings before evaluating 2025. It
refits using March–October 2005–2024 and compares the result with the
previous CNN, ridge and neighborhood boosting on the original 2025
scoring masks. The 2025 results are a repeat comparison with the
established test year.

```powershell
python scripts/tune_spatial_cnn.py
```

The tuning run uses the same input cache and nonlinear benchmark as
the CNN benchmark, plus the original CNN checkpoints and manifest.
Under `outputs/spatial_validation/clayton_cnn_rolling_2005_2025/`,
`cross_validation.json` records all candidate scores, dates, and
training support. `run.json` contains the repeated 2025 evaluation,
including Lin's concordance correlation, average bias in low and high
observed NDVI, empirical signed-error percentiles, and paired MAE
differences with calendar-month bootstrap intervals. The saved
checkpoint is `twenty_full_cnn_tuned.pt`.

The folds selected a learning rate of 0.001 and eight epochs.
Mean validation-fold MAE was 0.04679, compared with 0.04703 for the
original 13-epoch settings across the same folds. The selected CNN's
MAE was 0.04168 on the 2022 fold, 0.04852 on 2023, and 0.05017 on
2024. The validation seasons differed more than the candidate averages.

| 2025 model | Cloud MAE | Stripe MAE | Cloud Lin concordance | Stripe Lin concordance |
| --- | ---: | ---: | ---: | ---: |
| Long-history ridge | 0.04753 | 0.04388 | 0.95057 | 0.95667 |
| Original long-history CNN | 0.04893 | 0.04429 | 0.94837 | 0.95627 |
| Tuned long-history CNN | 0.04854 | 0.04389 | 0.94890 | 0.95668 |
| Long-history neighborhood boosting | **0.04692** | **0.04269** | **0.95227** | **0.95916** |

Tuning lowered MAE relative to the original CNN by 0.00039 on clouds
and 0.00040 on stripes. The paired 95% month-block intervals for tuned
minus original were −0.00064 to −0.00018 for clouds and −0.00061 to
−0.00021 for stripes. Relative to neighborhood boosting, the tuned
CNN's MAE was higher by 0.00162 and 0.00120, respectively. The
intervals for those differences were positive on both masks. The
intervals use eight months of the same previously reviewed Clayton
benchmark. An independent year or site is needed to check transfer.

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
