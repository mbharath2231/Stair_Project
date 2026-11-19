# Clayton fusion comparison using two training histories

## Experiment completed on November 18, 2026

The first trainable extension fits a date-balanced ridge regression to the
NIR and red errors of the existing two-pair residual fusion. Both histories
use the same seven features, fixed physical scaling, and ridge strength of
0.001. Features comprise an intercept, fused NIR and red, daily MODIS NIR
and red, and sine/cosine of the calendar-year phase. The fitted correction
is added to the existing fused reflectance. NDVI is subsequently derived
from the corrected NIR and red.

| Version | Supervision period | Supported acquisitions | Training pixel-dates |
| --- | --- | ---: | ---: |
| Long history | March–October, 2005–2024 | 354 | 6,827,745 |
| Short history | March–August 2024 | 25 | 532,215 |

The archive contains 21 calendar years from 2005 through 2025. Twenty years
provide training labels and 2025 provides the shared evaluation. The short
period was selected by the user. March–October is the archived season.

Each training acquisition is withheld while its fused predictors are
constructed. Both training versions use the same cached context for a
given acquisition, including up to 32 days on either side. Labels are
restricted to the respective supervision period. Each supported date has
total fitting weight one, distributed equally over its valid pixels.
All coefficient fitting and training contexts precede 2025. The ridge
strength and features were fixed before evaluation.

Evaluation is retrospective leave-one-acquisition-out reconstruction.
Other 2025 Landsat dates provide the local anchors. The target Landsat
acquisition is excluded from its own prediction and is used for scoring
after prediction. This evaluates the intended reconstruction of imagery
on days when Landsat is missing. A future forecasting experiment would
require a causal anchor policy. Chronological evaluation follows the
time-series guidance in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split).

## Shared 2025 evaluation

The runner evaluated 32 acquisitions with at least 60% QA-clear Landsat
coverage. Twenty-five had common NDVI support across all methods. Results
below use identical reference pixels for every method within each variable.

| Method | NIR RMSE | Red RMSE | NDVI RMSE | NDVI MAE | NDVI Wasserstein-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing residual fusion | 0.04578 | 0.02094 | 0.07266 | 0.04793 | 0.00609 |
| Six-month-trained correction | 0.04701 | 0.02101 | 0.07327 | 0.05212 | 0.02068 |
| Twenty-year-trained correction | 0.04326 | 0.02045 | 0.07246 | 0.05073 | 0.01693 |
| Landsat interpolation with the fusion pairs | 0.04480 | 0.02068 | 0.07047 | 0.04599 | 0.00850 |
| Nearest-clear Landsat interpolation | 0.04704 | 0.02063 | 0.06896 | 0.04594 | 0.00691 |
| MODIS resampled to the Clayton grid | 0.06234 | 0.03281 | 0.14072 | 0.11041 | 0.05068 |

NIR and red each have 561,221 shared pixel-date comparisons. NDVI has
561,056, representing 66.6% of the 842,196 QA-clear reference pixel-dates.
NDVI requires finite, nonnegative NIR and red with a positive sum. Every
method's coverage is recorded separately in `run.json` and
`per_date_metrics.csv`. Negative finite reflectance remains available for
the reflectance evaluation.

## Insights from this comparison

- Twenty-year training reduced NIR RMSE by about 5.5% relative to existing
  fusion and 8.0% relative to six-month training. Red RMSE also improved.
- NDVI RMSE improved by about 1.1% relative to six-month training and 0.3%
  relative to existing fusion. The paired month-block bootstrap interval
  for long-history minus short-history NDVI RMSE is approximately
  −0.00153 to +0.00010. It spans zero. The bootstrap uses seven supported
  calendar-month blocks and 5,000 resamples within this chip and year.
- Existing fusion has lower NDVI MAE and a closer empirical NDVI
  distribution than either learned correction. NIR/red RMSE and the NDVI
  ratio respond differently to the fitted corrections. Density agreement
  and paired error measure different aspects of reconstruction quality.
- Nearest-clear Landsat interpolation achieved the lowest NDVI RMSE in
  this test. The same-pair Landsat control also outperformed the fusion
  variants on NDVI. The MODIS adjustment merits further evaluation.
- Restricting the test to March–August gives NDVI RMSE of 0.07399 for
  long-history training and 0.07403 for short-history training. Their
  difference is approximately 0.00004. September and October lie outside
  the short model's seasonal training coverage, which affects the full
  season comparison.
- Coverage remains an important development task. April 1, April 8,
  June 3 and June 11 lack valid target MODIS retrievals under the archived
  QA rule. March 15, May 18 and June 27 lack supported two-sided matched
  pairs. Their gaps remain explicit in the exported rasters and report.

## Scope and remaining evaluation

These results apply to the selected ridge correction, one Clayton chip,
and one test year. The two histories differ in length, recency, seasonal
coverage, and sensor composition. Further controlled comparisons can
isolate those effects. Independent spatial transfer remains untested.

The archived Landsat 5/7/8/9 reflectance retains its mission-native
spectral response. Independent Landsat–MODIS registration remains
unverified. MCD43A4 uses overlapping 16-day retrieval windows with daily
output dates, so adjacent products share temporal information. QA-clear
Landsat serves as a satellite reference with its own uncertainty.
[MCD43A4 product documentation](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD43A4)

Additional independent withheld years, sensor-era sensitivity, and
quality-screened anchor comparisons are the next evaluation steps. The
coverage gaps and Landsat-only control remain part of that evaluation.

## Separate Landsat and MODIS inputs

Run `python scripts/compare_training_history.py --compare-sensors` to fit
three ridge corrections for each training history. The fused version uses
the existing Landsat–MODIS residual fusion and target-date MODIS. The
Landsat-only version uses two-sided interpolation of nearby Landsat
observations. The MODIS-only version uses target-date MODIS resampled to
the Landsat grid. Each learns the difference between its input estimate
and QA-screened Landsat NIR and red. The MODIS-only version uses Landsat
as training labels and for scoring. Its predictions use MODIS and the
calendar date. Each history uses the same dates and pixels for all three
fits. The fused coefficients match the original run.

On 561,051 shared NDVI pixel-dates in 2025, the linear models achieved
the following RMSE. Coverage is each method's share of eligible Landsat
reference pixels with valid predictions, measured before intersecting
the masks for scoring.

| Prediction inputs | Six-month fit | Twenty-year fit | NDVI coverage |
| --- | ---: | ---: | ---: |
| Landsat and MODIS | 0.07327 | 0.07246 | 66.6% |
| Landsat only | 0.07367 | 0.06962 | 97.4% |
| MODIS only | 0.15768 | 0.14040 | 79.3% |

Uncorrected Landsat interpolation scored 0.06896 on those same pixels.
The 20-year fused correction had the lowest NIR and red RMSE, 0.04326
and 0.02045 respectively. The 20-year Landsat-only correction had the
lowest NDVI RMSE among the fitted models. The paired month-block interval
for the 20-year fused minus Landsat-only NDVI RMSE spans zero. These
results cover one chip and one evaluation year. The six-month model's
September and October test dates are outside its training season.

The sensor run writes six model files, per-date metrics, comparative
GeoTIFFs and `run.json` to
`outputs/fusion_validation/sensor_comparison_2005_2025/`. Its QGIS project
uses the builder below with this sensor output directory. Under
**Project → Layouts**, choose a sensor comparison for NIR, red, or NDVI,
then choose the six-month or twenty-year training history.

## Reproduce and review in QGIS

Run from the repository root with the project Python environment.

```powershell
python -m pytest -q
python scripts/compare_training_history.py
```

The runner uses the local QA-screened two-band caches at
`data/ndvi_inputs/march_october/`. It performs training and evaluation
offline. The default output is
`outputs/fusion_validation/history_comparison_2005_2025/`. For a rerun,
choose a fresh directory with `--output`.

Build the QGIS project with the Python supplied by QGIS.

```powershell
$env:QT_QPA_PLATFORM = 'windows'
& 'C:\Program Files\QGIS 3.44.15\bin\python-qgis-ltr.bat' scripts/build_history_comparison_qgis.py outputs/fusion_validation/history_comparison_2005_2025
```

Open `training_history_comparison.qgz`. Choose NIR, Red, or NDVI in the
layer tree, then a date and model. Project layouts compare the reference,
existing fusion, short-history correction, and long-history correction.
Overview dates are selected using common coverage. All dates remain
available in the layer tree. Each variable has a shared fixed display
scale across all dates and methods.

Each exported TIFF contains NIR in band 1, red in band 2, and NDVI in band 3.
The QGIS project assigns the appropriate band and renderer explicitly.
The three variable views use the stored quantitative values. The project
uses relative paths and should travel with its `rasters/` folder.

The output also includes the two reloadable model NPZ files, input and
source-code SHA-256 fingerprints, per-date metrics, pooled and monthly
scores, input-support diagnostics, and probability-density plots.

On this device, the existing `.venv` launcher refers to an unavailable
Python installation. The run and tests used QGIS Python 3.12 with the
existing project packages appended to `sys.path`. The QGIS project was
built and reopened with all 672 styled layers and three layouts.

The same fallback can rerun the experiment into a fresh folder on this
device.

```powershell
@'
from pathlib import Path
import os, runpy, sys
paths = [str(Path(p).resolve()) for p in (".venv/Lib/site-packages", "scratch/ndvi_python_deps")]
sys.path.extend(paths)
os.environ["PYTHONPATH"] = os.pathsep.join(paths)
sys.argv = ["scripts/compare_training_history.py", "--output", "outputs/fusion_validation/history_comparison_rerun"]
runpy.run_path(sys.argv[0], run_name="__main__")
'@ | & 'C:\Program Files\QGIS 3.44.15\bin\python-qgis-ltr.bat' -
```
