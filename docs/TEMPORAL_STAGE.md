# Temporal interpolation: first maintained STAIR stage

Implemented and evaluated on 2026-11-06. This is the baseline for subsequent
global and adaptive spatial correction. Spatial correction and fusion are not
applied in this workflow.

## First real land-area experiment

- Area: Clayton Township farmland, Michigan, centered at -84.2333, 41.8611.
- Requested size: 5 km square. Actual aligned grid: 167x167 pixels, 5010 m square,
  EPSG:32616, 30 m pixels. Exact transform is recorded in each `run.json`.
- Target: July 22, 2022. The target passes the selected QA screen over this chip.
- Inputs: original Landsat 8/9 Collection 2 Level 2 observations from June 20
  through August 23. Fourteen acquisition dates are cached. Reference rasters
  from earlier imputation/fusion experiments are never used.
- Bands: NIR (`SR_B5`), Red (`SR_B4`), Green (`SR_B3`).
- Main validation: a fixed cloud-shaped area withholding 6,385 pixels, selected
  by coordinates independently of image brightness and prediction errors.
- Temporal reconstruction covers all 6,385 withheld pixels on this target.
  All allowed clear target values are preserved exactly.

| Band | Held-out RMSE | MAE | Bias (prediction - truth) | Pearson r |
|---|---:|---:|---:|---:|
| NIR | 0.03377 | 0.02561 | -0.01871 | 0.8677 |
| Red | 0.01912 | 0.01285 | -0.00714 | 0.8755 |
| Green | 0.01548 | 0.01092 | -0.00770 | 0.8524 |

These errors are in surface-reflectance units. The baseline has a downward
bias, particularly in NIR. Future spatial stages must demonstrate their actual
effect on this same evaluation; an improvement is not assumed.

Additional masks on the same target:

| Holdout | Eligible/scored pixels | NIR RMSE | Red RMSE | Green RMSE |
|---|---:|---:|---:|---:|
| Cloud shape | 6,385 / 6,385 | 0.03377 | 0.01912 | 0.01548 |
| Diagonal stripes | 8,489 / 8,489 | 0.03742 | 0.01715 | 0.01409 |
| Entire scene | 27,889 / 27,889 | 0.03663 | 0.01724 | 0.01401 |

These are three evaluations of one date/site, not evidence of universal
accuracy. Full-scene holdout evaluates temporal prediction; a later global
correction cannot estimate target clear-pixel bias in that scenario without
an explicit fallback policy.

## Defined algorithm

For each pixel and band, find the nearest QA-clear observation strictly before
the target calendar day and the nearest strictly after it. Both must be within
32 days of the target by default, inclusively. Thus the maximum permitted total
interval is 64 days. The limit is configurable and reported, not an assertion
that all vegetation changes remain linear over that interval.

```text
w = (target_day - before_day) / (after_day - before_day)
prediction = before_value + w * (after_value - before_value)
filled = observed_target where available, otherwise prediction
```

Dates use UTC calendar days, not float32 epoch milliseconds. Every target-day
acquisition is excluded, even if supplied accidentally in the reference input.
Daily mosaics use source IDs sorted ascending, with the last valid pixel
winning; all requested bands share one valid mask.

Defaults require two-sided support. If only one side exists, the prediction
remains missing. `--allow-one-sided` permits carrying an available nearby value,
but labels it separately. No observations means an unresolved prediction, not
zero. A genuine zero reflectance is allowed. There is no extrapolation,
clipping, feathering, texture injection, or spatial adjustment.

The compatibility function `impute_temporal()` still returns the pair expected
by earlier scripts. Its safer defaults now reject stale/one-sided support;
historical workflows may consequently contain more explicit unresolved pixels.
`interpolate_temporal_ee()` also provides diagnostics. The NumPy implementation
can be imported and tested without Earth Engine or plotting dependencies.

## Preprocessing and holdout separation

`stair/landsat.py` scales optical stored values by `0.0000275` and adds `-0.2`.
QA_PIXEL bits 0-5 reject fill, dilated cloud, cirrus, cloud, shadow, and snow.
QA_RADSAT must be zero, a conservative saturation/occlusion screen. All three
bands must be valid. Negative surface reflectance is retained; no arbitrary
brightness threshold or lake exemption is applied. This new loader is separate
from the old raw-DN loaders so existing DN-dependent scripts are not silently
reinterpreted. Official reference: [Landsat C2 L2 catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2).

The runner removes the entire target from the reference stack. Artificially
withheld values become NaN in the allowed target. The interpolation engine sees
only references and allowed target values. It never receives held-out truth.
Truth is used in a separate scoring/plotting layer. Scores use originally
QA-clear withheld pixels and common finite support across the bands; missing
predictions reduce reported coverage. Missing metrics are `null`, never zero.

## Running it on Windows

The project now has a local `.venv` with the necessary libraries installed.
For a new environment with a normal Python installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\earthengine.exe authenticate
```

Fetch original Landsat observations and generate the main report:

```powershell
.\.venv\Scripts\python.exe scripts/run_temporal_validation.py --fetch
```

Regenerate from the cache without network or authentication:

```powershell
.\.venv\Scripts\python.exe scripts/run_temporal_validation.py
.\.venv\Scripts\python.exe scripts/run_temporal_validation.py --pattern stripes --output outputs/temporal_validation/clayton_20220722_stripes
.\.venv\Scripts\python.exe scripts/run_temporal_validation.py --pattern scene --output outputs/temporal_validation/clayton_20220722_full_scene
```

The project defaults to `stair-project-509517`, overridable with `--project` or
`EE_PROJECT_ID`. Downloads go directly to a local cache; no Drive exports or
Earth Engine batch tasks are submitted. `--lon`, `--lat`, `--size-m`, `--crs`,
`--target-date`, `--max-distance-days`, and `--input` configure another chip.
Use distinct input/output paths for new experiments. The initial renderer
expects NIR, Red, Green in that order.

## Visual and analytical deliverables

Open `outputs/temporal_validation/clayton_20220722/index.html`.

- `temporal_overview.png`: withheld truth, allowed target with gaps, temporal
  reconstruction, NIR signed errors, support classes, and support interval.
- `temporal_bands.png`: truth/reconstruction/error for each band, with identical
  truth/prediction scales and a fixed 0-0.1 absolute-error scale.
- `temporal_trace.png`: a fixed center pixel's allowed observation dates,
  selected bracketing interval, predicted target, and withheld target truth.
- Analytical GeoTIFFs: QA-clear truth, allowed target, temporal-only prediction,
  filled target, signed held-out error, holdout mask, per-band support counts,
  prediction status, before/after ages, and fill-origin labels.
- `run.json`: exact grid, source IDs/dates, QA policy, reference dates, cache
  SHA-256, scientific module SHA-256 hashes, software versions, and metrics.

No-data is -9999 in float GeoTIFFs and 255 in categorical GeoTIFFs. Category
zero remains valid. Colors are display transformations only; values are never
derived from display products. Gray in error plots is outside evaluation or
unsupported, rather than zero error. Support maps describe prediction support
over the whole scene, including pixels where the final fill retains observations.

Prediction status: 0 unresolved, 1 interpolated, 2 prior carry, 3 future carry.
Fill origin: 0 unresolved, 1 observed, 2 interpolated, 3 prior carry, 4 future carry.

## Checks performed

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_temporal_ee.py --real-chip data/temporal/clayton_20220722.npz
```

- 21 automated tests passed, covering nearest clear pixels, masks, missing/empty
  series, one-sided options, distance bounds, year boundaries, exact linear
  reconstruction, preservation of observations, target exclusion, scoring,
  output CRS/transform/no-data, and cached report generation.
- Actual Earth Engine execution matched NumPy for 112 values across eight
  tiny edge-case experiments, including empty collections and stale support.
- Actual Earth Engine execution matched NumPy over **all 27,889 real chip
  pixels for 15 fields**: three bands each of prediction, before age, after age,
  support count, and status. Absolute/relative tolerance was 1e-7.
- The server check caught and resolved a band-naming restriction and unbounded
  mosaic geometry issue that graph serialization alone could not detect.
- Figures were opened and visually checked. Observed-value preservation is
  asserted during every report run. Source arrays are cached without pickle.

Runtime validation records are `outputs/temporal_validation/ee_parity.json`
and `ee_real_chip_parity.json`. Cache and generated reports are ignored by Git
and remain available locally. The full development source and tests are tracked
as changes for review; no commit was made automatically.

## Next stage

Retain this date, exact grid, input cache, holdout, and fixed display scales.
First specify and test global bias correction; then class-specific adaptive
correction using a reference independent of the withheld target. Evaluate
each against the temporal baseline and export signed stage differences alongside
held-out error. Enforce an unchanged-scene invariant and define sparse/fully
cloudy-class fallbacks before accepting either correction. The ambiguous
centering in the paper's printed correction equation still needs reconciliation;
the earlier implementation is not silently accepted as the correct formula.
