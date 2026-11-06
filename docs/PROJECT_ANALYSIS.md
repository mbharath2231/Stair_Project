# Research and implementation review

Reviewed 2026-11-06. Repository baseline: `56bda16` (`After the Fusion, updated the code`).

## Assessment

This project is an exploratory implementation of satellite gap filling and Landsat-MODIS fusion for the Raisin River/Clayton Township study area in Michigan. It contains substantial work on data acquisition, temporal interpolation, spatial correction, visualization, and exports. It is not yet a reproducible implementation of the complete cover-crop study, and several recent demonstration scripts implement materially different algorithms under the same STAIR label.

The strongest starting point for further work is the modular `stair/` package, combined with the two-reference residual-fusion equation in `stair/fusion.py`. The immediate priority is to establish a single scientifically specified pipeline and independent validation before expanding the application.

No existing scientific source code, inputs, or outputs were changed during this review. Added review files are this document, `audit_project.py`, and `project_audit.json` in `docs/`.

## Review scope and evidence

- Read both supplied papers: the 13-page Luo et al. (2018) STAIR paper and the 12-page Zhou et al. (2022) cover-crop paper. The latter has no usable extracted text; its pages were rendered and read visually. Inspected the STAIR method figures and equations visually as well as through extracted text.
- Parsed all 54 pre-existing Python files, totaling 8,299 lines, and inspected the core package and active scientific workflows in detail. Inspected archived implementations and migration/API compatibility.
- Inspected all 179 GeoTIFF headers. They include 120 eight-bit display products and 59 other rasters; display composites must not be treated as quantitative reflectance inputs.
- Decoded the saved July 22, 2022 benchmark rasters, checked their grids, recomputed metrics, and compared selected same-date output rasters.
- Reviewed the README, configuration, dependency declarations, diagrams, shapefile metadata, and all 16 Git commit subjects/dates.
- Verified satellite scaling, product definitions, and quality-mosaic semantics against official Earth Engine documentation.

Evidence levels are distinguished below: observed source behavior, measured saved-artifact behavior, and concerns needing a live Earth Engine check. Syntax checks do not establish remote graph correctness.

The bundled Python has NumPy, Pillow, PDFium, and PDF readers, but lacks Earth Engine, GeoPandas, Rasterio, and Matplotlib. I did not authenticate, run Earth Engine jobs, submit Drive exports, or regenerate scientific outputs. The original externally referenced Downloads dataset is not in this repository. The online cover-crop supplement was located, but its download returned HTTP 403; its detailed time-window rules and data table remain unverified. The 2018 supplement was not supplied or reviewed.

## 1. How the papers fit together

### Luo et al. (2018): constructing the satellite time series

Title: *STAIR: A generic and fully-automated method to fuse multiple sources of optical satellite data to generate a high-resolution, daily and cloud-/gap-free surface reflectance product*. DOI: [10.1016/j.rse.2018.04.042](https://doi.org/10.1016/j.rse.2018.04.042).

The paper expands STAIR as **SaTellite dAta IntegRation**. The repository's expansion, “Spatio-Temporal Adaptive Image Restoration,” is a project-specific name and should be distinguished from the publication's name.

The method has two principal stages:

1. Fill missing pixels independently in the satellite time series, using temporal information followed by correction within homogeneous image segments.
2. On matching Landsat-MODIS dates, calculate fine/coarse differences, interpolate those differences through time, and add daily MODIS reflectance.

Its test setting was Champaign County, Illinois, 2000-2015, April 1-November 1. It used Landsat 5/7/8 and 500 m MODIS MCD43A4 NBAR, with six spectral bands: Blue, Green, Red, NIR, SWIR1, and SWIR2. These are important distinctions from the project's local 2021/2022, predominantly three-band demonstrations.

**Temporal imputation, Section 2.2.1.** For a pixel with clear observations at nearby times, a local linear model estimates missing reflectance. The paper permits interpolation using one/two references and regression when more references are available. Therefore a correctly implemented nearest-before/after interpolation is a reasonable restricted case; it should not be described as a multi-observation regression.

**Spatial correction, Section 2.2.2.** The method assumes gap and clear communities retain approximately stable relative reflectance over short periods, exemplified as less than two to three weeks. A reference-date gap/clear community difference supports correction. K-means separates homogeneous land covers so each class can have its own correction. The paper searches cluster counts 2-8 and selects the highest gap statistic, rather than always using four clusters.

The exact correction equation needs deliberate reconciliation. The published page 90 prints a sum involving the temporal prediction, target clear-community mean, and reference community difference. The repository instead adds a clear-pixel temporal residual plus the reference community difference. The paper's printed notation does not explain the repository's subtraction of the temporal clear mean. That subtraction must be documented as an interpretation or modification, with explicit invariants and reference evidence, rather than asserted to be an exact transcription. See finding F7 below for a counterexample to the current implementation.

**Fusion, Section 2.3.** For co-registered, equally scaled images:

```text
epsilon(Tj) = L(Tj) - M(Tj)
Delta(tp)   = temporal interpolation of epsilon at tp
I(tp)       = M(tp) + Delta(tp)
```

Between two references, linear interpolation becomes:

```text
w1 = (T2 - tp) / (T2 - T1)
w2 = (tp - T1) / (T2 - T1)
I(tp) = M(tp) + w1 * [L(T1)-M(T1)] + w2 * [L(T2)-M(T2)]
```

Two references are sufficient within a single interval of a piecewise-linear residual series. What is missing here is the automated construction and selection of a full-season series of valid pairs, not a requirement to use every pair simultaneously for every prediction.

The paper also performs image co-registration, quality screening, and produces per-pixel quality flags. Bilinear resampling alone does not estimate an actual geolocation shift. A 30 m output grid carries inherited Landsat detail; it does not establish an independent daily 30 m observation.

**Evidence and limits.** The paper evaluates held-out 200x200 patches from scenes with over 60% clear pixels, compares with two-pair STARFM, reports Pearson correlation and RMSE, and uses common comparison support. Its significance tests span multiple dates/years/bands. These results do not transfer automatically to this implementation. Its discussion stresses spectral compatibility, registration, BRDF effects, and the need for adequate observations. “Cloud free” is an output goal requiring verification, not an invariant established by naming a stage.

### Zhou et al. (2022): converting that series into cover-crop detections

Title: *Recent Rapid Increase of Cover Crop Adoption Across the U.S. Midwest Detected by Fusing Multi-Source Satellite Data*. DOI: [10.1029/2022GL100249](https://doi.org/10.1029/2022GL100249).

This is a downstream application of STAIR, rather than another definition of its fusion equation. It studies corn/soybean regions from 2000-2021. The essential application chain is:

```text
daily 30 m Red/NIR -> NDVI time series
-> separate soil baseline and cash-crop growth
-> accumulate remaining cover-crop signal
-> learn environmental thresholds
-> classify pixels and aggregate to fields/counties
-> validate maps and analyze trends
```

The NDVI decomposition is:

```text
NDVI(d) = soil(d) + cash_crop(d) + cover_crop(d)
cover_crop(d) = NDVI(d) - soil(d) - cash_crop(d)
cSign = sum of cover_crop(d) over the cover-crop window
```

The soil baseline is a minimum NDVI in a dynamically selected window. A logistic curve models cash-crop growth using peak-growing-season observations. Dormant, cash-crop, and cover-crop windows are determined dynamically at the pixel level. Their exact operational definitions are delegated to the unavailable supplement, so they should not be replaced silently with arbitrary calendar windows.

The threshold model is a Random Forest predicting a **cover-feature threshold**, not simply a classifier trained on fused images. Inputs include temperature, precipitation, VPD, soil texture, soil organic carbon, and latitude/longitude. County thresholds derived using NASS 2012/2017 cover-crop rates provide training targets. The paper uses ten-fold validation. Threshold inversion and preprocessing must be isolated within training folds in a new implementation.

The study controls winter wheat and weeds using corn/soybean crop masks, counties with over 40% corn/soybean coverage, relative cash/cover crop peak conditions, and a reported peak cover NDVI condition of 0.38. It classifies a field as cover cropped when over 40% of its pixels qualify, excluding fields smaller than 150x150 m. These are study rules to reproduce or validate for a new area, rather than universally established thresholds.

Reported results include threshold prediction R-squared 0.94, county adoption R-squared 0.63 with RMSE 2.27 percentage points in 2017 (Figure 2, N=825), and field detection values labeled “precision” of 87%, 83%, and 65% in three validation settings. The printed field metric denominator warrants careful interpretation; a new evaluation should explicitly define TP, FP, FN, precision, and recall. The adoption increase and association with conservation funding are observational findings, not a causal intervention estimate.

The paper acknowledges its March 15-October 15 fusion window misses some early spring, late fall, and winter signals, indirectly estimates cover-crop phenology, and depends on crop-mask accuracy. A growing-season-only NDVI product is not sufficient for every winter cover-crop use case.

The [associated Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.4xgxd25dg) contains two period-average county adoption rasters and a README. It is not a complete field-label training set or the complete daily fusion archive. The paper states that RMA labels are not public under the relevant agreements.

## 2. Repository architecture and development history

### Current module map

| File | Actual responsibility | Status |
|---|---|---|
| `config.py` | Paths, project ID, older collection/date defaults | Useful starting point; partially bypassed by scripts |
| `stair/core.py` | EE initialization, shapefile AOI, Landsat/MODIS loading, masks, NDVI arithmetic | Central preprocessing needs consolidation |
| `stair/temporal.py` | Nearest-before/after time-weighted interpolation | Important masking and empty-series defects |
| `stair/adaptive.py` | Global/class corrections, Weka clustering, optional shared segments, smoothing | Genuine clustering implementation; equation/null handling unresolved |
| `stair/fusion.py` | Single-reference difference transfer, hard-coded seasonal helper, two-reference EE/NumPy fusion | Dual residual equation is the preferred foundation |
| `stair/diagnostics.py` | Fill-stage tracker, CDL pixel selection, buffered bounding boxes | Useful utilities; not a cover-crop detector |
| `stair/visualization.py` | Folium maps and local plots | Uses older raw-DN display conventions |
| `stair/__init__.py` | Public exports, version 1.0.0 | Does not expose the newly added dual-fusion helpers |
| `scripts/` | 15 active exports, validation, figure/diagram workflows | Scientific behavior differs substantially between runners |
| `archive/` | 31 Python prototypes/runners | Historical reference; many runners incompatible with current API |
| `outputs/` | Figures, diagrams, analytical and display rasters | Evidence of past execution; provenance incomplete |

The supplied boundary contains one polygon record and is declared in WGS84 / UTM zone 17N. `load_aoi` uses only the first geometry: harmless for this particular file, but an issue for a future multi-feature AOI. Most recent experiments bypass the boundary and use a hard-coded point near `[-84.2333, 41.8611]` with a nominal 5 or 7 km chip, exporting in EPSG:32616. These are local experiments rather than demonstrated full-watershed production.

### Development progression visible in Git

| Period | Evidence in commit history and source |
|---|---|
| June 23-July 10, 2026 | Initial satellite access, Red/NIR work, six-month data, temporal regression and diagnostics |
| July 17-27 | Fill-origin tracking, adaptive K-means, local 5 km experiments |
| September 22 | Further prototype updates |
| October 12 | Fusion work, modularization/archive reorganization, expanded MODIS masking |
| November 6 | New 2022 evaluations, dual-reference helpers, extensive visual demonstrations/diagrams |

The archive confirms real algorithm changes: older temporal scripts used `linearFit`; the older spatial-fusion prototype used moving-window regression on MOD09GQ. Current `learn_spatial_rules` merely stores a fine/coarse reference pair; `kernel_radius` is unused. A radius of 15 would in any event describe a 31x31 square, not the README's stated 15x15 window.

### Active workflow distinctions

| Workflow | What it actually establishes |
|---|---|
| `export_golden_day.py`, `find_best_golden_day.py` | Pull/audit reference data for a fixed 2021 local AOI |
| `export_august_fusion.py`, `validation_test.py` | Single-reference MOD09GA difference-transfer/export route |
| `generate_paper_figures.py` | Uses modular interpolation/adaptive correction with shared multiband clustering |
| `run_simulated_gap_validation.py`, `run_multi_day_validation.py` | Local-ring cluster residual correction, but boundary blending leaks held-out truth |
| `run_5day_cloudy_interpolation.py` | Actual Weka class correction on 2022 dates; benchmark uses the target date as its reference |
| `run_stair_fusion.py` | Actual two-reference MCD43A4 residual interpolation, with an external pre-imputed T2 raster |
| `run_visual_stair_fusion_progress.py` | Stage visualization; same-date residual reconstruction returns imputed Landsat |
| `run_real_cloud_fusion_single_day.py`, `run_all_layers_real_clouds_july29.py` | Heuristic texture correction and same-date residual reconstruction |
| `run_5km_stage_by_stage_progression.py` | Synthetic cloud test with heuristic corrections and same-date reconstruction |
| `run_5km_5day_analysis.py` | Single-reference transfer on days without Landsat; same-date reconstruction on overpass days |
| `update_clean_drawio.py` | Diagram authoring, not a processing engine |

No implementation was found for NDVI soil/cash/cover decomposition, dynamic phenology windows, environmental predictors, NASS threshold inversion, Random Forest training, field aggregation, cover-crop map validation, or trend analysis. Generic CDL corn/soy sampling should not be confused with these stages.

## 3. Findings to resolve before scientific expansion

### F1. The single-reference public route mixes incompatible units

`stair/core.py:80` returns Landsat C2 stored values without converting to reflectance. `get_modis_collection` also returns MODIS stored values. `stair/fusion.py:70-73` adds the MODIS numerical change directly to Landsat stored values.

Landsat reflectance is `DN * 0.0000275 - 0.2`; MODIS uses `DN * 0.0001`. Equal integer changes therefore represent different physical changes. For example, a MODIS change of 1000 means 0.1 reflectance, but adding 1000 to a Landsat DN represents only 0.0275. NDVI computed from Landsat DNs also fails to account for the additive offset. This affects the older helper/export route; the newer local NumPy benchmark explicitly scales both inputs correctly.

Normalize to physical reflectance in one preprocessing layer and declare units on outputs. Preserve QA bands separately. Official sources: [Landsat L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2), [MOD09GA](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA).

### F2. Nearest-clear temporal selection is not implemented by the current quality band

`stair/temporal.py:36-37` adds an unmasked constant timestamp. `qualityMosaic('t')`/`qualityMosaic('neg_t')` at lines 55/61 select by that timestamp, rather than a timestamp masked to the requested reflectance band's valid pixels. The latest scene can therefore win even where its reflectance is cloudy, bypassing an older clear observation.

This is a source-level defect inferred from [official qualityMosaic semantics](https://developers.google.com/earth-engine/apidocs/ee-imagecollection-qualitymosaic): all output bands come from the image selected by the quality band. Confirm with a tiny server-side example once EE is available. Mask each quality-time band to the corresponding valid observation, then carry its matching time/value through the selection.

### F3. Empty-side temporal fallbacks introduce valid zeros

The dummy image at `stair/temporal.py:51` has unmasked zero values/timestamps. Arithmetic can produce a valid erroneous interpolant, so later `unmask(v_before).unmask(v_after)` does not trigger the intended one-sided fallback. If both sides are empty, valid zero can also defeat the reference fallback in `adaptive.py:299`.

Represent missing support with masks and branch explicitly among two-sided interpolation, one-sided carry, reference fallback, and unresolved gap. Check per-pixel support as well as empty collections. Impose a scientifically chosen maximum temporal distance and export the actual support distances.

### F4. Multiple “fusion” scripts cancel MODIS algebraically

Examples: `run_5km_stage_by_stage_progression.py:206-209`, `run_real_cloud_fusion_single_day.py:203-207`, `run_all_layers_real_clouds_july29.py:224-230`, and `run_visual_stair_fusion_progress.py:166-170`.

```text
epsilon = L_imputed(tp) - M(tp)
fused(tp) = M(tp) + epsilon = L_imputed(tp)
```

This identity can be a legitimate consistency check at a known matching pair. It does not test temporal prediction or demonstrate MODIS-driven changes on an unobserved date. The July 29 real-cloud FCC output and the 5 km progression FCC output were each decoded and found exactly equal to their imputed Landsat FCC input, with maximum pixel difference zero.

Use a distinct held-out target date and interpolate residuals from allowed reference dates to evaluate fusion. Describe existing same-date demonstrations as gap filling or endpoint consistency.

### F5. Held-out truth leaks into simulated-cloud predictions

`run_simulated_gap_validation.py:144` and `run_multi_day_validation.py:145` multiply the **unmasked** `gt_band` by a feathering weight that extends into the artificial cloud gap. Some scored gap pixels therefore contain their true values. Those adaptive scores cannot be considered independent holdout results.

In `run_5day_cloudy_interpolation.py`, the July 22 benchmark also uses July 22 as `ref_date` (`main`, line 404). Its unmasked target scene is available for clustering, water decisions, and reference gap/clear means. Excluding the target day in temporal interpolation alone does not remove this leakage.

There is an additional provenance question for the July 29 imputed raster used in `run_stair_fusion.py`: the associated 2022 imputation workflow uses July 22 as a reference and includes that scene in its training archive. The externally loaded raster's exact generation lineage is unavailable, so target-date leakage into this saved fusion benchmark cannot be ruled out. Regenerate all reference imputation under the holdout policy.

### F6. Several “adaptive K-means” stages do not run K-means

The latest local visualization/real-cloud scripts add `0.35 * reference_detail` to filled Landsat pixels. This is a texture-injection heuristic, not class-specific adaptive-average correction. Their MODIS “adaptive” stage often only clips the global result to `[0,1]`. In `run_visual_stair_fusion_progress.py`, the global MODIS offset compares clear pixels against a stage that already copied those same clear pixels, so the offset is identically zero when finite.

Keep actual Weka correction in the modular package and shared multiband clustering. Treat fixed texture injection as an explicitly named experimental baseline if retained. Do not assign it evidence from the STAIR paper.

### F7. The spatial correction can add an existing land-cover contrast twice

The implemented equation in `adaptive.py` is:

```text
prediction + [mean(clear target) - mean(clear prediction)]
           + [mean(reference gap) - mean(reference clear)]
```

If temporal prediction is already exact, the first bracket is zero. The second bracket can still be nonzero simply because the gap covers different land cover. For a true/predicted gap value 0.6, clear value 0.2, and unchanged reference values, the code produces 1.0 rather than 0.6. The same problem can occur within a class if its gap/clear means differ.

This counterexample establishes that the current equation does not preserve an already correct temporal estimate. It does not by itself establish the authors' intended replacement. Reconcile the published equation, its centering convention, and any available reference implementation; test constant scenes, unchanged scenes, mixed classes, and known biases before selecting a correction.

### F8. Null statistics and sparse classes need explicit support handling

`adaptive.py` uses dictionary `.contains()` as a proxy for a valid numerical mean. A key can exist with a null value. Entirely cloudy targets, empty reference groups, or masked temporal support can invalidate subtraction even when the key exists. The adaptive fallback computes its reference gap/clear difference without the same joint guard as the global routine. Script-level `if mean_target_f` also treats a valid zero as missing, and `...getInfo() or 0` can report missing validation as zero error.

Use explicit valid counts on the common support for each statistic. Distinguish no-data, inadequate support, successful fallback, and actual zero. Class labels are spectral clusters, not verified crop identities.

### F9. MODIS product/QA policy changes by execution path

`config.py` names MOD09GQ (250 m), `core.py` hard-codes MOD09GA (500 m), and newer local scripts use MCD43A4 (500 m NBAR). These are not interchangeable products. New MCD43A4 scripts generally select reflectance directly without applying mandatory per-band quality flags. Some demonstrations paint a Landsat cloud mask onto MODIS; that illustrates a shared missing region, but does not measure independent MODIS cloud/quality coverage.

MCD43A4 is issued daily from a 16-day retrieval window, rather than being an independent instantaneous clear observation every day. Account for that temporal support in interpretation and retrospective/operational design. See the [official MCD43A4 catalog](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD43A4).

The core Landsat cloud/shadow variable names have bits 3/4 reversed; because both are removed, that swap alone does not change the mask. The core still omits dilated cloud/cirrus and other QA policies used in newer scripts. Its extra MOD09GA Red `<0.15` cutoff is an unvalidated land-surface filter that can remove legitimate bright soil; it requires an ablation rather than acceptance as a universal cloud rule.

### F10. Grid/provenance and metrics are not yet reliable contracts

The five saved benchmark reflectance rasters have matching dimensions, pixel scale, tie points, and CRS keys. That is useful positive evidence. However, `run_stair_fusion.py` reads the external T2 array without checking its transform/CRS and writes it using the target profile. Current output alignment does not prove that the original external source was aligned correctly.

`stair_dual_fusion_numpy` implements the residual equation correctly for valid, consistently aligned inputs; three isolated numerical checks passed. It does not validate shapes, units, masks, reference chronology, or that weights sum to one. EE fusion similarly needs a declared common transform, not just CRS and nominal scale. Co-registration should be measured separately from resampling.

The scripts label squared Pearson correlation as R-squared. This measures association and differs from prediction R-squared `1-SSE/SST`. In the saved Red benchmark, Pearson r is about 0.663 (r-squared about 0.439), while prediction R-squared is approximately -0.0004. Both can be reported with accurate labels.

NDVI code using `maximum(NIR+Red, 1e-4)` can turn nonpositive denominators into extreme values. The stored prior scene contains negative reflectance; reproducing this formula yields a prior NDVI RMSE around 2.38 under the original broad mask. That is a reason to audit QA and NDVI validity, not persuasive evidence of a large fusion improvement. Specify denominator/quality policy and apply identical evaluation support to all models.

### F11. Reproducibility and documentation lag behind the implementation

- Several active scripts contain `/Users/bharathmikkilineni/...` output/input paths. They cannot locate the original dataset on this Windows machine and may attempt unintended root-relative output directories.
- `fuse_date` always loads May-October **2021**, regardless of the requested year.
- `config.OUTPUT_DIR` is `data/outputs`, while most saved/current runs use top-level `outputs/`.
- README commands refer to runners moved into `archive/legacy_scripts/`. Those runners also import removed `compute_temporal_regression`, call the old interpolation signature, or select nonexistent `synthetic_ndvi`/`synthetic_red` bands.
- Moving archived runners one level deeper invalidated their old `sys.path` parent assumptions.
- `requirements.txt` uses minimum bounds, despite the README calling dependencies pinned; there is no environment lock or package metadata. Pillow is directly imported but not explicitly declared.
- No maintained automated scientific test suite was found. Files named validation/test are mainly EE experiments/export scripts, not isolated tests.
- Repeated download, export, plotting, and correction logic has drifted between scripts. Many `.first()` calls lack empty-collection and scene-selection checks. `is None` on a lazy EE object is not a reliable server-side existence check.
- The package import eagerly loads geospatial/visual dependencies; configuration creates directories during import; initialization treats every failure as an authentication problem.
- `apply_edge_blending` smooths all filled pixels before restoring clear pixels, rather than restricting smoothing to an explicit boundary ring.
- Quality tracking currently does not encode all correction/fallback/support states used by the outputs. Existing diagrams describe an intended composite workflow rather than one executable canonical pipeline.

## 4. Independent check of the saved two-reference benchmark

Source directory: `outputs/STAIR_Fusion_Evaluation/`. Target July 22, 2022; references July 14/29. Grid: 241x242 pixels, EPSG:32616, 30 m. Bands follow the script's NIR/Red/Green order.

For a fair numerical comparison, the following table uses each band's **same finite-pixel intersection** across truth, prior Landsat, MODIS, and fusion, with positive target-band reflectance. These are diagnostics of the saved files, without independently validated QA, land-cover restriction, or leakage-free generation.

| Band | Pixels | Prior Landsat RMSE | MODIS RMSE | Fused RMSE | Fused MAE | Fused Pearson r |
|---|---:|---:|---:|---:|---:|---:|
| NIR | 54,704 | 0.09341 | 0.07261 | 0.05399 | 0.03913 | 0.8473 |
| Red | 54,705 | 0.08194 | 0.03221 | 0.03671 | 0.01629 | 0.6629 |
| Green | 54,705 | 0.07559 | 0.02300 | 0.03414 | 0.01392 | 0.5613 |

Fusion improves NIR RMSE relative to both baselines. For Red and Green it improves MAE and correlation but has worse RMSE than MODIS, indicating larger error tails. Improvement depends on the metric and band.

The saved fused and post-reference arrays each contain 3,617 pixels with missing values out of 58,322 total pixels, or **6.20%**. Their matching count is consistent with gap propagation from T2; it contradicts an unconditional “100% cloud-/gap-free” description. The exact cause requires source-raster provenance and QA inspection.

The audit also confirms identical display arrays for two “imputed Landsat” / “final fusion” pairs, as predicted by their source algebra. FCC identity verifies those displays; it is not a substitute for comparing every analytical output.

Machine-readable inventory, counts, ranges, full metrics, and numerical checks are in `project_audit.json`. Run `python docs/audit_project.py` from a Python environment with NumPy to regenerate the offline audit. The TIFF decoder is narrowly scoped to these artifacts, supports the tested uncompressed/Deflate layouts, and fails on unsupported layouts rather than being a production geospatial reader. Header inventory includes LZW rasters; those were not all decoded.

## 5. Recommended development sequence

### Milestone 1: one reproducible STAIR experiment

1. Establish a Windows-compatible project environment and declare/lock dependencies.
2. Define a single configuration for AOI, exact grid, dates, sensors, bands, MODIS product, QA policy, and output directory. Keep the existing artifacts as historical evidence.
3. Use MCD43A4 for a paper-aligned baseline; retain MOD09GA only as an explicit alternate experiment.
4. Convert reflectance once, preserve metadata and source identifiers, and centralize scene availability checks and I/O.
5. Correct temporal masks/fallbacks and specify correction mathematics. Reuse one multiband clustering map and test fixed k versus gap-statistic selection.
6. Build a seasonal series of imputed matching pairs, interpolate residuals for requested dates, and export reflectance, NDVI, and support/quality flags together.

Success criterion: a documented command reproduces one small-area date series using known inputs, exact grids, traceable fallback states, and explicit unresolved gaps.

### Milestone 2: independent validation

Maintain two different experiments:

- **Gap filling:** remove known clear pixels with synthetic masks; only allowed observed target pixels can calibrate corrections. Ground truth is passed to scoring only. Test random pixels, contiguous clouds, and stripes across several dates/seasons.
- **Fusion:** remove the entire target Landsat observation from reference selection, temporal fitting, segmentation/reference statistics, and preparation of every imputed reference. Predict with target MODIS and allowed matching pairs.

Use the same QA/common support across methods. Report sample counts, missing fraction, RMSE, MAE, bias, Pearson r, prediction R-squared, and NDVI errors separately. Include MODIS resampling, prior/nearest Landsat, Landsat temporal interpolation, single-reference residual transfer, dual-reference interpolation, and later STARFM/GNSPI where relevant. Compare imputation components through ablations, including smoothing and the 0.35 heuristic if retained. Use multiple sites/years and spatially separated evaluation units before general claims.

Success criterion: automated checks catch target leakage, unchanged-scene distortions, invalid interpolation support, mask loss, unit mismatch, and grid mismatch; reported results include failure and no-data rates.

### Milestone 3: cover-crop application

After the satellite series is trustworthy, obtain and verify the 2022 supplement and data availability. Implement dynamic phenology windows, soil baseline, logistic cash-crop fit, cover-signal accumulation, crop masks, climate/soil covariates, NASS-calibrated threshold modeling, and field aggregation. Preserve fold isolation and evaluate independent years/counties/fields. The first useful local product is a validated field-level detection for one season, before extending to regional trends.

Decisions for the next implementation session: exact replication versus a documented adaptation; target AOI/years; retrospective versus near-real-time processing; available field labels/boundaries; and whether fall/winter coverage is required. The current files establish a local retrospective STAIR prototype, so that is the reasonable baseline until those choices are specified.

## Sources and continuity

Primary research inputs are the two user-supplied PDFs in Downloads. Paper methods/results above refer to those documents; external links are additional verification, not instructions to execute code. No instructions embedded in research documents were treated as user authorization.

Official technical references used:

- [Landsat 8 Collection 2 Level 2 catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2)
- [MOD09GA Collection 6.1 catalog](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA)
- [MCD43A4 Collection 6.1 catalog](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD43A4)
- [ImageCollection.qualityMosaic](https://developers.google.com/earth-engine/apidocs/ee-imagecollection-qualitymosaic)
- [Cover-crop publisher record and supplement link](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022GL100249)
- [Associated Dryad data listing](https://datadryad.org/dataset/doi:10.5061/dryad.4xgxd25dg)

This review is a project baseline for future sessions. Update findings and measured evidence as implementation changes; do not treat historical figures or this static audit as certification of a later pipeline.
