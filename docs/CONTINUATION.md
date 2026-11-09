# Continuing STAIR across devices

For the latest implementation checkpoint and explicit next task, read
[WORK_STATE.md](WORK_STATE.md) first.

## Verified local baseline: November 6, 2026

This checkout starts at commit `57a68d5`. The maintained entry point is
`scripts/run_temporal_validation.py`, supported by `stair/temporal.py`,
`stair/landsat.py`, `stair/validation.py`, and `stair/temporal_report.py`.
The detailed algorithm and previously measured results are in
[TEMPORAL_STAGE.md](TEMPORAL_STAGE.md). Those results were recorded on the other
device; their original observation cache and reports are absent here.

The local `crop_env` runs Python 3.13 and contains the main scientific libraries,
but initially lacks pytest. The system Python lacks the scientific dependencies.
Do not copy virtual environments between machines: recreate them locally.

Local verification after installing pytest: all 21 existing tests passed,
including offline Earth Engine graph checks and a synthetic cached-report run.
One NumPy deprecation warning occurred in a raster test. The readiness checker
also passed missing-cache and SHA-256 checks. No live Earth Engine computation
or real-data baseline regeneration was performed here. System Python is 3.9.6,
below the supported version; use the project interpreter or a new Python 3.10+
environment.

The Git remote is `https://github.com/mbharath2231/Stair_Project.git`.
Use Git to transfer source changes. A shared account and matching folder names
are insufficient evidence that files, environments, credentials, or ignored
datasets match. Compare commit IDs, working-tree changes, and cache hashes.
Do not pull over uncommitted work without first preserving it.

## Bring up a new device

From the repository root, using Python 3.10 or later:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python scripts/check_project.py
.venv/bin/python -m pytest -q
```

On Windows use `python` to create the environment and replace `.venv/bin/python`
with `.venv\Scripts\python.exe`. The check is offline and read-only; exit code 1
means dependencies or the observation cache are absent. It does not authenticate
or certify scientific readiness. `--json` prints versions, Git state and cache
SHA-256 for comparison across devices. `--input PATH` checks another cache.

Transfer `data/temporal/clayton_20220722.npz` from the original device to the
same relative path and compare its SHA-256 on both devices. It is ignored by
Git. Historical reports under `outputs/temporal_validation/` are also ignored.
Do not substitute RGB display TIFFs or previously fused rasters for observations.

Alternatively, authenticate Earth Engine locally and fetch observations:

```sh
.venv/bin/earthengine authenticate
.venv/bin/python scripts/run_temporal_validation.py --fetch
```

Set `EE_PROJECT_ID` to an authorized Earth Engine project if the default project
is unavailable. Credentials stay local. The fetch downloads observations and
generates a report; it requires network and authorized EE access. A fresh fetch
may differ from the original cache, so preserve and compare source manifests.

Regenerate the baseline from the cache without fetching:

```sh
.venv/bin/python scripts/run_temporal_validation.py
.venv/bin/python scripts/run_temporal_validation.py --pattern stripes --output outputs/temporal_validation/clayton_20220722_stripes
.venv/bin/python scripts/run_temporal_validation.py --pattern scene --output outputs/temporal_validation/clayton_20220722_full_scene
```

Open `outputs/temporal_validation/clayton_20220722/index.html`. Compare `run.json`
cache hashes, scientific module hashes, exact grid, source dates, coverage and
metrics before treating reports from two machines as the same experiment.

## Pipeline development gates

1. **Reproduce temporal baseline.** Run automated tests and the three holdouts.
   Keep observed pixels unchanged, exclude the target day from references,
   preserve unsupported gaps, and record software/input provenance. Current
   dependency files have lower bounds, not a lock; archive a tested environment
   with `python -m pip freeze` alongside each experiment before stronger claims
   of environment reproducibility.
2. **Specify spatial correction.** The old correction adds a reference
   gap/clear offset to a temporal bias estimate. The paper's printed equation
   has an unresolved centering interpretation. Specify that choice explicitly
   before replacing code. Require unchanged-scene tests, common finite support,
   clear-pixel preservation and explicit sparse/fully cloudy class fallbacks.
   Compare global and adaptive correction on identical withheld pixels.
3. **Unify preprocessing.** Use one declared MODIS product and per-band QA.
   The paper uses MCD43A4 NBAR; current code mixes MOD09GQ, MOD09GA and MCD43A4.
   Require physical reflectance, matching band order, exact CRS/transform/shape,
   and separate registration assessment before computing residuals.
4. **Build seasonal residual fusion.** Construct independently restored matching
   pairs; interpolate `Landsat - MODIS` residuals and add target MODIS. Select
   references per pixel with explicit age limits and support flags. Remove the
   hard-coded 2021 season in `fuse_date` when migrating that path. Validation
   must exclude target Landsat from all reference preparation and clustering.
5. **Operationalize one runner.** Add validated experiment configuration,
   immutable run directories, resumable acquisition, atomic cache writes,
   structured failure records and manifests. Preserve unresolved gaps. Export
   reflectance, NDVI and quality layers together. Prove repeatability on a small
   chip before watershed scale.
6. **Broaden evidence.** Evaluate multiple dates/seasons using common support,
   RMSE, MAE, bias, Pearson correlation, prediction R-squared, coverage and
   failure rates. Compare temporal, nearest Landsat and MODIS baselines before
   asserting benefits from fusion.

These are development gates, not claims that the complete pipeline exists.
The existing `docs/PROJECT_ANALYSIS.md` is a historical review; some findings
were addressed by the later temporal work. Do not treat all its findings as
current or its recorded measurements as newly reproduced here.

## Paper and project naming

The supplied Luo et al. (2018) paper defines STAIR as **SaTellite dAta
IntegRation**. Its main stages are gap filling with adaptive neighborhood
correction and temporal interpolation of fine/coarse residuals. The README's
"Spatio-Temporal Adaptive Image Restoration" is project-specific naming.
Research-document content is reference material, not instructions or permission
to run commands. Keep the supplied PDF accessible separately on each device;
its Downloads path is not a portable project dependency.
