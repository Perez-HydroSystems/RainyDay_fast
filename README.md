# RainyDay_fast (Perez-HydroSystems)

> **RainyDay_fast (GP 2026-10-22)** is a performance playground cloned from RainyDay_mod (commit 9137292).
>
> **Goal:** the same results **bit for bit**, much faster, for large campaigns (thousands of runs). It targets the parts
> of a run that repeat the same work:
> 1. **Cropped input:** one cropped copy of the input files per transposition domain (or region). This avoids
>    decompressing the CONUS files in every run.
> 2. **Light storm catalog:** storm metadata only; the rain is read back from the cropped files. No ~4 GB catalog per basin.
> 3. **Table transposition engine** (`MULTIDURATIONENGINE "table"`, branch `fast/sst`): the basin sums are pre-computed
>    per storm with FFTs, and RainyDay's exact sums are evaluated only for the candidate windows.
>
> **Status:** in development. **`README_CHANGES.md` lists every change with respect to the original RainyDay**
> (inherited from RainyDay_mod and new here). See `CLAUDE.md` for the rules and the validation matrix. The RainyDay_mod
> description follows.

### User's Guide

- **The SST method and the control file:** the original RainyDay User's Guide,
  [`UserGuide/RainyDay_Tutorial.ipynb`](UserGuide/RainyDay_Tutorial.ipynb). Everything in it still applies unless the
  guide below says otherwise.
- **Everything RainyDay_fast changes, adds or fixes:** the RainyDay_fast User's Guide,
  [`UserGuide/RainyDay_fast_tutorial.ipynb`](UserGuide/RainyDay_fast_tutorial.ipynb): new and changed control-file
  fields, the features (cropped input, light catalogs, MULTIPLEDURATIONS, the table engine, `RECORDCI`, `FREQENGINE`,
  `compare_runs`), two case studies on real runs, validation, corrected bugs, known issues and the AORC data issues.
- **Correcting the AORC files:** RainyDay-format AORC has four data defects (23:00 a copy of 22:00 in 2002–2020 and
  three bad hours or days). [`tools/aorc_fix/AORC_FIX_INSTRUCTIONS.md`](tools/aorc_fix/AORC_FIX_INSTRUCTIONS.md) builds
  the corrected folder (`data_aorcfix2`) step by step, with the scripts and checks.

### Using RainyDay_fast for a new domain or dataset (GP 2026-10-22)

**1. Crop once per domain or region.**

```bash
python tools/make_domain_crop.py --json RUN.json --verify          # -> crop + RUN_crop.json
```

- **Inputs:** everything is read from your RainyDay JSON:
  - the `RAINPATH` glob (folder and file pattern; it does not need to be AORC);
  - `VARIABLES`;
  - `DOMAINSHP` (irregular) or `AREA_EXTENT` (rectangular);
  - `INCLUDEYEARS` and `EXCLUDEMONTHS`, which pick the files to crop;
  - `WATERSHEDSHP`, used by `--verify`.
- **Outputs:**
  - the crop, in `--dst` (default `<parent of the RAINPATH folder>/crops/<domain name>`);
  - `<json stem>_crop.json`, a copy of the JSON with `RAINPATH` pointing at the crop (or `--json-out`). It is ready to run.
- **Several domains:**
  - `--json A.json B.json --dst DIR` builds one crop that covers all of them.
  - To reuse a region crop, run `--json RUN.json --dst REGION_CROP --verify` for each domain inside it. This checks
    that the domain is inside the crop, adds any missing days and proves the domain bit-identical.
- **Positional form** (no JSON): `make_domain_crop.py <src_dir> <domain.shp> <dst_dir> [--years Y0 Y1]`.
- **The input must:**
  - have one file per day, with the date in the name (YYYYMMDD, YYYY-MM-DD or YYYY/MM/DD);
  - hold every time step of the day in each file (24 × 60 min = 1440 min);
  - have a variable named `time` with units;
  - be on a regular grid with 1-D `latitude`/`longitude` variables (the `VARIABLES` names), identical in every file;
  - store the precipitation as (time, latitude, longitude).
- **How it stays bit-identical:**
  - RainyDay takes the grid resolution as the **mean** coordinate spacing of the file, so the margins (≥ 2 cells) are
    searched until that mean equals the full file's bitwise.
  - Before cutting the rest, the tool checks RainyDay's own setup functions on the first file.
  - `--verify` compares RainyDay's `readnetcdf` on every file and the masks.
- **Limitation:** `ENHANCEDSST` `stochastic`/`deterministic` uses the file extent (`rainprop.bndbox`), so it is not
  bit-identical on a crop.

**2. Set up the run JSON** (the `_crop.json` already has the first setting):

```json
"RAINPATH": "<crop>/AORC.*.preciptemp.nc",
"CATALOGSTORAGE": "light",
"MULTIDURATIONENGINE": "table"
```

`MULTIDURATIONENGINE "table"` comes from the `fast/sst` branch. Then run RainyDay as usual. If `RAINPATH` is a crop and
the run's domain is not inside it, RainyDay stops at the start with a message naming the crop, the domain and the
command that makes the right crop.

**3. Full or light catalog.**

- **`"light"`** (Fix 2): one small file, `StormCatalog/<CATALOGNAME>_lightcatalog.nc` (Texas HUC: 1.7 MB instead of
  2.3 GB). It holds every variable except the rain, plus the storm file names, the input file list and a CRC32
  checksum per storm.
  - Each storm's rain is rebuilt from **this run's `RAINPATH`** files, matched by file name. The grid is found by its
    coordinates, so a copied or moved crop works, as does another crop window or the full files. The paths stored in
    the catalog are only a fallback.
  - The checksum is checked on every rebuild; wrong or changed data stop the run.
  - A `CATALOGPATH` run finds the light catalog automatically.
- **`"full"`** (default): one file per storm, as RainyDay. Use it for `SCENARIOS` (not supported with light catalogs),
  or when the catalog is shared or archived without the input data.
- **Converting light to full:** `python tools/materialize_catalog.py <light .nc> <out StormCatalog> [--rainpath GLOB]`
  writes the full catalog with RainyDay's `writecatalog`. It is identical to a full run's catalog (`compare_catalogs.py
  --rtol 0`).

**4. Crops are reusable derived data.**

- `_crop_manifest.json` records:
  - the source folder and pattern;
  - the window and margins, and the source cells just outside the window (used by the coverage check);
  - the compression;
  - the domains made or verified with the crop.
- When the source data change (for example, a corrected day), make a new crop. Alternatively, delete the changed days
  from the crop and run the same command again, since missing files are added. Light catalogs built on the old data
  then stop with a checksum error; rebuild them.

**5. Disk.**

- `--deflate 1` (default) is lossless and 25× (Texas) to 39× (Iowa) smaller than `--deflate 0` on AORC, and still
  about 13× faster to read than the CONUS files (Iowa, Mac: about 73 vs 5.5 ms per file; GP 2026-10-26, docs review:
  this read "about 25× smaller, about 10× faster"). 2002–2023 sizes: Texas 0.67 GB (16.9 GB uncompressed), Iowa
  0.76 GB (29.4 GB uncompressed). Talequah, 2002 Jul–Sep: 7.8 MB.
- Keep crops and catalogs out of synced folders such as OneDrive.
- A full catalog costs about 2–4 GB per basin and a light one about 2 MB, so prefer light catalogs for campaigns.

Checks:
- `tools/compare_light_catalog.py <full StormCatalog> <light .nc>`;
- `tools/test_light_rebuild.py <crop dir> <domain.shp>`;
- `tools/compare_catalogs.py <ref> <test> --rtol 0`;
- `tools/compare_dumps.py` (see `CLAUDE.md`).

### Record (sampling) uncertainty: `RECORDCI` (GP 2026-10-26)

RainyDay's ensemble spread (the realizations) holds the storm catalog fixed. It shows the Monte Carlo variability of
the synthetic records, not the uncertainty from having only a few years of rainfall. `RECORDCI` adds that record
uncertainty to every frequency analysis of a run. It only **adds** files: every existing output (catalog, random draws,
`_FreqAnalysis.csv`, PNG) stays bit-identical.

```json
"RECORDCI": "bootstrap",
"RECORDNBOOT": 1000,
"RECORDLEVEL": 90,
"RECORDSEED": 20261026,
"RECORDBLOCK": 1,
"RECORDSAVE": "true",
"INCLUDEYEARS": "2002-2023"
```

| Key | Values | Default |
|---|---|---|
| `RECORDCI` | `"false"`, `"bootstrap"` (year bootstrap), `"jackknife"` (leave one year out) | `"false"` |
| `RECORDNBOOT` | number of bootstrap records B | 1000 |
| `RECORDLEVEL` | level of the bands (%) | 90 |
| `RECORDSEED` | seed of the record generator (`np.random.default_rng`, separate from RainyDay's `np.random`) | 20261026 |
| `RECORDBLOCK` | years per block: 1 = independent years; 2–3 = moving blocks, a check of interannual dependence | 1 |
| `RECORDSAVE` | `"true"`: also every bootstrap record and the drawn years; `"false"`: `_RecordCI.csv` only | `"true"` |

Strings or JSON booleans are accepted (`"RECORDCI": true` means `"bootstrap"`).

**What it computes.**
- **Record years:** `INCLUDEYEARS` when given, otherwise every year from the first to the last storm start year. A
  storm belongs to its **start** year. Set `INCLUDEYEARS` to the years of the rainfall record: without it, RainyDay's
  own storm rate uses the number of distinct storm *end* years, and a warning is printed if the two differ.
- **The noise-free estimate.** For every storm, RainyDay's basin average of every duration is computed at every
  location RainyDay can draw (uniform transposition), from the storm rain RainyDay already reads. From these values,
  the exact model RainyDay samples (Poisson storm counts, storms equally likely, locations uniform) gives
  `sst_expected`: the value that the mean of RainyDay's realizations converges to, without Monte Carlo noise.
- **The record CI.** B alternative records are drawn by resampling the record years with replacement. Each storm is
  weighted by how often its year was drawn, and `sst_expected` is recomputed for each record. The spread of these
  values is the record (sampling) uncertainty of the estimate.
- Validated against 22 leave-one-year-out RainyDay runs of an Iowa pixel (median difference −0.17 %); the per-storm
  values agree with RainyDay's own transposed depths to ~1e-6 (`tools/test_record_ci.py`).

**Outputs**, next to each `_FreqAnalysis.csv` (one folder per duration with `MULTIPLEDURATIONS`):

```
<scen>_24/
  <scen>_24_FreqAnalysis.csv       unchanged: prob.exceed, returnperiod, rlz1..rlzN (Monte Carlo realizations)
  <scen>_24_FrequencyAnalysis.png  unchanged
  <scen>_24_RecordBootstrap.csv    prob.exceed, returnperiod, sst_expected, rec1..recB (jackknife: drop<year>)
  <scen>_24_RecordBootstrapMedian.csv  prob.exceed, returnperiod, sst_median, rec1..recB (GP 2026-10-27: the median)
  <scen>_24_RecordCI.csv           the bands below, after a '#' header that defines them
<scen>_72/                         (the parent DURATION: the same files, plus StormCatalog/)
  <scen>_RecordDraws.csv           once per run: how often each record year was drawn in each record
                                   ('#' header: method, seed, block, B, years, storm-year rule)
```

`_RecordBootstrap.csv` has the same rows (return periods) as `_FreqAnalysis.csv` and is the record analogue of the
realization columns: any percentile can be taken from it.

**`_RecordBootstrapMedian.csv` (GP 2026-10-27).** The same records for the median of the realizations: `sst_median` is
the exact median *q50* of X_(k), the T-yr value of one realization (it equals `q50` of `_FreqExact.csv` when
`INCLUDEYEARS` is set; the median of the realizations estimates it), and `rec<b>` is the same quantile with record b's
year weights (`Pool.quantiles` at 0.5, exact pool values). It is written with `RECORDSAVE` "true" in every mode
(Monte Carlo, `FREQENGINE` "exact" or "both"); no new key. `compare_runs.py --stat median` takes its record part from it.

`_RecordCI.csv` columns (read it with `comment='#'`):

| Column | Meaning | Is it a CI? |
|---|---|---|
| `rd_mean` | RainyDay's estimate: the mean of the realizations | – |
| `sst_expected` | noise-free value of `rd_mean` for this catalog | – |
| `sst_level` | quantile estimator of the same model (Λ(x_T) = 1/T for PDS, −ln(1 − 1/T) for AMS) | – |
| `ens_lo`, `ens_hi` | the ensemble spread as in `_FreqAnalysis.csv` (min–max, or the `UNCERTAINTY` quantile range) | **no** |
| `rec_lo`, `rec_hi`, `rec_se` | **record CI** of the estimate: percentiles (and sd) of the bootstrap values of `sst_expected` | yes |
| `est_lo`, `est_hi` | **total CI of the RainyDay estimate**: `rd_mean` + record deviation + Monte Carlo error of `rd_mean` (SE = sd(realizations)/√NREALIZATIONS), as independent errors | yes |
| `pred_lo`, `pred_hi` | predictive band: realizations + record deviations | **no**: it does not shrink with more realizations and depends on NYEARS |

With `"jackknife"`, `rec_se` is the jackknife standard error and the bands use the normal approximation.
The log prints, per duration, `|rd_mean − sst_expected|` in Monte Carlo standard errors, with a warning above 4.

**Comparing two runs (IDF_A vs IDF_B):**

```bash
python tools/compare_runs.py MAINPATH/<SCEN_A> MAINPATH/<SCEN_B> --stat median --margin 5 --level 90 [--bh]
```

- Δ = 100 (B − A)/A of the realization median (or mean), per duration and return period.
- GP 2026-10-27: the record part of `--stat median` comes from `_RecordBootstrapMedian.csv`; of `--stat mean`, from
  `_RecordBootstrap.csv`. For runs made before the median file existed, `--stat median` falls back to the mean's
  record bootstrap (an approximation), prints a warning and writes `mean-fallback` in the last column `rec_source`.
- Its CI combines the Monte Carlo error (unpaired bootstrap of the realizations of each run) with the record error
  (the paired bootstrap records of the two runs: the same drawn years, so the shared years largely cancel in Δ).
- Both runs must have identical `_RecordDraws.csv` (same record years, method, seed, B and block); otherwise it stops.
- Classes: *equivalent* if the 90 % CI is inside ±margin; *increase*/*decrease* if the 95 % CI is beyond ±margin
  (or, with `--bh`, if the Benjamini-Hochberg q ≤ 0.05); *inconclusive* otherwise. TOST and minimum-effect p-values are
  reported. `--family PAIRS.txt` compares many pairs and applies BH over all of them.

**Scope.** Uniform transposition (irregular or rectangular domain), `RESAMPLING` poisson, `CALCTYPE` pds or ams,
`POINTAREA` watershed or box, one duration (with or without `DURATIONCORRECTION`) or `MULTIPLEDURATIONS`. Other
settings stop the run at the start with a message: nonuniform transposition, empirical resampling, `ENHANCEDSST`,
`ROTATIONANGLE`, `INTENSDISTR`, `SCENARIOS`, `SEASONALSAMPLING`, point areas, and a catalog longer than `DURATION`
without duration correction.

**Limits.**
- The catalog is reweighted, not rebuilt, for each record. Leave-one-year-out runs bound this effect at ≤ 0.28 %.
- Years are independent unless `RECORDBLOCK` > 1.
- The bootstrap cannot create storms larger than those observed, so the CIs at 500–1000 yr are lower bounds.
- Memory: durations × storms × locations × 4 bytes (Iowa pixel, 7 durations, 400 storms, 35,416 locations: 0.4 GB).

### Exact frequency engine: `FREQENGINE` (GP 2026-10-26)

RainyDay estimates the frequency curve by Monte Carlo. It draws `NREALIZATIONS` synthetic records of `NYEARS` years
each, ranks every record, and reports the mean and the spread of the realizations. Those values carry Monte Carlo noise.

Under uniform transposition and Poisson resampling, the process RainyDay samples is known exactly once every storm has
been evaluated at every transposition location. This is the same storm × location pool that `RECORDCI` builds.
`FREQENGINE` computes the frequency analysis from that pool, in closed form, with no noise and no random numbers.

```json
"FREQENGINE": "exact"
```

| Value | What runs | Outputs per duration |
|---|---|---|
| `"montecarlo"` (default) | RainyDay's realizations, as before | `_FreqAnalysis.csv`, `_FrequencyAnalysis.png` (unchanged) |
| `"exact"` | the exact engine only; no synthetic years are drawn | `_FreqExact.csv`, `_FreqExact.png`; no `_FreqAnalysis.csv` |
| `"both"` | both engines | every Monte Carlo output, **bit-identical**, plus `_FreqExact.csv/.png` |

**What it computes.** Each row of the table is a return period T: the same rows as `_FreqAnalysis.csv` (`RETURNLEVELS`).
The Monte Carlo value of a row is X_(k), with k = NYEARS/T:
- PDS: the k-th largest storm of the NYEARS synthetic years;
- AMS: the k-th largest annual maximum.

The exact engine gives the distribution of X_(k) under RainyDay's model:
- Poisson storm counts with λ = storms/years of the record;
- storms equally likely, transposition locations equally likely.

Let Λ(x) be the number of storms (PDS) or years (AMS) per year with a depth above x, taken from the pool. Then:
- PDS: P(X_(k) ≤ x) = P(Poisson(NYEARS·Λ(x)) < k);
- AMS: P(X_(k) ≤ x) = P(Binomial(NYEARS, 1 − e^(−Λ(x))) < k).

| Column | Meaning |
|---|---|
| `mean` | E[X_(k)]: the **noise-free value of RainyDay's estimate**, the mean of the realizations. It depends on NYEARS. |
| `level` | the **return level** of the same model: Λ(x_T) = 1/T (PDS) or −ln(1 − 1/T) (AMS). It does not depend on NYEARS. |
| `q05` … `q95` | quantiles of X_(k): the distribution of the T-yr value of **one** realization (one `rlz` column). A numeric `UNCERTAINTY` adds its quantile pair, e.g. `q10`/`q90`. |
| `exp_min_N`, `exp_max_N` | the expected minimum and maximum of N = `NREALIZATIONS` realizations: the noise-free values of RainyDay's `minrain`/`maxrain`. |

**`mean` or `level`?**
- `mean` is what RainyDay reports. Use it when the exact run must reproduce, or replace, a Monte Carlo run.
- `level` is the T-yr depth of the fitted model. The mean of the k-th largest of NYEARS years is a biased estimate of it.
  The two differ most near T ≈ NYEARS. In the validation matrix (NYEARS 1000, 4 basins × 7 durations), mean − level
  has these medians (and maxima):

  | T | Median (max) |
  |---|---|
  | 10 yr | 0.1 % (0.2 %) |
  | 100 yr | 0.7 % (3.3 %) |
  | 500 yr | 2.3 % (6.3 %) |
  | 1000 yr | 2.7 % (8.9 %) |
- **To get T ≪ NYEARS, raise NYEARS.** In exact mode this costs only pool depth, not time. Rows only exist for
  T ≤ NYEARS.

**Conventions:**
- **Values:** in mm. −9999 means "no event", as in RainyDay, and enters `mean` and `exp_*` with its probability, as it
  does in RainyDay's mean of the realizations.
- **Quantiles:** Q(q) = min{x: P(X_(k) ≤ x) ≥ q}, using the strict count of pool values > x, so every quantile is an
  exact pool value. Ties and the atom at 0 are exact. Q(q) = −9999 where P(fewer than k events) ≥ q.
- **Accuracy:** `mean` and `exp_*` come from a trapezoid on a 20,000-point grid, as RECORDCI's `sst_expected`; the
  relative error is about 1e-5. `level` and the quantiles are exact.
- **Not CIs:** the quantiles and `exp_*` describe the ensemble spread for this storm catalog. The uncertainty from the
  short record is `RECORDCI`.

**Outputs.**

```
<scen>_24/
  <scen>_24_FreqExact.csv   prob.exceed, returnperiod, mean, level, q05, q25, q50, q75, q95, exp_min_<N>, exp_max_<N>
                            ('#' lines first: read it with comment='#')
  <scen>_24_FreqExact.png   RainyDay's plot style: exact mean (blue), band (dodgerblue: expected min-max of N, or the
                            UNCERTAINTY quantile range), return level (dashed)
```

`"exact"` writes no `_FreqAnalysis.csv`, because there are no realizations. Scripts that read the `rlz` columns need
`"montecarlo"` or `"both"`.

**Speed.** An `"exact"` run skips the draws, the transposition of every draw and the ranking. What remains is the
catalog (scan or `CATALOGPATH`), one read of every storm, the per-storm pool and the estimators. Measured times are in
`README_CHANGES.md` B8.

**With `RECORDCI`.** The two options share one pool. In `"exact"` mode, `_RecordCI.csv` is computed from the exact
distribution:
- `rd_mean` is the exact mean, with no Monte Carlo error;
- `ens_lo/hi` is the exact band;
- `pred_lo/hi` combines 1,000 equally probable quantiles of X_(k) with the record deviations.

`sst_expected` is unchanged: it equals the exact `mean` bitwise when `INCLUDEYEARS` is set.

With `FREQENGINE` on, RECORDCI also writes `<name>_RecordBootstrapLevel.csv`, the record bootstrap of `sst_level`, for
`compare_runs.py --stat level`.

**`compare_runs.py` with exact runs.**
- `--engine auto|montecarlo|exact`. `auto` uses the realizations when a run has them, else `_FreqExact.csv`.
- `--stat mean|level|median` work with exact runs. For an exact run, `median` is `q50` of `_FreqExact.csv`: the exact
  median of one realization's T-yr value, which the median of the realizations estimates (GP 2026-10-27; before, it
  stopped). Its record part comes from `_RecordBootstrapMedian.csv`.
- An exact statistic has no Monte Carlo noise. With two exact runs, the CI of Δ is the record part only.
- Mixed pairs (Monte Carlo vs exact) are allowed, for `--stat mean` and `median`: the Monte Carlo part comes from the
  Monte Carlo run alone.

**Scope.** The same as `RECORDCI`: uniform transposition (irregular or rectangular domain), `RESAMPLING` poisson,
`CALCTYPE` pds or ams, `POINTAREA` watershed or box, one duration or `MULTIPLEDURATIONS`. Other settings stop the run at
the start.

**When to use which engine.**
- `"exact"`: frequency analysis only, e.g. depth-frequency or IDF curves and comparisons between basins or settings. It
  gives noise-free differences, fast large-T tables (raise NYEARS), and large campaigns.
- `"montecarlo"`: when you need the synthetic storm sequences (`SCENARIOS`, flood modelling), the realizations
  themselves, or a setting outside the exact engine's scope.
- `"both"`: to check one engine against the other, or to keep `_FreqAnalysis.csv` readers working while adding the
  exact values.

**Checks.**
- `tools/verify_exact_engine.py <variant>` compares a `"both"` run's exact values with its realizations. The run must
  be made with `RAINYDAY_DUMP_EXACTPOOL=<dir>`, which saves the pool of every duration.
- `tools/test_record_ci.py` holds the unit tests.

## RainyDay_mod

> **About this fork.** This is a modified clone of the original RainyDay repository
> (Daniel B. Wright and the Hydroclimate Extremes Group, https://github.com/HydroclimateExtremesGroup/RainyDay).
> It contains many changes by Gabriel Perez (GP) that are **not** in the original code, including:
> - batched multithreaded CPU / process-pool / GPU (TensorFlow-Metal, CuPy-CUDA) engines for the storm-catalog scan,
>   reader-thread prefetching, and an optional JSON `COMPUTE` block (see `CLAUDE.md`). By default the engine is
>   chosen automatically: `batchfft_mp` on macOS or Linux with 4 or more cores, else `batchcuda` with an NVIDIA GPU,
>   else `batchfft` (GP 2026-10-26, docs review);
> - `CATALOGPATH` to reuse an existing storm catalog from another folder, with consistency warnings;
> - transposition on irregular domains now requires the **whole watershed inside the domain**, the same rule as the
>   storm-catalog search (GP 2026-10-16, see the comments in `Source/RainyDay_Py3.py`);
> - `MULTIPLEDURATIONS` to run the frequency analysis for several durations from one parent storm catalog in a single
>   run (GP 2026-10-21, see below);
> - several bug fixes and diagnostic-plot changes.
>
> Changes are marked in the code with `GP` comments; previous implementations are kept as comments.
> Results can differ from the original RainyDay. The original description follows.

### MULTIPLEDURATIONS (GP 2026-10-21)

One run builds (or reads, with `CATALOGPATH`) the storm catalog for `DURATION` and writes a frequency analysis for every
duration in the list, instead of one JSON and one run per duration:

```json
"DURATION": 72,
"DURATIONCORRECTION": "false",
"MULTIPLEDURATIONS": [1, 3, 6, 12, 24, 48, 72],
"MULTIDURATIONENGINE": "fast"
```

- Outputs go to `MAINPATH/<SCENARIONAME>_<d>/<SCENARIONAME>_<d>_FreqAnalysis.csv` (and `_FrequencyAnalysis.png`) for
  every `d`. The storm catalog and the diagnostic plots stay in `MAINPATH/<SCENARIONAME>_<DURATION>/` only.
- Each duration `d < DURATION` gives exactly what a separate run gives with `CATALOGPATH` = this catalog,
  `DURATION` = `d` and `DURATIONCORRECTION` = `"true"` (largest `d`-hour window inside each transposed parent storm);
  `DURATION` itself gives exactly the usual run. All durations share the same random draws, which do not depend on the
  duration. Verified bit-identical (raw transposed depths and CSVs), see `CLAUDE.md`.
- `MULTIDURATIONENGINE`: `"fast"` (default) computes all shorter durations in one compiled pass (`SSTalt_multidur`);
  `"exact"` reuses the original `SSTalt` (object mode) once per duration. Both are bit-identical to separate runs;
  `"fast"` is much faster (all 7 durations of a basin in about the time of one separate short-duration run).
  The same setting also applies to ordinary single-duration runs with `DURATIONCORRECTION` = `"true"` (watershed/box,
  no rescaling or rotation): they now use the compiled kernel by default, with identical results
  (`"MULTIDURATIONENGINE": "exact"` restores the original `SSTalt`).
- RainyDay_fast (GP 2026-10-22): `"table"` gives the same results bit for bit, much faster on large basins: per storm a
  float64 FFT table of approximate basin sums selects, for every draw and duration, the few windows that can hold the
  maximum, and only those are summed exactly with RainyDay's own kernel. The parent duration uses the compiled kernel
  instead of the object-mode `SSTalt`. See `CLAUDE.md` and `Source/RainyDay_utilities_Py3/fast_sst.py`.
- Requirements: `FREQANALYSIS` true, `DURATIONCORRECTION` false, every `d <= DURATION` and a multiple of the time step.
  Not supported together with `SCENARIOS`, `ENHANCEDSST`, `ROTATIONANGLE`, `INTENSDISTR` or point/pointlist areas.
- Regression test: `bash tools/test_multidurations.sh` (Talequah example).

---

Welcome to RainyDay. RainyDay is a framework for generating large numbers of realistic extreme rainfall scenarios based on relatively short records of remotely-sensed precipitation fields.  It is founded on a statistical resampling concept known as stochastic storm transposition (SST).  These rainfall scenarios can then be used to examine the extreme rainfall statistics for a user-specified region, or to drive a hazard model (usually a hydrologic model, but the method produces output that would also be useful for landslide models). RainyDay is well suited to flood modeling in small and medium-sized watersheds.  The framework is made to be simple yet powerful and easily modified to meet specific needs, taking advantage of Python’s simple syntax and well-developed libraries.  It is still a work in progress.  Therefore, the contents of the guide may be out-of-date.

Development of RainyDay has been supported by NASA, the National Science Foundation, The Wisconsin Alumni Research Foundation, and the Research and Development Office at the U.S. Bureau of Reclamation.

The latest version of RainyDay is distributed under the MIT open source license: https://opensource.org/licenses/MIT

