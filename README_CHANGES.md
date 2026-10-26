# RainyDay_fast — changes with respect to the original RainyDay

This file records what RainyDay_fast changes compared with the original RainyDay, and why. It is kept up to date with
every change. In the code, every change is marked with a `GP YYYY-MM-DD` comment, and the previous code is kept as a
comment next to it.

## Lineage

| Step | Repository | Base | Date |
|---|---|---|---|
| 1 | **Original RainyDay**: Daniel B. Wright and the Hydroclimate Extremes Group, https://github.com/HydroclimateExtremesGroup/RainyDay | the last upstream commit included is `b14a011` ("Changed example folder") | 2025-04-17 |
| 2 | **RainyDay_mod**: Gabriel Perez (GP), private `Perez-HydroSystems/RainyDay_mod` | clone of the original, plus GP's changes in section A | 2025-05 → 2026-10-21 (`9137292`) |
| 3 | **RainyDay_fast** (this repo): GP, private `Perez-HydroSystems/RainyDay_fast` | **clone of RainyDay_mod at `9137292`**, plus the performance changes in section B | from 2026-10-22 |

**In one paragraph.** RainyDay_fast is RainyDay_mod made fast for large campaigns, meaning thousands of runs over the
same rainfall data and domains. Its results are **bit-identical** to RainyDay_mod. It speeds runs up in three ways:
1. it reads each domain from a cropped, uncompressed copy of the input, instead of decompressing the CONUS files in
   every run;
2. it keeps the storm catalog as a small metadata file, instead of about 4 GB of rain fields per basin;
3. it computes the transposition with a per-storm FFT table, plus exact sums only for the candidate windows.

RainyDay_mod already differs from the original RainyDay in a few places that change results, notably the strict
transposition rule. Those differences are listed in section C.

---

## A. Changes inherited from RainyDay_mod (GP, 2025-05 to 2026-10-21)

| # | Change | Why | Effect on results vs original | Commit |
|---|---|---|---|---|
| A1 | `UNCERTAINTY: "all_realizations"` writes every realization to the FreqAnalysis CSV; fixed the selection of the printed return periods | Uncertainty analysis needs all realizations; the selected return periods were wrong | New option; the bug fix changes the printed rows | `07021dd` |
| A2 | A missing input day (no file) is filled with zeros instead of failing | Datasets with missing days, e.g. leap years | Only when files are missing | `5185abb` |
| A3 | Bug fixes: storm-probability plot alignment, index sorting when printing realizations, diagnostic-plot padding (storms at the domain edge now detected) | Wrong plots and indices | Plots and diagnostics only | `645d0cb`, `1e9ea84`, `0cc1f2c` |
| A4 | Parallel loop in `catalogNumba_irregular` | Speed on large domains | None | `2bf6a54` |
| A5 | **Catalog-scan engines** (`RAINYDAY_CATALOG_ENGINE`): `batchfft` (batched, multithreaded CPU FFT, the default), `batchfft_mp` (process pool), `batchtf` (TensorFlow-Metal GPU), `batchcuda` (CuPy GPU); reader-thread prefetch; benchmark tools | The catalog scan dominated long records | None: every engine is verified bit-identical to the per-step FFT; GPU engines screen candidates, then the CPU confirms them exactly | `5536ca0` |
| A6 | Optional JSON `COMPUTE` block (engine, workers, blockfiles, GPU settings), auto-detected from the machine; precedence is env var > JSON > auto | Portable performance settings | None | `df2c909`, `f5ca299` |
| A7 | **`CATALOGPATH`**: reuse an existing storm catalog from any folder, with warnings if it was built with different settings | Many durations or settings from one catalog | None | `42bd1e0` |
| A8 | The temporary hard-coded skip of August 2017 (Hurricane Harvey) in the file list is commented out; no-Harvey runs now use a data folder without August 2017 | The RainyDay source should not hide data | Runs on the same files are unchanged | `42bd1e0` |
| A9 | **Strict transposition**: on irregular domains a location is valid only if the **whole watershed bounding box** lies inside the domain, the same rule as the catalog search | The original only required the anchor cell inside the domain, so transposed basins could hang up to about 55 % outside it and collect rain there. In SE Texas this made 100-yr depths grow with basin area (Harvey's core lies on the domain edge) | **Changes results** wherever basins reached the domain edge | `4ae1ace` |
| A10 | Fix for one-step windows in the batched scan (`build_windowed_block`) | Native 1 h catalogs: unbounded memory, a stalled scan and wrong windows | Fixes 1 h catalogs; 72 h catalogs unchanged | `3453cbc` |
| A11 | **`MULTIPLEDURATIONS`** with `MULTIDURATIONENGINE "fast"/"exact"`: one run writes the frequency analysis for several durations (e.g. 1–72 h) from one parent catalog. Each duration equals a separate duration-corrected run. Single duration-corrected runs also use the compiled kernel | One run per basin instead of 7; about 8× faster frequency analysis | None: bit-identical to separate runs | `583d697`, `b6bc42c`, `9137292` |
| A12 | Smaller fixes:<br>• the storm-writing loop no longer silently swallows non-permission `OSError`s;<br>• `batchfft_mp` uses `fork` because the script has no `__main__` guard;<br>• a stray import was removed | Silent catalog losses; crashes | None | in `5536ca0` |
| A13 | Verification tools:<br>• `tools/compare_catalogs.py`;<br>• the Talequah references;<br>• `tools/test_multidurations.sh` and `compare_multidurations.py`;<br>• `tools/test_windowed_block.py` and `test_domainaware_equivalence.py`;<br>• the `RAINYDAY_DUMP_WHICHRAIN` hook, which dumps the raw transposed depths | Proving that changes are exact | None | several |

---

## B. New in RainyDay_fast (GP, from 2026-10-22)

**Rule:** with the same machine, environment and data, every result is **bit-identical** to RainyDay_mod. That means
the same catalogs, the same random draws, the same raw transposed depths of every duration, and byte-identical
FreqAnalysis CSVs. No `np.random` call was added, removed or moved. Any difference is treated as a bug.

### B1. Cropped input per transposition domain (fix 1). Status: done and verified.
- **Tool:**
  ```
  tools/make_domain_crop.py <source folder> <domain.shp> <crop folder> [--years] [--verify] [--watershed ...]
  ```
- **What it writes.** A copy of the daily input files, cropped to the domain plus a small margin, with the same file
  names, values, attributes and times.
- **How it is used.** RainyDay itself is **not changed** for this: set `RAINPATH` to `<crop folder>/AORC.*.preciptemp.nc`.
- **Why the margin matters.** RainyDay derives the grid resolution from the mean coordinate spacing of the file, and
  the masks are built from that value. The tool therefore picks margins that keep that mean bitwise equal to the
  CONUS value, and asserts it.
- **What `--verify` checks:**
  - every file is read identically;
  - the grid set-up, the domain mask and the watershed masks are identical.
- **Why.** Each run used to decompress the same 8,035 compressed CONUS files. The scan spent about 85 % of its time
  reading.
- **Measured (Mac).** Reading a file goes from 78 to 4.6 ms (Iowa). The Iowa catalog scan goes from 11.0 to 2.0 min.
- **Size, uncompressed:** Iowa 29.4 GB, Texas 16.9 GB for 2002–2023.
- **Verified.** Talequah, Iowa pixel and Texas HUC all PASS (catalogs, raw depths and CSVs).
- **Limits.** `ENHANCEDSST` stochastic or deterministic depends on the file extent, so it is not bit-identical on a
  crop.

### B2. Light storm catalog, `CATALOGSTORAGE: "full" | "light"` (fix 2). Status: done and verified.
- **What is stored.** With `"light"`, the catalog is one small file, `StormCatalog/<CATALOGNAME>_lightcatalog.nc`, of
  about 2 MB. It holds every catalog variable except the rain fields: storm times, locations, depths and masks. It
  also stores the file list and a CRC32 checksum of each storm's rain.
- **How the rain comes back.** Whenever a storm's rain is needed (diagnostics, resampling), it is rebuilt exactly as
  RainyDay wrote it, from the input files, and checked against the checksum.
- **Default.** `"full"` stays the default, so behaviour is unchanged.
- **Why.** The full catalog wrote about 4 GB per basin (Iowa) and was then read twice.
- **Measured.** Iowa catalog write goes from 2.6 to 0.6 min. The catalog shrinks from 4.0 GB to 1.8 MB.
- **Reuse.** Light catalogs can be kept and reused with `CATALOGPATH`, for example to rerun the frequency analysis
  with other years or durations.
- **Verified.** Every storm and variable is bitwise identical to the full catalog, and runs all PASS, including
  `CATALOGPATH` runs that read a light catalog.
- **Limits.** `SCENARIOS` needs full catalogs.

### B3. Table transposition engine, `MULTIDURATIONENGINE: "table"` (fix 3). Status: done and verified.
- **Why.** For each of about 1.8 million random draws (a storm and a location), RainyDay summed the basin's bounding
  box for each of 344 sub-duration windows. That is about 10¹² operations for a 16,000 km² basin.
- **What the engine does, per storm:**
  1. an FFT gives approximate basin sums for every window at every location;
  2. the zero-rain rule is evaluated exactly, with sliding box maxima;
  3. for each draw, only the candidate windows (within a proven rounding bound of the maximum) are summed **exactly**,
     with RainyDay's own compiled function, in the same order.
- **Also:**
  - the parent 72 h sum uses the compiled kernel, not object mode;
  - the storm selection is done once per storm.
- **Verify mode.** `RAINYDAY_VERIFY_TABLE=<fraction>` brute-forces a random sample of draws and stops on any mismatch.
- **Each distinct location is evaluated once per storm.** Duplicate draws copy the value, which is exact.
- **Compiled-code identity is checked.** `numba_multimask_calc` is called, not inlined, and has the same machine code in
  both callers (`tools/verify_fast_sst.py --asm`).
- **Driver overheads removed** (bit-identical, every engine):
  - `tracemalloc.start()`, inherited from the original RainyDay (2024), is switched off. It was never read, but it
    traced every array allocation, about 620 M per run, and took most of the frequency-analysis time.
    `RAINYDAY_TRACEMALLOC=1` restores it.
  - `StormSelector` builds one index per storm instead of about 10 boolean scans of `whichstorms`.
- **Verified:** all 5 matrix shapes ALL PASS. With verify mode at 1 %, about 18,000 brute-forced draws per shape had
  0 mismatches.
- **Speed.** On average only 1.15–1.72 exact window sums are needed per draw and duration, instead of 344. That is
  35–62× fewer sums.
- **Frequency-analysis time, `"fast"` → `"table"`:**

  | Shape | `"fast"` | `"table"` |
  |---|---|---|
  | Iowa pixel | 579 s | 36 s |
  | Iowa 16,000 km² circle | 4,048 s | 69 s |
  | R10 ellipse | ~9,200 s | 140 s |
  | Texas HUC | 705 s | 37 s |

### B4. Usability for other users, datasets and domains. Status: done and verified.
- **`make_domain_crop.py --json RUN.json`.** Takes the data path, variable names, domain and years from the user's
  JSON (any dataset RainyDay reads), and writes a ready-to-run `<stem>_crop.json`.
- **Compression.** Lossless deflate becomes the default: about 30× smaller, still about 12× faster to read than the
  CONUS files.
- **Coverage check at run start.** A crop that does not contain the run's domain stops the run with a clear message.
- **Portable light catalogs.** The rain is rebuilt from the current run's `RAINPATH`, with the checksum checked per
  storm, so a crop can be moved to another folder or machine.
- **`tools/materialize_catalog.py`.** Converts a light catalog to a full one (bit-identical), for scenarios, sharing
  or archiving.
- **README guide.** "Using RainyDay_fast for a new domain or dataset".

### B6. End-to-end result (2026-10-22, commit `6e9c316`)

**Setup.**
- All three fixes on: cropped input (deflate 1), light catalog, table engine.
- Production settings: NYEARS 1000, 1–72 h, 10 return levels, `data_aorcfix2`.
- The 5 shapes ran in parallel on the Mac M3 Max, for both RainyDay_mod and RainyDay_fast.

**Validation.** `compare_dumps ref_mod final_all --csv-ref`:
- ALL PASS, 89 checks: every raw depth of every duration bitwise equal, and every CSV byte-identical. That includes
  the Mac dry-run CSVs used for the 2026-10-21 Ubuntu campaign.
- `compare_light_catalog`: every storm and variable bitwise equal to RainyDay_mod's full catalogs.

| Shape | RainyDay_mod 9137292 | RainyDay_fast | Speed-up | Catalog size | Peak memory |
|---|---|---|---|---|---|
| Talequah (smoke test) | 32 s | 14 s | 2.3× | 300 KB (was 92 MB) | 1.6 GB |
| Texas HUC 120401020101 | 1,198 s | 149 s | 8.0× | 1.7 MB (was 2.3 GB) | 1.5 GB (was 2.4) |
| Iowa pixel | 1,436 s | 209 s | 6.9× | 1.7 MB (was 4.0 GB) | 2.2 GB (was 2.9) |
| Iowa 16,000 km² circle | 4,957 s | 241 s | 20.6× | 1.7 MB (was 4.0 GB) | 2.1 GB (was 2.7) |
| Iowa R10 ellipse, rotated 45° | 10,066 s | 325 s | 31× | 1.7 MB (was 4.0 GB) | 2.2 GB (was 2.9) |

**What remains in a RainyDay_fast run** (Iowa pixel, 209 s):

| Part | Time |
|---|---|
| Catalog scan | 123 s (the FFT itself; reading is no longer a cost) |
| Catalog write | 22 s |
| Diagnostics | 22 s (they re-read all storms) |
| Transposition and frequency analysis | 35 s |

The next levers are batching several basins per scan, or a GPU scan, plus caching the storms for the diagnostics.

### B5. Validation and profiling tools (all in `tools/` unless noted)
- **`make_test_jsons.py`, `run_matrix.sh`, `compare_dumps.py`.** The validation matrix against RainyDay_mod:
  - five shapes: Talequah, Iowa pixel, Iowa 16,000 km² circle, Iowa R = 10 ellipse rotated 45°, Texas HUC basin;
  - production settings;
  - bitwise checks of the dumps, CSVs and catalogs.
- **`RAINYDAY_TIMERS=1`** (`Source/RainyDay_utilities_Py3/fast_prof.py`). Prints a wall-clock table of the run's
  parts. It does not change results.
- **`compare_light_catalog.py`, `test_light_rebuild.py`.** Checks for the light catalog.
- **`verify_fast_sst.py`, `bench_fast_sst.py`.** Checks and benchmarks for the table engine.
- **`compare_dumps.py --catalogs`** (GP 2026-10-26). Light catalogs are now compared too: every variable bitwise and
  every global attribute except `history` and `description`. Before, `--catalogs` skipped them silently. A run pair
  without comparable catalogs is now a FAIL.
- **`test_record_ci.py`, `compare_runs.py`** (GP 2026-10-26). Unit tests of `RECORDCI`, and IDF_A vs IDF_B with the
  combined Monte Carlo + record CI (B7).

### B7. Record (sampling) uncertainty, `RECORDCI` (GP 2026-10-26). Status: done and verified (branch `fast/record`).
- **Commits:** `2da68ef` (module, hooks, unit tests), `874db70` (`compare_runs.py`, light catalogs in `compare_dumps`),
  `caa319b` (docs).
- **What.** An optional JSON setting, off by default, that adds the record (sampling) uncertainty of every frequency
  analysis of a run: the uncertainty from having only NY years of rainfall, which RainyDay's ensemble spread does not
  contain (the spread holds the storm catalog fixed). Per duration it writes `<name>_RecordBootstrap.csv` (every
  bootstrap record, like `"all_realizations"`) and `<name>_RecordCI.csv` (record CI, total CI of the estimate,
  predictive band, and the unchanged ensemble spread side by side, with a `#` header that defines each band); once per
  run `<scenario>_RecordDraws.csv` (the drawn years). Keys: `RECORDCI`, `RECORDNBOOT`, `RECORDLEVEL`, `RECORDSEED`,
  `RECORDBLOCK`, `RECORDSAVE` (section D). New tool `tools/compare_runs.py`: IDF_A vs IDF_B with the combined Monte
  Carlo + record CI of their difference, the paper's three-way classes and TOST / minimum-effect p-values.
- **How.** Port of the validated prototype of the paper project (`24_record_uncertainty/scripts/record_boot.py`,
  `22_huc_ellipse_diagnostic/scripts/04_correlation_emulator.py`) into `Source/RainyDay_utilities_Py3/fast_record.py`:
  1. Inside RainyDay's resampling loop, from the storm rain RainyDay has already read (a copy), the basin-average
     maximum of every duration at every location RainyDay can draw: float64 FFT correlation with RainyDay's `trimmask`,
     one cumulative sum per storm, `timeres/60/mnorm`, RainyDay's zero rule. No extra I/O.
  2. The exact SST model RainyDay samples (Poisson counts at λ = storms/NY, storms equally likely, locations uniform)
     gives, per FreqAnalysis row (k-th largest of NYEARS years), the noise-free value of the realization mean
     E[X_(k)] (PDS: `gammainc`; AMS: `betainc`) and a quantile estimator.
  3. Year bootstrap with its own generator `np.random.default_rng(RECORDSEED)` (the prototype's draws for block 1;
     moving blocks for `RECORDBLOCK` > 1) or leave-one-year-out jackknife; a storm's weight is the count of its start
     year. Record years = `INCLUDEYEARS`, else the span of the storm start years.
  4. Hooks in `RainyDay_Py3.py`, all insertions marked `GP 2026-10-26` (no existing line changed): import; parse and
     option checks before the catalog (unsupported runs stop at the start); the duration check next to the
     MULTIPLEDURATIONS checks; the run object before the resampling loop; one call per storm; one write after each
     `write_frequency_outputs` (parent and every sub-duration), before the SCENARIOS block.
- **Why.** The paper's comparisons need the record uncertainty of depths and of differences between configurations,
  computed in the same run as RainyDay's estimate, for any basin or shape.
- **Effect on results.** None: outputs are only added. No `np.random` call was added, removed or moved; RainyDay's
  arrays are only read.
- **Verified** (2026-10-26, Mac M3 Max, RainyDay_Env; outputs in `/Users/gjperezm/Databases/RainyDay_fast_tests`):
  - **Bit-identity.** The validation matrix (5 shapes, production settings) was run with main `b63865a` (`ref`), and
    with this branch with the option off (`off`) and on (`on`: bootstrap, B = 200). `compare_dumps ref off
    --catalogs` and `compare_dumps ref on --catalogs`: ALL PASS, 80 checks each (35 raw-depth dumps bitwise, 35
    FreqAnalysis CSVs byte-identical, 5 light catalogs bitwise with the new light-catalog check). All 55 PNGs are
    byte-identical too.
  - **Unit tests** (`tools/test_record_ci.py`, 28 PASS):
    - the pool equals RainyDay's `SSTalt_multidur` at every anchor of synthetic storms (timeres 60 and 30, fractional
      and 1×1 masks, parent and sub-durations, duration correction): the same zero values, other depths within
      3.3e-7;
    - `levels` equals `np.partition` of the expanded values bitwise (PDS and AMS, with pool extension);
    - `expected` matches a direct simulation of the PDS and AMS processes (|z| ≤ 2.2);
    - `Pool` equals a verbatim copy of the prototype's `Pool` bitwise;
    - block 1 reproduces `default_rng(seed).integers(0, NY, (B, NY))`.
  - **Prototype, Iowa pixel** (catalog `C72_N400_all`, B = 1000, seed 20261026; `CATALOGPATH` run whose
    FreqAnalysis CSVs equal the original run's byte for byte):
    - `sst_expected`, `rec_lo` and `rec_hi` equal `24_record_uncertainty/csv/pixel_record_ci.csv` to every printed
      decimal (70 cells);
    - every bootstrap value equals the prototype's full-precision cache within the 4-decimal rounding (≤ 5e-5 mm);
      offline, on the same pool values, all four estimators are bitwise equal to the cache;
    - jackknife: the 22 drop-year values equal `jackknife_validation.csv` to every printed decimal (7 durations ×
      10 return periods).
  - **Noise-free check.** |`rd_mean` − `sst_expected`| in Monte Carlo SE: PDS, 70 cells, median 0.60, max 1.74;
    AMS max 1.75; matrix shapes max 2.1.
  - **`compare_runs.py`**, Iowa HUC-8 07100006 raw vs rg (each run on its own catalog, T = 100, median, margin 5) vs
    `classes_mc_vs_combined.csv`: the Monte Carlo part is identical (Δ and all its CIs); the classes are identical
    (6 equivalent, 1 h inconclusive). The record part differs as expected, because the prototype evaluated the rg mask
    on the raw basin's catalog with the quantile estimator: the record half-width is 1.0–3.5 pp here vs 0.9–2.3 pp,
    the largest change at 1 h, where the rg run's own catalog moves the noise-free Δ from −1.6 % (shape only) to
    −3.0 %, closer to RainyDay's −3.8 %.
  - **Runtime and memory.** Matrix with B = 200 (5 jobs in parallel):

    | Shape | Off | On | Added | Per-storm pool | Estimators + files | Peak memory off → on |
    |---|---|---|---|---|---|---|
    | Talequah | 10 s | 19 s | +9 s | 0.6 s | 7.6 s | 1.64 → 1.64 GB |
    | Texas HUC 120401020101 | 146 s | 172 s | +26 s | 12.8 s | 11.1 s | 1.51 → 1.72 GB |
    | Iowa pixel | 202 s | 232 s | +30 s | 12.9 s | 9.8 s | 2.24 → 2.53 GB |
    | Iowa 16,000 km² circle | 235 s | 279 s | +44 s | 23.6 s | 13.7 s | 2.15 → 2.49 GB |
    | Iowa R10 ellipse, rotated 45° | 317 s | 365 s | +48 s | 28.0 s | 11.3 s | 2.16 → 2.34 GB |

    With B = 1000: Iowa pixel PDS 12.7 + 31.2 s, AMS 12.7 + 48.4 s, jackknife 12.7 + 5.6 s; Iowa HUC-8 25 + 57 s.
- **Limits.**
  - Scope: uniform transposition (irregular or rectangular domain), Poisson resampling, PDS or AMS, watershed or box,
    one duration (with or without duration correction) or MULTIPLEDURATIONS. Everything else stops with a message
    (nonuniform, empirical, ENHANCEDSST, ROTATIONANGLE, INTENSDISTR, SCENARIOS, SEASONALSAMPLING, point/pointlist, a
    catalog longer than DURATION without duration correction).
  - The catalog is reweighted, not rebuilt, per record (bounded at ≤ 0.28 % by the leave-one-year-out runs); years are
    independent unless `RECORDBLOCK` > 1; the CIs at 500–1000 yr are lower bounds (no storm larger than observed).
  - Storm year = start year. Without `INCLUDEYEARS`, RainyDay's own storm rate uses the number of distinct storm
    **end** years (RainyDay_Py3.py, `nyears`), while RECORDCI uses the span of the start years; a warning is printed
    when the two counts differ (then `sst_expected` is not the noise-free value of `rd_mean`). Set `INCLUDEYEARS`.
  - `expected` uses the prototype's trapezoid on a 20000-point geometric grid: quadrature error ~1e-5 relative
    (unit test against the exact step-function integral), far below the Monte Carlo error.
  - Memory: durations × storms × locations × 4 bytes (Iowa pixel, 7 × 400 × 35,416: 378 MB).

---

## C. Which changes alter results relative to the original RainyDay
- **Change results:**
  - A9, strict transposition, wherever basins reached the domain edge;
  - A1, the printed return-period selection fix;
  - A2, only when input files are missing;
  - A10, 1 h catalogs only.
- **Identical by construction and verified:**
  - all catalog engines (A5);
  - `MULTIPLEDURATIONS` (A11);
  - everything in section B, which is bit-identical to RainyDay_mod.
- **Adds outputs only:** `RECORDCI` (B7) writes new files next to the frequency analysis; with it off or on, every
  existing output (catalogs, raw transposed depths, `_FreqAnalysis.csv`) is verified identical to RainyDay_fast
  without it (`compare_dumps ref off|on --catalogs`, ALL PASS).

## D. New settings reference

| Setting | Where | Values | Default |
|---|---|---|---|
| `UNCERTAINTY` | JSON | adds `"all_realizations"` | original options |
| `CATALOGPATH` | JSON | `<folder>/StormCatalog/<CATALOGNAME>` | none |
| `MULTIPLEDURATIONS` | JSON | list of durations in hours, each ≤ `DURATION` | none |
| `MULTIDURATIONENGINE` | JSON | `"fast"`, `"exact"`, `"table"` (B3) | `"fast"` |
| `CATALOGSTORAGE` | JSON | `"full"`, `"light"` (B2) | `"full"` |
| `COMPUTE` | JSON block | engine, nworkers, blockfiles, GPU settings | `"auto"` |
| `RECORDCI` | JSON | `"false"`, `"bootstrap"`, `"jackknife"` (B7; JSON booleans accepted) | `"false"` |
| `RECORDNBOOT` | JSON | number of bootstrap records | 1000 |
| `RECORDLEVEL` | JSON | level of the bands (%) | 90 |
| `RECORDSEED` | JSON | seed of the record generator (`np.random.default_rng`) | 20261026 |
| `RECORDBLOCK` | JSON | years per moving block (1 = independent years) | 1 |
| `RECORDSAVE` | JSON | `"true"`: also `_RecordBootstrap.csv` and `_RecordDraws.csv`; `"false"`: `_RecordCI.csv` only | `"true"` |
| `RAINYDAY_CATALOG_ENGINE` | env | `batchfft`, `batchfft_mp`, `fft_perstep`, `batchtf`, `batchcuda` | auto (`batchfft`) |
| `RAINYDAY_BLOCKFILES` | env | files per FFT batch | 16 / auto |
| `RAINYDAY_DUMP_WHICHRAIN` | env | folder for the raw transposed depths | off |
| `RAINYDAY_TIMERS` | env | `1` prints timers | off |
| `RAINYDAY_VERIFY_TABLE` | env | fraction of draws brute-forced (B3) | off |

More detail: `README.md` (usage), `CLAUDE.md` (rules, correctness model, harness, benchmarks) and the `GP` comments in
the code.
