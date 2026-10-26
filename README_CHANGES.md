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

### B8. Exact frequency engine, `FREQENGINE` (GP 2026-10-26). Status: done and verified (branch `fast/exact`).
- **Commit:** `e859896` (module, driver hooks, tools, unit tests, docs).
- **What.** An optional JSON setting: `"montecarlo"` (default, unchanged), `"exact"` or `"both"`.
  - The exact engine computes the frequency analysis of RainyDay's SST model from the complete storm × location pool,
    without Monte Carlo noise.
  - For every FreqAnalysis row, X_(k) (k-th largest event of NYEARS synthetic years, PDS; or k-th largest annual
    maximum, AMS), it writes:
    - `mean` = E[X_(k)], the noise-free value of RainyDay's realization mean;
    - `level`, the return level (Λ(x_T) = 1/T or −ln(1 − 1/T));
    - the 5/25/50/75/95 % quantiles of X_(k) (plus the numeric `UNCERTAINTY` pair);
    - the expected minimum and maximum of NREALIZATIONS realizations.
  - Files per duration: `<name>_FreqExact.csv` (with `#` definition lines) and `<name>_FreqExact.png` (RainyDay's
    plot style).
  - With RECORDCI in `"exact"` mode, `_RecordCI.csv` uses the exact distribution. With FREQENGINE on, RECORDCI also
    writes `_RecordBootstrapLevel.csv`.
- **How** (`Source/RainyDay_utilities_Py3/fast_record.py`):
  1. `StormPool`: the per-storm pool of RECORDCI, moved out of `RecordCI` with unchanged arithmetic. One pool per run,
     shared when both options are on.
  2. New `Pool` methods; `levels` and `expected` are unchanged:
     - `quantiles`: the inverse through `gammaincinv` (PDS) or `betaincinv` (AMS), then the pool value of rank
       floor(c*) + 1. This is exact, including ties and the atom at 0, and gives −9999 where P(fewer than k events) ≥ q;
     - `cdf`;
     - `expected_minmax`: survival Pᴺ and 1 − (1 − P)ᴺ in the `expected` trapezoid.
  3. `ExactFA`: per duration, a fresh `Pool` with the same construction and call order as `RecordCI.estimates`, and
     RainyDay's own `nyears` as the divisor of the storm rate. So `mean` equals `sst_expected` bitwise when RECORDCI's
     record years count the same.
  4. Hooks in `RainyDay_Py3.py`, marked `GP 2026-10-26`, with every replaced line kept as a comment and no block
     re-indented:
     - in exact mode, an `if freqexact:` branch ahead of the sampling `elif` chain, and one-line or `and not freqexact`
       guards on the draws, allocations, `StormSelector`, −9999 marks, whichrain dump, ranking and Monte Carlo writes;
     - in the storm loop, `add_storm` then `continue` right after reading `catrain`;
     - `exactfa.write_all(...)` before the Monte Carlo writes.
  5. Tools:
     - `tools/verify_exact_engine.py`: exact vs realizations of a `"both"` run;
     - `compare_runs.py --engine auto|montecarlo|exact --stat mean|median|level`;
     - `RAINYDAY_DUMP_EXACTPOOL`: the pool dump for the verification;
     - 16 new unit tests in `tools/test_record_ci.py`.
- **Why.** The Monte Carlo estimate carries sampling noise. For frequency analysis only (IDF curves, comparisons
  between basins), the exact engine removes it at almost no cost, and gives the ensemble spread as exact bands. Large T
  only needs a larger NYEARS, which costs pool depth, not draws.
- **Effect on results.**
  - `"montecarlo"` and `"both"`: none. Every Monte Carlo output is bit-identical; no `np.random` call was added,
    removed or moved.
  - `"exact"`: **by design, the random draws are skipped**. No `np.random` call happens after `np.random.seed(0)`, and
    no realizations or `_FreqAnalysis.csv` are written. The catalogs are built before any random call and are
    identical.
- **Verified** (2026-10-26, Mac M3 Max, RainyDay_Env; outputs in `/Users/gjperezm/Databases/RainyDay_fast_tests`):
  - **Bit-identity**, validation matrix (5 shapes, production settings), `ref` = main `b63865a`:
    - `compare_dumps ref off2 --catalogs` (this branch, option absent) and `ref both --catalogs`: ALL PASS, 80 checks
      each (35 raw-depth dumps, 35 FreqAnalysis CSVs, 5 light catalogs);
    - all 55 PNGs of `both` byte-identical to `ref`.
    - With RECORDCI (B = 200), variants `on2` (Monte Carlo), `on_both` and `on_exact`:
      - `compare_dumps ref on2|on_both --catalogs` ALL PASS;
      - `_RecordCI.csv` and `_RecordBootstrap.csv` of `on2` and `on_both` byte-identical (35/35) to `fast/record`'s
        `on` (the StormPool refactor changes nothing);
      - `_RecordBootstrap.csv` and `_RecordBootstrapLevel.csv` of `on_exact` byte-identical to `on_both`.
  - **`"exact"`:**
    - 5/5 light catalogs bitwise equal to `ref` (`compare_dumps.compare_light_catalogs`);
    - 35 `_FreqExact.csv`, and no `_FreqAnalysis.csv`, PNG or raw-depth dump;
    - every `_FreqExact.csv` byte-identical to `both`'s (35/35).
  - **Exact vs Monte Carlo** (`tools/verify_exact_engine.py both`; 322 cells: 4 basins × 7 durations × 10 T and Talequah
    7 × 6):
    - rows, realizations and quantile columns consistent in all 35 durations;
    - **mean**: |z| ≤ 2.64 for the 280 Iowa/Texas cells (z = (rd_mean − mean)/(sd/√100); 1.8 % beyond 2). Talequah has
      5 realizations: max |t| = 3.87 (t, 4 df: p = 0.018; 3 durations of the same realizations);
    - **quantiles**: of the 5 × 322 cells, 1603/1610 lie inside the order-statistic 99 % binomial band of the 100
      realizations, and 1570/1610 inside the 95 % band;
    - **observed minimum / maximum** of the realizations inside the central 90 % of the exact distribution of the
      min/max of N draws: 288/322 (89.4 %) and 289/322 (89.8 %).
  - **mean − level** (NYEARS 1000):

    | T | Median (max) |
    |---|---|
    | 10 yr | 0.09 % (0.22 %) |
    | 100 yr | 0.7 % (3.3 %) |
    | 500 yr | 2.3 % (6.3 %) |
    | 1000 yr | 2.7 % (8.9 %) |

  - **Unit tests:** `tools/test_record_ci.py`, 44 PASS (28 existing + 16 new). They check `quantiles`, `cdf`,
    `expected_minmax` and `expected` against the direct simulation `simulate()`, for PDS and AMS:
    - all-ones and sparse weights (rows with many −9999);
    - gamma pools with ties and zeros, and pools of synthetic storms at timeres 60 and 30.

    They also check `_values_at` against the sorted expanded values, the rows against `rank_frequency` bitwise, the
    `mean`/`level` of `ExactFA` against RecordCI's `sst_expected`/`sst_level` bitwise, and the option parsing.
  - **RECORDCI + exact vs RECORDCI + both** (28 Iowa/Texas files):
    - `sst_expected`, `sst_level`, `rec_lo/hi/se` identical;
    - `rd_mean` = the exact mean = `sst_expected` bitwise.
    - Relative to the Monte Carlo values: `rd_mean` within 0.7 % (median; max 3.0 %, Monte Carlo noise). The `ens`
      bands differ by 3–4 % (median; max 18 %): the exact expected min/max of 100 vs one observed min/max. `pred`
      differs by about 1 % (max 7 %), and the `est` width ratio is 0.99.
  - **`compare_runs.py`**, Iowa HUC-8 07100006 raw vs rg (`CATALOGPATH`, RECORDCI B = 1000, seed 20261026;
    `runs/hucexact`):
    - exact pair: Δ(`mean`) equals Δ(`sst_expected`) exactly; the Monte Carlo CI has zero width; the combined CI equals
      the record CI within 0.12 pp (resampling of the record deviations); `--stat level` runs too;
    - classes vs the Monte Carlo pair (`runs/val/HUC`, `--stat mean`): 66/70 equal. The 4 others are
      inconclusive → equivalent with the exact engine (3 h 200 yr, 6/12/48 h 1000 yr), where the Monte Carlo noise had
      widened the CI;
    - the mixed pair (Monte Carlo raw vs exact rg) runs;
    - the Monte Carlo pair gives exactly the old `compare/huc07100006_T100.csv` (`engine_a`/`engine_b` columns added).
  - **Runtime** (validation matrix, RECORDCI off, CREATECATALOG; 5 jobs in parallel, one variant at a time on a quiet
    machine, 13:41–13:56; `RAINYDAY_TIMERS`).
    - "Frequency analysis" is:
      - Monte Carlo: `resampling: total` + `frequency: write`, i.e. storm reads, draws, transposition of 7 durations
        and ranking;
      - exact: `resampling: total` + `exact: estimators + write`, i.e. storm reads, per-storm pool and estimators.
    - Storm reads (light-catalog rebuild, 9–13 s) are in both.

    | Shape | Total, montecarlo | Total, exact | Total, both | FA part, montecarlo | FA part, exact |
    |---|---|---|---|---|---|
    | Talequah | 9.8 s | 8.7 s | 12.2 s | – | – |
    | Texas HUC 120401020101 | 139 s | 133 s | 155 s | 30 s | 24 s |
    | Iowa pixel | 197 s | 197 s | 216 s | 34 s | 34 s |
    | Iowa 16,000 km² circle | 233 s | 206 s | 263 s | 66 s | 41 s |
    | Iowa R10 ellipse, rotated 45° | 320 s | 220 s | 352 s | 143 s | 44 s |

    - The catalog scan dominates the totals.
    - With the table engine, the Monte Carlo cost grows with the number of draws (NYEARS × NREALIZATIONS) and the box
      size; the exact cost does not depend on NYEARS or NREALIZATIONS.
    - `both` adds the pool and estimators to the Monte Carlo run (+20–32 s here, including the
      `RAINYDAY_DUMP_EXACTPOOL` dump).
    - Peak memory, exact vs montecarlo: −0.17 to +0.21 GB.
    - Iowa HUC-8 07100006 (`CATALOGPATH`, RECORDCI B = 1000), exact + RECORDCI: 1.75 min. The Monte Carlo +
      RECORDCI runs of `runs/val` took 2.50–2.62 min, at another time and not concurrent with other jobs.
    - A frequency-analysis-only rerun of both engines from the `off2` catalogs (`fa_mc`, `fa_exact`, 13:56–14:05) is not
      reported: it overlapped a system incident (load average 63–114).
- **Limits.**
  - Scope: the same as RECORDCI (uniform transposition, Poisson, PDS/AMS, watershed/box, one duration or
    MULTIPLEDURATIONS); `"exact"` and `"both"` stop on other settings, including SCENARIOS.
  - `"exact"` has no realizations: no SCENARIOS, no `_FreqAnalysis.csv`, no ensemble statistics other than the exact
    ones (the median of the realizations is not available; `q50` is the median of one realization's T-yr value).
  - Rows only exist for T ≤ NYEARS, and `mean` is the NYEARS-dependent realization mean. For large T, raise NYEARS
    (cheap in exact mode) or use `level`.
  - `mean` and `exp_*` use the 20,000-point trapezoid of `expected` (relative error about 1e-5); `level` and the
    quantiles are exact pool values.
  - Without `INCLUDEYEARS`, RainyDay's `nyears` counts storm end years; the exact engine uses it, as RainyDay does.
  - Memory: the RECORDCI pool (durations × storms × locations × 4 bytes), once even when both options are on.

### B9. Bug fixes of 2026-10-26 (GP, branch `fast/fixes`). Status: done and verified; validation matrix pending.
- **Commits:** pending.
- **Where.** New parsing code is in `Source/RainyDay_utilities_Py3/fast_parse.py`, called from one-line hooks in
  `RainyDay_Py3.py` (marked `GP 2026-10-26`, the replaced code kept as comments). No `np.random` call was added,
  removed or moved.
- **Rule kept.** Every input that already parsed correctly gives the same objects as before (values, list vs array,
  `int` vs `np.int32`). `tools/test_parsing.py` checks this against verbatim copies of the old code; the paper's
  configuration and the 5 matrix JSONs parse to identical objects.
- **Tests** (no data needed): `tools/test_parsing.py` and `tools/test_rect_catalog.py`.
- **Validation matrix** (5 shapes, production settings, on top of `fast/exact`): pending.

#### B9.1 Rectangular-domain storm catalog search
- **What.** With `DOMAINTYPE "rectangular"`, the catalog search (`RainyDay_Py3.py`, storm scan) and the
  duration-rewrite search (catalog longer than `DURATION`, `DURATIONCORRECTION` false) now call
  `catalogFFT_irregular_domainaware` with an all-ones domain mask: the exhaustive search of the irregular domains,
  every anchor `0..W-mw`, `0..H-mh`. The old `catalogNumba` call is kept as a comment.
- **Why.**
  - **Fork regression (`0cc1f2c`).** That commit set `xlen = W-mw+1`, `ylen = H-mh+1` for the irregular kernel. The
    mirrored `catalogNumba` of rectangular domains then sliced `mw-1` columns (`mh-1` rows): a numba "unable to
    broadcast" ValueError for any mask 3 or more cells wide or tall.
  - **Masks of 1–2 cells** ran, but returned wrong anchors: three mirrored quadrants stored the sum one cell away, the
    last row and column got an empty slice (0), and the middle anchors were never evaluated and stayed 0.
  - **The original RainyDay (`b14a011`)** used `xlen = W-mw`, `ylen = H-mh-2` with the same mirrored kernel. It never
    searched the last anchor rows and also stored mirrored sums one cell away (`tools/test_rect_catalog.py` checks
    both old calls).
- **Consistency of the rest of the rectangular branch.** `xlen`/`ylen` are now just the anchor counts. They match the
  search range, the uniform draws (`randint(0, W-mw+1)`) and the KDE grid (`kx`, `ky`). The KDE `checkind` therefore
  drops storms on the first and last anchor, as intended. Nonuniform transposition samples the same anchor grid.
  `xloop`, `yloop`, `rainsum` and `catalogstride` fed only `catalogNumba`.
- **`CATALOGACCELERATOR` has no effect on either domain type.** Irregular domains have ignored it since the FFT search
  (`5536ca0`). The run now prints that it is ignored, instead of "Good for you!".
- **Engines.** Rectangular domains use the per-step search, the same one as the `fft_perstep` engine. The batched
  engines (`RAINYDAY_CATALOG_ENGINE`, `COMPUTE`) stay irregular-only, which is the smallest change. A rectangle can use
  them as `DOMAINTYPE "irregular"` with a rectangle polygon: the catalog is identical (below).
- **Effect.** Rectangular catalogs and the frequency analyses on them change; irregular domains are unchanged
  (section C says for which runs).
- **Verified.**
  - `tools/test_rect_catalog.py`, brute force with plain numpy at every anchor. Masks 1×1, 2×3, 5×7 and a fractional
    6×9 basin mask with a hole; 26 random fields each (25 of 40×50, 1 of 184×184). The argmax anchor is identical in
    every case. The FFT sum map is within 3.4e-5 mm absolute (2.4e-7 relative) of the float64 direct sums. The smallest
    top-2 gap was 1.2e-4 relative, so there were no near-ties.
  - Maxima planted at all 8 edge and corner anchors, including `(H-mh, W-mw)`, are found exactly.
  - The fork's old call raises the ValueError for 2×3, 5×7 and the basin mask, and returns wrong anchors for 1×1 and
    2×2. The original's call returns wrong anchors for planted maxima on the last row and the right edge.
  - Talequah box (180×180) as `"rectangular"` (`AREA_EXTENT`): the catalog builds in 0.13 min and the frequency analysis
    finishes for 1–72 h. The fork's old code stops with the ValueError at the first time step.
  - The same box as `"irregular"`, with a rectangle polygon: `compare_catalogs.py --rtol 0` ALL PASS (10 storms, every
    variable including the masks). The rectangular catalog is also bitwise equal to the matrix `ref` Talequah catalog
    (`compare_light_catalog.py`, built by `batchfft` on the original polygon).
  - There is no grid difference: `findsubbox` selects the same 180×180 grid for the float32 polygon bounding box, the
    float32-exact `AREA_EXTENT`, the float64 polygon corners and the rounded corners. `rastermask(..., all_touched=True)`
    of the rectangle polygon is all ones. In general, a box edge within float32 rounding of a grid line could select a
    different row or column for the two domain types.
  - Box area 6×8 on the rectangular Talequah domain: storms found at `x = W-mw`, `y = H-mh`; finishes 1, 6, 72 h.
- **Limits.** The per-step search is slower than the batched engines on large domains, and is not multithreaded. The
  duration-rewrite path keeps its other known problems (its window search skips the last window, and the times of all
  storms but the last are left uninitialised).

#### B9.2 `EXCLUDEMONTHS`: wrap-around ranges, single months, lists of strings
- **What.** `fast_parse.parse_excludemonths`:
  - `"12-3"` becomes `[12, 1, 2, 3]`;
  - `"8"` becomes `[8]`;
  - a list of strings (`["12","1"]`) becomes integers.
  Everything else is as before: comma strings, ordinary ranges, `"none"`, JSON lists of numbers, other JSON types (`[]`).
  A string that cannot be read (`"summer"`, `"1,3-5"`, `"13-2"`) now stops with the accepted formats.
- **Why.**
  - `range(12, 4)` is empty, so nothing was excluded, silently.
  - A single string fell into `else: []`.
  - A list of strings never matched the integer months, in `createfilelist` or in the catalog filter.
- **Effect.** Runs that used these spellings now exclude the months they named. Before, they ran with every month
  (section C).
- **Verified.**
  - `test_parsing.py`: one check per spelling, plus a battery of previously valid inputs with identical objects.
  - New Talequah catalog with `"9-7"`: 31 August files and 3 storms. The old code read all 92 Jul–Sep files.
  - Iowa pixel `ref` catalog (`CATALOGPATH`, no `INCLUDEYEARS`) with `"12-3"`: 377 of 400 storms kept, months 04–11.
    The old code kept all 400. The frequency analysis (1, 24, 72 h) finishes in 31 s.
  - Lists of month strings with `INCLUDEYEARS ["2002"]`: 62 Jul–Aug files. The old code found 0 files and stopped with
    an IndexError.

#### B9.3 `INCLUDEYEARS`: `"all"`, lists of strings
- **What.** `fast_parse.parse_includeyears`. `"all"`, `"none"` (the original User's Guide uses both) and `""` explicitly
  mean all years (`False`). A list of strings becomes `np.int32`. Ranges, comma strings, single years and JSON lists are
  as before; the range `"2002-2023"` still gives Python `int`s. An unreadable string (`"2002to2010"`) now stops with a
  message.
- **Why.** `includeyr.lower == "all"` compared the method, which is never equal. `"all"` only worked because
  `np.int32("all")` raised inside the surrounding `try`. Any unreadable string silently meant all years.
- **Effect.** None for `"all"` and `"none"` (the same `False` as before). Lists of strings used to crash (empty file
  list); unreadable strings used to run with all years and now stop.
- **Verified.** `test_parsing.py`, and a catalog built with `INCLUDEYEARS ["2002"]` (B9.2).

#### B9.4 `EXCLUDEMONTHS` on an existing catalog without `INCLUDEYEARS`
- **What.** In the `CreateCatalog==False` filter, `catinclude` compares `np.int32(month)` with `excludemonths`, as
  `stormlist` and the `INCLUDEYEARS` branch already did. This is a one-line change in `RainyDay_Py3.py`.
- **Why.** The file-name month is the string `"08"`. It is never in a list of integers, so `catinclude` kept every
  storm while `stormlist` dropped the excluded months. `catmax`/`catx`/`caty`/`cattime` no longer lined up with
  `stormlist`: storm i was resampled with another storm's locations and times, and the Poisson rate `len(catmax)/nyears`
  still counted the excluded storms.
- **Effect.** Such runs change (section C). With `INCLUDEYEARS` set, or without `EXCLUDEMONTHS`, nothing changes.
- **Verified.**
  - Real runs on the Iowa pixel `ref` catalog with the driver itself (runpy, arrays inspected afterwards). Old code,
    `"12,1,2,3"`: `stormlist` 377 storms, `catmax` 400, 372 rows belonging to other storms. New code, `"12-3"` and
    `"12,1,2,3"`: 377 storms, every array row equal to its storm's catalog row, start months 04–11 only.
  - `test_parsing.py` runs the driver's filter block on a synthetic catalog and checks the same.

#### B9.5 `EXCLUDESTORMS`: a single storm number as a string
- **What.** `fast_parse.parse_excludestorms`. `"17"` becomes `[17]`; everything else is as before. An unreadable string
  stops with a message.
- **Why.** A single string fell into `else: []`, so nothing was excluded.
- **Effect.** Runs with such a value now exclude the storm (section C).
- **Verified.** On the Talequah `ref` catalog, `"3"` gives 9 storms, aligned. The old code kept 10.

#### B9.6 Unknown `CALCTYPE` stops before the catalog is built
- **What.** `fast_parse.parse_calctype`. `ams`, `annmax`, `pds` and `partialduration` (any case) are read as before,
  with the same messages. A missing key or JSON `null` gives `'ams'`, as before. Anything else stops at parse time,
  before the storm catalog, with the valid values listed.
- **Why.** The old code printed "Defaulting to Annual Maxima Series" but never set `calctype`. The run built the whole
  catalog and then stopped with a NameError. A non-string value (e.g. a JSON number) silently ran as AMS.
- **Effect.** Only inputs that crashed, or non-string values that silently ran as AMS, behave differently.
- **Verified.** With an unknown value, the new code stops after 1 s and creates no `StormCatalog` folder. The old code
  built the catalog (0.12 min), then raised `NameError: calctype`. `test_parsing.py` checks the hook precedes the
  catalog block.

#### B9.7 `UNCERTAINTY` as a number
- **What.**
  - `fast_parse.parse_uncertainty`: a JSON number `90`, or the string `"90"`, gives the quantile band.
  - `RainyDay_functions.write_frequency_outputs`: one added line sets `meanrain` (the realization mean) in the
    quantile branch.
  - `"ensemble"`, `"all_realizations"` and other strings are read as before. Other strings, e.g. `"false"` in the
    Talequah example, still fall back to `"ensemble"` with a message.
- **Which quantiles.** For `X = 90` the CSV columns `minrain` and `maxrain` are `np.percentile` `(100-X)/2 = 5` and
  `X + (100-X)/2 = 95` of the realizations, i.e. the 5th–95th percentiles, as the run prints. The original User's
  Guide says "the range between the X/2 and 100-X/2 quantiles", which for 90 would be the 45th–55th. Its example
  (realizations 50 and 950 of 1000) is right; the formula is wrong.
- **Why.** A JSON number failed on `.lower()` and silently gave `"ensemble"`. The string path never set `spreadmean`,
  so the run stopped at the very end: an `UnboundLocalError` in `write_frequency_outputs` (since A11), and a
  `NameError` in the original RainyDay, where the same code ran at module level.
- **Effect.** JSON numbers now give the 5th/95th band instead of min/max (section C). Numeric strings now finish.
- **Verified.**
  - Talequah `ref` catalog, `UNCERTAINTY 90` and `"90"`: both finish. All 14 CSVs (7 durations each) equal, as text,
    the 5th percentile, mean and 95th percentile of the dumped realizations.
  - The draws are bitwise identical to an `"all_realizations"` run.
  - Old code: `90` wrote min/max; `"90"` stopped with `UnboundLocalError: spreadmean`.

#### B9.8 Missing required keys
- **What.** `fast_parse.check_required`, called right after the JSON is read, stops with one message that names every
  missing key and what it is for.
  - Always required (the driver reads them with no default): `MAINPATH`, `SCENARIONAME`, `CATALOGNAME`,
    `CREATECATALOG`, `DURATION`, `VARIABLES`, `POINTAREA`.
  - When the settings need them: `RAINPATH` (`CREATECATALOG` true); `AREA_EXTENT` (`CREATECATALOG` true and
    `DOMAINTYPE` not `irregular`); `WATERSHEDSHP` (watershed/basin); `POINTLAT`/`POINTLON` (point/grid);
    `BOX_YMIN/YMAX/XMIN/XMAX` (box/rectangle).
- **Why.** The reads catch `ImportError`/`ValueError`/`IndexError`, never `KeyError`, so a missing key gave a raw
  traceback.
- **Effect.** Only runs that crashed.
- **Verified.** `test_parsing.py`: every key above, plus the paper, Talequah and matrix JSONs pass. A run without
  `POINTAREA` gives a clear message (old: `KeyError: 'POINTAREA'`).

#### B9.9 `NSTORMS` absent with an existing catalog
- **What.** `fast_parse.default_nstorms`. Without `NSTORMS`, a `CREATECATALOG false` run uses every storm of the catalog.
  This is the same run as `NSTORMS` = the catalog's storm count, as the original User's Guide describes.
- **Why.** `nstorms` was undefined (NameError at the catalog-size check), and `nstorms_default` later too.
- **Effect.** Only runs that crashed.
- **Verified.** Talequah `ref` catalog without `NSTORMS`: CSVs and dumps of all 7 durations are byte-identical to the
  same run with `NSTORMS 10`. The old code stopped with `NameError: nstorms`.

#### B9.10 `RETURNLEVELS` as a comma string (fork regression of `07021dd`)
- **What.** `fast_parse.returnlevels_from_string`. `"2,5,10"` becomes `[2., 5., 10.]` before the existing list code,
  which gives the same `np.float32` array as the JSON list `[2, 5, 10]`. JSON lists and `"all"` are unchanged. Any
  other string still stops with the old message.
- **Why.** The original User's Guide documents the comma string; the list fix of `07021dd` made it stop the run.
- **Effect.** Only runs that stopped.
- **Verified.** Talequah `ref` catalog with `"2,5,10,25,50,100"`: CSVs and dumps of all 7 durations are byte-identical
  to the JSON-list run.

**Not changed.** Other known problems of the original and the fork were left as they are, for example the substring
key matching that still surrounds the `UNCERTAINTY` and `RETURNLEVELS` hooks, and the duration-rewrite path (B9.1,
Limits).

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
- **`FREQENGINE` (B8).**
  - `"montecarlo"` (default) and `"both"` keep every Monte Carlo output bit-identical; `"both"` only adds
    `_FreqExact.csv/.png`.
  - `"exact"` replaces the Monte Carlo frequency analysis with its noise-free values, and by design skips the random
    draws (no `np.random` call after the seed). Its catalogs are identical; it writes no realizations and no
    `_FreqAnalysis.csv`.
- **B9, the bug fixes of 2026-10-26, change results only for these configurations.** Every other configuration that
  already ran, including the paper's, parses to identical objects and gives identical outputs.

  | Fix | Configurations that ran before and now give different results |
  |---|---|
  | B9.1 rectangular catalog search | **Relative to the original RainyDay (`b14a011`):** every `DOMAINTYPE "rectangular"` catalog in general, because the original search never searched the last anchor rows and stored mirrored sums one cell away; hence the frequency results too. **Relative to RainyDay_mod and earlier RainyDay_fast:** masks 3 or more cells wide or tall crashed (fork regression of `0cc1f2c`) and now run; masks at most 2 × 2 cells (e.g. a 1×1 point or pixel) ran with wrong anchors and now change. The same holds for the duration-rewrite search. `CATALOGACCELERATOR` > 1 is now ignored. |
  | B9.2 EXCLUDEMONTHS spellings | `EXCLUDEMONTHS` written as a range across the new year (`"12-3"`), a single month string (`"8"`) or a list of strings (`["8"]`): these months are now excluded (before: every month was used). An unreadable string now stops the run. |
  | B9.3 INCLUDEYEARS spellings | None for `"all"` and `"none"`. An unreadable `INCLUDEYEARS` string (it meant all years) now stops the run. |
  | B9.4 EXCLUDEMONTHS on an existing catalog | Existing catalog (`CREATECATALOG false` or `CATALOGPATH`) with `EXCLUDEMONTHS` and without `INCLUDEYEARS`: the storm arrays now line up with the storm files, and the Poisson rate counts only the kept storms. |
  | B9.5 EXCLUDESTORMS single string | `EXCLUDESTORMS` as a single number string (`"17"`): the storm is now excluded. An unreadable string now stops the run. |
  | B9.6 unknown CALCTYPE | A non-string `CALCTYPE` (e.g. a JSON number), which ran as AMS, now stops. Unknown strings used to crash after the catalog. |
  | B9.7 numeric UNCERTAINTY | `UNCERTAINTY` as a JSON number (`90`): the CSV and PNG band is now the 5th–95th percentile instead of min–max. The mean column is unchanged. |
  | B9.8–B9.10 required keys, NSTORMS absent, RETURNLEVELS string | None: these inputs crashed or stopped before. |

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
| `FREQENGINE` | JSON | `"montecarlo"`, `"exact"` (no realizations, no draws), `"both"` (B8) | `"montecarlo"` |
| `RAINYDAY_DUMP_EXACTPOOL` | env | folder for the exact engine's pool of every duration (`<name>_exactpool.npz`, for `tools/verify_exact_engine.py`; B8) | off |
| `RAINYDAY_CATALOG_ENGINE` | env | `batchfft`, `batchfft_mp`, `fft_perstep`, `batchtf`, `batchcuda` | auto (`batchfft`) |
| `RAINYDAY_BLOCKFILES` | env | files per FFT batch | 16 / auto |
| `RAINYDAY_DUMP_WHICHRAIN` | env | folder for the raw transposed depths | off |
| `RAINYDAY_TIMERS` | env | `1` prints timers | off |
| `RAINYDAY_VERIFY_TABLE` | env | fraction of draws brute-forced (B3) | off |

More detail: `README.md` (usage), `CLAUDE.md` (rules, correctness model, harness, benchmarks) and the `GP` comments in
the code.
