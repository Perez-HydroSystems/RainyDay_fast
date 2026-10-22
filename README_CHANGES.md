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

### B3. Table transposition engine, `MULTIDURATIONENGINE: "table"` (fix 3). Status: in validation.
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
- **Validation and final timings:** to be added after the merge.

### B4. Usability for other users, datasets and domains. Status: in development.
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

### B5. Validation and profiling tools (all in `tools/` unless noted)
- **`make_test_jsons.py`, `run_matrix.sh`, `compare_dumps.py`.** The validation matrix against RainyDay_mod:
  - five shapes: Talequah, Iowa pixel, Iowa 16,000 km² circle, Iowa R = 10 ellipse rotated 45°, Texas HUC basin;
  - production settings;
  - bitwise checks of the dumps, CSVs and catalogs.
- **`RAINYDAY_TIMERS=1`** (`Source/RainyDay_utilities_Py3/fast_prof.py`). Prints a wall-clock table of the run's
  parts. It does not change results.
- **`compare_light_catalog.py`, `test_light_rebuild.py`.** Checks for the light catalog.
- **`verify_fast_sst.py`, `bench_fast_sst.py`.** Checks and benchmarks for the table engine.

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

## D. New settings reference

| Setting | Where | Values | Default |
|---|---|---|---|
| `UNCERTAINTY` | JSON | adds `"all_realizations"` | original options |
| `CATALOGPATH` | JSON | `<folder>/StormCatalog/<CATALOGNAME>` | none |
| `MULTIPLEDURATIONS` | JSON | list of durations in hours, each ≤ `DURATION` | none |
| `MULTIDURATIONENGINE` | JSON | `"fast"`, `"exact"`, `"table"` (B3) | `"fast"` |
| `CATALOGSTORAGE` | JSON | `"full"`, `"light"` (B2) | `"full"` |
| `COMPUTE` | JSON block | engine, nworkers, blockfiles, GPU settings | `"auto"` |
| `RAINYDAY_CATALOG_ENGINE` | env | `batchfft`, `batchfft_mp`, `fft_perstep`, `batchtf`, `batchcuda` | auto (`batchfft`) |
| `RAINYDAY_BLOCKFILES` | env | files per FFT batch | 16 / auto |
| `RAINYDAY_DUMP_WHICHRAIN` | env | folder for the raw transposed depths | off |
| `RAINYDAY_TIMERS` | env | `1` prints timers | off |
| `RAINYDAY_VERIFY_TABLE` | env | fraction of draws brute-forced (B3) | off |

More detail: `README.md` (usage), `CLAUDE.md` (rules, correctness model, harness, benchmarks) and the `GP` comments in
the code.
