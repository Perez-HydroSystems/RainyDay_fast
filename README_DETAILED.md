# RainyDay_fast: complete description

> This is the complete description of RainyDay_fast. For a one-page summary, see [`README.md`](README.md).

**RainyDay_fast is a modified version of [RainyDay](https://github.com/HydroclimateExtremesGroup/RainyDay)**, the
stochastic storm transposition (SST) framework developed by Daniel B. Wright and the Hydroclimate Extremes Group
(University of Wisconsin–Madison). The modifications are by Gabriel Perez (Perez-HydroSystems).

- **Base:** the original RainyDay was taken from commit
  [`b14a011`](https://github.com/HydroclimateExtremesGroup/RainyDay/commit/b14a011) (17 April 2025). Everything described
  here is a change from that commit. Changes made upstream after it are **not** included.
- **Relation to RainyDay:** some of these changes may be integrated into RainyDay over time. For the current version
  of RainyDay, see the [RainyDay repository](https://github.com/HydroclimateExtremesGroup/RainyDay).
- **How the code is marked:** every change carries a `GP YYYY-MM-DD` comment, with the previous code kept as a comment
  next to it. [`README_CHANGES.md`](README_CHANGES.md) records every change, why it was made, its effect on results and
  its commit.

## What is different from the original RainyDay

RainyDay_fast adds three new capabilities and four ways of making runs faster. Each is switched on with a setting in
the run's JSON file (or a separate tool); **with the settings left at their defaults, a run behaves as described in the
original RainyDay User's Guide**, apart from the corrections listed in
[Changes that affect results](#changes-that-affect-results).

| # | Modification | Setting | What it gives you |
|---|---|---|---|
| **New capabilities** | | | |
| 1 | [Multiple durations in one run](#1-multiple-durations-in-one-run-multipledurations) | `MULTIPLEDURATIONS` | The frequency analysis for many durations (e.g. 1–72 h) from one storm catalog, in one run |
| 2 | [Record (sampling) uncertainty](#2-record-sampling-uncertainty-recordci) | `RECORDCI` | Confidence intervals for the uncertainty that comes from the short rainfall record |
| 3 | [Exact frequency engine](#3-exact-frequency-engine-freqengine) | `FREQENGINE` | The frequency curve RainyDay's realizations converge to, computed without Monte Carlo noise |
| **Faster runs** (same results, bit for bit) | | | |
| 4 | [Table transposition engine](#4-table-transposition-engine-multidurationengine-table) | `MULTIDURATIONENGINE: "table"` | 16–66× faster transposition and frequency analysis |
| 5 | [Light and full storm catalogs](#5-light-and-full-storm-catalogs-catalogstorage) | `CATALOGSTORAGE` | A storm catalog of about 2 MB instead of 2–4 GB per basin |
| 6 | [Cropped, compressed input files](#6-cropped-compressed-input-files-toolsmake_domain_croppy) | `tools/make_domain_crop.py` | Each run reads small files cut to its domain instead of the national files |
| 7 | [Batched, multithreaded catalog search](#7-batched-multithreaded-catalog-search-compute) | `COMPUTE` | The storm-catalog search runs on all CPU cores (optionally on a GPU) |

With 4–7 together, a full run (catalog, 7 durations from 1 to 72 h, 1,000 synthetic years) of a circular Iowa basin
of 16,000 km² went from 4,957 s to 241 s (21×), and that of an elongated ellipse of the same area from 10,066 s to
325 s (31×), with every output bit-identical (Apple M3 Max; details in [`README_CHANGES.md`](README_CHANGES.md), section B6).

---

## New capabilities

### 1. Multiple durations in one run: `MULTIPLEDURATIONS`

In the original RainyDay, every rainfall duration needs its own JSON file and its own run. With `MULTIPLEDURATIONS`,
one run builds (or reuses) the storm catalog for the longest duration and writes the frequency analysis for every
duration in the list.

```json
"DURATION": 72,
"DURATIONCORRECTION": "false",
"MULTIPLEDURATIONS": [1, 3, 6, 12, 24, 48, 72]
```

- **Outputs:** one folder per duration, `MAINPATH/<SCENARIONAME>_<d>/`, each with the usual `_FreqAnalysis.csv` and
  `_FrequencyAnalysis.png`. The storm catalog and the diagnostic plots are written once, in the folder of `DURATION`.
- **Same results as separate runs.** Each duration *d* gives exactly what a separate run gives with `DURATION` = *d*,
  `DURATIONCORRECTION` = `"true"` and this catalog: the largest *d*-hour window inside each transposed parent storm.
  All durations share the same random draws. This was verified bit for bit.
- **Speed:** the frequency analysis is about 8× faster than in seven separate runs, because the shorter durations are
  computed in one compiled pass.
- **Requirements:** `FREQANALYSIS` true, `DURATIONCORRECTION` false, and every duration ≤ `DURATION` and a multiple of
  the time step of the data. Not available with `SCENARIOS`, `ENHANCEDSST`, `ROTATIONANGLE`, `INTENSDISTR` or point
  areas.

### 2. Record (sampling) uncertainty: `RECORDCI`

RainyDay's band of realizations shows how much a synthetic record of `NYEARS` years varies for **one** storm catalog.
It does not include the uncertainty that comes from having only a few years of observed rainfall: a different 20-year
record would give a different catalog and a different frequency curve. `RECORDCI` estimates that record (sampling)
uncertainty.

```json
"RECORDCI": "bootstrap",
"RECORDNBOOT": 1000,
"RECORDLEVEL": 90,
"INCLUDEYEARS": "2002-2023"
```

- **How it works.** Every storm in the catalog is evaluated at every transposition location. The record years are then
  resampled with replacement (a year bootstrap) to make `RECORDNBOOT` alternative records, and the noise-free
  frequency estimate (see 3) is recomputed for each. The spread of those estimates is the record uncertainty. A
  leave-one-year-out jackknife (`"jackknife"`) is also available.
- **Outputs**, next to each `_FreqAnalysis.csv`: `_RecordCI.csv` (the record CI, the total CI of RainyDay's estimate
  and the ensemble band side by side, defined in a `#` header) and, with `RECORDSAVE` "true", every bootstrap record
  (`_RecordBootstrap.csv`, `_RecordBootstrapMedian.csv`) and the drawn years (`_RecordDraws.csv`).
- **It only adds files.** The catalog, the random draws and every existing output stay bit-identical.
- **Comparing two runs:** `tools/compare_runs.py` gives the difference between two runs (for example, two basin
  shapes) with a CI that combines the Monte Carlo and the record uncertainty, and classifies it as equivalent,
  increase, decrease or inconclusive.
- **Set `INCLUDEYEARS`** to the years of the rainfall record, so that the record years are defined exactly.
- **Limits:** the catalog is reweighted, not rebuilt, for each bootstrap record; years are treated as independent
  (moving blocks with `RECORDBLOCK` > 1); the bootstrap cannot create storms larger than those observed, so the CIs at
  500–1,000 years are lower bounds.

### 3. Exact frequency engine: `FREQENGINE`

RainyDay estimates the frequency curve by Monte Carlo: it draws `NREALIZATIONS` synthetic records and reports their
mean and spread, which carry Monte Carlo noise. Under uniform transposition and Poisson resampling, the process RainyDay
samples is known exactly once every storm has been evaluated at every location. `FREQENGINE` computes the frequency
analysis from it in closed form, with no noise and no random numbers.

```json
"FREQENGINE": "exact"
```

| Value | What runs | Output per duration |
|---|---|---|
| `"montecarlo"` (default) | RainyDay's realizations, as in the original | `_FreqAnalysis.csv`, `_FrequencyAnalysis.png` |
| `"exact"` | the exact engine only | `_FreqExact.csv`, `_FreqExact.png` |
| `"both"` | both engines | all Monte Carlo outputs, unchanged, plus `_FreqExact.csv/.png` |

`_FreqExact.csv` gives, for every return period:
- `mean`: the noise-free value of RainyDay's estimate (the mean of the realizations);
- `level`: the return level of the same model, which does not depend on `NYEARS`;
- `q05` … `q95`: the distribution of the value of one realization;
- the expected minimum and maximum of `NREALIZATIONS` realizations.

Use `"exact"` for frequency analysis (IDF curves, comparisons between basins or settings, large campaigns); use
`"montecarlo"` when you need the synthetic storm sequences themselves (`SCENARIOS`, flood modelling). `RECORDCI` and
`FREQENGINE` support uniform transposition, Poisson resampling, PDS or AMS, and watershed or box areas; other settings
stop the run at the start with a message.

---

## Faster runs

These options change **how** a run is computed, not **what** it computes: on the same machine and environment, every
catalog, random draw, transposed depth and output file is bit-identical to a run without them. A regression harness in
`tools/` checks this on five basin shapes.

One speed-up needs no setting: the original RainyDay starts Python's memory tracing (`tracemalloc`) in every run and
never reads it, which took most of the frequency-analysis time. RainyDay_fast leaves it off (`RAINYDAY_TRACEMALLOC=1`
turns it back on).

### 4. Table transposition engine: `MULTIDURATIONENGINE: "table"`

For every random draw (a storm and a location) and every duration, RainyDay sums the rain over the basin for each
possible time window and keeps the largest. On a large basin with many durations this is about 10¹² operations.

```json
"MULTIDURATIONENGINE": "table"
```

- **What it does.** For each storm, a fast Fourier transform (FFT) gives approximate basin totals for every window at
  every location. Only the few windows that can hold the maximum are then summed **exactly**, with RainyDay's own
  function and in the same order, so the result is identical.
- **Speed** (transposition and frequency analysis, 7 durations, vs the default `"fast"`): Iowa 1-km² pixel 579 → 36 s;
  Iowa 16,000 km² circle 4,048 → 69 s; 16,000 km² ellipse about 9,200 → 140 s; Texas HUC-12 705 → 37 s.
- **Other values:** `"fast"` (default; one compiled pass for all durations) and `"exact"` (the original object-mode
  code, one duration at a time). All three give identical results.
- **Check mode:** `RAINYDAY_VERIFY_TABLE=0.01` recomputes 1 % of the draws by brute force and stops on any mismatch.

### 5. Light and full storm catalogs: `CATALOGSTORAGE`

A RainyDay **storm catalog** is the list of the largest storms found in the rainfall record over the transposition
domain. The original RainyDay saves each storm as a netCDF file that holds its full rain field: every time step of the
storm's window (e.g. 72 hours) over the whole domain. For a large domain and a 20-year record, this is 2–4 GB per
basin, written once and read back during the run.

```json
"CATALOGSTORAGE": "light"
```

| | `"full"` (default) | `"light"` |
|---|---|---|
| What is stored | one file per storm, with its rain field (as the original RainyDay) | one small file, `StormCatalog/<CATALOGNAME>_lightcatalog.nc`, with everything except the rain fields: storm times, locations, totals, masks, the list of input files and a checksum (CRC32) of each storm's rain |
| Size (Iowa basin, 2002–2023) | 4.0 GB | 1.7 MB |
| Where the rain comes from | the catalog files | read again from the input files (`RAINPATH`) whenever it is needed, and checked against the stored checksum |
| Needs the input data | no | yes, the same data the catalog was built from |
| Use it for | `SCENARIOS`; sharing or archiving a catalog without the input data | everything else, in particular campaigns with many basins |

- **Safety:** if the input data have changed since the catalog was built, the checksum does not match and the run
  stops. Rebuild the catalog after correcting the input data.
- **Portable:** the light catalog finds its files by name and its grid by coordinates, so the input folder can be
  moved or replaced by a crop (see 6).
- **Reuse:** `CATALOGPATH` reuses an existing catalog, full or light, in another run.
- **Conversion:** `tools/materialize_catalog.py <light .nc> <output folder>` writes the equivalent full catalog, file
  for file identical to one built with `"full"`.

### 6. Cropped, compressed input files: `tools/make_domain_crop.py`

RainyDay reads each daily input file (e.g. AORC, about 885 × 1,770 cells over the United States) and decompresses the
whole file, although a run only needs its transposition domain. A campaign of many runs over the same domain repeats
this work in every run.

```bash
python tools/make_domain_crop.py --json RUN.json --verify     # writes the crop and RUN_crop.json
```

- **What it writes:** a copy of every daily file, cut to the domain plus a small margin, with the same names, values,
  times and attributes, and `RUN_crop.json`, a copy of your JSON with `RAINPATH` pointing to the crop. RainyDay itself
  needs no change.
- **Compression (`--deflate`).** The crop is written with lossless netCDF/zlib compression, level 1 by default:
  - `--deflate 1` (default): 25–39× smaller than uncompressed and about 13× faster to read than the national files
    (5.5 vs 73 ms per file, Iowa). 2002–2023: 0.76 GB for Iowa, 0.67 GB for SE Texas.
  - `--deflate 0`: uncompressed, slightly faster to read (4.6 ms per file) but 17–29 GB per domain.
  - Compression is lossless, so the values are identical either way.
- **Bit-identical by construction.** RainyDay derives the grid spacing from the mean spacing of the coordinates, so the
  tool chooses margins that keep that mean exactly equal to the full file's, and checks it. `--verify` also checks that
  every file reads identically and that the domain and basin masks are identical.
- **Any dataset RainyDay reads:** one file per day with the date in its name, a regular latitude–longitude grid, and
  the precipitation stored as (time, latitude, longitude).
- **Limit:** `ENHANCEDSST` "stochastic" or "deterministic" depends on the file extent and is not bit-identical on a crop.

### 7. Batched, multithreaded catalog search: `COMPUTE`

Building a storm catalog means scanning every time step of the rainfall record and finding where, inside the domain,
the basin would collect the most rain. The original RainyDay does one search per time step, on one core.
RainyDay_fast processes many time steps per batch with multithreaded FFTs, and spreads the batches over a pool of
processes that read and compute in parallel.

```json
"COMPUTE": {"engine": "auto", "nworkers": "auto", "blockfiles": "auto"}
```

| Engine | What it is |
|---|---|
| `batchfft_mp` | batched CPU FFT over a pool of processes; **the default on macOS and Linux with 4 or more cores** |
| `batchfft` | batched, multithreaded CPU FFT in one process (the default when neither of the other two applies) |
| `batchcuda` | NVIDIA GPU (CuPy), with every accepted storm confirmed exactly on the CPU (the default with a CUDA GPU when `batchfft_mp` is not available, e.g. on Windows) |
| `batchtf` | Apple GPU (TensorFlow-Metal), with the same CPU confirmation |
| `fft_perstep` | the reference: one search per time step |

- **Every field is optional;** `"auto"` (or no `COMPUTE` block) picks the engine and the number of workers from the
  machine's cores and free memory. An environment variable (`RAINYDAY_CATALOG_ENGINE`, `RAINYDAY_NWORKERS`,
  `RAINYDAY_BLOCKFILES`) overrides the JSON value.
- **Same catalog:** storm dates, positions and rain fields are identical to `fft_perstep`; the storm totals agree to
  rounding (the measured difference is zero for the CPU engines).
- **Speed** of the catalog search, 8 workers vs one search per step: 10× (Talequah example) and 17× (a domain covering
  the conterminous United States).
- **Scope:** irregular domains (`DOMAINTYPE` "irregular"). Rectangular domains use the per-step search.

---

## Changes that affect results

All the options above either add outputs or give bit-identical results. A few corrections, made on the way, do change
results relative to the original RainyDay (`b14a011`). They are always on:

- **Strict transposition on irregular domains.** A transposition location is valid only if the whole basin (its
  bounding box) lies inside the domain, the same rule as the catalog search. The original only required one anchor
  cell inside the domain, so a transposed basin could hang partly outside the domain and collect rain there. This
  changes results wherever transposed basins reach the domain edge.
- **Catalog search on irregular domains.** Every location where the whole basin fits inside the domain is searched
  exactly, by FFT. The original search skipped or misplaced some locations. Catalogs can therefore hold other storms or
  positions.
- **Catalog search on rectangular domains.** The same exact search replaces the original one, which missed the last
  rows of locations.
- **Input parsing.** Several spellings of `EXCLUDEMONTHS`, `INCLUDEYEARS`, `EXCLUDESTORMS`, `RETURNLEVELS` and
  `UNCERTAINTY` that the original read wrongly or ignored now work; unreadable values stop the run with a message
  instead of being silently ignored.
- **Smaller fixes:** missing time steps in the input are filled with zeros (with a message) instead of failing; errors
  while writing the catalog are no longer swallowed; fixes to the diagnostic plots and to `SCENARIOS` with one
  realization.

[`README_CHANGES.md`](README_CHANGES.md) (sections A, B9 and C) lists every change, with the configurations it affects.

### Other additions

- `CATALOGPATH`: reuse a storm catalog from another folder or run, with warnings if it was built with other settings.
- `UNCERTAINTY: "all_realizations"`: write every realization to `_FreqAnalysis.csv`.
- `DIAGNOSTICPLOTS: "only-statistics"`: the catalog statistics maps without the per-storm maps.
- `RAINYDAY_TIMERS=1`: print how long each part of the run took.
- Comparison and verification tools in `tools/` (catalogs, runs, light catalogs, the exact engine).

---

## Getting started

1. **Environment.** RainyDay_fast uses the same Python environment as RainyDay ([`RainyDay.yml`](RainyDay.yml)) and
   needs no extra package. The optional GPU engines need CuPy (NVIDIA) or TensorFlow-Metal (Apple).
   ```bash
   conda env create -n RainyDay_Env -f RainyDay.yml
   conda activate RainyDay_Env
   ```
2. **Run** RainyDay as usual, with the JSON passed as an absolute path:
   ```bash
   python Source/RainyDay_Py3.py /absolute/path/to/RUN.json
   ```
3. **A typical fast configuration** for a campaign (after making a crop with `tools/make_domain_crop.py`):
   ```json
   "RAINPATH": "<crop folder>/AORC.*.preciptemp.nc",
   "DURATION": 72,
   "DURATIONCORRECTION": "false",
   "MULTIPLEDURATIONS": [1, 3, 6, 12, 24, 48, 72],
   "MULTIDURATIONENGINE": "table",
   "CATALOGSTORAGE": "light",
   "FREQENGINE": "both",
   "RECORDCI": "bootstrap",
   "INCLUDEYEARS": "2002-2023"
   ```

## Documentation

| Document | Contents |
|---|---|
| [`UserGuide/RainyDay_Tutorial.ipynb`](UserGuide/RainyDay_Tutorial.ipynb) | The original RainyDay User's Guide: the SST method and every control-file setting. It still applies, except where the guide below says otherwise. |
| [`UserGuide/RainyDay_fast_tutorial.ipynb`](UserGuide/RainyDay_fast_tutorial.ipynb) | The RainyDay_fast User's Guide: every new or changed setting, the seven modifications in detail, two worked case studies, validation, corrected bugs and known issues. |
| [`README_CHANGES.md`](README_CHANGES.md) | Every change with respect to the original RainyDay: what, why, effect on results, commit, and a reference of all new settings (section D). |
| [`tools/aorc_fix/AORC_FIX_INSTRUCTIONS.md`](tools/aorc_fix/AORC_FIX_INSTRUCTIONS.md) | The RainyDay-format AORC files have four known data defects (for example, hour 23:00 UTC is a copy of 22:00 in 2002–2020). These scripts build a corrected copy from NOAA AORC v1.1, step by step, with checks. |

## How to cite

If you use RainyDay_fast, please cite the RainyDay papers:

- Wright, D. B., Mantilla, R., & Peters-Lidard, C. D. (2017). A remote sensing-based tool for assessing rainfall-driven
  hazards. *Environmental Modelling & Software*, 90, 34–54. https://doi.org/10.1016/j.envsoft.2016.12.006
- Wright, D. B., Yu, G., & England, J. F. (2020). Six decades of rainfall and flood frequency analysis using stochastic
  storm transposition: Review, progress, and prospects. *Journal of Hydrology*, 585, 124816.
  https://doi.org/10.1016/j.jhydrol.2020.124816

and this repository: Perez, G. RainyDay_fast: a modified version of RainyDay with multi-duration, record-uncertainty
and exact frequency analysis and faster runs. https://github.com/Perez-HydroSystems/RainyDay_fast

## License

MIT, as the original RainyDay (see [`LICENSE`](LICENSE)). The original copyright notice is kept; the modifications are
distributed under the same license.

---

## About RainyDay (from the original README)

Welcome to RainyDay. RainyDay is a framework for generating large numbers of realistic extreme rainfall scenarios based on relatively short records of remotely-sensed precipitation fields.  It is founded on a statistical resampling concept known as stochastic storm transposition (SST).  These rainfall scenarios can then be used to examine the extreme rainfall statistics for a user-specified region, or to drive a hazard model (usually a hydrologic model, but the method produces output that would also be useful for landslide models). RainyDay is well suited to flood modeling in small and medium-sized watersheds.  The framework is made to be simple yet powerful and easily modified to meet specific needs, taking advantage of Python’s simple syntax and well-developed libraries.  It is still a work in progress.  Therefore, the contents of the guide may be out-of-date.

Development of RainyDay has been supported by NASA, the National Science Foundation, The Wisconsin Alumni Research Foundation, and the Research and Development Office at the U.S. Bureau of Reclamation.

The latest version of RainyDay is distributed under the MIT open source license: https://opensource.org/licenses/MIT
