# RainyDay_fast (GP 2026-10-22)

**This repo is RainyDay_fast**, a performance playground cloned from RainyDay_mod at 9137292. RainyDay_mod is not
touched. The remote is `fast` = private `Perez-HydroSystems/RainyDay_fast`. There is no `mod`/`origin` remote here, so
nothing can reach RainyDay_mod by accident.

**Goal:** fixes 1–3, see `README.md` "RainyDay_fast":
1. cropped, uncompressed input per domain;
2. light storm catalog;
3. table transposition engine.

**Hard rule:** results must be **bit-identical** to RainyDay_mod on the same machine and environment. That means:
- the same catalogs;
- the same random draws;
- the same `whichrain`;
- byte-identical FreqAnalysis CSVs.

Any difference is a bug.

**Rules:**
- **Every change is recorded in `README_CHANGES.md`** (what, why, effect on results, commit), as GP requested on
  2026-10-22.
- **New code goes in new modules:** `Source/RainyDay_utilities_Py3/fast_*.py` and `tools/`. Edits to
  `RainyDay_Py3.py` are small hooks marked `GP YYYY-MM-DD`, with the old code kept as comments.
- **Do not add, remove or reorder any `np.random` call.** Verification sampling uses `np.random.default_rng`.
- **Commits** are allowed on the `fast/*` branches and on `main` within the approved plan. Push to GitHub (`fast`) only
  when GP asks.
- **Large outputs stay out of OneDrive:**
  - runs, catalogs and dumps go in `/Users/gjperezm/Databases/RainyDay_fast_tests/`;
  - crops go in `/Users/gjperezm/Databases/AORC_RainyDay/crops/`.

**Validation matrix** (`tools/make_test_jsons.py`, `tools/run_matrix.sh`, `tools/compare_dumps.py`):
- **Shapes:**
  - Talequah (smoke test, 30 s);
  - Iowa `Pixel_r1_A1km2_rot0`;
  - Iowa `Circular_r1_A16000km2_rot0`;
  - Iowa `Elliptical_r10_A16000km2_rot45`;
  - Texas HUC `Basin_raw_Texas_HUC_120401020101`.

  All use production settings: NYEARS 1000, MULTIPLEDURATIONS 1–72 h, 10 return levels, `data_aorcfix2`.
- **Reference variant `ref_mod`:** RainyDay_mod 9137292 with `RAINYDAY_DUMP_WHICHRAIN` set and the catalogs kept.

```bash
python tools/make_test_jsons.py <variant> [--rainpath Iowa=DIR HUC_Texas=DIR Talequah=DIR] [--catalog-from ref_mod] [--set KEY=VALUE]
bash   tools/run_matrix.sh <repo_dir> <variant> [name ...]        # parallel, with dumps, wall time, peak memory
python tools/compare_dumps.py ref_mod <variant> [name ...] [--catalogs] [--csv-ref <ubuntu_kit/reference_md>]   # must print ALL PASS
```

**Timers:** `RAINYDAY_TIMERS=1` prints a wall-clock table at the end of a run (`Source/RainyDay_utilities_Py3/fast_prof.py`;
hooks in `RainyDay_Py3.py`). The timers do not change results: Talequah stays bit-identical to `ref_mod`.

**Fix 1, cropped input** (`tools/make_domain_crop.py`, branch `fast/io`; full workflow in README "Using RainyDay_fast
for a new domain or dataset"):
- **Usage:** `make_domain_crop.py --json RUN.json [--dst DIR] [--verify]`.
  - It reads RAINPATH, VARIABLES, the domain, INCLUDEYEARS/EXCLUDEMONTHS and WATERSHEDSHP from the JSON.
  - It writes the crop and `<stem>_crop.json`, a copy with RAINPATH pointing at the crop.
  - `--deflate 1` (lossless) is the default.
  - An existing `--dst` crop must contain the domain; the tool then only adds missing files and verifies.
- **Location:** crops go in `/Users/gjperezm/Databases/AORC_RainyDay/crops/<domain>/`. Each crop has a
  `_crop_manifest.json` with its window, margins and the source cells just outside it.
- **Grid resolution:** `rainprop_setup` takes it as the **mean** coordinate spacing over the whole file, so the margins
  are searched until that mean equals the full file's bitwise.
- **Coverage check:** `FIO.check_crop_coverage`, a hook after `createfilelist`. It stops a run whose domain is not
  inside the RAINPATH crop.
- **Not bit-identical on a crop:** `ENHANCEDSST` `stochastic`/`deterministic`, because of `rainprop.bndbox`.

**Fix 2, light catalog** (`Source/RainyDay_utilities_Py3/fast_io.py`, JSON `CATALOGSTORAGE "light"`):
- **Rebuild:** `rebuild_storm_rain` is a copy of the catalog write loop. If that loop changes, change the copy too,
  then re-run `tools/test_light_rebuild.py` and `tools/compare_light_catalog.py`.
- **Re-rooting:** the rain comes from **this run's RAINPATH** files (`FIO.set_rainpath`, matched by file name).
  - The grid is found by an exact coordinate match. The stored paths and indices are only a fallback.
  - The CRC32 is checked on every rebuild.
- **SCENARIOS:** stops with a light catalog. `tools/materialize_catalog.py` writes the full catalog, identical to a
  full run's (`compare_catalogs --rtol 0`).
- **Hooks:** the `FIO.*` calls in `RainyDay_Py3.py`.
- **Disk:** delete validated test outputs (`runs/<variant>`, `dumps/<variant>`, temporary crops). The free space is
  only about 50 GB.

# CLAUDE.md — RainyDay GPU/acceleration testbed

## What this is

Private research fork of RainyDay (stochastic storm transposition for extreme-rainfall
scenarios). This working copy is a testbed for accelerating the **storm-catalog
creation scan** — the part that reads every timestep of the precipitation record and
finds the max-rainfall storm placement — which dominates runtime for long records.

**Ground rules**
- Commit or push only when the user asks. The remote for this fork is `mod`
  (private `Perez-HydroSystems/RainyDay_mod`); `origin` is the older `Perez-HydroSystems/RainyDay`.
- Mark code changes with `GP YYYY-MM-DD` comments and keep the previous implementation commented.
- Always pass the parameter JSON as an **absolute path** (`writecatalog` re-opens it
  by path later in the run; see "Bugs found" below).
- Any change to catalog-scan code must re-run the verification harness (below) and
  get ALL PASS before it counts.

## Catalog engines

Selected via env var `RAINYDAY_CATALOG_ENGINE` (read near the top of
`Source/RainyDay_Py3.py`). Only active for `DOMAINTYPE: irregular`; rectangular
domains always use the original `catalogNumba` path.

| Engine | What it is | Status |
|---|---|---|
| `batchfft` (default) | Batched multithreaded CPU FFT over blocks of files | verified bit-identical |
| `batchfft_mp` | `batchfft` over a process pool (parallel reads + compute) | verified bit-identical; fastest CPU option |
| `fft_perstep` | Original loop: one `catalogFFT_irregular_domainaware` call per timestep | reference |
| `batchtf` | TensorFlow-Metal GPU screen + CPU exact-confirm | verified bit-identical; slower than batchfft on Apple GPU |
| `batchcuda` | CuPy/cuFFT (NVIDIA) GPU FFT screen + CPU exact-confirm | verified bit-identical to `batchfft` on the Windows/CUDA machine (Talequah + CONUS) |

Tuning: `RAINYDAY_BLOCKFILES` (files per FFT batch, default 16), `RAINYDAY_NWORKERS`
(mp workers, default cores/2), `RAINYDAY_GPU_DELTA` (GPU error screen, default 1e-3),
`RAINYDAY_GPU_CHUNK` (`batchcuda` timesteps per GPU FFT, default 256; lower it on
small-VRAM GPUs — CONUS needs ~64-128 on an 8 GB card, see CUDA section below),
`RAINYDAY_SCAN_PREFETCH` (`batchcuda` reader-thread lookahead, default 2; a background
thread reads the next block(s) while the GPU works — overlaps I/O with compute, ~1.5×
on the CUDA scan, bit-identical output; default 0 for the CPU engines).

**JSON `COMPUTE` block (preferred over env vars for reproducibility).** The parameter
file may carry an optional `"COMPUTE"` block; every field is optional and `"auto"`
(or a missing field/block) triggers machine auto-detection:
```json
"COMPUTE": {"engine": "auto", "nworkers": "auto", "blockfiles": "auto",
            "gpu_chunk": "auto", "gpu_delta": 0.001, "prefetch": "auto"}
```
`RainyDay.resolve_compute_config` (called before the scan dispatch, irregular domains
only) resolves each field with precedence **env var > JSON value > auto**, then injects
the result into the `RAINYDAY_*` env vars the engine code already reads (so nothing
downstream changed), and prints the resolved config. Auto-detection: engine by
availability (`fork`+≥4 cores → `batchfft_mp`, else CUDA → `batchcuda`, else
`batchfft`), with graceful fallback + warning if a requested engine is unavailable
(e.g. `batchfft_mp` on Windows → `batchcuda`); `nworkers`/`blockfiles` sized from CPU
count + free RAM (workers capped at 8 — single-disk saturation); `gpu_chunk` sized from
free VRAM. `gpu_delta` is a tolerance (not machine-auto). Keep shared JSONs portable by
leaving `"auto"`; pin values only for machine-specific reproducibility. JSON has no
comments, so the block may carry `_`-prefixed note keys (e.g. `"_engine_options": "..."`)
to document the choices inline — the resolver reads only the known keys via `.get()` and
ignores anything starting with `_` (see `Examples/Talequah/Talequah.json`).

Key functions, all in `Source/RainyDay_utilities_Py3/RainyDay_functions.py`:
- `catalogFFT_precompute` — one-time: kernel FFT + domain-validity map. The
  "full storm footprint inside domain" rule (`footprint=ones`, `>= sum-0.5`) is
  **identical** to `catalogFFT_irregular_domainaware`; proven placement-identical on
  an adversarial concave/holed domain by `tools/test_domainaware_equivalence.py`.
- `catalogFFT_batch` — batched rfft2/irfft2 + masked argmax for a (T,H,W) block.
- `read_file_block` — reads files onto a contiguous hourly grid (gaps zero-filled,
  matching the original time-window semantics); returns per-step source-file index.
- `build_windowed_block` — rolling 72-h (nwindow) sums via float64 cumsum.
- `catalog_scan_files` — block loop (context carry-over, gap handling); used by the
  inline engine, the mp workers, and the exact-confirm recompute. For `batchcuda` it
  reads blocks through `_iter_read_blocks` (see below) to overlap I/O with GPU compute.
- `_iter_read_blocks` — yields file blocks with an optional reader-thread lookahead
  (`RAINYDAY_SCAN_PREFETCH`): a daemon thread reads ahead while the caller windows +
  runs the GPU. Block order/contents are identical to a sequential read, so results
  are unchanged; only read latency is hidden. In-process (no IPC) because shipping
  windowed blocks between processes is infeasible (~tens of GB/yr).
- `catalog_scan_worker` — picklable mp wrapper.
- `catalogTF_batch` — TF conv2d screen (values only, in effect).
- `catalogCUDA_batch` — CuPy/cuFFT port of `catalogFFT_batch` (keeps the FFT, no
  direct-conv penalty); constants cached on `pre['_cuda']`, complex64 pipeline,
  masked argmax on device. Forces `CUPY_ACCELERATORS=''` before importing cupy
  (see CUDA section). Same GPU-screen role as `catalogTF_batch`.
- `catalog_select_confirmed` — GPU-screen + CPU-exact-confirm selection (see below).

## Correctness model (why "bit-identical" is achievable)

- Per-storm `rain` arrays are re-read from the raw files for the *selected* dates, so
  if storm selection matches (dates, x/y, cattime), rain fields are bitwise equal.
  Only `basinrainfall` (=catmax) passes through the FFT; compared at rtol 1e-5
  (measured diff has been exactly 0 for all CPU engines).
- Adjacent 72-h windows of one storm are often **exact float ties** (marginal hours
  add zero rain → identical window arrays). Legacy rule `rainmax>=catmax[checkind]`
  keeps the *last* tied hour. Any backend with different float rounding (Metal, CUDA)
  breaks these ties differently — same precision, different summation order; neither
  is "more accurate" (the windows differ by ~1e-4 mm). You cannot reproduce CPU-FFT
  bit patterns on another backend.
- Fix: `catalog_select_confirmed`. GPU values are only a screen with error bound δ.
  Invariant: catalog slots only ever hold values from accepted timesteps → force every
  accepted timestep to be CPU-exact → all decisions are exact-vs-exact or
  "GPU value below running min by more than δ" (safe either way). Iterate: selection
  pass → mark non-exact steps with `v >= tempmin − δ·scale` → recompute marked steps'
  files with the CPU engine (grouped into consecutive-file runs, ctx-seeded) → restart;
  a pass with no new marks is exact by induction. Measured Metal error ~8e-7 vs δ=1e-3.
  On short records most files get recomputed (low catalog min) — artifact that
  shrinks on multi-year records.

## Verification harness (`RainyDay/tools/`)

```bash
bash tools/run_talequah.sh <conda_env> <engine>          # runs Examples/Talequah
python tools/compare_catalogs.py tools/reference_catalog_fftperstep \
    Examples/Talequah/Results/Talequah_3days/StormCatalog   # must print ALL PASS
```
- `compare_catalogs.py`: exact match on dates/x/y/cattime/rain/masks; rtol on
  basinrainfall only. Exit 0/1.
- Reference archives: `tools/reference_catalog_fftperstep/` (Talequah),
  `tools/conus_catalogs/reference_fftperstep/` (CONUS July-2002).
- `tools/test_domainaware_equivalence.py`: unit-level per-step vs batched equivalence
  on a nasty synthetic domain (run after touching mask/validity logic).
- CONUS case: `Examples/CONUS_Benchmark/CONUS_bench.json` (July 2002, CONUS-mainland
  shapefile from `tools/make_conus_shapefile.py`, Talequah watershed 28×33 mask).
- Conda envs: `RainyDay_Env` (benchmarks, py3.8), `RainyDay_Env_GPU` (tensorflow-metal).
  Data: `/Users/gjperezm/Databases/AORC_RainyDay/data/AORC.*.preciptemp.nc`
  (12,765 daily files, hourly, CONUS at 1/30°: 885×1770; paths differ on other machines).

## Benchmarks (Apple M3 Max, 16 cores; scan timer = the part engines accelerate)

Talequah (180×180 domain, 92 files): fft_perstep 0.30 min → batchfft 0.12 (2.5×) →
batchfft_mp×8 0.03 (10×) → batchtf 0.30 (1×, exact-confirm dominated).

CONUS domain (747×1734, 31 files): fft_perstep 2.16 min → batchfft 0.37 (5.8×) →
batchfft_mp×8 0.13 (16.6×) → batchtf 0.80 (2.7×).

Takeaways: batching pays more as domain grows (compute overtakes I/O); Metal GPU
crossed from useless (small domain) to 2.7× (CONUS) despite doing direct conv
(~30× the arithmetic of FFT — TF-Metal has no good FFT path). Full details and
notes: `tools/benchmark_results.md`.

### Windows/CUDA machine (RTX 4060 Laptop 8 GB, 22-core CPU, 34 GB RAM, single disk)

Full-year 2002 CONUS (729×1732 domain, 365 files, NSTORMS=10; `CONUS_bench_win_allmonths.json`,
*before* the prefetch optimization): batchfft 7.92 min, batchfft_mp×8 3.15 (2.5×),
batchcuda 10.05 (slower — serial reads + 44% exact-confirm recompute).

**Two optimizations then added ("levers"):**
- *Lever A — reader-thread prefetch* (`RAINYDAY_SCAN_PREFETCH`, default 2 for batchcuda;
  `_iter_read_blocks`): overlaps file I/O with GPU compute. Validated **1.53×** on the
  CUDA scan with **bit-identical output** (`tools/bench_prefetch.py`). Threads can't
  parallelize the HDF5 reads *themselves* (measured flat, 1→8 threads), but one reader
  thread overlaps I/O with the GPU. Multi-*process* readers are infeasible for a single
  GPU: shipping windowed blocks over IPC is ~tens of GB/yr, and shared-memory readers
  would just reimplement `batchfft_mp` (the GPU is only ~12% of the work).
- *Lever B — longer record*: synthetic 2-year 2002+2003 set (`tools/make_synthetic_2003.py`
  → `C:/Test_RainyDay/AORC_2yr`, 730 files; 2003 = 2002 with `time` +365 d, identical rain).
  Doubling the record raised the catalog min and cut the exact-confirm recompute from
  **44% (161/365) to 25% (185/730)**.

2-year 2002+2003 CONUS (730 files, 17520 timesteps; `CONUS_bench_win_2yr.json`), scan time:

| engine / config | min |
|---|---|
| batchfft single (bf16) | 14.55 |
| batchcuda prefetch (bf16, chunk 64) | 12.39 ← now beats single batchfft |
| batchfft_mp×4 (bf2) | 5.79 |
| **batchfft_mp×8 (bf2)** | **4.30 ← fastest** |
| batchfft_mp×12 (bf2) | 4.74 |
| batchfft_mp×8 (bf4) | 7.15 |
| batchfft_mp×16 (bf1) | 8.98 |

All bit-identical (batchcuda ≡ batchfft ALL PASS, basinrainfall diff 0.0; Talequah too).

**More cores? (the ">8 cores" question).** At fixed `RAINYDAY_BLOCKFILES=2`, mp scales to
~8 workers then plateaus (×4 5.79 → ×8 4.30 → ×12 4.74): the single disk saturates around
8 concurrent readers. **Block size matters more than extra cores** — ×8 at bf2 (4.30) beat
×8 at bf4 (7.15) because the smaller float64 windowed block is cache/RAM-friendlier; and
>8 workers forces tiny blocks anyway (RAM: 16 × a bf16 block ≈ 74 GB → OOM). Sweet spot:
**mp×8 with a small blockfiles (~2–4)**.

**Verdict:** `batchcuda` (prefetch) is verified bit-identical and, with the two levers,
faster than single-thread `batchfft` (12.39 vs 14.55 min, 2-yr) — but `batchfft_mp×8`
(4.30) is still ~2.9× faster. The scan is ~88% read+window (CPU/IO), only ~12% GPU FFT,
so parallel-CPU (mp) wins on this box. batchcuda would lead where compute is the
bottleneck: far larger domains, faster/parallel storage, or long records (exact-confirm
amortizes). **GPU-windowing** (moving the 44% window onto the GPU) was evaluated and
*declined*: it would make batchcuda read-bound (~2.7 min/yr) — competitive with mp×8 but
not dominant — at real OOM risk on the 8 GB card (float64 cumsum), not worth it while mp×8 wins.

Benchmark artifacts (not the shipped macOS references), all in `tools/`: `bench_mp_scan.py`
(mp over a spawn pool — has its own `__main__` guard; `batchfft_mp` can't run through
`RainyDay_Py3.py` on Windows, no `fork`), `bench_prefetch.py` (prefetch speedup +
identical-output check), `profile_scan.py` (read/window/compute split), `make_synthetic_2003.py`.
The env-gated dump hook `RAINYDAY_DUMP_SCANINPUTS` in `RainyDay_Py3.py` feeds them.

## MULTIPLEDURATIONS — frequency analysis for many durations in one run (GP 2026-10-21)

JSON: `"DURATION": 72, "DURATIONCORRECTION": "false", "MULTIPLEDURATIONS": [1,3,6,12,24,48,72]` (+ optional
`"MULTIDURATIONENGINE": "fast"|"exact"`, default `fast`). One run builds (CREATECATALOG) or reads (CATALOGPATH) the
DURATION catalog and writes `MAINPATH/<SCENARIONAME>_<d>/<SCENARIONAME>_<d>_FreqAnalysis.csv/.png` for every d; the
catalog and diagnostics live only in `<SCENARIONAME>_<DURATION>/` (that is `fullpath`; no bare SCENARIONAME folder).

**Why it can be exact.** The only seed is `np.random.seed(0)`; the draws (Poisson counts, storm indices, the location
per storm) depend on NSTORMS/NYEARS/NREALIZATIONS/valid anchors, never on DURATION or DURATIONCORRECTION, and no random
call happens before resampling in either mode. Once the catalog exists, a d-hour analysis differs from the parent only
in `passrain` (`rolling_sum` of d steps instead of the `nansum`) and `SSTalt(durcheck=True)`. So every d < DURATION
reuses the parent's draws and reproduces the separate run (CATALOGPATH = this catalog, DURATION d,
DURATIONCORRECTION true). With CREATECATALOG the run checks that the masks stored in the new catalog equal the ones in
memory and that the CATALOGPATH year/month file filter would keep every storm (else it exits).

**Engines.** `exact`: the original `SSTalt` per duration. It runs in numba **object mode** (`nopython_signatures`
empty: it contains `sys.exit`/`erfinv`), ~10 µs Python dispatch per window, cost ∝ draws × Σ(73−d) windows.
`fast`: `SSTalt_multidur` (nopython) — same arithmetic (zero rule: all box cells < 0.5 in every window → 0; strict
max from 0. over `numba_multimask_calc`), all shorter durations in one compiled pass over the stacked
`rolling_sum` windows. Bit-identical to `exact` (unit tests on dense and sparse fields + all runs below).

**Code.** M: parse at the MAINPATH/SCENARIONAME block; checks before `if FreqAnalysis:`; per-storm sub-duration block at
the end of the resampling loop (`whichrain_md`); `-9999` no-storm mask; outputs right after the parent's. The parent's
CSV/PNG block moved verbatim into F `write_frequency_outputs` (old code commented); F `rank_frequency` replicates the
watershed/box ranking (pds/ams + RETURNLEVELS). Not supported (exits): SCENARIOS, ENHANCEDSST, ROTATIONANGLE,
INTENSDISTR, point/pointlist, DURATIONCORRECTION true, d > DURATION, d not a multiple of the time step.
Hook: `RAINYDAY_DUMP_WHICHRAIN=<dir>` saves the raw transposed depths (`<name>_whichrain.npy`) of every duration.

**Verification (2026-10-21, macOS, RainyDay_Env, numba 0.55.2).**
- `tools/test_multidurations.sh` (Talequah, CREATECATALOG + CATALOGPATH, exact & fast vs separate 1/6/72 h): ALL PASS.
- Production-size cases (NSTORMS 400, NYEARS 500, 100 realizations, CATALOGPATH on the `_aorcfix_test` catalogs):
  every FreqAnalysis.csv of the multi runs (fast and exact) byte-identical to the separate runs made **with the old
  code** — Iowa pixel 7/7 durations, Iowa circle 16000 km² 7/7, Texas circle 8000 km² 72 h + 1 h; raw depths of
  `fast` and `exact` bitwise equal for all 3 basins × 7 durations, and equal to new-code separate runs (pixel
  1/24/72 h, 16000 km² 3/72 h, 8000 km² 1/72 h).
- CREATECATALOG multi run (Texas 8000 km²): rebuilt catalog ALL PASS vs the existing one (`compare_catalogs.py`,
  400 storms), all 7 CSVs identical to the CATALOGPATH multi run, raw depths bitwise equal to separate runs.
- Talequah harness ALL PASS; `test_windowed_block.py` and `test_domainaware_equivalence.py` PASS (scan untouched).
- Wall time, all 7 durations from an existing catalog, 14 jobs sharing 16 cores (`fast` / `exact`): Iowa pixel
  382 / 2,988 s (the separate 1 h run alone took 686 s; separate d < 72 runs of a shape ~3,200 s in production);
  Texas 8000 km² 1,031 / 3,906 s; Iowa 16000 km² 2,058 / 4,567 s (separate d < 72 runs ~7,000 s). The `fast` gain
  shrinks for large basins because each window then sums a large mask box (~1,500 cells) in compiled code.

**Single duration-corrected runs (GP 2026-10-21).** Without MULTIPLEDURATIONS, a run with DURATIONCORRECTION true
(watershed/box, no rescaling/rotation) now also calls `SSTalt_multidur` (one duration) when `MULTIDURATIONENGINE` is
`fast` (default); `exact` keeps the original `SSTalt`. `whichstep` (filled but used by no output) stays 0 in the fast
path. Verified on the `_aorcfix_test` catalogs (CATALOGPATH, production settings), fast and exact run side by side:
raw depths bitwise equal and CSVs byte-identical to the old-code runs. Wall time old/fast: Iowa pixel 1 h 595 / 86 s
(6.9×), pixel 24 h 432 / 73 s (5.9×), Iowa 16000 km² 1 h 879 / 278 s (3.2×). `tools/test_multidurations.sh` ALL PASS
(its separate 1/6 h runs now use the fast path and still match the exact multi run bitwise).

## Bugs found & fixed along the way

- `RainyDay_Py3.py` storm-writing loop: `except OSError` silently swallowed
  non-permission errors → storms silently missing from the catalog (triggered by a
  relative parameterfile path). Non-13 errors now print + re-raise.
- macOS multiprocessing: this script has **no `__main__` guard**, so `spawn` and
  `forkserver` re-execute the whole script in workers (infinite recursion / crash).
  `batchfft_mp` must use the `fork` start method (workers touch only
  numpy/scipy/netCDF4 — fork-safe here). Same constraint applies on Linux only if
  the start method is changed from its default (`fork`).
- Removed stray `from unittest import result` (IDE auto-import).
- `build_windowed_block` (batched engines, 2026-10-20): with a one-step window (catalog
  DURATION equal to the data time step, e.g. a native 1 h catalog of hourly AORC)
  `nwindow - 1 == 0` and the returned context `full[-(nwindow-1):]` was `full[-0:]`, i.e.
  the whole array. The context then grew with every block (13–38 GB, a scan that crawled
  and stalled) and later windows were read from the start of the record. Now sliced from
  the end by length; identical for `nwindow > 1`. Talequah 1 h and 3 day: `batchfft` and
  `batchfft_mp` ALL PASS vs `fft_perstep` (1 h FAILED before the fix). Regression test:
  `tools/test_windowed_block.py`.

## CUDA engine `batchcuda` — DONE (built + verified on the Windows/CUDA machine)

`catalogCUDA_batch` (CuPy/cuFFT port of `catalogFFT_batch`) is implemented and wired
into `catalog_scan_files` and both `RainyDay_Py3.py` batch branches, routed through
`catalog_select_confirmed` exactly like `batchtf`. Verified **bit-identical to
`batchfft`** on this machine: Talequah (ALL PASS, basinrainfall diff 0.0) and CONUS
July-2002 (ALL PASS, basinrainfall diff 0.0). Converge lines: Talequah 2 passes / 82
files / max GPU err 3.95e-07; CONUS 2 passes / 31 files / 2.70e-07 — both ≪ δ=1e-3.

**Two Windows-specific gotchas (both handled in-code / documented, not open work):**
- **CuPy reductions (`argmax`/`max`) need NVRTC, not nvcc.** CuPy 12.x's default CUB
  accelerator compiles reductions with `nvcc`, which needs an MSVC host compiler
  (`cl.exe`) that isn't installed here → `NVCCException`. `catalogCUDA_batch` does
  `os.environ.setdefault('CUPY_ACCELERATORS','')` *before* its local `import cupy`
  (cupy reads that env var at import), forcing the NVRTC path. cuFFT and elementwise
  ops never hit this. If you ever run cupy elsewhere in the process first, set
  `CUPY_ACCELERATORS=` in the environment too.
- **VRAM.** This box is an RTX 4060 Laptop = **8 GB**, not the 16-24 GB the earlier
  notes assumed. CONUS spectra are complex64 ~33 MB/timestep, so the default
  chunksize 256 (~8.5 GB) OOMs. Use `RAINYDAY_GPU_CHUNK=64` (~2 GB) for CONUS;
  128 (~4 GB) also fits. Talequah is tiny — 256 is fine.

**Benchmark reality — batchcuda beats single-thread batchfft but not batchfft_mp.**
After the two optimizations below (reader-thread prefetch + longer record), `batchcuda`
is faster than single-process `batchfft` on multi-year data but still trails
`batchfft_mp`. On the synthetic 2-year 2002+2003 CONUS case (730 files): batchfft 14.55
min, **batchcuda (prefetch) 12.39**, best **batchfft_mp×8 4.30** (bf2). The GPU can't win
here because the scan is only **~12% GPU compute** (profile: read 44% / window 44% / GPU
FFT 12%, via `tools/profile_scan.py`) — `batchfft_mp` parallelizes the other 88% across
processes, which a single GPU fed by one machine's serial reads cannot match. batchcuda
would win on compute-heavy regimes instead: much larger domains, faster/parallel storage,
or very long records (exact-confirm amortizes). Full table in the Benchmarks section above.

### Running the harness on the Windows/CUDA machine
- Python: `C:\Users\gpere11\.conda\envs\RainyDay_Env_GPU\python.exe` (env
  `RainyDay_Env_GPU`; CuPy 12.3.0, CUDA 12.9). `tools/run_talequah.sh` hardcodes a
  macOS python path — invoke python directly on Windows instead.
- AORC data here: `C:/Test_RainyDay/AORC_2002/AORC.*.preciptemp.nc` — real data is
  **2002 only** (365 daily files). Talequah (Jul-Sep) = 92 files, CONUS (Jul) = 31 files.
  `C:/Test_RainyDay/AORC_2yr/` = **730 files** = 2002 (hardlinked) + a synthetic 2003
  (`tools/make_synthetic_2003.py`: 2002 copies with `time` +365 d) for multi-year tests.
- Windows param files (absolute paths, forward slashes): `Examples/Talequah/Talequah_win.json`,
  `Examples/CONUS_Benchmark/CONUS_bench_win.json` (July), `_allmonths.json` (full-year 2002),
  `_2yr.json` (2002+2003, `INCLUDEYEARS:"2002,2003"`). Run e.g.:
  `RAINYDAY_CATALOG_ENGINE=batchcuda CUPY_ACCELERATORS= RAINYDAY_GPU_CHUNK=64 \
   python Source/RainyDay_Py3.py <abs path>/CONUS_bench_win_2yr.json`  (prefetch is
  auto-on for batchcuda; set `RAINYDAY_SCAN_PREFETCH=0` to disable).
- **Do not write large (CONUS) catalogs into the OneDrive-synced tree** — OneDrive
  locks the big storm `.nc` files and the script's `shutil.rmtree` of the old
  catalog dies with `WinError 32`. `CONUS_bench_win.json` points `MAINPATH` at
  `C:/Test_RainyDay/Results` for this reason. Talequah's small files are fine in OneDrive.
- **The macOS reference catalogs are not portable to Windows.** `batchfft` on
  Windows already diverges from `tools/reference_catalog_fftperstep` (cattime encoding
  on every storm; 3 storms pick a different 72-h window — a platform float tie-break,
  same diffs from *both* engines). So the correctness bar here is **batchcuda ≡
  batchfft on the same machine**, verified above — not the shipped macOS reference.

### Optional next rungs
- **Faster read floor (biggest lever left).** The scan is ~44% reads + 44% window; both
  are CPU/IO. A one-time Zarr conversion (consolidated store, time-block chunks, Zstd/LZ4,
  rain var only) + a Zarr-aware `read_file_block` would cut the read floor ~5× and help
  *every* engine, including lifting the single-disk ceiling that caps `batchfft_mp` at
  ~8 workers. Best ROI for repeated large runs.
- **Real multi-year data.** Lever B was tested with a *synthetic* 2003 (identical rain,
  shifted dates), which raises the catalog min and cuts exact-confirm recompute (44%→25%)
  but is not physically new weather. Stage real additional AORC years to confirm the
  recompute fraction keeps dropping (→ batchcuda's exact-confirm tax shrinks further).
- **GPU-windowing (evaluated, not built).** Moving `build_windowed_block` onto the GPU
  would remove the 44% CPU window and make batchcuda read-bound (~2.7 min/yr), competitive
  with `batchfft_mp×8` while freeing CPU cores — but the float64 cumsum is an OOM risk on
  the 8 GB card (needs `blockfiles≤8` or chunked/float32-screen windowing) and it still
  wouldn't dominate mp. Revisit if a larger-VRAM GPU or a compute-heavier domain appears.
- **Wire process-parallelism into `RainyDay_Py3.py`** (would enable in-product `batchfft_mp`
  on Windows): needs a `__main__` guard so `spawn` workers don't re-run the 3000-line
  script. Deferred — the body has mixed tabs/spaces (line 1476) that make a mechanical
  tail-indent risky; `tools/bench_mp_scan.py` sidesteps it with its own guard for now.
