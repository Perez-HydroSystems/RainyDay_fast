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

- `--deflate 1` (default) is lossless and about 25× smaller than `--deflate 0` on AORC, and still about 10× faster to
  read than the CONUS files. 2002–2023 sizes: Texas 0.67 GB (16.9 GB uncompressed), Iowa 0.76 GB (29.4 GB
  uncompressed). Talequah, 2002 Jul–Sep: 7.8 MB.
- Keep crops and catalogs out of synced folders such as OneDrive.
- A full catalog costs about 2–4 GB per basin and a light one about 2 MB, so prefer light catalogs for campaigns.

Checks:
- `tools/compare_light_catalog.py <full StormCatalog> <light .nc>`;
- `tools/test_light_rebuild.py <crop dir> <domain.shp>`;
- `tools/compare_catalogs.py <ref> <test> --rtol 0`;
- `tools/compare_dumps.py` (see `CLAUDE.md`).

## RainyDay_mod

> **About this fork.** This is a modified clone of the original RainyDay repository
> (Daniel B. Wright and the Hydroclimate Extremes Group, https://github.com/HydroclimateExtremesGroup/RainyDay).
> It contains many changes by Gabriel Perez (GP) that are **not** in the original code, including:
> - batched multithreaded CPU / process-pool / GPU (TensorFlow-Metal, CuPy-CUDA) engines for the storm-catalog scan,
>   reader-thread prefetching, and an optional JSON `COMPUTE` block (see `CLAUDE.md`);
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
- Requirements: `FREQANALYSIS` true, `DURATIONCORRECTION` false, every `d <= DURATION` and a multiple of the time step.
  Not supported together with `SCENARIOS`, `ENHANCEDSST`, `ROTATIONANGLE`, `INTENSDISTR` or point/pointlist areas.
- Regression test: `bash tools/test_multidurations.sh` (Talequah example).

---

Welcome to RainyDay. RainyDay is a framework for generating large numbers of realistic extreme rainfall scenarios based on relatively short records of remotely-sensed precipitation fields.  It is founded on a statistical resampling concept known as stochastic storm transposition (SST).  These rainfall scenarios can then be used to examine the extreme rainfall statistics for a user-specified region, or to drive a hazard model (usually a hydrologic model, but the method produces output that would also be useful for landslide models). RainyDay is well suited to flood modeling in small and medium-sized watersheds.  The framework is made to be simple yet powerful and easily modified to meet specific needs, taking advantage of Python’s simple syntax and well-developed libraries.  It is still a work in progress.  Therefore, the contents of the guide may be out-of-date.

Development of RainyDay has been supported by NASA, the National Science Foundation, The Wisconsin Alumni Research Foundation, and the Research and Development Office at the U.S. Bureau of Reclamation.

The latest version of RainyDay is distributed under the MIT open source license: https://opensource.org/licenses/MIT

