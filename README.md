# RainyDay_fast (Perez-HydroSystems)

> **RainyDay_fast (GP 2026-10-22)** is a performance playground cloned from RainyDay_mod (commit 9137292).
>
> **Goal:** the same results **bit for bit**, much faster, for large campaigns (thousands of runs). It targets the parts
> of a run that repeat the same work:
> 1. **Cropped input:** one cropped, uncompressed copy of the AORC files per transposition domain. This avoids
>    decompressing the CONUS files in every run.
> 2. **Light storm catalog:** storm metadata only; the rain is read back from the cropped files. No ~4 GB catalog per basin.
> 3. **Table transposition engine** (`MULTIDURATIONENGINE "table"`): the basin sums are pre-computed per storm with
>    FFTs, and RainyDay's exact sums are evaluated only for the candidate windows.
>
> **Status:** in development. See `CLAUDE.md` for the rules and the validation matrix. The RainyDay_mod description
> follows.

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

