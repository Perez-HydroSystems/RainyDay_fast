# RainyDay_fast

RainyDay_fast is a modified version of [RainyDay](https://github.com/HydroclimateExtremesGroup/RainyDay), the
stochastic storm transposition (SST) software of Daniel B. Wright and the Hydroclimate Extremes Group (University of
Wisconsin–Madison).

- **Starting point:** the original RainyDay at commit
  [`b14a011`](https://github.com/HydroclimateExtremesGroup/RainyDay/commit/b14a011) (17 April 2025).
- **Modifications:** by Gabriel Perez (Perez-HydroSystems). This is an independent research version, not the official
  RainyDay release.

**The complete description is in [`README_DETAILED.md`](README_DETAILED.md).**

## What it adds

**Three new capabilities**

1. **Many durations in one run** (`MULTIPLEDURATIONS`). One run gives the frequency analysis for every duration you
   list (e.g. 1 to 72 hours), instead of one run per duration.
2. **Record (sampling) uncertainty** (`RECORDCI`). Confidence intervals that account for the short rainfall record,
   which RainyDay's band of realizations does not include.
3. **Exact frequency engine** (`FREQENGINE`). The frequency curve that RainyDay's realizations converge to, computed
   directly, without Monte Carlo noise.

**Four ways to run faster, with identical results**

4. **Table transposition engine** (`MULTIDURATIONENGINE: "table"`). Transposition and frequency analysis 16–66×
   faster.
5. **Light storm catalogs** (`CATALOGSTORAGE: "light"`). The catalog stores the storm information but not the rain
   fields, which are read again from the input files when needed: about 2 MB instead of 2–4 GB per basin.
6. **Cropped, compressed input** (`tools/make_domain_crop.py`). The national rainfall files are cut once to the
   domain and stored with lossless compression, so each run reads small files.
7. **Parallel catalog search** (`COMPUTE`). The search for storms uses all CPU cores (or a GPU) instead of one core.

Together, options 4–7 made a full run for a 16,000 km² basin 21–31× faster, with every output identical.

**Are the results the same as the original RainyDay?** The faster options (4–7) give identical results, and options
1–3 only add outputs. Some corrections made along the way do change results, mainly the rule that a transposed basin
must lie entirely inside the domain. They are listed in
[`README_DETAILED.md`](README_DETAILED.md#changes-that-affect-results).

## Quick start

```bash
conda env create -n RainyDay_Env -f RainyDay.yml     # the same environment as RainyDay
conda activate RainyDay_Env
python Source/RainyDay_Py3.py /absolute/path/to/RUN.json
```

To use the new options, add them to the run's JSON file, for example:

```json
"MULTIPLEDURATIONS": [1, 3, 6, 12, 24, 48, 72],
"MULTIDURATIONENGINE": "table",
"CATALOGSTORAGE": "light",
"FREQENGINE": "both",
"RECORDCI": "bootstrap"
```

## Documentation

- [`README_DETAILED.md`](README_DETAILED.md): every modification, with settings, outputs, speed and limits.
- [`UserGuide/RainyDay_Tutorial.ipynb`](UserGuide/RainyDay_Tutorial.ipynb): the original RainyDay User's Guide.
- [`UserGuide/RainyDay_fast_tutorial.ipynb`](UserGuide/RainyDay_fast_tutorial.ipynb): the RainyDay_fast User's Guide,
  with worked examples.
- [`README_CHANGES.md`](README_CHANGES.md): the change log, with every change, its reason and its effect on results.
- [`tools/aorc_fix/AORC_FIX_INSTRUCTIONS.md`](tools/aorc_fix/AORC_FIX_INSTRUCTIONS.md): how to correct four known
  defects in the RainyDay-format AORC rainfall files.

## Citation and license

Please cite the RainyDay papers (Wright et al., 2017, *Environmental Modelling & Software* 90, 34–54; Wright et al.,
2020, *Journal of Hydrology* 585, 124816) and this repository. Full references are in
[`README_DETAILED.md`](README_DETAILED.md#how-to-cite).

MIT license, as the original RainyDay (see [`LICENSE`](LICENSE)).
