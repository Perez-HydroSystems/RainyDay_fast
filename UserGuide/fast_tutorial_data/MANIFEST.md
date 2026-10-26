# fast_tutorial_data: MANIFEST

Real data for the two case studies of the RainyDay_fast User's Guide (`RainyDay_fast_tutorial.ipynb`). The notebook only reads these files; no RainyDay run happens inside it. File and folder names are RainyDay's own (`<SCENARIONAME>_<d>/<SCENARIONAME>_<d>_<kind>.csv`), so the folders of each case are a trimmed copy of the run's MAINPATH. The CSVs are byte copies of the run outputs (same md5); only the JSONs were edited (paths replaced by placeholders).

- **Made:** 2026-10-26, on the Mac (Apple M3 Max, 16 cores).
- **Code:** RainyDay_fast branch `fast/exact` at **7ab20cf** (FREQENGINE + RECORDCI + the parsing and rectangular-domain fixes), run from a detached worktree. The files were generated at 74b3c03 and **re-verified byte-identical at 7ab20cf**: all three runs and all three compare_runs calls were repeated at 7ab20cf into new folders, and every output file (CSV and PNG) and every compare table has the same md5.
- **Rain data:** AORC `data_aorcfix2`, Iowa crop `SST_domain_rect_buffer_0p5deg` (tools/make_domain_crop.py, deflate 1), via the light catalogs.
- **Catalogs (reused, no scan; `CREATECATALOG` false, `CATALOGPATH`):**
  - Case 1: the paper's Iowa 72 h pixel catalog `C72_N400_all` (23_catalog_sensitivity; 400 storms, 2002-2023). Its storms, windows, locations, masks and CRCs are identical to the validation matrix's `Pixel_r1_A1km2_rot0` catalog (only the storm file names differ).
  - Case 2: each run reuses its own 72 h catalog (`Basin_raw_Iowa_HUC_07100006_72` and `Basin_rg_Iowa_HUC_07100006_72` in RainyDay_diag_huc), exactly as the earlier FREQENGINE verification runs (`hucexact`, `val`). The two catalogs hold different storm sets (storms per record year: 12-26 raw, 13-24 rg).
- **Case 1 pixel location:** the run's WATERSHEDSHP is the production 1-cell pixel `Pixel_Area_1km2_Rotation_0deg.shp` (centre lon -93.3515, lat 42.3158, the Iowa domain centroid), the same pixel as the paper's Atlas 14 comparison and the validation matrix. It is NOT placed at the HUC-12 070600040101 centroid (lon -92.4132, lat 43.3637). With uniform transposition a 1-cell watershed is moved to all 35,416 domain cells, so its curve does not depend on where it sits; the paper compares it with Atlas 14 at the HUC-12 centroid, and so does the tutorial.

Total: 95 files, 3,439,963 bytes (3.44 MB).

## Settings (all three runs)

`DURATION` 72 with `MULTIPLEDURATIONS` [1, 3, 6, 12, 24, 48, 72]; `NSTORMS` 400; `NYEARS` 1000; `NREALIZATIONS` 100; `UNCERTAINTY` "all_realizations"; `CALCTYPE` "PDS"; `TRANSPOSITION` "uniform"; `RESAMPLING` "poisson"; `INCLUDEYEARS` "2002-2023"; `DURATIONCORRECTION` "false"; `MULTIDURATIONENGINE` "table"; `CATALOGSTORAGE` "light"; `FREQENGINE` "both"; `RECORDCI` "bootstrap", `RECORDNBOOT` 1000, `RECORDSEED` 20261026, `RECORDLEVEL` 90, `RECORDSAVE` "true"; `RETURNLEVELS` [2, 3, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 500, 1000].

RainyDay snaps each `RETURNLEVELS` value to the nearest NYEARS/k (k = 1 ... NYEARS), so the `returnperiod` column holds 2, 3.003, 5, 10, 14.925, 20, 25, 50, 76.923, 100, 142.857, 200, 250, 500, 1000. A requested 750 was left out: it snaps to the 1000-yr row (k = 1) and would duplicate it.

## Commands

Paths: `{path}` = the RainyDay_fast folder (or the folder of the JSON); `{MAINPATH}` = the run's MAINPATH, which must exist before the run (RainyDay creates only the `<SCENARIONAME>_<d>` folders inside it).

```
RAINYDAY_TIMERS=1 python {path}/Source/RainyDay_Py3.py {path}/Pixel_r1_A1km2_rot0.json
RAINYDAY_TIMERS=1 python {path}/Source/RainyDay_Py3.py {path}/Basin_raw_Iowa_HUC_07100006.json
RAINYDAY_TIMERS=1 python {path}/Source/RainyDay_Py3.py {path}/Basin_rg_Iowa_HUC_07100006.json

python {path}/tools/compare_runs.py {MAINPATH}/Basin_raw_Iowa_HUC_07100006 {MAINPATH}/Basin_rg_Iowa_HUC_07100006 \
       --stat median --engine montecarlo --out compare_median_montecarlo.csv
python {path}/tools/compare_runs.py {MAINPATH}/Basin_raw_Iowa_HUC_07100006 {MAINPATH}/Basin_rg_Iowa_HUC_07100006 \
       --engine exact --stat mean --out compare_mean_exact.csv
python {path}/tools/compare_runs.py {MAINPATH}/Basin_raw_Iowa_HUC_07100006 {MAINPATH}/Basin_rg_Iowa_HUC_07100006 \
       --stat level --out compare_level.csv
```

compare_runs defaults: `--margin 5`, `--level 90`, `--seed 42`, `--nboot-mc 10000`, no `--bh`. `RAINYDAY_TIMERS` only prints a timing table; it does not change results.

## Run times

At 7ab20cf (re-check, 14:41, quiet machine with load average 10-13; the three runs in parallel):

| Run | RainyDay "Elapsed time" | Wall (s) | Start (1-min load) | End (1-min load) |
|---|---|---|---|---|
| `Pixel_r1_A1km2_rot0` | 1.50 min | 91.80 | 14:41:03 (12.01) | 14:42:35 (12.93) |
| `Basin_raw_Iowa_HUC_07100006` | 2.97 min | 179.89 | 14:41:03 (12.01) | 14:44:03 (10.12) |
| `Basin_rg_Iowa_HUC_07100006` | 2.90 min | 175.81 | 14:41:03 (12.01) | 14:43:59 (10.48) |

At 74b3c03 (first run, 14:23; the three runs in parallel; from 14:24 eight other RainyDay jobs shared the machine, load average 150-185, so these times are upper bounds):

| Run | RainyDay "Elapsed time" | Wall (s) | Start (1-min load) | End (1-min load) |
|---|---|---|---|---|
| `Pixel_r1_A1km2_rot0` | 1.73 min | 106.75 | 14:23:33 (165.47) | 14:25:19 (188.06) |
| `Basin_raw_Iowa_HUC_07100006` | 3.57 min | 216.63 | 14:23:33 (165.47) | 14:27:09 (134.49) |
| `Basin_rg_Iowa_HUC_07100006` | 3.49 min | 212.35 | 14:23:33 (165.47) | 14:27:05 (134.49) |

Peak memory (`/usr/bin/time -l`) at 74b3c03: 1.74 GB (pixel), 1.86 GB (raw), 1.91 GB (rg); at 7ab20cf: 1.75, 1.85, 1.89 GB. `RAINYDAY_TIMERS`, raw run at 74b3c03: resampling 111 s, of which the sub-durations 58 s; frequency analysis plus RECORDCI estimators 97 s; per-storm pool 29 s (shared by RECORDCI and the exact engine); exact estimators 5.3 s.

compare_runs.py (run one after the other): `--stat median --engine montecarlo` 3.8 s, `--engine exact --stat mean` 0.7 s, `--stat level` 0.7 s at 74b3c03 (14:28); 3.4 s, 0.6 s, 0.6 s at 7ab20cf (14:43).

## Checks

- Every duration of every run has all outputs; no NaN, no -9999, 15 rows, depths increase with T.
- `rd_mean` = the realization mean, `sst_expected` = the exact `mean` and `sst_level` = the exact `level` (bitwise); `rec_lo/hi` = the 5th/95th percentiles of `_RecordBootstrap.csv`.
- Exact vs Monte Carlo: |mean of realizations - exact mean| at most 2.54 Monte Carlo standard errors over the 315 cells (3 runs x 7 durations x 15 T); 88-93 % of the realizations fall inside the exact q05-q95.
- Identical, at the 10 shared return periods, to earlier runs of the same configurations: case 1 `_FreqAnalysis.csv` vs the paper's `C72_N400_all` run and the matrix `both` run, `_FreqExact.csv` vs the matrix `both` run; case 2 `_FreqAnalysis.csv`, `_RecordCI.csv`, `_RecordBootstrap.csv` vs the earlier `val` (Monte Carlo + RECORDCI) runs, and `_FreqExact.csv`, `_RecordBootstrap.csv`, `_RecordBootstrapLevel.csv`, `_RecordDraws.csv` vs the earlier `hucexact` runs. The three compare_runs tables equal the earlier ones at the shared rows.
- Re-run at 7ab20cf: byte-identical to 74b3c03 (every RainyDay output file, every compare_runs table, every packaged CSV; the JSONs differ only in MAINPATH).
- Case 1 `sst_expected`, `rec_lo/hi` and `sst_level` equal the paper's `24_record_uncertainty/csv/pixel_record_ci.csv` (Iowa) to 4 decimals.

## Files

**File kinds** (RainyDay names: `<SCENARIONAME>_<d>_<kind>.csv`; the `#` header lines of FreqExact, RecordCI and RecordDraws define every column: read them with `comment='#'`):

| Kind | What it is |
|---|---|
| `_FreqAnalysis.csv` | Monte Carlo frequency analysis (RainyDay's own output): prob.exceed, returnperiod, rlz1..rlz100 (the T-yr depth of each of the 100 realizations, mm) |
| `_FreqExact.csv` | exact engine: mean, level, q05..q95, exp_min_100, exp_max_100 (mm; '#' header lines) |
| `_RecordCI.csv` | record CI table: rd_mean, sst_expected, sst_level, ens_lo/hi, rec_lo/hi/se, est_lo/hi, pred_lo/hi (mm; '#' header lines) |
| `_RecordBootstrap.csv` | sst_expected of each of the 1000 bootstrap records (rec1..rec1000, mm) |
| `_RecordDraws.csv` | how often each record year 2002-2023 was drawn in each bootstrap record ('#' header: method, seed, block, B, years) |

Every CSV has 15 rows, one per return period (see Settings), except RecordDraws (1000 rows), the Atlas 14 table (190 rows), the compare tables (105 rows) and `validation/verify_exact_both.csv` (322 rows). Values in mm.

| File | What it is | Run | Bytes | md5 |
|---|---|---|---|---|
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0.json` | control file (JSON), paths as placeholders | Case 1 run `Pixel_r1_A1km2_rot0` | 1,574 | `786c2806686129fa21198fa46d1a54bc` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_1/Pixel_r1_A1km2_rot0_1_FreqAnalysis.csv` | 1 h, `_FreqAnalysis.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 11,328 | `5eccced3734cbb2a06db028ead1efd64` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_1/Pixel_r1_A1km2_rot0_1_FreqExact.csv` | 1 h, `_FreqExact.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 2,772 | `66a2e36f764a1b1652e2197a290506e3` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_1/Pixel_r1_A1km2_rot0_1_RecordCI.csv` | 1 h, `_RecordCI.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 2,926 | `cb92eb6110cdd6c5b78b383476b39436` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_1/Pixel_r1_A1km2_rot0_1_RecordBootstrap.csv` | 1 h, `_RecordBootstrap.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 127,310 | `932e0dda3071d5908298826d38669a5a` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_3/Pixel_r1_A1km2_rot0_3_FreqAnalysis.csv` | 3 h, `_FreqAnalysis.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 11,328 | `8b0380fd42c0702e15291c493158f2bf` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_3/Pixel_r1_A1km2_rot0_3_FreqExact.csv` | 3 h, `_FreqExact.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 2,810 | `4877f04e344c34764620eff7d9d5ff71` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_3/Pixel_r1_A1km2_rot0_3_RecordCI.csv` | 3 h, `_RecordCI.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 2,972 | `d643df90ab3b65218150a97cc8e1283a` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_3/Pixel_r1_A1km2_rot0_3_RecordBootstrap.csv` | 3 h, `_RecordBootstrap.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 131,882 | `291661cc9923a71e95341e03bc6d63d0` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_6/Pixel_r1_A1km2_rot0_6_FreqAnalysis.csv` | 6 h, `_FreqAnalysis.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 11,328 | `b303870e174ca16d4fbdbd0fcdff4eed` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_6/Pixel_r1_A1km2_rot0_6_FreqExact.csv` | 6 h, `_FreqExact.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 2,843 | `18250bcc3942c432b0411ace6453540a` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_6/Pixel_r1_A1km2_rot0_6_RecordCI.csv` | 6 h, `_RecordCI.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 3,012 | `0fcc0e83862fb7ddaa069cc176203f4e` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_6/Pixel_r1_A1km2_rot0_6_RecordBootstrap.csv` | 6 h, `_RecordBootstrap.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 135,330 | `64bbc0b61052fd13b9bd9135abf63ff0` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_12/Pixel_r1_A1km2_rot0_12_FreqAnalysis.csv` | 12 h, `_FreqAnalysis.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 11,328 | `c5632765eb99ac4b12322ba0d74f07cd` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_12/Pixel_r1_A1km2_rot0_12_FreqExact.csv` | 12 h, `_FreqExact.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 2,867 | `559b2d7d48930445b66b1d236c74bc86` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_12/Pixel_r1_A1km2_rot0_12_RecordCI.csv` | 12 h, `_RecordCI.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 3,042 | `cf6089a2e671247cbaf4ba36e95ad226` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_12/Pixel_r1_A1km2_rot0_12_RecordBootstrap.csv` | 12 h, `_RecordBootstrap.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 137,847 | `dc94c9d4fb666c5d24b38051f2261458` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_24/Pixel_r1_A1km2_rot0_24_FreqAnalysis.csv` | 24 h, `_FreqAnalysis.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 11,328 | `11947a0a5c36c1671fc037b1c02b94a4` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_24/Pixel_r1_A1km2_rot0_24_FreqExact.csv` | 24 h, `_FreqExact.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 2,880 | `df8c8068e0072db33995faed44badadd` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_24/Pixel_r1_A1km2_rot0_24_RecordCI.csv` | 24 h, `_RecordCI.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 3,059 | `c5db1cf1836f568eb3c2896da5eb4f9e` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_24/Pixel_r1_A1km2_rot0_24_RecordBootstrap.csv` | 24 h, `_RecordBootstrap.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 139,304 | `07c4a62021f0e355df3d801331acc251` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_48/Pixel_r1_A1km2_rot0_48_FreqAnalysis.csv` | 48 h, `_FreqAnalysis.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 11,328 | `dec02a43f95ddb080aa0b47b7c9aa14d` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_48/Pixel_r1_A1km2_rot0_48_FreqExact.csv` | 48 h, `_FreqExact.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 2,888 | `63dacaa32b8a85bec0b3c4f7875cb4d0` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_48/Pixel_r1_A1km2_rot0_48_RecordCI.csv` | 48 h, `_RecordCI.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 3,064 | `20a722b98b58a39a85648df876a298ee` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_48/Pixel_r1_A1km2_rot0_48_RecordBootstrap.csv` | 48 h, `_RecordBootstrap.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 140,218 | `aaa97d1a9e0cacd9ccec23237fe10f2c` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_72/Pixel_r1_A1km2_rot0_72_FreqAnalysis.csv` | 72 h, `_FreqAnalysis.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 11,328 | `ad08d172cca0480bac8698f5294cba3a` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_72/Pixel_r1_A1km2_rot0_72_FreqExact.csv` | 72 h, `_FreqExact.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 2,890 | `a3e7b572e4bf82c9edc52dbfddbfc60d` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_72/Pixel_r1_A1km2_rot0_72_RecordCI.csv` | 72 h, `_RecordCI.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 3,072 | `da3b51394a7ca0b1527b2516b29d00e3` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_72/Pixel_r1_A1km2_rot0_72_RecordBootstrap.csv` | 72 h, `_RecordBootstrap.csv` | Case 1 run `Pixel_r1_A1km2_rot0` | 140,529 | `b979c2c26e028e3ba629f1b495a1b30d` |
| `case1_iowa_pixel/Pixel_r1_A1km2_rot0_72/Pixel_r1_A1km2_rot0_RecordDraws.csv` | `_RecordDraws.csv` (once per run, in the `_72` folder) | Case 1 run `Pixel_r1_A1km2_rot0` | 51,399 | `405d38f2c0e1ed6dd4d80f594eb04944` |
| `case1_iowa_pixel/atlas14_070600040101.csv` | NOAA Atlas 14 (Vol 8 v2, PDS) point depths and 90 % CI at the HUC-12 070600040101 centroid, all 19 durations x 10 ARIs; columns duration_h, ari_yr, depth_mm, lower90_mm, upper90_mm | PUBLIC/DATA/ATLAS14 | 5,488 | `c92fd987b44ab6993465e9fb8ed8ac00` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006.json` | control file (JSON), paths as placeholders | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 1,625 | `76f64c407839215dd8bead18d7b7aa43` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_1/Basin_raw_Iowa_HUC_07100006_1_FreqAnalysis.csv` | 1 h, `_FreqAnalysis.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 11,328 | `ae498fa2204a6f6ecaa403a83593adb3` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_1/Basin_raw_Iowa_HUC_07100006_1_FreqExact.csv` | 1 h, `_FreqExact.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 2,778 | `130ad2b032fbe7ec734e7cb498f9cc4d` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_1/Basin_raw_Iowa_HUC_07100006_1_RecordCI.csv` | 1 h, `_RecordCI.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 2,932 | `f12049d26a0846737b4329442ac4c242` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_1/Basin_raw_Iowa_HUC_07100006_1_RecordBootstrap.csv` | 1 h, `_RecordBootstrap.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 127,310 | `393b73caa22c1031cf5315e0550bfac6` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_3/Basin_raw_Iowa_HUC_07100006_3_FreqAnalysis.csv` | 3 h, `_FreqAnalysis.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 11,328 | `29f101018b3f47fa759cd7c940a48257` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_3/Basin_raw_Iowa_HUC_07100006_3_FreqExact.csv` | 3 h, `_FreqExact.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 2,778 | `e05f23b320dd528aa05115ffb5ca3332` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_3/Basin_raw_Iowa_HUC_07100006_3_RecordCI.csv` | 3 h, `_RecordCI.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 2,932 | `050efcc05c0932732a96eb1ec3870c48` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_3/Basin_raw_Iowa_HUC_07100006_3_RecordBootstrap.csv` | 3 h, `_RecordBootstrap.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 127,310 | `c9c7a90d52e4cb59ac05b5d50ae61f2e` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_6/Basin_raw_Iowa_HUC_07100006_6_FreqAnalysis.csv` | 6 h, `_FreqAnalysis.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 11,328 | `abb5c3a02e467984782fb821ee1872fd` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_6/Basin_raw_Iowa_HUC_07100006_6_FreqExact.csv` | 6 h, `_FreqExact.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 2,778 | `ba6c611d24eaa340fb9739cc2bf517d4` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_6/Basin_raw_Iowa_HUC_07100006_6_RecordCI.csv` | 6 h, `_RecordCI.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 2,932 | `339360ebacdbeefc0415091c9c0b8d45` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_6/Basin_raw_Iowa_HUC_07100006_6_RecordBootstrap.csv` | 6 h, `_RecordBootstrap.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 127,310 | `86b49de3a9b56bc836e517c401b5e3ba` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_12/Basin_raw_Iowa_HUC_07100006_12_FreqAnalysis.csv` | 12 h, `_FreqAnalysis.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 11,328 | `3dae0249a219445158bdadb285b6e996` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_12/Basin_raw_Iowa_HUC_07100006_12_FreqExact.csv` | 12 h, `_FreqExact.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 2,815 | `a2fa3dac9ab4b16de178ea9fc5f7d172` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_12/Basin_raw_Iowa_HUC_07100006_12_RecordCI.csv` | 12 h, `_RecordCI.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 2,974 | `53062ffcb6d362aa6605e35f83d5d88c` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_12/Basin_raw_Iowa_HUC_07100006_12_RecordBootstrap.csv` | 12 h, `_RecordBootstrap.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 131,515 | `2204aa55cd6f9e64e5c3e2aa339f9bef` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_24/Basin_raw_Iowa_HUC_07100006_24_FreqAnalysis.csv` | 24 h, `_FreqAnalysis.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 11,328 | `6cd6dc35c2a98c553112a000c7a31a73` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_24/Basin_raw_Iowa_HUC_07100006_24_FreqExact.csv` | 24 h, `_FreqExact.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 2,855 | `fa75d2bbde82c8c1c34556043afed842` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_24/Basin_raw_Iowa_HUC_07100006_24_RecordCI.csv` | 24 h, `_RecordCI.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 3,031 | `afa68e65c24a12296f81e460536d3430` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_24/Basin_raw_Iowa_HUC_07100006_24_RecordBootstrap.csv` | 24 h, `_RecordBootstrap.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 135,838 | `acd511ceca00d8a78e40958313af4971` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_48/Basin_raw_Iowa_HUC_07100006_48_FreqAnalysis.csv` | 48 h, `_FreqAnalysis.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 11,328 | `12ac9174163204acaef27f565dd32949` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_48/Basin_raw_Iowa_HUC_07100006_48_FreqExact.csv` | 48 h, `_FreqExact.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 2,886 | `dfa5ff2740d57ab9b230ca382bfcde50` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_48/Basin_raw_Iowa_HUC_07100006_48_RecordCI.csv` | 48 h, `_RecordCI.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 3,061 | `19a25f632efe20f1acd6937d23e8101d` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_48/Basin_raw_Iowa_HUC_07100006_48_RecordBootstrap.csv` | 48 h, `_RecordBootstrap.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 139,091 | `29eea2e202e81ff5fd5e16772fe900b7` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_72/Basin_raw_Iowa_HUC_07100006_72_FreqAnalysis.csv` | 72 h, `_FreqAnalysis.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 11,328 | `e0e3da9b63460e01c022c54c3ebaa420` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_72/Basin_raw_Iowa_HUC_07100006_72_FreqExact.csv` | 72 h, `_FreqExact.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 2,889 | `af18286bb2cc18e5a89f393879b1560d` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_72/Basin_raw_Iowa_HUC_07100006_72_RecordCI.csv` | 72 h, `_RecordCI.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 3,069 | `475334c2b0c55bf37f679716023774ad` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_72/Basin_raw_Iowa_HUC_07100006_72_RecordBootstrap.csv` | 72 h, `_RecordBootstrap.csv` | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 139,688 | `2456a0da63727731a02e5bee00c3bc5c` |
| `case2_huc07100006/Basin_raw_Iowa_HUC_07100006_72/Basin_raw_Iowa_HUC_07100006_RecordDraws.csv` | `_RecordDraws.csv` (once per run, in the `_72` folder) | Case 2 run `Basin_raw_Iowa_HUC_07100006` (real basin) | 51,399 | `405d38f2c0e1ed6dd4d80f594eb04944` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006.json` | control file (JSON), paths as placeholders | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 1,620 | `5f3ace889b81e8d2824592809c8d1da8` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_1/Basin_rg_Iowa_HUC_07100006_1_FreqAnalysis.csv` | 1 h, `_FreqAnalysis.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 11,328 | `6d6765a0bb2cd5000f60a101fb5eaa85` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_1/Basin_rg_Iowa_HUC_07100006_1_FreqExact.csv` | 1 h, `_FreqExact.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 2,777 | `2ce1506ac14a3ef3c66eff3542e78fc3` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_1/Basin_rg_Iowa_HUC_07100006_1_RecordCI.csv` | 1 h, `_RecordCI.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 2,931 | `b7a1b2c4c023e90cc96498e7db7a150a` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_1/Basin_rg_Iowa_HUC_07100006_1_RecordBootstrap.csv` | 1 h, `_RecordBootstrap.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 127,310 | `07787ceb9965b85187de20f4c36fc6dc` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_3/Basin_rg_Iowa_HUC_07100006_3_FreqAnalysis.csv` | 3 h, `_FreqAnalysis.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 11,328 | `13252fc7045c74ec0e0b85c60041834c` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_3/Basin_rg_Iowa_HUC_07100006_3_FreqExact.csv` | 3 h, `_FreqExact.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 2,777 | `8b2fe41ed0f6a4df439df88a07702f69` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_3/Basin_rg_Iowa_HUC_07100006_3_RecordCI.csv` | 3 h, `_RecordCI.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 2,931 | `520655b5c2aae1d52c716eb09d5904d9` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_3/Basin_rg_Iowa_HUC_07100006_3_RecordBootstrap.csv` | 3 h, `_RecordBootstrap.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 127,310 | `effa8006ec5add86cff8e9f0a73863f9` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_6/Basin_rg_Iowa_HUC_07100006_6_FreqAnalysis.csv` | 6 h, `_FreqAnalysis.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 11,328 | `f53785263360d652dea0972ea65b1a7b` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_6/Basin_rg_Iowa_HUC_07100006_6_FreqExact.csv` | 6 h, `_FreqExact.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 2,777 | `4be21ea81bbf27dc6eb197b4ebfe6919` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_6/Basin_rg_Iowa_HUC_07100006_6_RecordCI.csv` | 6 h, `_RecordCI.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 2,931 | `2dbb253ff2cf065d099b3259a5f2b94c` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_6/Basin_rg_Iowa_HUC_07100006_6_RecordBootstrap.csv` | 6 h, `_RecordBootstrap.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 127,310 | `9b91d2c820272af2972712f93b4632eb` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_12/Basin_rg_Iowa_HUC_07100006_12_FreqAnalysis.csv` | 12 h, `_FreqAnalysis.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 11,328 | `c680f9f7917b76a21314997f6b998c9f` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_12/Basin_rg_Iowa_HUC_07100006_12_FreqExact.csv` | 12 h, `_FreqExact.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 2,816 | `b552f4fdc83b8588e401f301890d388a` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_12/Basin_rg_Iowa_HUC_07100006_12_RecordCI.csv` | 12 h, `_RecordCI.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 2,977 | `f62954274b85eaf4970db62213e5cb67` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_12/Basin_rg_Iowa_HUC_07100006_12_RecordBootstrap.csv` | 12 h, `_RecordBootstrap.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 131,722 | `3221ce1c39c0594c74c9de4a34ab5982` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_24/Basin_rg_Iowa_HUC_07100006_24_FreqAnalysis.csv` | 24 h, `_FreqAnalysis.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 11,328 | `489fafae331e5dafbfe158d9e24abb8a` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_24/Basin_rg_Iowa_HUC_07100006_24_FreqExact.csv` | 24 h, `_FreqExact.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 2,854 | `6bb3a7f54d3806f231fff809921a58e1` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_24/Basin_rg_Iowa_HUC_07100006_24_RecordCI.csv` | 24 h, `_RecordCI.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 3,030 | `f065eeccf2f175af3143329e937575c3` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_24/Basin_rg_Iowa_HUC_07100006_24_RecordBootstrap.csv` | 24 h, `_RecordBootstrap.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 135,690 | `ae5d45ad3b7b7c6351e4552eb0f66443` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_48/Basin_rg_Iowa_HUC_07100006_48_FreqAnalysis.csv` | 48 h, `_FreqAnalysis.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 11,328 | `35d368b4ee886105a350a5006f49bbd7` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_48/Basin_rg_Iowa_HUC_07100006_48_FreqExact.csv` | 48 h, `_FreqExact.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 2,885 | `28e2bd011e97d30d1d42daba650377ea` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_48/Basin_rg_Iowa_HUC_07100006_48_RecordCI.csv` | 48 h, `_RecordCI.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 3,060 | `a804a9fdf6906f79a66ec1fd80d494c4` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_48/Basin_rg_Iowa_HUC_07100006_48_RecordBootstrap.csv` | 48 h, `_RecordBootstrap.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 139,118 | `5f428f9fb9aa7ab056a752c140fdb8a8` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_72/Basin_rg_Iowa_HUC_07100006_72_FreqAnalysis.csv` | 72 h, `_FreqAnalysis.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 11,328 | `1e99702ad06faf165f75bdc008e96627` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_72/Basin_rg_Iowa_HUC_07100006_72_FreqExact.csv` | 72 h, `_FreqExact.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 2,888 | `6618898e05c4bce77dbd46f2c096f599` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_72/Basin_rg_Iowa_HUC_07100006_72_RecordCI.csv` | 72 h, `_RecordCI.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 3,068 | `f1525a4ffeda359d47da45a0bbf86f19` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_72/Basin_rg_Iowa_HUC_07100006_72_RecordBootstrap.csv` | 72 h, `_RecordBootstrap.csv` | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 139,684 | `1216fb96fc65cc00f4f6711f5755ff12` |
| `case2_huc07100006/Basin_rg_Iowa_HUC_07100006_72/Basin_rg_Iowa_HUC_07100006_RecordDraws.csv` | `_RecordDraws.csv` (once per run, in the `_72` folder) | Case 2 run `Basin_rg_Iowa_HUC_07100006` (spread-matched ellipse) | 51,399 | `405d38f2c0e1ed6dd4d80f594eb04944` |
| `case2_huc07100006/compare_level.csv` | tools/compare_runs.py raw vs rg, `--stat level`, margin 5 %, level 90 %: delta (%), Monte Carlo, record and combined CIs, p-values and class per duration x T | Case 2 pair | 26,662 | `f81adc3533b0b758066867201cab8a6d` |
| `case2_huc07100006/compare_mean_exact.csv` | tools/compare_runs.py raw vs rg, `--engine exact --stat mean`, margin 5 %, level 90 %: delta (%), Monte Carlo, record and combined CIs, p-values and class per duration x T | Case 2 pair | 26,548 | `00a8fbba7c7c29a7b2f6a7cde753bca8` |
| `case2_huc07100006/compare_median_montecarlo.csv` | tools/compare_runs.py raw vs rg, `--stat median --engine montecarlo`, margin 5 %, level 90 %: delta (%), Monte Carlo, record and combined CIs, p-values and class per duration x T | Case 2 pair | 27,842 | `d33a619b478aa840c2ac4a53176677e3` |
| `validation/verify_exact_both.csv` | validation-matrix cells of the exact engine vs the Monte Carlo realizations (FREQENGINE "both" runs: 4 shapes x 7 durations x 10 T + Talequah 7 x 6 = 322 rows); columns shape, duration, T, N, rd_mean, exact_mean, z_mean, q_ok99, q_ok95, min_ok, max_ok: an 11-column subset of compare/verify_exact_both.csv (39 columns, md5 88101864219df8e67f45049807f69193; tools/verify_exact_engine.py), prepared for the validation code cell | validation matrix (`runs/both`) | 25,384 | `b756d100d63003f965bd4218b5d831d0` |

Not copied (to stay under 5 MB): the PNGs (`_FrequencyAnalysis.png`, `_FreqExact.png`) and `_RecordBootstrapLevel.csv` (the record bootstrap of `level`, which `compare_runs.py --stat level` needs). With the files above, `compare_runs.py` can be re-run on the case 2 folders for `--stat median` and `--stat mean`; `--stat level` needs the full run output.
