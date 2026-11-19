# Getting the AORC rainfall files in RainyDay format (1979–2025)

These instructions give you the AORC rainfall record as the daily files that RainyDay reads
(`AORC.YYYYMMDD.preciptemp.nc`), for any period between 1979 and 2025. The files come from two places:

| Days | Where they come from | What you do | Part |
|---|---|---|---|
| 1979-02-01 to 2023-12-31 | Ready-made files in the public bucket of the Hydroclimate Extremes Group (University of Wisconsin–Madison), `https://web.s3.wisc.edu/aorc` | Download them | A |
| 2024-01-01 to 2025-12-31 | Not in that bucket. NOAA AORC version 1.1 on the AWS Open Data registry, `s3://noaa-nws-aorc-v1-1-1km` | Build them with the script in this folder | B |

Part C then joins the two sets in one folder for RainyDay. The state of both sources was checked on 16 November 2026:
the bucket ends on 2023-12-31 and the NOAA archive ends on 2025-12-31. When NOAA publishes a new year, Part B builds
it with the same command.

**What you need:** an internet connection, conda, and free disk space: about 20 MB per day, so **7 GB per year**
(160 GB for 2002–2023, 14 GB for 2024–2025). The scripts were run on macOS; they use nothing specific to it.

**Time:** Part A depends on your connection (about 1 s per file on a fast one). Part B took about 1 h 50 min for
2024–2025 with 8 workers.

---

## 1. The files

One file per UTC day, the same for both sources:

- **Grid:** CONUS, 885 × 1770 cells of 1/30° (about 4 km). It is a nearest-neighbour subsample of the NOAA AORC 1 km
  grid, every 4th cell: RainyDay cell (i, j) = NOAA v1.1 cell (424 + 4i, 544 + 4j).
- **Time:** 24 hourly steps, stamped at the end of the hour: 01:00, 02:00, … and 00:00 of the next day (UTC).
- **Variables:** `RAINRATE` (mm/h) and `TMP_2maboveground` (K), both float32. Cells outside the AORC coverage are
  missing, at the same 562,547 cells in every hour (the "coverage mask", stored in `coverage_mask_nan.npz`).

The files built in Part B follow this format exactly: on the test days they are identical, value for value, to the
ready-made files (step B2).

## 2. Rules

1. The scripts never modify or delete an existing file. A file that already exists is skipped, so an interrupted
   download or build continues when you run the same command again.
2. Keep the downloaded files and the built files in **separate folders** (`data/` and `data_2024_2025/` below). Part C
   joins them with links, without copying.
3. Run the test of step B2 before the full build of step B3, and stop at every **STOP IF** line that applies.

## 3. Set up

All commands are for a zsh or bash terminal, run from the RainyDay_fast folder. Set these once per terminal session
(change the path to yours):

```bash
export AORC_ROOT=/path/to/AORC            # where all AORC folders go
export SRC=$AORC_ROOT/data                # the downloaded files (Part A)
export NEW=$AORC_ROOT/data_2024_2025      # the files built from NOAA AORC v1.1 (Part B)
export DL=$PWD/tools/aorc_download
mkdir -p $AORC_ROOT/aorc_logs
```

Part A and Part C need only Python 3. Part B needs a small conda environment that can read the NOAA archive, the
same one as the AORC corrections of [`tools/aorc_fix`](../aorc_fix/AORC_FIX_INSTRUCTIONS.md):

```bash
conda env create -f tools/aorc_fix/aorcfix_env.yml       # creates the env "aorcfix"; skip if you already have it
conda activate aorcfix
python -c "import zarr, s3fs, xarray, netCDF4; print('zarr', zarr.__version__)"
```

STOP IF zarr is not 2.x (the scripts use the zarr 2 API; the env file pins 2.18.3).

---

## Part A. Download 1979–2023

Choose the first and the last day you need. For 2002–2023:

```bash
python $DL/download_aorc_wisc.py --start 20020101 --end 20231231 --dst $SRC --workers 4 \
    | tee $AORC_ROOT/aorc_logs/download.log
```

Expected last lines for 2002–2023:

```
downloaded 8035, already there 0, not in the bucket 0, failed 0
```

- **`failed` is not 0:** run the same command again. Only the missing files are downloaded.
- **`not in the bucket` is not 0:** the script lists those days. Days after 2023-12-31 are built in Part B.

STOP IF `failed` is still not 0 after a second run; keep the messages.

**The same download with `wget`.** The bucket has a list of all its files. This is the procedure given by the
Hydroclimate Extremes Group:

```bash
wget https://web.s3.wisc.edu/aorc/aorc_urls.txt
grep -E "/(2002|2003)/" aorc_urls.txt > download.txt     # the years you need
wget -nc -P $SRC -i download.txt
```

**Before you use these files.** The files of 2002–2023 have four known defects (for example, in 2002–2020 the hour
ending 23:00 UTC is a copy of 22:00). Correct them with
[`tools/aorc_fix/AORC_FIX_INSTRUCTIONS.md`](../aorc_fix/AORC_FIX_INSTRUCTIONS.md), which builds the corrected folder
`data_aorcfix2`. You can do that before or after Part B.

---

## Part B. Build 2024–2025 from NOAA AORC v1.1

The script reads NOAA AORC v1.1 directly from AWS (anonymous access, no account), takes the cells of the RainyDay
grid, and writes daily files in the RainyDay format. It needs one existing RainyDay-format file as a **template**:
the new files copy its grid, compression and attributes. Use a file of 2022 or 2023 from Part A:

```bash
conda activate aorcfix
export TPL=$SRC/AORC.20231231.preciptemp.nc
ls -la $TPL
```

If you did not download 2023, get that one file:
`python $DL/download_aorc_wisc.py --start 20231231 --end 20231231 --dst $AORC_ROOT/template`
and set `TPL` to it.

### Step B1. Check the access to the NOAA archive

```bash
python -c "import s3fs; print([p.split('/')[-1] for p in s3fs.S3FileSystem(anon=True).ls('noaa-nws-aorc-v1-1-1km') if p.endswith('.zarr')][-3:])"
```

Expected: `['2023.zarr', '2024.zarr', '2025.zarr']` (more years later). STOP IF it fails: a VPN or a campus proxy can
block AWS.

### Step B2. Test: rebuild two days that you already have (about 2 min)

```bash
python $DL/make_aorc_from_v11.py --days 20231230 20231231 --dst $AORC_ROOT/aorc_test/new --template $TPL --workers 2
python $DL/download_aorc_wisc.py --start 20231230 --end 20231231 --dst $AORC_ROOT/aorc_test/ref
python $DL/compare_daily_files.py $AORC_ROOT/aorc_test/new $AORC_ROOT/aorc_test/ref
```

Expected:

```
template grid = v1.1 subsample: True; template missing cells = coverage mask: True
...
AORC.20231230.preciptemp.nc OK ... RAINRATE: identical=True max|d|=0 ... TMP_2maboveground: identical=True max|d|=0 ...
AORC.20231231.preciptemp.nc OK ...
ALL IDENTICAL
```

STOP IF the last line is not `ALL IDENTICAL`, or if the first line has a `False`: the built files would not match the
ready-made ones.

### Step B3. Build 2024–2025 (about 1 h 50 min with 8 workers)

```bash
python $DL/make_aorc_from_v11.py --start 20240101 --end 20251231 --dst $NEW --template $TPL --workers 8 \
    | tee $AORC_ROOT/aorc_logs/build_2024_2025.log
```

Expected: one line per segment of 48 days, then `done`, and 731 files (14 GB) in `$NEW`:

```bash
ls $NEW/AORC.202[45]*.preciptemp.nc | wc -l          # 731
```

- The script downloads about 60 GB in total but stores only the 14 GB of the result. It stops at the start if the
  disk has less than 25 GB free.
- Each worker holds several days of the full grid in memory. If the computer runs out of memory, use fewer workers.
- One line will say `v1.1 hours missing (written as NaN): ['2026-01-01 00:00:00']`. That hour is the last one of the
  file of 31 December 2025 and belongs to the 2026 archive, which did not exist on 16 November 2026 (see section 4).
- **Interrupted?** Run the same command again. Finished files are skipped; a file is complete as soon as it has its
  final name (unfinished ones end in `.tmp` and are rewritten).

STOP IF the file count is not 731 after `done`.

### Step B4. Check the new files (about 10 min)

```bash
python $DL/qa_daily.py $NEW 20240101 20251231 $AORC_ROOT/aorc_logs/qa_2024_2025.csv --workers 8 \
    | tee $AORC_ROOT/aorc_logs/qa_2024_2025.log
```

Expected (build of 12 November 2026):

```
QA ... 20240101-20251231: 731 days expected, 731 present
  times ok: 731/731; grid identical across files: True
  all-NaN hours: [24 hours of 2024-06-18, and 2026-01-01 00:00:00]
  hours with NaN off the static mask: 13872; TMP NaN == mask: False
  negative: 4, -9999: 0, inf: 0, not multiple of 1/16: 0
  rain max 2900.000 mm/h on 20240611; ...
  uniform-field hours: []
  TMP range 225.88-328.38 K
  duplicated non-dry hours (any): 11 [...]; consecutive: 0 []
```

The files are correct copies of the NOAA archive, and the check shows that **the archive itself has problems in
2024–2025**: a missing day, impossible rain rates, and cells without data. Section 4 lists them and says what to do.
If NOAA corrects the archive, these numbers will change.

STOP IF days are not present, `times ok` is not 731/731, the grid is not identical, or `not multiple of 1/16` is not
0: those would be errors of the build, not of the data.

---

## Part C. One folder for RainyDay

RainyDay reads one folder. `make_view.py` builds it with a link for every day, taken from the first source folder that
has that day. Nothing is copied.

With the corrected 2002–2023 files of `tools/aorc_fix` (recommended):

```bash
python $DL/make_view.py --start 20020101 --end 20251231 --dst $AORC_ROOT/data_aorcfix2_2025 \
    --src $AORC_ROOT/data_aorcfix2 --src $NEW
```

Expected:

```
.../data_aorcfix2_2025: 8766 links, 20020101 to 20251231, no day missing
    8035 from .../data_aorcfix2
     731 from .../data_2024_2025
```

Without the corrections, use `--src $SRC --src $NEW`. The script stops without writing anything if a day of the period
is in no source folder.

Then, in RainyDay_fast:

1. Point RAINPATH at the new folder: `"RAINPATH": "<AORC_ROOT>/data_aorcfix2_2025/AORC.*.preciptemp.nc"`, and set
   the years of the run (for example `"INCLUDEYEARS": "2002-2025"`). RainyDay follows the links.
2. Make **new** crops from that RAINPATH and rebuild the storm catalogs
   ([`AORC_FIX_INSTRUCTIONS.md`, section 5](../aorc_fix/AORC_FIX_INSTRUCTIONS.md#5-using-the-corrected-folders-in-rainyday_fast)).
   Crops and catalogs made from another folder do not contain the new years.
3. Links are absolute paths. Do not move or rename the source folders afterwards; if you must, build the view again
   in a new folder. The disk must be able to hold symbolic links (an exFAT drive cannot).

---

## 4. Known problems of the 2024–2025 data

These problems are in NOAA AORC v1.1 itself (verified in the archive on 12 November 2026). The scripts copy the
values as they are and correct nothing.

| Problem | Where | Effect in RainyDay |
|---|---|---|
| **A missing day.** 2024-06-18 00:00 to 23:00 UTC has no data at any cell | last hour of file `20240617`, first 23 hours of file `20240618` | Read as no rain: a storm in those hours is lost or underestimated |
| **A missing hour.** 2026-01-01 00:00 UTC is not in the archive yet | last hour of file `20251231` | Read as no rain. Rebuild that file when NOAA publishes 2026 (move the old file aside first) |
| **Impossible rain rates.** Up to 2,900 mm/h in South Florida (near 26.1°N, 81.5°W) from 2024-06-10 13:00 to 2024-06-12 12:00 UTC, and bursts above 1,000 mm/h on 2025-09-19, 2025-11-07, 2025-11-14/15 and 2025-11-23. In total, 75 hours have a cell above 300 mm/h (2022–2023: none; maximum 252 mm/h) | 14 daily files: `20240610`–`20240612`, `20250301`, `20250601`, `20250918`, `20250919`, `20251106`, `20251107`, `20251113`–`20251115`, `20251122`, `20251123` | These can enter a storm catalog as false extreme storms |
| **Negative values.** 4 cell-hours, in the same Florida event | files `20240610`–`20240612` | Negligible, but a sign of the same fault |
| **Cells without data.** From 2024-06-01, every hour has about 1,300 cells without data outside the coverage mask (scattered between 25–43°N and 76–125°W). A few hours have much larger gaps (up to 179,000 cells in November 2024) | all files from `20240531` | Read as no rain at those cells |
| **Repeated hours.** 11 hours are exact copies of the hour six hours earlier | 2024-02-25 and 2024-06-11 (08:00–12:00 = 02:00–06:00), 2025-11-14 (12:00 = 06:00) | Small; the same kind exists in November 2022 |

The defect of the 2002–2020 files (hour 23:00 a copy of 22:00) does **not** occur in 2024–2025: no hour is a copy of
the hour before it.

**What to do.** Look at the diagnostic plots of every storm of 2024–2025 that enters your catalog, above all in
Florida and in autumn 2025. Remove a false storm with `EXCLUDESTORMS`, or end the run in 2023 or on 2024-05-31
(`INCLUDEYEARS`) if your study area is affected. `qa_daily.py` writes one row per day
(`rain_max`, `allnan_hours`, `nan_off_mask_hours`, `n_neg`), which shows the days to look at.

## 5. Building other days from NOAA AORC v1.1

`make_aorc_from_v11.py` works for any day of the NOAA archive (1979 onward), for example a day that is missing from
your folder. Two limits:

- **Tested period.** The built files were compared with the ready-made files on 21 days of 2022–2023, all identical
  (and on the two days of step B2 with this version of the script). Earlier years were not compared.
- **Before 2022, prefer the download.** The ready-made files of earlier years can differ from v1.1: they contain the
  defects listed in `tools/aorc_fix`, and their temperature field has no missing cells in 2002–2021. A period that
  mixes built and downloaded days would not be uniform.

To rebuild a day, move the old file out of the destination folder first; the script does not overwrite.

## 6. Troubleshooting

| Problem | What to do |
|---|---|
| S3 errors, timeouts or a hang during Part B | Ctrl-C and run the same command again (resumable). Check step B1. A VPN or a campus proxy can block AWS. |
| `Only ... GB free` or `Not enough free space` | Free space on the disk of `$AORC_ROOT`, or choose another disk. |
| `AssertionError: template grid != v1.1 subsample` | The template is not on the RainyDay AORC grid (885 × 1770). Use a file downloaded in Part A. |
| `WARNING: the first hour of the template is not missing exactly on the coverage mask` | Use a file of 2022 or 2023 as template (not 2022-10-31 or 2022-12-31, which have extra cells without data). |
| `zarr` errors such as `S3Map` not found | zarr 3 was installed. Recreate the env from `tools/aorc_fix/aorcfix_env.yml`. |
| `make_view.py`: `... days are in no source folder` | The message gives the first and last missing day. Download or build them, then run it again. |
| `make_view.py`: `already holds files` | Choose a new folder name, or move the old view aside. |
| Broken links (`No such file` when RainyDay opens a file) | A source folder was moved or renamed. Move it back, or build the view again in a new folder. |

## 7. Files in this folder

| File | What it is |
|---|---|
| `download_aorc_wisc.py` | Part A: downloads the ready-made files of 1979–2023 (resumable, parallel; Python standard library only) |
| `make_aorc_from_v11.py` | Part B: builds RainyDay-format daily files from NOAA AORC v1.1 (resumable, parallel) |
| `compare_daily_files.py` | Step B2: compares built files with existing ones (values, missing cells, encoding) |
| `qa_daily.py` | Step B4: checks a folder of daily files over a period, one row per day |
| `make_view.py` | Part C: one folder of links from several source folders (Python standard library only) |
| `coverage_mask_nan.npz` | The coverage mask: the 562,547 cells without data in every hour of the 2002–2023 files |

The conda environment is `tools/aorc_fix/aorcfix_env.yml` (Python 3.10, zarr 2.18.3, s3fs, xarray, netCDF4).

**Data credits.** AORC (Analysis of Record for Calibration) is a product of the NOAA Office of Water Prediction;
version 1.1 is distributed through the AWS Open Data registry. The RainyDay-format files of 1979–2023 are prepared and
hosted by the Hydroclimate Extremes Group, University of Wisconsin–Madison.

Scripts by Gabriel Perez (November 2026), first used to extend a research study to 2025; they were run on macOS.
