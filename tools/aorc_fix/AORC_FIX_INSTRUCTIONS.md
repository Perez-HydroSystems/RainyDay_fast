# Correcting the RainyDay-format AORC files (aorcfix2)

These instructions rebuild, on your own Mac, the corrected AORC folder that our RainyDay_fast runs use
(`data_aorcfix2`). The result is the same as ours, value for value. They are written for you and for Claude Code: every
step has the exact command, what you should see, and when to stop.

To hand the job to Claude Code (VS Code), open the RainyDay_fast folder and type:

> Read `tools/aorc_fix/AORC_FIX_INSTRUCTIONS.md` and follow it step by step. My raw AORC folder is `<path>` and the
> corrected folders should go in `<path>`. Stop and tell me if any check fails.

**What you need:** macOS, the RainyDay-format AORC daily files `AORC.YYYYMMDD.preciptemp.nc` for 2002–2023 (from
`web.s3.wisc.edu/aorc`), at least **160 GB free** on the disk that will hold the corrected folder, an internet
connection, and conda. **Time:** about 1.5 h in total, most of it downloading.

---

## 1. Why the files need fixing

The RainyDay-format AORC files have four defects that change SST results. They are in the data, not in RainyDay, so
RainyDay_fast neither fixes nor detects them ([User's Guide, Data issues (AORC)](../../UserGuide/RainyDay_fast_tutorial.ipynb#fast-data)).

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | In every file of 2002–2020, the hour ending 23:00 UTC is an exact copy of 22:00 | 1 h: the real 23:00 hour is lost; 2–3 h sums are inflated | 23:00 replaced by NOAA AORC v1.1 |
| 2 | `AORC.20211231` holds the rain of 2021-12-30 | 31 December 2021 lost, 30 December counted twice | all 24 hours replaced by v1.1 |
| 3 | 2011-07-23 00:00 UTC (last hour of file 20110722) is −9999 everywhere | one hour treated as dry | replaced by v1.1 |
| 4 | 2022-05-29 20:00 UTC is 22.875 mm/h at every CONUS cell (in v1.1 too) | pins the 1 h depths of large basins at ~22.9 mm | mean of 19:00 and 21:00 (rain and temperature) |

Two known defects are **not** fixed: a 96.7 mm streak at 2010-12-31 16:00 UTC in NW Iowa (probably a radar
artifact; check your diagnostic plots, and use EXCLUDESTORMS if it enters your catalog), and the 23:00 copies before
2002 (outside this period).

**Where the new values come from.** NOAA AORC version 1.1 on the AWS Open Data registry
(`s3://noaa-nws-aorc-v1-1-1km/<year>.zarr`, anonymous access, no account). The RainyDay files are a nearest-neighbour
subsample of the v1.1 1 km grid, every 4th cell: RainyDay cell (i, j) = v1.1 cell (424 + 4i, 544 + 4j). The script
checks this mapping before it writes anything. Patched values match v1.1 within storage rounding (≤ 0.035 mm).

## 2. How the corrected folders are built ("alias" folders)

Your original files are **never modified**. Each corrected folder holds real files only for the days that change, and
a symbolic link (alias) to the original for every other day, so RainyDay sees a complete folder without a second copy
of the data:

```
data/                        your original files (untouched)
data_aorcfix/                step 1: 2 interpolated days (20110722, 20220529) + links to data/
data_aorcfix2/               step 2: 6,942 patched days (2002–2020, 20211231, 20220529) + links to data/
                             -> RAINPATH for runs WITH Hurricane Harvey
data_aorcfix2_no_aug2017/    step 4 (optional): links to data_aorcfix2/ and data/, without August 2017
                             -> RAINPATH for runs WITHOUT Hurricane Harvey
```

Step 1 must run before step 2: step 2 takes days 20110722 and 20220529 from `data_aorcfix/`. Without them it silently
loses fixes 3 and 4.

## 3. Rules for Claude Code

1. Never write into, rename or delete anything in the original AORC folder.
2. Never delete files or folders. If something must be redone, move it aside (e.g. `mv data_aorcfix2 data_aorcfix2_old`).
3. Do not edit the scripts in `tools/aorc_fix/`. Their checksums are part of the check (step 0.3). If a step fails,
   stop and report the full error.
4. Run the commands exactly as written, in order, and stop at every **STOP IF** line that applies.
5. Save the output of every step in `$AORC_ROOT/aorcfix_logs/` (the commands below do it with `tee`).
6. Run the test of step 2a before the full run of step 2b.
7. Links are absolute paths. Do not move or rename the original folder afterwards; if you must, rebuild the views.

## 4. Steps

All commands are for the zsh/bash terminal of macOS, run from the RainyDay_fast folder.

### Step 0. Set up

**0.1 Folders.** Set these once per terminal session (change the two paths to yours):

```bash
export SRC=/path/to/AORC/data            # your original RainyDay-format files
export AORC_ROOT=/path/to/AORC           # where the corrected folders go (>= 160 GB free)
export PREV=$AORC_ROOT/data_aorcfix
export DST=$AORC_ROOT/data_aorcfix2
export NH=$AORC_ROOT/data_aorcfix2_no_aug2017
export FIX=$PWD/tools/aorc_fix
mkdir -p $AORC_ROOT/aorcfix_logs
```

The disk of `$AORC_ROOT` must be APFS or Mac OS Extended (HFS+); an exFAT external drive cannot hold symbolic links.
Check with `diskutil info $AORC_ROOT | grep "File System"`.

**0.2 Python environment** (a small one, separate from RainyDay's; it adds zarr and s3fs to read v1.1):

```bash
conda env create -f $FIX/aorcfix_env.yml       # creates the env "aorcfix"
conda activate aorcfix
python -c "import zarr, s3fs, xarray, netCDF4; print('zarr', zarr.__version__)"
```

STOP IF zarr is not 2.x (the scripts use the zarr 2 API; the env file pins 2.18.3).

**0.3 Check the scripts and your files.**

```bash
md5 -q $FIX/make_aorcfix.py $FIX/make_aorcfix2.py $FIX/check_aorcfix2.py
ls $SRC/AORC.20{02..23}*.preciptemp.nc | wc -l
df -h $AORC_ROOT
ncdump -h $SRC/AORC.20100715.preciptemp.nc | head -20
```

Expected:
- md5: `54d8dff011491ed6562f09c6d6bb215f`, `b95ccef9e70743f43beb35046a7f34fd`, `037ed821fce379484a285637dffb3812`.
- **8035** files for 2002–2023 (every day). Files of other years in `$SRC` are fine; they are linked unchanged.
- At least 160 GB available.
- `latitude = 885`, `longitude = 1770`, `time = 24`, and a float variable `RAINRATE`.

STOP IF an md5 differs, the file count is not 8035, or the grid is not 885 × 1770.

**0.4 Confirm the defects** (prints `True` four times on the original files):

```bash
python $FIX/check_defects.py $SRC
```

If some lines print `False`, your copy differs from ours; note which, and continue only after telling GP.

### Step 1. Interpolate the two corrupt hours (`data_aorcfix`, about 1 min)

```bash
python $FIX/make_aorcfix.py $SRC $PREV "2022-05-29T20:00" "2011-07-23T00:00" 2>&1 | tee $AORC_ROOT/aorcfix_logs/step1.log
```

Expected (paths shortened):

```
AORC.20110722.preciptemp.nc -> .../data_aorcfix/AORC.20110722.preciptemp.nc
  RAINRATE 2011-07-23T00:00: was uniform -9999.000 (std 0.000); replaced by mean of 23:00 and 01:00
  TMP_2maboveground 2011-07-23T00:00: was uniform -3397.636 (std 4941.598); replaced by mean of 23:00 and 01:00
AORC.20220529.preciptemp.nc -> .../data_aorcfix/AORC.20220529.preciptemp.nc
  RAINRATE 2022-05-29T20:00: was uniform 22.875 (std 0.000); replaced by mean of 19:00 and 21:00
  TMP_2maboveground 2022-05-29T20:00: was uniform 290.875 (std 0.000); replaced by mean of 19:00 and 21:00
.../data_aorcfix: 8035 files (2 corrected copies)
```

("was uniform" is just the label; the 2011 temperature line has a nonzero std, as expected. The file count is larger
if `$SRC` holds other years.)

### Step 2. Corrected view (`data_aorcfix2`)

**2a. Test on three days** (about 2 min; writes to a separate test folder, makes no links):

```bash
python $FIX/make_aorcfix2.py $SRC $PREV $AORC_ROOT/aorcfix2_test --only-days 20020615 20110722 20211231 \
    2>&1 | tee $AORC_ROOT/aorcfix_logs/step2_test.log
python $FIX/check_aorcfix2.py $SRC $PREV $AORC_ROOT/aorcfix2_test --only-days 20020615 20110722 20211231 \
    2>&1 | tee -a $AORC_ROOT/aorcfix_logs/step2_test.log
```

Expected: the check prints PASS for C1b, C2, C3, C3b, C4a and C4b, then `ALL PASS`. STOP IF it says
`SOME CHECKS FAILED`, or if the first command cannot reach AWS (see Troubleshooting). Afterwards move the test folder
aside: `mv $AORC_ROOT/aorcfix2_test $AORC_ROOT/_aorcfix2_test_done`.

**2b. Full run** (about 1 h with 4 workers; it downloads about 90 GB and writes about 130 GB):

```bash
caffeinate -i python $FIX/make_aorcfix2.py $SRC $PREV $DST --years 2002 2023 --workers 4 \
    2>&1 | tee $AORC_ROOT/aorcfix_logs/step2_full.log
```

`caffeinate -i` keeps the Mac awake. Expected output:
- `... GB free; building ...` and `symlinks created: 1093` (8035 − 6942; more if `$SRC` holds other years);
- one line per year, `2002: ... s (v1.1-missing cells kept from original: N)` up to 2020, plus 2021, in any order
  (N is in the tens of thousands for 2002–2009 and 0 from 2010 on);
- `done`.

The run is resumable: each finished year leaves a marker in `$DST/_aorcfix2_done/`. If it stops (network, sleep,
Ctrl-C), run the same command again; finished years are skipped and unfinished ones are redone from the originals.
STOP IF it exits with `Only ... GB free` (free space first; do not lower `--min-free-gb`).

### Step 3. Verify (about 15 min)

```bash
python $FIX/check_aorcfix2.py $SRC $PREV $DST 2>&1 | tee $AORC_ROOT/aorcfix_logs/step3_check.log
diff <(cut -f1,3 $DST/_aorcfix2_manifest.tsv) <(cut -f1,3 $FIX/reference/aorcfix2_manifest_reference.tsv) \
    && echo "MANIFEST IDENTICAL TO REFERENCE"
```

Expected: the same lines as our reference, `$FIX/reference/check_aorcfix2_reference.log`:

```
PASS  C1 completeness  0 source files without an entry in dst
PASS  C1b patched days are real files with the aorcfix2 attribute  0 bad []
PASS  C2 23:00 no longer a copy of 22:00  0 days still identical []
PASS  C3 patched hour == NOAA v1.1  max |diff| 0.0250 mm over 20 sampled days
PASS  C3b other hours and TMP byte-identical to the source  []
PASS  C4a 2021-12-31 != 2021-12-30 and == v1.1  max |diff| vs v1.1 0.0031
PASS  C4b 2011-07-23 00:00 not all missing  valid fraction 0.641
PASS  C4c 2022-05-29 20:00 not uniform  std 0.422
C5 manifest: 6942 patched files -> .../_aorcfix2_manifest.tsv
ALL PASS
```

and then `MANIFEST IDENTICAL TO REFERENCE`. The manifest's third column is a checksum of the patched rain values of
each of the 6,942 files; it is the same on every machine (our Mac and Ubuntu builds matched 6942/6942). The second
column, a checksum of the whole file, depends on the HDF5 library and may differ; that is expected.

STOP IF any line is FAIL or `diff` prints lines. A `diff` line names a day whose patched rain differs from ours:
report the days, and check that the original file of that day is the same as GP's.

### Step 4 (optional). View without August 2017 (runs without Hurricane Harvey)

```bash
bash $FIX/make_noaug_view.sh $DST $NH | tee $AORC_ROOT/aorcfix_logs/step4_noaug.log
```

Expected: `.../data_aorcfix2_no_aug2017: 8004 files (8004 new links), 0 files of August 2017 (must be 0)`
(8035 − 31; more if `$SRC` holds other years).

### Step 5. Final check of the folders

```bash
python $FIX/check_defects.py $DST                 # must print False four times
python $FIX/check_defects.py $NH                  # only if you made step 4; False four times
ls $DST | grep -c 'preciptemp.nc$'                # 8035 entries ...
find $DST -maxdepth 1 -type f -name 'AORC.*' | wc -l    # ... of which 6942 are real files
```

(More entries if `$SRC` holds other years; the real files are always 6942.)

## 5. Using the corrected folders in RainyDay_fast

1. Point RAINPATH at the corrected view, with Harvey or without:
   `"RAINPATH": "<AORC_ROOT>/data_aorcfix2/AORC.*.preciptemp.nc"` or `".../data_aorcfix2_no_aug2017/AORC.*.preciptemp.nc"`.
   RainyDay follows the links.
2. Make **new** crops from that RAINPATH, in a folder named after the data, so they cannot be mixed with crops of the
   original files:
   ```bash
   conda activate RainyDay_Env
   python tools/make_domain_crop.py --json RUN.json --dst <AORC_ROOT>/crops/<domain>_aorcfix2 --verify
   ```
   and run with the `RUN_crop.json` it writes ([README](../../README.md), "Using RainyDay_fast for a new domain").
3. Rebuild storm catalogs. Never reuse crops or catalogs made from the original files. A light catalog built on other
   data stops with "doesn't match the catalog (CRC32): wrong or changed input data": that means mixed data, not a bug.

## 6. Troubleshooting

| Problem | What to do |
|---|---|
| S3 errors, timeouts or a hang while downloading | Ctrl-C and rerun the same command (step 2b is resumable). Check that `python -c "import s3fs; print(s3fs.S3FileSystem(anon=True).ls('noaa-nws-aorc-v1-1-1km')[:3])"` lists the bucket. A VPN or a campus proxy can block AWS. |
| `Only ... GB free` | Free space on the disk of `$AORC_ROOT` or choose another disk. Do not lower `--min-free-gb`: the result needs about 130 GB. |
| `AssertionError` about latitude or longitude | Your files are not on the RainyDay 1/30° AORC grid (885 × 1770). Stop and report. |
| `zarr` errors such as `S3Map` not found | zarr 3 was installed. Recreate the env from `aorcfix_env.yml`. |
| `readlink: illegal option -- f` | macOS before 12.3; the script falls back to python3 automatically. |
| Broken links (`No such file` when RainyDay opens a file) | The original folder was moved or renamed. Move it back, or build the views again into new folders. |
| Very slow disk, Spotlight busy | Exclude `$AORC_ROOT` from Spotlight (System Settings > Siri & Spotlight > Spotlight Privacy) and from Time Machine. |
| The 4 defect lines of step 0.4 do not all print `True` | Your copy is not the one we fixed. Tell GP before continuing. |

## 7. Files in this folder

| File | What it is |
|---|---|
| `make_aorcfix.py` | Step 1: links + the two interpolated days |
| `make_aorcfix2.py` | Step 2: links + days patched from NOAA AORC v1.1 (resumable, parallel per year) |
| `check_aorcfix2.py` | Step 3: checks C1–C5 and the manifest |
| `check_defects.py` | Steps 0.4 and 5: quick test of a folder for the four defects |
| `make_noaug_view.sh` | Step 4: the view without August 2017 |
| `aorcfix_env.yml` | conda env `aorcfix` (Python 3.10, zarr 2.18.3, s3fs, xarray, netCDF4) |
| `reference/aorcfix2_manifest_reference.tsv` | Our manifest (name, file md5, rain-data md5) for the 6,942 patched files |
| `reference/check_aorcfix2_reference.log` | Our output of step 3 |

Scripts by Gabriel Perez (October 2026), first used for the Basin-IDF study; they were run on macOS and Ubuntu with
identical patched data.
