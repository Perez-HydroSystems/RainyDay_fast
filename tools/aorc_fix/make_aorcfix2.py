"""
GP 2026-10-21: corrected AORC view for RainyDay (v2): fixes the RainyDay-format AORC repackaging defects with NOAA AORC
v1.1 values, without touching the original files.

    python make_aorcfix2.py <src_dir> <prev_fix_dir> <dst_dir> [--years 2002 2023] [--workers 4] [--min-free-gb 160]

src_dir       original daily files AORC.YYYYMMDD.preciptemp.nc (RAINRATE mm/h, 0.033332 deg, hours 01:00..24:00 UTC)
prev_fix_dir  the earlier corrected view (data_aorcfix, make_aorcfix.py): its two repaired days are the starting point
              for those days (2011-07-22, 2022-05-29); everything else is read from src_dir
dst_dir       new view: patched copies + symlinks to src_dir for every untouched file (complete drop-in RAINPATH)

Fixes (RAINRATE only; TMP has no copied hour):
  1. every day of the selected years up to 2020: the hour stamped 23:00 UTC is an exact copy of 22:00 in the
     RainyDay-format files -> replaced by NOAA AORC v1.1 23:00 at the same cells
  2. 2021-12-31 is a byte copy of 2021-12-30 -> all 24 hours replaced by v1.1 (01:00 .. next-day 00:00)
  3. 2011-07-23 00:00 (all -9999, inside file 20110722) -> v1.1 value (if v1.1 is valid there), replacing the earlier
     linear interpolation (which had used the copied 23:00 hour)
  4. 2022-05-29 20:00 (uniform 22.875 mm/h, corrupt in v1.1 too) -> kept as interpolated in prev_fix_dir
Source: s3://noaa-nws-aorc-v1-1-1km/<year>.zarr (anonymous), APCP_surface int16 x 0.1 mm, fill -32767. The RainyDay grid
is a nearest-neighbour subsample of the native 30" grid: cell (i, j) = native (424 + 4 i, 544 + 4 j) (asserted).
Cells that are missing (< 0) in the original keep their missing value; if v1.1 is missing where the original is valid,
the original value is kept (counted and reported).
Each patched file: byte copy of its source, then the patched hours written in place (netCDF4 r+, same variable,
attributes and encoding), plus a global attribute `aorcfix2` documenting the change. Resumable per year
(dst_dir/_aorcfix2_done/<year>); parallel per year.
"""
import argparse
import os
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

NI0, NJ0, STEP = 424, 544, 4          # RainyDay grid -> native v1.1 indices
NLAT, NLON = 885, 1770
STAMP = "GP 2026-10-21 make_aorcfix2.py"


def file_times(path):
    with nc.Dataset(path) as f:
        tv = f.variables["time"]
        return pd.to_datetime([x.isoformat() for x in nc.num2date(tv[:], tv.units, only_use_cftime_datetimes=False)])


def open_v11(year):
    import s3fs
    import xarray as xr
    fs = s3fs.S3FileSystem(anon=True)
    ds = xr.open_zarr(s3fs.S3Map(f"noaa-nws-aorc-v1-1-1km/{year}.zarr", s3=fs), consolidated=True, mask_and_scale=False)
    lat, lon = ds.latitude.values, ds.longitude.values
    assert abs(lat[NI0] - 23.533192) < 1e-4 and abs(lon[NJ0] - (-125.466848)) < 1e-4, (lat[NI0], lon[NJ0])
    assert abs(lat[NI0 + STEP * (NLAT - 1)] - 52.99868) < 1e-3 and abs(lon[NJ0 + STEP * (NLON - 1)] - (-66.50254)) < 1e-3
    return ds


def v11_fields(ds, stamps):
    """v1.1 precipitation (mm/h, float32, NaN = missing) on the RainyDay grid for the given hour-ending stamps."""
    t = pd.to_datetime(ds.time.values)
    idx = [int(np.where(t == s)[0][0]) for s in stamps]
    da = ds.APCP_surface.isel(time=idx, latitude=slice(NI0, NI0 + STEP * NLAT, STEP),
                              longitude=slice(NJ0, NJ0 + STEP * NLON, STEP))
    a = np.asarray(da.values).astype(np.float32)
    out = np.where(a <= -32000, np.nan, a * np.float32(0.1)).astype(np.float32)
    assert out.shape[1:] == (NLAT, NLON)
    return out


def patch_file(src, dst, stamps, new, note):
    """Copy src -> dst (real file) and write the given stamps' RAINRATE fields. Returns (n_kept_orig, n_cells)."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    shutil.copy2(os.path.realpath(src), dst)
    t = file_times(dst)
    kept = 0
    with nc.Dataset(dst, "r+") as f:
        v = f.variables["RAINRATE"]
        v.set_auto_mask(False)
        nanfill = "_FillValue" in v.ncattrs() and np.isnan(v.getncattr("_FillValue"))
        for s, field in zip(stamps, new):
            k = int(np.where(t == s)[0][0])
            orig = np.asarray(v[k], np.float32)
            missing = ~np.isfinite(orig) | (orig < 0)
            bad_new = ~np.isfinite(field) & ~missing
            kept += int(bad_new.sum())
            out = np.where(bad_new, orig, field)
            out = np.where(missing, np.nan if nanfill else orig, out)
            v[k] = out.astype(np.float32)
        f.setncattr("aorcfix2", f"{STAMP}: {note}")
    return kept


def do_year(args):
    year, src, prev, dst, y_last_dup = args
    marker = dst / "_aorcfix2_done" / str(year)
    if marker.exists():
        return year, "skipped (done)"
    t0 = time.time()
    ds = open_v11(year)
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    log = []
    kept_tot = 0
    if year <= y_last_dup:                                  # fix 1: the copied 23:00 hour, every day
        for b in range(0, len(days), 6):                    # 6 days = one v1.1 time chunk (144 h)
            block = days[b:b + 6]
            stamps = [d + pd.Timedelta(hours=23) for d in block]
            fields = v11_fields(ds, stamps)
            for d, s, fld in zip(block, stamps, fields):
                name = f"AORC.{d:%Y%m%d}.preciptemp.nc"
                srcf = prev / name if (prev / name).exists() and not (prev / name).is_symlink() else src / name
                if not srcf.exists():
                    continue
                extra_stamps, extra = [], []
                if d == pd.Timestamp("2011-07-22"):         # fix 3: 2011-07-23 00:00 (all -9999) from v1.1
                    s3 = pd.Timestamp("2011-07-23 00:00")
                    f3 = v11_fields(ds, [s3])[0]
                    if np.isfinite(f3).sum() > 0.3 * f3.size:
                        extra_stamps, extra = [s3], [f3]
                kept_tot += patch_file(srcf, dst / name, [s] + extra_stamps, [fld] + extra,
                                       "RAINRATE at " + ", ".join(f"{x:%Y-%m-%dT%H:%M}" for x in [s] + extra_stamps)
                                       + " replaced by NOAA AORC v1.1 (23:00 was a copy of 22:00 in the RainyDay-format files)")
    if year == 2021:                                        # fix 2: 2021-12-31 duplicated 2021-12-30
        stamps = list(pd.date_range("2021-12-31 01:00", "2021-12-31 23:00", freq="h"))
        f21 = v11_fields(ds, stamps)
        f22 = v11_fields(open_v11(2022), [pd.Timestamp("2022-01-01 00:00")])
        name = "AORC.20211231.preciptemp.nc"
        kept_tot += patch_file(src / name, dst / name, stamps + [pd.Timestamp("2022-01-01 00:00")],
                               list(f21) + list(f22), "all 24 hours replaced by NOAA AORC v1.1 (the file was a copy of 2021-12-30)")
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(f"{time.time() - t0:.0f} s, v1.1-missing cells kept from original: {kept_tot}\n")
    return year, f"{time.time() - t0:.0f} s (v1.1-missing cells kept from original: {kept_tot})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("prev"); ap.add_argument("dst")
    ap.add_argument("--years", nargs=2, type=int, default=[2002, 2023])
    ap.add_argument("--last-dup-year", type=int, default=2020)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=160.)
    ap.add_argument("--only-days", nargs="*", default=None, help="test mode: patch only these YYYYMMDD days")
    a = ap.parse_args()
    src, prev, dst = Path(a.src), Path(a.prev), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dst).free / 1e9
    need = 0 if a.only_days else a.min_free_gb
    if free < need:
        sys.exit(f"Only {free:.0f} GB free on the disk of {dst}; need >= {need:.0f} GB. Free space or pass --min-free-gb.")
    print(f"{free:.0f} GB free; building {dst} from {src} (+ repaired days of {prev}), years {a.years}", flush=True)
    y0, y1 = a.years
    if a.only_days:                                         # test mode: a few days, no symlinks, no markers
        for day in a.only_days:
            d = pd.Timestamp(day); ds = open_v11(d.year)
            name = f"AORC.{d:%Y%m%d}.preciptemp.nc"
            srcf = prev / name if (prev / name).exists() and not (prev / name).is_symlink() else src / name
            if name == "AORC.20211231.preciptemp.nc":
                stamps = list(pd.date_range("2021-12-31 01:00", "2021-12-31 23:00", freq="h"))
                f = list(v11_fields(ds, stamps)) + list(v11_fields(open_v11(2022), [pd.Timestamp("2022-01-01 00:00")]))
                stamps.append(pd.Timestamp("2022-01-01 00:00"))
            else:
                stamps = [d + pd.Timedelta(hours=23)]; f = list(v11_fields(ds, stamps))
                if day == "20110722":
                    stamps.append(pd.Timestamp("2011-07-23 00:00")); f.append(v11_fields(ds, [stamps[-1]])[0])
            k = patch_file(srcf, dst / name, stamps, f, "TEST " + ", ".join(f"{x:%Y-%m-%dT%H:%M}" for x in stamps))
            print(f"  {name}: patched {len(stamps)} h, v1.1-missing kept {k}", flush=True)
        return
    # symlinks for every source file that will not be patched (all years, so the view is complete)
    patched_names = set()
    for y in range(y0, min(y1, a.last_dup_year) + 1):
        patched_names |= {f"AORC.{d:%Y%m%d}.preciptemp.nc" for d in pd.date_range(f"{y}-01-01", f"{y}-12-31")}
    if y0 <= 2021 <= y1:
        patched_names.add("AORC.20211231.preciptemp.nc")
    nlink = 0
    for f in sorted(src.glob("AORC.*.preciptemp.nc")):
        out = dst / f.name
        if f.name in patched_names and (out.exists() and not out.is_symlink()):
            continue
        if f.name in patched_names:
            continue
        srcf = prev / f.name if (prev / f.name).exists() and not (prev / f.name).is_symlink() else f
        if out.is_symlink() or out.exists():
            continue
        if srcf != f:                                       # an earlier repaired day not patched now (2022-05-29)
            shutil.copy2(srcf, out)
        else:
            os.symlink(os.path.realpath(f), out)
            nlink += 1
    print(f"symlinks created: {nlink}", flush=True)
    jobs = [(y, src, prev, dst, a.last_dup_year) for y in range(y0, y1 + 1) if y <= a.last_dup_year or y == 2021]
    with get_context("spawn").Pool(a.workers) as pool:
        for y, msg in pool.imap_unordered(do_year, jobs):
            print(f"  {y}: {msg}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
