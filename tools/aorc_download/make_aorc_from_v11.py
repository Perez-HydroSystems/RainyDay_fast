"""
Build RainyDay-format AORC daily files (AORC.YYYYMMDD.preciptemp.nc) from NOAA AORC v1.1 on the AWS Open Data registry
(s3://noaa-nws-aorc-v1-1-1km/<year>.zarr, anonymous access). Use it for the days that web.s3.wisc.edu/aorc does not
have (2024 onward). Instructions: AORC_DOWNLOAD_INSTRUCTIONS.md in this folder.

    python make_aorc_from_v11.py --start 20240101 --end 20251231 --dst DIR --template FILE [--workers 6]
    python make_aorc_from_v11.py --days 20231230 20231231 --dst DIR --template FILE          (test mode)

FILE is any existing RainyDay-format AORC daily file of 2022 or 2023 (e.g. AORC.20231231.preciptemp.nc). The new files
copy its grid, encoding and attributes. It is only read.

Chain (bit-identical to the web.s3.wisc.edu/aorc files on the test days of 2022-2023):
  * grid: nearest-neighbour subsample of the v1.1 30" grid, RainyDay cell (i, j) = v1.1 (424 + 4 i, 544 + 4 j)
    (the mapping of tools/aorc_fix/make_aorcfix2.py); latitude/longitude copied from the template file and asserted
    equal to the v1.1 coordinates;
  * time: hour-ending stamps 01:00 .. 24:00 (= next day 00:00) UTC, v1.1 stamp s -> file stamp s, seconds since
    1970-01-01;
  * RAINRATE (mm/h) = APCP_surface (int16, x 0.1 kg/m2) rounded to the nearest 1/16 mm; TMP_2maboveground (K) =
    TMP (int16 x 0.1) rounded to the nearest 1/16 K;
  * NaN where v1.1 is missing (-32767) or where the existing files are always NaN (static coverage mask,
    coverage_mask_nan.npz in this folder; --mask to use another one);
  * encoding copied from the template: netCDF-4 classic, chunks, shuffle, deflate levels, attributes.
A v1.1 hour that does not exist (e.g. the first hour after the end of the archive) is written as NaN everywhere and
reported. Files are written to <name>.tmp and renamed, so an existing file is complete; existing files are skipped
(resumable: rerun the same command).
"""
import argparse
import os
import sys
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

NI0, NJ0, STEP, NLAT, NLON = 424, 544, 4, 885, 1770
CHUNK_T = 144
STAMP = "RainyDay_fast tools/aorc_download/make_aorc_from_v11.py"
DEFAULT_MASK = Path(__file__).resolve().parent / "coverage_mask_nan.npz"
_DS = {}


def load_mask(path):
    """Static coverage mask (True = always missing), 885 x 1770, from a .npz (key 'mask') or a .npy file."""
    m = np.load(path)
    m = m["mask"] if hasattr(m, "files") else m
    assert m.dtype == bool and m.shape == (NLAT, NLON), f"{path}: not a boolean {NLAT} x {NLON} mask"
    return m


def open_v11(year):
    if year not in _DS:
        import s3fs
        import xarray as xr
        fs = s3fs.S3FileSystem(anon=True)
        if not fs.exists(f"noaa-nws-aorc-v1-1-1km/{year}.zarr/.zmetadata"):
            ds = None                                        # year not in the archive (e.g. 2026)
        else:
            ds = xr.open_zarr(s3fs.S3Map(f"noaa-nws-aorc-v1-1-1km/{year}.zarr", s3=fs),
                              consolidated=True, mask_and_scale=False)
        _DS[year] = ds
    return _DS[year]


def read_hours(h0, h1):
    """v1.1 APCP and TMP (int16, RainyDay grid) for hour stamps h0..h1 inclusive; missing hours -> all -32767.
    Returns (stamps, apcp, tmp, missing_stamps)."""
    stamps = pd.date_range(h0, h1, freq="h")
    ap = np.full((len(stamps), NLAT, NLON), -32767, np.int16)
    tp = np.full((len(stamps), NLAT, NLON), -32767, np.int16)
    missing = []
    for y in sorted(set(stamps.year)):
        sel = np.where(stamps.year == y)[0]
        ds = open_v11(y)
        if ds is None:
            missing += list(stamps[sel]); continue
        t = pd.DatetimeIndex(ds.time.values)
        pos = t.get_indexer(stamps[sel])
        ok = pos >= 0
        missing += list(stamps[sel][~ok])
        if not ok.any():
            continue
        p = pos[ok]
        assert np.all(np.diff(p) == 1)
        k0, k1 = int(p[0]), int(p[-1]) + 1
        isel = dict(time=slice(k0, k1), latitude=slice(NI0, NI0 + STEP * NLAT, STEP),
                    longitude=slice(NJ0, NJ0 + STEP * NLON, STEP))
        for _ in range(5):                                   # retry transient S3 errors
            try:
                a = ds.APCP_surface.isel(**isel).values
                b = ds.TMP_2maboveground.isel(**isel).values
                break
            except Exception as e:                           # noqa
                print(f"    retry {y} {k0}:{k1}: {e}", flush=True); time.sleep(20)
        else:
            raise RuntimeError(f"S3 read failed {y} {k0}:{k1}")
        ap[sel[ok]] = a; tp[sel[ok]] = b
    return stamps, ap, tp, missing


def check_coords(tlat, tlon):
    ds = open_v11(2024)
    vlat = ds.latitude.values[NI0:NI0 + STEP * NLAT:STEP]
    vlon = ds.longitude.values[NJ0:NJ0 + STEP * NLON:STEP]
    assert np.array_equal(tlat, vlat) and np.array_equal(tlon, vlon), "template grid != v1.1 subsample"


def q16(a16, scale, mask):
    out = (np.round(a16.astype(np.float64) * (scale * 16.0)) / 16.0).astype(np.float32)
    out[(a16 == -32767) | mask[None]] = np.nan
    return out


def write_day(path, tmpl, times_s, rain, tmp, note):
    tmpf = path.with_suffix(".nc.tmp")
    if tmpf.exists():
        tmpf.unlink()
    with nc.Dataset(tmpl) as t, nc.Dataset(tmpf, "w", format=t.data_model) as o:
        for d in t.dimensions.values():
            o.createDimension(d.name, len(d))
        for name, v in t.variables.items():
            filt = v.filters()
            ch = v.chunking()
            o.createVariable(name, v.dtype, v.dimensions, zlib=bool(filt.get("zlib")), complevel=filt.get("complevel", 0),
                             shuffle=bool(filt.get("shuffle")), fletcher32=bool(filt.get("fletcher32")),
                             chunksizes=None if ch == "contiguous" else ch, endian=v.endian(),
                             fill_value=v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else False)
            ov = o.variables[name]
            ov.set_auto_mask(False)
            ov.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
        ga = {k: t.getncattr(k) for k in t.ncattrs()}
        ga["history"] = f"Created {datetime.now()}"
        ga["source"] = ("NOAA AORC v1.1, s3://noaa-nws-aorc-v1-1-1km (AWS Open Data), subsampled and quantized to "
                        "match the web.s3.wisc.edu/aorc files")
        ga["aorc_v11_build"] = f"{STAMP}: {note}"
        o.setncatts(ga)
        tv = t.variables
        o.variables["latitude"][:] = tv["latitude"][:]
        o.variables["longitude"][:] = tv["longitude"][:]
        o.variables["time"][:] = times_s
        o.variables["RAINRATE"][:] = rain
        o.variables["TMP_2maboveground"][:] = tmp
    os.replace(tmpf, path)


def do_segment(args):
    days, dst, tmpl, mask_path = args
    mask = load_mask(mask_path)
    done, log = 0, []
    todo = [d for d in days if not (dst / f"AORC.{d:%Y%m%d}.preciptemp.nc").exists()]
    if not todo:
        return days[0], 0, ["all done already"]
    # read by v1.1 time chunks: hour range of the segment split at chunk boundaries
    h_first = todo[0] + pd.Timedelta(hours=1)
    h_last = todo[-1] + pd.Timedelta(hours=24)
    buf_st, buf_a, buf_t = [], [], []
    cur = h_first
    todo_set = list(pd.date_range(todo[0], todo[-1], freq="D"))   # contiguous; existing files are not rewritten
    while cur <= h_last:
        y0 = pd.Timestamp(f"{cur.year}-01-01")
        k = int((cur - y0) / pd.Timedelta(hours=1))
        end_chunk = y0 + pd.Timedelta(hours=(k // CHUNK_T + 1) * CHUNK_T - 1)
        h1 = min(end_chunk, h_last, pd.Timestamp(f"{cur.year}-12-31 23:00"))
        st, a, b, miss = read_hours(cur, h1)
        if miss:
            log.append(f"v1.1 hours missing (written as NaN): {[str(x) for x in miss]}")
        buf_st += list(st); buf_a.append(a); buf_t.append(b)
        A = np.concatenate(buf_a); T = np.concatenate(buf_t); buf_a, buf_t = [A], [T]
        cur = h1 + pd.Timedelta(hours=1)
        # emit complete days
        while todo_set and len(buf_st) >= 24 and buf_st[0] == todo_set[0] + pd.Timedelta(hours=1):
            d = todo_set.pop(0)
            if len(buf_st) < 24:
                break
            stamps = buf_st[:24]
            assert stamps[-1] == d + pd.Timedelta(hours=24)
            rain = q16(buf_a[0][:24], 0.1, mask)
            tmp = q16(buf_t[0][:24], 0.1, mask)
            secs = np.array([(s - pd.Timestamp("1970-01-01")).total_seconds() for s in stamps], np.float64)
            missing_h = [str(s) for s, x in zip(stamps, buf_a[0][:24]) if (x == -32767).all()]
            note = (f"hours {stamps[0]:%Y-%m-%dT%H:%M}..{stamps[-1]:%Y-%m-%dT%H:%M} from NOAA AORC v1.1"
                    + (f"; v1.1 hours not available, written as NaN: {missing_h}" if missing_h else ""))
            out = dst / f"AORC.{d:%Y%m%d}.preciptemp.nc"
            if not out.exists():
                write_day(out, tmpl, secs, rain, tmp, note)
                done += 1
            buf_st = buf_st[24:]; buf_a = [buf_a[0][24:]]; buf_t = [buf_t[0][24:]]
    return days[0], done, log


def main():
    ap = argparse.ArgumentParser(description="RainyDay-format AORC daily files from NOAA AORC v1.1 (AWS)")
    ap.add_argument("--start", help="first day, YYYYMMDD"); ap.add_argument("--end", help="last day, YYYYMMDD")
    ap.add_argument("--dst", required=True, help="folder for the new files (created if needed)")
    ap.add_argument("--template", required=True, help="an existing RainyDay-format AORC daily file of 2022 or 2023")
    ap.add_argument("--mask", default=str(DEFAULT_MASK), help="static coverage mask (default: the one in this folder)")
    ap.add_argument("--days", nargs="*", default=None, help="only these YYYYMMDD days (test mode)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seg-days", type=int, default=48, help="days per work segment")
    ap.add_argument("--min-free-gb", type=float, default=25.)
    a = ap.parse_args()
    if not a.days and not (a.start and a.end):
        ap.error("give --start and --end, or --days")
    dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    import shutil
    free = shutil.disk_usage(dst).free / 1e9
    if free < a.min_free_gb:
        sys.exit(f"Only {free:.0f} GB free on {dst}")
    mask = load_mask(a.mask)
    with nc.Dataset(a.template) as t:
        t.set_auto_mask(False)
        check_coords(t["latitude"][:], t["longitude"][:])
        r0 = t["RAINRATE"][0]
        same = np.array_equal(np.isnan(r0) | (r0 == -9999), mask)
    print(f"template grid = v1.1 subsample: True; template missing cells = coverage mask: {same}", flush=True)
    if not same:
        print("  WARNING: the first hour of the template is not missing exactly on the coverage mask. Use a file of "
              "2022 or 2023 as template (not 2022-10-31 or 2022-12-31).", flush=True)
    if a.days:
        segs = [[pd.Timestamp(d)] for d in a.days]
    else:
        days = list(pd.date_range(a.start, a.end, freq="D"))
        segs = [days[i:i + a.seg_days] for i in range(0, len(days), a.seg_days)]
    print(f"{free:.0f} GB free; {sum(map(len, segs))} days in {len(segs)} segments -> {dst}", flush=True)
    jobs = [(s, dst, a.template, a.mask) for s in segs]
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        for d0, n, log in pool.imap_unordered(do_segment, jobs):
            print(f"  segment {d0:%Y-%m-%d}: {n} files written ({time.time() - t0:.0f} s) {'; '.join(log)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
