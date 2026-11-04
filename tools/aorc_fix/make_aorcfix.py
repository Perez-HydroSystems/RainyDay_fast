"""
GP 2026-10-20: build a corrected AORC view for RainyDay without touching the original files.

NOAA AORC (v1.0 as distributed by web.s3.wisc.edu/aorc and v1.1 on AWS) contains corrupt hours in which every CONUS cell
has the same value, e.g. 2022-05-29 20:00: RAINRATE = 22.875 mm/h and TMP = 290.87 K everywhere.
For each corrupt hour the field is replaced by the linear interpolation in time of the previous and next hour.

    python make_aorcfix.py <src_dir> <dst_dir> "YYYY-MM-DDTHH:MM" [...]
dst_dir gets one symlink per source file, except the days with corrupt hours, which get a corrected copy
(same variables, attributes and encoding, plus an attribute documenting the fix).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
bad = sorted(pd.Timestamp(t) for t in sys.argv[3:])
dst.mkdir(parents=True, exist_ok=True)


def day_file(ts):
    # file AORC.YYYYMMDD holds hours 01:00 ... 24:00 (= next day 00:00) of that day
    d = (ts - pd.Timedelta(minutes=1)).strftime("%Y%m%d")
    return src / f"AORC.{d}.preciptemp.nc"


def field(ts, var):
    with xr.open_dataset(day_file(ts)) as ds:
        return ds[var].sel(time=ts).values


fix_files = {}
for ts in bad:
    fix_files.setdefault(day_file(ts).name, []).append(ts)

for f in sorted(src.glob("AORC.*.preciptemp.nc")):
    link = dst / f.name
    if f.name in fix_files or link.exists() or link.is_symlink():
        continue
    os.symlink(os.path.realpath(f), link)

for name, hours in fix_files.items():
    ds = xr.open_dataset(src / name).load()
    enc = {v: {k: ds[v].encoding[k] for k in ("zlib", "complevel", "shuffle", "chunksizes", "_FillValue", "dtype") if k in ds[v].encoding}
           for v in ds.data_vars}
    notes = []
    for ts in hours:
        prev_ts, next_ts = ts - pd.Timedelta(hours=1), ts + pd.Timedelta(hours=1)
        for v in ("RAINRATE", "TMP_2maboveground"):
            before = np.array(ds[v].sel(time=ts).values, copy=True)
            new = 0.5 * (field(prev_ts, v) + field(next_ts, v))
            ds[v].loc[dict(time=ts)] = new
            notes.append(f"{v} {ts:%Y-%m-%dT%H:%M}: was uniform {np.nanmean(before):.3f} (std {np.nanstd(before):.3f}); "
                         f"replaced by mean of {prev_ts:%H:%M} and {next_ts:%H:%M}")
    ds.attrs["aorcfix"] = "GP 2026-10-20 corrupt-hour fix (make_aorcfix.py): " + " | ".join(notes)
    out = dst / name
    if out.is_symlink() or out.exists():
        out.unlink()
    ds.to_netcdf(out, encoding=enc)
    print(name, "->", out, "\n  " + "\n  ".join(notes))
print(f"{dst}: {len(list(dst.glob('*.nc')))} files ({len(fix_files)} corrected copies)")
