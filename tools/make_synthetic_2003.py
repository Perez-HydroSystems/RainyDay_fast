#!/usr/bin/env python
"""
Build a synthetic 2-year AORC record (2002 + 2003) for testing the storm-catalog
engines on a longer record than the single 2002 year staged on this machine.

2003 files are byte copies of the 2002 files with the internal `time` coordinate
shifted forward by exactly 365 days (both years are non-leap, so 2002-mm-dd maps
cleanly to 2003-mm-dd). Rain data is identical to 2002. This is NOT physically a
new year; it exists only to (a) double the record length and (b) raise the catalog
minimum (each top storm now has an identical twin one year later), which shrinks the
GPU exact-confirm recompute fraction so the multi-year regime can be measured.

Output: C:/Test_RainyDay/AORC_2yr/ with 730 files (2002 hardlinked, 2003 generated).
Run once: python tools/make_synthetic_2003.py
"""
import glob
import os
import shutil
import sys

import netCDF4
import numpy as np

SRC = r"C:/Test_RainyDay/AORC_2002"
OUT = r"C:/Test_RainyDay/AORC_2yr"
SHIFT_MIN = 365 * 24 * 60  # 525600 minutes = 365 days, in the files' "minutes since 1970" units


def main():
    os.makedirs(OUT, exist_ok=True)
    src_files = sorted(glob.glob(os.path.join(SRC, "AORC.2002*.preciptemp.nc")))
    if not src_files:
        sys.exit("no 2002 source files found in " + SRC)
    print("found {0} source files".format(len(src_files)))

    for i, f in enumerate(src_files):
        base = os.path.basename(f)                    # AORC.20020101.preciptemp.nc
        # 2002 twin: hardlink into the combined dir (no extra disk)
        link2002 = os.path.join(OUT, base)
        if not os.path.exists(link2002):
            os.link(f, link2002)
        # 2003 copy with shifted time
        base2003 = base.replace("AORC.2002", "AORC.2003", 1)
        dst = os.path.join(OUT, base2003)
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
            nc = netCDF4.Dataset(dst, "r+")
            try:
                t = nc.variables["time"]
                t[:] = t[:] + SHIFT_MIN
            finally:
                nc.close()
        if (i + 1) % 60 == 0 or i + 1 == len(src_files):
            print("  {0}/{1} done".format(i + 1, len(src_files)))

    n = len(glob.glob(os.path.join(OUT, "AORC.*.preciptemp.nc")))
    print("combined dir has {0} files".format(n))

    # verify a sample 2003 file decodes to 2003 dates
    import xarray as xr
    sample = os.path.join(OUT, "AORC.20030715.preciptemp.nc")
    ds = xr.open_dataset(sample)
    print("sample", os.path.basename(sample), "time[0]:", str(ds["time"].values[0]),
          "time[-1]:", str(ds["time"].values[-1]))
    assert str(ds["time"].values[0]).startswith("2003-07-15"), "time shift wrong!"
    ds.close()
    print("OK: synthetic 2003 verified")


if __name__ == "__main__":
    main()
