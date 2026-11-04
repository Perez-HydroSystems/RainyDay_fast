"""
GP 2026-11-04: quick test of an AORC folder for the four defects fixed by make_aorcfix.py + make_aorcfix2.py.

    python check_defects.py <aorc_folder>

Prints True four times for the original RainyDay-format files and False four times for a corrected view
(data_aorcfix2 or data_aorcfix2_no_aug2017). Same test as the User's Guide, "Data issues (AORC)".
"""
import sys

import netCDF4 as nc
import numpy as np

folder = sys.argv[1]


def rain(day):   # the 24 hourly fields of one daily file, 01:00 ... 24:00 UTC
    with nc.Dataset(f"{folder}/AORC.{day}.preciptemp.nc") as f:
        v = f.variables["RAINRATE"]
        v.set_auto_mask(False)
        return np.asarray(v[:])


r = rain("20100715")
print("23:00 is a copy of 22:00:   ", np.array_equal(r[21], r[22], equal_nan=True))
print("20211231 is a copy of 1230: ", np.array_equal(rain("20211230"), rain("20211231"), equal_nan=True))
h = rain("20220529")[19]
print("2022-05-29 20:00 is uniform:", bool(np.nanmin(h) == np.nanmax(h)))
print("2011-07-23 00:00 is missing:", bool(np.all(rain("20110722")[23] < 0)))
