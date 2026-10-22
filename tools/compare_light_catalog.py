#!/usr/bin/env python
"""
GP 2026-10-22: RainyDay_fast Fix 2 acceptance check: a light storm catalog must give, for EVERY storm, exactly what
RainyDay.readcatalog gives on the storm file of a full catalog.

    python tools/compare_light_catalog.py <full StormCatalog dir> <light catalog .nc> [--storms N ...]

Per storm (fast_io.readstorm on the virtual path vs RainyDay.readcatalog on the storm file), bitwise:
  rain (values, dtype, dims, attributes, time/latitude/longitude coordinates), storm time, latitude, longitude,
  xlocation, ylocation, basinrainfall, gridmask (values, dtype, dims, attributes), domainmask, cattime of all storms,
  time resolution (value and type); plus: same storm file names, stored CRC32 == CRC32 of the full storm's rain.
Prints PASS/FAIL per storm and ALL PASS at the end; exit 0 = ALL PASS.
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Source"))
import RainyDay_utilities_Py3.RainyDay_functions as RD   # noqa: E402
import RainyDay_utilities_Py3.fast_io as FIO              # noqa: E402

STORM_RE = re.compile(r"_storm_(\d+)_(\d{8})\.nc$")


def same(x, y):
    """Bitwise equality of values and dtype (floats compared through their bit patterns)."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.dtype.kind == 'f':
        u = np.dtype('u%d' % x.dtype.itemsize)
        return np.array_equal(np.ascontiguousarray(x).view(u), np.ascontiguousarray(y).view(u))
    return np.array_equal(x, y)


def same_da(a, b, coords=True):
    ok = same(a.values, b.values) and a.dims == b.dims and a.name == b.name and dict(a.attrs) == dict(b.attrs)
    if coords:
        ok = ok and set(a.coords) == set(b.coords) and all(same(a.coords[c].values, b.coords[c].values)
                                                           and dict(a.coords[c].attrs) == dict(b.coords[c].attrs)
                                                           for c in a.coords)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fulldir"); ap.add_argument("light"); ap.add_argument("--storms", nargs="*", type=int)
    a = ap.parse_args()
    full = {}
    for f in glob.glob(os.path.join(a.fulldir, "*_storm_*.nc")):
        m = STORM_RE.search(os.path.basename(f))
        if m:
            full[int(m.group(1))] = f
    c = FIO._load(a.light)
    names = c['names']
    ldir = os.path.dirname(os.path.abspath(a.light))
    res = []

    def rep(ok, msg):
        res.append(bool(ok)); print(("PASS  " if ok else "FAIL  ") + msg, flush=True)

    rep(sorted(os.path.basename(f) for f in full.values()) == sorted(names),
        "storm file names: %d full / %d light" % (len(full), len(names)))
    labels = ["ntuple", "rain", "stormtime", "latitude/longitude", "xlocation", "ylocation", "basinrainfall",
              "gridmask", "domainmask", "cattime", "timeresolution"]
    for n in sorted(full):
        if a.storms and n not in a.storms:
            continue
        name = os.path.basename(full[n])
        if name not in c['index']:
            rep(False, "storm %d: %s not in the light catalog" % (n, name)); continue
        R = RD.readcatalog(full[n])
        vpath = os.path.join(ldir, name)
        assert FIO.is_virtual(vpath)
        L = FIO.readstorm(vpath)
        checks = [len(R) == len(L)]
        checks.append(same_da(R[0], L[0]))                                   # rain
        checks.append(same(R[1], L[1]))                                      # storm time
        checks.append(same_da(R[2], L[2]) and same_da(R[3], L[3]))           # latitude, longitude
        checks += [same(R[k], L[k]) for k in (4, 5, 6)]                      # x, y, basinrainfall
        checks.append(same_da(R[7], L[7]))                                   # gridmask
        checks += [same(R[8], L[8]), same(R[9], L[9])]                       # domainmask, cattime
        checks.append(type(R[10]) is type(L[10]) and R[10] == L[10])         # time resolution
        crc_ok = FIO.rain_crc32(np.asarray(R[0].values)) == int(c['crc'][c['index'][name]])
        bad = [k for k, ok in zip(labels, checks) if not ok]
        rep(all(checks) and crc_ok, "storm %3d %s: %s%s" % (n, name, "all variables bitwise" if not bad else
                                                              "DIFFER " + ",".join(bad), "" if crc_ok else ", CRC32 differs"))
    print("ALL PASS" if res and all(res) else "SOME CHECKS FAILED (%d of %d)" % (res.count(False), len(res)))
    sys.exit(0 if res and all(res) else 1)


if __name__ == "__main__":
    main()
