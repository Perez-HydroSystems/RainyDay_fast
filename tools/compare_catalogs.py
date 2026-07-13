#!/usr/bin/env python
"""
Compare two RainyDay storm-catalog directories (per-storm NetCDF files) and
verify they describe the SAME storm catalog.

Equality criterion:
  - same set of storm numbers, and same storm date in each file name
  - xlocation, ylocation, cattime, time/latitude/longitude coords: exact match
  - rain, gridmask, domainmask: bitwise match (NaN-aware)
  - basinrainfall: relative tolerance (FFT batching changes float summation
    order, so the max metric can differ in the last digits)

Usage:
    python compare_catalogs.py REF_DIR TEST_DIR [--rtol 1e-5]
Exit code 0 = catalogs identical under the criterion, 1 = mismatch.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import xarray as xr

STORM_RE = re.compile(r"_storm_(\d+)_(\d{8})\.nc$")


def index_dir(path):
    files = {}
    for f in glob.glob(os.path.join(path, "*.nc")):
        m = STORM_RE.search(os.path.basename(f))
        if m:
            files[int(m.group(1))] = (m.group(2), f)
    if not files:
        sys.exit(f"ERROR: no '*_storm_N_YYYYMMDD.nc' files found in {path}")
    return files


def arrays_equal(a, b):
    """Bitwise equality, treating NaN==NaN as equal."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, np.inf
    if np.issubdtype(a.dtype, np.floating):
        eq = np.array_equal(a, b, equal_nan=True)
        diff = 0.0 if eq else float(np.nanmax(np.abs(a - b)))
        return eq, diff
    eq = np.array_equal(a, b)
    return eq, 0 if eq else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refdir")
    ap.add_argument("testdir")
    ap.add_argument("--rtol", type=float, default=1e-5,
                    help="relative tolerance for basinrainfall (default 1e-5)")
    args = ap.parse_args()

    ref, test = index_dir(args.refdir), index_dir(args.testdir)
    ok = True

    if set(ref) != set(test):
        print(f"FAIL: storm numbers differ: ref={sorted(ref)} test={sorted(test)}")
        sys.exit(1)

    exact_vars = ["xlocation", "ylocation", "cattime", "rain", "gridmask", "domainmask"]
    print(f"{'storm':>5} {'date':>10} {'x,y':>9} {'cattime':>8} {'rain':>10} "
          f"{'masks':>6} {'basinrain(maxrel)':>18}  verdict")

    for n in sorted(ref):
        refdate, reffile = ref[n]
        testdate, testfile = test[n]
        row_ok = True
        notes = []

        if refdate != testdate:
            print(f"{n:>5} FAIL: storm date differs: ref={refdate} test={testdate}")
            ok = False
            continue

        dref = xr.open_dataset(reffile)
        dtest = xr.open_dataset(testfile)

        results = {}
        for v in exact_vars:
            eq, diff = arrays_equal(dref[v].values, dtest[v].values)
            results[v] = eq
            if not eq:
                row_ok = False
                notes.append(f"{v} differs (maxabs={diff:g})")

        for c in ["time", "latitude", "longitude"]:
            eq, _ = arrays_equal(dref[c].values, dtest[c].values)
            if not eq:
                row_ok = False
                notes.append(f"coord {c} differs")

        br_ref = dref["basinrainfall"].values.astype("float64")
        br_test = dtest["basinrainfall"].values.astype("float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.nanmax(np.abs(br_test - br_ref) / np.where(br_ref == 0, 1, np.abs(br_ref)))
        br_ok = bool(np.allclose(br_test, br_ref, rtol=args.rtol, equal_nan=True))
        if not br_ok:
            row_ok = False
            notes.append(f"basinrainfall exceeds rtol={args.rtol:g} (maxrel={rel:.3e})")

        mark = lambda b: "OK" if b else "DIFF"
        print(f"{n:>5} {refdate:>10} "
              f"{mark(results['xlocation'] and results['ylocation']):>9} "
              f"{mark(results['cattime']):>8} {mark(results['rain']):>10} "
              f"{mark(results['gridmask'] and results['domainmask']):>6} "
              f"{rel:>18.3e}  {'PASS' if row_ok else 'FAIL'}")
        for msg in notes:
            print(f"       -> {msg}")

        dref.close()
        dtest.close()
        ok = ok and row_ok

    print()
    if ok:
        print(f"ALL PASS: {len(ref)} storms identical "
              f"(basinrainfall within rtol={args.rtol:g})")
        sys.exit(0)
    print("MISMATCH: catalogs are NOT identical — see rows above")
    sys.exit(1)


if __name__ == "__main__":
    main()
