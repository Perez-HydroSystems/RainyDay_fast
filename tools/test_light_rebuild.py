#!/usr/bin/env python
"""
GP 2026-10-22: unit test of fast_io.rebuild_storm_rain against a verbatim copy of the catalog write loop of
RainyDay_Py3.py, on cases the validation runs never reach: dates without a file (the loop keeps the previous file and
fills zeros), windows across a gap, both date rules (check_time True/False), windows at the end of the list.

    python tools/test_light_rebuild.py <dir with AORC.YYYYMMDD.preciptemp.nc> <domain.shp> [--n 150]

Uses the first year of files found; removes some dates from the list to create gaps. Prints ALL PASS / FAIL.
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import shapefile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Source"))
import RainyDay_utilities_Py3.RainyDay_functions as RainyDay   # noqa: E402
import RainyDay_utilities_Py3.fast_io as FIO                    # noqa: E402


def write_loop_verbatim(flist, cattime, i, nsteps, shape, readtime, timeres, variables, idxes, droplist, calendar,
                        time_units):
    """Lines of RainyDay_Py3.py "This part saves each storm as single file" (rainprop.timeres -> timeres), + the NaN
    replacement of writecatalog."""
    start_time = cattime[i,0]
    end_time   = cattime[i,-1]
    current_datetime = start_time
    dataset_date = None
    catrain = np.zeros((nsteps,shape[0],shape[1]),dtype='float32')
    k  = 0
    stm_file = None
    count = 0
    while current_datetime <= end_time:
        if RainyDay.check_time(readtime[0]):
            current_date = np.datetime_as_string(current_datetime, unit='D')
        else:
            current_date = np.datetime_as_string(current_datetime - timeres, unit='D')
        if current_date != dataset_date:
            dataset_date = current_date
            # This loop searches for the right file to open from the filelist
            for file in flist:
                match = re.search(r'\d{4}(?:\d{2})?(?:\d{2}|\-\d{2}\-\d{2}|\/\d{2}/\d{2})',\
                                  os.path.basename(file))
                if current_date.replace("-","") == match.group().replace("-","").replace("/",""):
                    stm_file = file
                    break
            stm_rain,stm_time = RainyDay.readnetcdf(stm_file,variables,idxes,dropvars=droplist,calendar=calendar,time_units=time_units)
        match = np.where(stm_time == current_datetime)[0] if len(stm_time) > 0 else []
        if len(match) == 0:
            catrain[k, :] = 0.0
        else:
            cind = match[0]
            catrain[k, :] = stm_rain[cind, :]
        current_datetime += timeres
        k += 1
    catrain[np.isnan(catrain)] = -9999.
    return catrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rain_dir"); ap.add_argument("domain_shp"); ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    variables = {"precip": "RAINRATE", "latname": "latitude", "longname": "longitude"}
    files = sorted(glob.glob(os.path.join(a.rain_dir, "AORC.*.preciptemp.nc")))
    year = os.path.basename(files[0])[5:9]
    files = [f for f in files if os.path.basename(f)[5:9] == year]
    gaps = {year + d for d in ("0715", "0716", "0801", "0902", "0903", "0904", "0905")}
    flist = [f for f in files if os.path.basename(f)[5:13] not in gaps]
    bb = shapefile.Reader(a.domain_shp).bbox
    inarea = np.array([bb[0], bb[2], bb[1], bb[3]], dtype='float32')
    idxes = RainyDay.find_indices(flist[0], inarea, variables)
    _, _, _, _, _, droplist, calendar, time_units = RainyDay.rainprop_setup(flist[0], None, dict(variables))
    _, readtime = RainyDay.readnetcdf(flist[0], variables, idxes, dropvars=droplist, calendar=calendar, time_units=time_units)
    timeres = np.int32(60)
    nsteps = 72
    ny, nx = idxes[1] - idxes[0] + 1, idxes[3] - idxes[2] + 1
    # storm windows: ends on random hours between July and the last file, plus windows hitting each gap edge
    rng = np.random.default_rng(1)
    t0 = np.datetime64(year + "-07-05T00:00", 'm')
    t1 = np.datetime64(year + "-12-31T23:00", 'm') if len(files) > 300 else readtime[0] + np.timedelta64(24 * len(files) - 1, 'h')
    ends = list(t0 + rng.integers(0, int((t1 - t0) / np.timedelta64(1, 'h')), a.n) * np.timedelta64(60, 'm'))
    for g in sorted(gaps):
        d = np.datetime64(g[:4] + "-" + g[4:6] + "-" + g[6:], 'm')
        ends += [d + np.timedelta64(h, 'h') for h in (0, 1, 23, 24, 25, 48, 70, 71, 72, 73, 95)]
    ends.append(np.datetime64(year + "-12-31T23:00", 'm'))
    ends.append(np.datetime64(str(int(year) + 1) + "-01-01T00:00", 'm'))
    cattime = np.array([np.arange(e - np.timedelta64(nsteps - 1, 'h'), e + np.timedelta64(1, 'm'), np.timedelta64(60, 'm'))
                        for e in ends], dtype='datetime64[m]')
    first = FIO.first_file_by_date(flist)
    nbad = 0
    for rule in (False, True):
        rt = readtime if not rule else np.array([np.datetime64(year + "-01-01T00:00", 'm')])
        assert RainyDay.check_time(rt[0]) == rule
        nzero, nfail = 0, 0
        for i in range(len(cattime)):
            try:
                ref = write_loop_verbatim(flist, cattime, i, nsteps, (ny, nx), rt, timeres, variables, idxes, droplist,
                                          calendar, time_units)
            except Exception:                    # first date without a file: the write loop reads file None
                ref = None
            try:
                new = FIO.rebuild_storm_rain(cattime[i], (nsteps, ny, nx), first, RainyDay.check_time(rt[0]), timeres,
                                             variables, idxes, droplist, calendar, time_units)
            except SystemExit:
                new = None
            if ref is None or new is None:
                nfail += 1
                if (ref is None) != (new is None):
                    nbad += 1
                    print("FAIL rule %s storm ending %s: only one of the two stops" % (rule, cattime[i, -1]))
                continue
            nzero += int(np.sum(~ref.reshape(nsteps, -1).any(axis=1)))
            if not (ref.dtype == new.dtype and ref.shape == new.shape
                    and np.array_equal(ref.view(np.uint32), new.view(np.uint32))):
                nbad += 1
                print("FAIL rule %s storm ending %s" % (rule, cattime[i, -1]))
        print("%s  check_time %s: %d windows bitwise equal to the write loop (%d all-zero steps from gaps/time "
              "mismatch), %d windows stop in both (first date without a file)"
              % ("PASS" if nbad == 0 else "FAIL", rule, len(cattime) - nfail, nzero, nfail))
    print("ALL PASS" if nbad == 0 else "FAILED")
    sys.exit(0 if nbad == 0 else 1)


if __name__ == "__main__":
    main()
