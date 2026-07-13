#!/usr/bin/env python
"""
Time the batchfft_mp catalog scan on Windows.

RainyDay_Py3.py pins batchfft_mp to the 'fork' start method, which does not exist
on Windows (only 'spawn'), and the main script has no __main__ guard, so spawn
workers would re-run the whole script. This standalone driver HAS a __main__
guard, so it can drive the same catalog_scan_worker over a spawn ProcessPoolExecutor
safely. It reproduces the exact task-splitting from the batchfft_mp branch.

Usage:
    # 1) dump the scan inputs from a real run (any engine; exits before scanning):
    RAINYDAY_DUMP_SCANINPUTS=/path/scaninputs.pkl \
        python Source/RainyDay_Py3.py <abs path>/CONUS_bench_win_allmonths.json
    # 2) time the mp scan:
    python tools/bench_mp_scan.py /path/scaninputs.pkl 8

Prints the scan wall time and the number of timesteps produced (a sanity check
that it scanned the whole record). Correctness of batchfft_mp vs batchfft is
already established elsewhere; this only measures speed.
"""
import concurrent.futures
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Source", "RainyDay_utilities_Py3"))
import RainyDay_functions as RF


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: bench_mp_scan.py SCANINPUTS.pkl [nworkers]")
    with open(sys.argv[1], "rb") as f:
        inp = pickle.load(f)
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) // 2)

    flist, nctx = inp["flist"], inp["nctx"]
    # each worker builds a float64 windowed block of ~blockfiles*24 steps; on a
    # RAM-limited box lower RAINYDAY_BLOCKFILES so nworkers blocks fit (the scan
    # result is independent of block size). Default keeps the dumped value.
    blockfiles = int(os.environ.get("RAINYDAY_BLOCKFILES", inp["blockfiles"]))
    bounds = np.linspace(0, len(flist), nworkers + 1).astype(int)
    tasks = []
    for c in range(nworkers):
        lo, hi = bounds[c], bounds[c + 1]
        if lo < hi:
            tasks.append((flist[lo:hi], flist[max(0, lo - nctx):lo], inp["variables"],
                          inp["idxes"], inp["droplist"], inp["calendar"], inp["time_units"],
                          inp["timestep"], inp["nwindow"], inp["trimmask"], inp["domainmask"],
                          blockfiles, False, "batchfft"))
    print("blockfiles={0}, nctx={1}".format(blockfiles, nctx))

    print("scanning {0} files with {1} workers (spawn)...".format(len(flist), len(tasks)))
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=nworkers, mp_context=mp.get_context("spawn")) as ex:
        parts = list(ex.map(RF.catalog_scan_worker, tasks))
    dt = time.time() - t0

    ntimes = sum(len(p[0]) for p in parts)
    print("batchfft_mp x{0} scan: {1:.2f} min ({2} timesteps)".format(len(tasks), dt / 60.0, ntimes))


if __name__ == "__main__":
    main()
