#!/usr/bin/env python
"""
Validate the threaded-prefetch win for the batchcuda scan: time catalog_scan_files
over the first N files with RAINYDAY_SCAN_PREFETCH=0 (serial read then compute) vs
the prefetch depth passed on argv, and confirm the two produce identical scan arrays.

Usage: python tools/bench_prefetch.py SCANINPUTS.pkl [nfiles] [prefetch]
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Source", "RainyDay_utilities_Py3"))
os.environ.setdefault("CUPY_ACCELERATORS", "")
os.environ.setdefault("RAINYDAY_GPU_CHUNK", "64")
import RainyDay_functions as RF


def scan(inp, files, prefetch):
    os.environ["RAINYDAY_SCAN_PREFETCH"] = str(prefetch)
    t = time.time()
    out = RF.catalog_scan_files(files, [], inp["variables"], inp["idxes"], inp["droplist"],
                                inp["calendar"], inp["time_units"], inp["timestep"], inp["nwindow"],
                                inp["trimmask"], inp["domainmask"], inp["blockfiles"], False,
                                "batchcuda", return_fileidx=True)
    return out, time.time() - t


def main():
    with open(sys.argv[1], "rb") as f:
        inp = pickle.load(f)
    nfiles = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    prefetch = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    files = inp["flist"][:nfiles]
    print("nfiles={0} blockfiles={1} chunk={2}".format(
        len(files), inp["blockfiles"], os.environ["RAINYDAY_GPU_CHUNK"]))

    (o0, t0) = scan(inp, files, 0)
    (o2, t2) = scan(inp, files, prefetch)
    print("prefetch=0 : {0:.1f}s".format(t0))
    print("prefetch={0} : {1:.1f}s  ({2:.2f}x)".format(prefetch, t2, t0 / t2))
    same = all(np.array_equal(a, b) for a, b in zip(o0, o2))
    print("scan arrays identical (prefetch changes only timing):", same)


if __name__ == "__main__":
    main()
