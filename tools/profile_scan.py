#!/usr/bin/env python
"""
Profile the serial catalog scan phase-by-phase (read / window / compute) using a
dumped scaninputs pickle (see RAINYDAY_DUMP_SCANINPUTS in RainyDay_Py3.py).

This isolates where the scan wall time actually goes, which decides whether a
reader/compute-overlap hybrid can help and by how much. It runs the same block
loop as catalog_scan_files but times each phase, for both the GPU (catalogCUDA_batch)
and CPU (catalogFFT_batch) compute kernels.

Usage: python tools/profile_scan.py SCANINPUTS.pkl [blockfiles]
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Source", "RainyDay_utilities_Py3"))
import RainyDay_functions as RF

os.environ.setdefault("CUPY_ACCELERATORS", "")


def run(engine, inp, blockfiles):
    fftpre = RF.catalogFFT_precompute(inp["trimmask"], inp["domainmask"])
    fh, fw = inp["domainmask"].shape
    nwindow, timestep = inp["nwindow"], inp["timestep"]
    files = inp["flist"]
    context = np.zeros((nwindow - 1, fh, fw), dtype=np.float32)
    context_endtime = None
    t_read = t_win = t_cmp = 0.0
    nsteps = 0
    t0all = time.time()
    for bstart in range(0, len(files), blockfiles):
        bfiles = files[bstart:bstart + blockfiles]
        t = time.time()
        blockrain, blocktimes, present, fileidx = RF.read_file_block(
            bfiles, inp["variables"], inp["idxes"], inp["droplist"],
            inp["calendar"], inp["time_units"], timestep)
        t_read += time.time() - t
        if context_endtime is not None:
            gapsteps = int((blocktimes[0] - (context_endtime + timestep)) / timestep)
            if gapsteps >= nwindow - 1:
                context[:] = 0.
            elif gapsteps > 0:
                context = np.concatenate([context[gapsteps:],
                                          np.zeros((gapsteps,) + context.shape[1:], dtype='float32')], axis=0)
        t = time.time()
        windowed, context = RF.build_windowed_block(blockrain, context, nwindow, inp["domainmask"])
        t_win += time.time() - t
        context_endtime = blocktimes[-1]
        t = time.time()
        if engine == "cuda":
            RF.catalogCUDA_batch(windowed, fftpre)
        else:
            RF.catalogFFT_batch(windowed, fftpre)
        t_cmp += time.time() - t
        nsteps += int(present.sum())
    total = time.time() - t0all
    print("--- engine={0}, blockfiles={1} ---".format(engine, blockfiles))
    print("  read   : {0:6.1f}s  ({1:4.1f}%)".format(t_read, 100 * t_read / total))
    print("  window : {0:6.1f}s  ({1:4.1f}%)".format(t_win, 100 * t_win / total))
    print("  compute: {0:6.1f}s  ({1:4.1f}%)".format(t_cmp, 100 * t_cmp / total))
    print("  TOTAL  : {0:6.1f}s  ({1:.2f} min), {2} timesteps".format(total, total / 60, nsteps))
    return t_read, t_win, t_cmp, total


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: profile_scan.py SCANINPUTS.pkl [blockfiles]")
    with open(sys.argv[1], "rb") as f:
        inp = pickle.load(f)
    blockfiles = int(sys.argv[2]) if len(sys.argv) > 2 else int(inp["blockfiles"])
    if len(sys.argv) > 3:                      # optional: profile only the first N files
        inp["flist"] = inp["flist"][:int(sys.argv[3])]
    print("files:", len(inp["flist"]), "domain:", inp["domainmask"].shape,
          "nwindow:", inp["nwindow"], "GPU_CHUNK:", os.environ.get("RAINYDAY_GPU_CHUNK", "256"))
    run("cuda", inp, blockfiles)
    run("fft", inp, blockfiles)


if __name__ == "__main__":
    main()
