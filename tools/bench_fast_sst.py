"""
GP 2026-10-22: per-storm benchmark of MULTIDURATIONENGINE "table" vs "fast" on real storm-catalog storms.

    python tools/bench_fast_sst.py <StormCatalog dir> [--storms N] [--draws N] [--durs 1 3 6 12 24 48] [--seed S]
                                   [--no-fast] [--no-parent]

Per storm, --draws locations (default 4550 = production 1000 years x 100 realizations x 400 storms / 22 years / 400)
drawn uniformly among the valid anchors (np.random.default_rng). Times, in seconds per storm:
  fast : rolling_sum per duration + concatenate + SSTalt_multidur ; parent SSTalt (numba object mode)
  table: window_stack | zero_maps + zero_hours | anchor_cumsums (FFT) | table_kernel ; parent_sst (compiled)
and checks the two engines bitwise on the same draws. Extrapolate x NSTORMS for a run.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Source"))
import RainyDay_utilities_Py3.RainyDay_functions as RDF  # noqa: E402
import RainyDay_utilities_Py3.fast_sst as FST  # noqa: E402
from verify_fast_sst import load_storm, trim, valid_anchors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog")
    ap.add_argument("--storms", type=int, default=5)
    ap.add_argument("--draws", type=int, default=4550)
    ap.add_argument("--durs", type=int, nargs="*", default=[1, 3, 6, 12, 24, 48])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-fast", action="store_true")
    ap.add_argument("--no-parent", action="store_true")
    a = ap.parse_args()
    FST.ALLOW_OBJECTMODE_SIG[0] = True
    rng = np.random.default_rng(a.seed)
    files = sorted(glob.glob(os.path.join(a.catalog, "*_storm_*.nc")))
    pick = sorted(rng.choice(len(files), size=min(a.storms, len(files)), replace=False))
    tot = {}
    warm = True
    for si in pick:
        rain, mask, domain = load_storm(files[si])
        tm = trim(mask)
        mh, mw = tm.shape
        xa, ya = valid_anchors(domain, mh, mw)
        k = rng.integers(0, len(xa), a.draws)
        xx, yy = np.ascontiguousarray(xa[k]), np.ascontiguousarray(ya[k])
        if warm:                                    # JIT outside the timings
            aw, o, _ = FST.window_stack(rain[:, :, :], a.durs[:1])
            FST.table_sst(rain, aw, o, a.durs[:1], xx[:5], yy[:5], tm, mh, mw)
            RDF.SSTalt_multidur(aw, o, xx[:5], yy[:5], tm, mh, mw)
            if not a.no_parent:
                RDF.SSTalt(np.nansum(rain, axis=0), xx[:5], yy[:5], tm, mh, mw, durcheck=False)
            warm = False
        t = {}
        c0 = time.perf_counter()
        allwin, off, _ = FST.window_stack(rain, a.durs)
        c1 = time.perf_counter()
        zero = FST.zero_maps(allwin, off, mh, mw)
        zh, dedup = FST.zero_hours(rain, mh, mw)
        c2 = time.perf_counter()
        P = FST.anchor_cumsums(rain, tm)
        c3 = time.perf_counter()
        tolrel, tolabs = FST.tolerances(mh, mw, P)
        hist = np.zeros(len(FST.HBINS), np.int64)
        tab, nexact, _, _ = FST.table_kernel(allwin, off, np.asarray(a.durs, np.int64), zero, P, zh, dedup, xx, yy,
                                             tm, mh, mw, tolrel, tolabs, hist)
        c4 = time.perf_counter()
        t.update(stack=c1 - c0, zero=c2 - c1, fft=c3 - c2, kernel=c4 - c3)
        passrain = np.nansum(rain, axis=0)
        if not a.no_parent:
            c0 = time.perf_counter()
            par = FST.parent_sst(passrain, xx, yy, tm, mh, mw)
            c1 = time.perf_counter()
            pref, _ = RDF.SSTalt(passrain, xx, yy, tm, mh, mw, durcheck=False)
            c2 = time.perf_counter()
            t.update(parent_table=c1 - c0, parent_fast=c2 - c1)
            assert np.array_equal(par.view(np.uint32), np.asarray(pref, np.float32).view(np.uint32))
        if not a.no_fast:
            c0 = time.perf_counter()
            mdwins = [np.array(RDF.rolling_sum(rain, s), dtype="float32") for s in a.durs]
            mdoff = np.cumsum([0] + [w.shape[0] for w in mdwins]).astype(np.int64)
            ref = RDF.SSTalt_multidur(np.ascontiguousarray(np.concatenate(mdwins, axis=0)), mdoff, xx, yy, tm, mh, mw)
            c1 = time.perf_counter()
            t.update(fast=c1 - c0)
            assert np.array_equal(ref.view(np.uint32), tab.view(np.uint32)), "table != fast"
        t["table"] = t["stack"] + t["zero"] + t["fft"] + t["kernel"]
        nz = max(int(hist.sum()), 1)
        print(f"storm {si + 1:4d} {rain.shape} box {mh}x{mw}: " + "  ".join(f"{k} {v:.3f}" for k, v in t.items())
              + f"  | exact sums/nonzero cell {nexact / nz:.2f}, single-candidate {100. * hist[0] / nz:.1f}%", flush=True)
        for kk, v in t.items():
            tot[kk] = tot.get(kk, 0.) + v
    n = len(pick)
    print("mean s/storm: " + "  ".join(f"{k} {v / n:.3f}" for k, v in tot.items()))
    if "fast" in tot:
        print(f"sub-durations speed-up table vs fast: {tot['fast'] / tot['table']:.1f}x")
    if "parent_fast" in tot:
        print(f"parent speed-up compiled vs object mode: {tot['parent_fast'] / tot['parent_table']:.1f}x")


if __name__ == "__main__":
    main()
