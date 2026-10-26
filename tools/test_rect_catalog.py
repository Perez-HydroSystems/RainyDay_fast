#!/usr/bin/env python
"""
GP 2026-10-26: B17, rectangular-domain storm-catalog search. Brute force (plain numpy, every anchor) vs the search the
driver now uses for DOMAINTYPE rectangular (RainyDay_functions.catalogFFT_irregular_domainaware with an all-ones domain
mask, in the catalog scan and in the duration-rewrite search). No input data needed.

    python tools/test_rect_catalog.py          (RainyDay_Env; ~20 s, mostly numba compilation; ALL PASS at the end)

Checks, on synthetic float32 fields (NaN-free, as the driver's nansum windows):
  1. masks 1x1, 2x3, 5x7 and an irregular (fractional, holed) basin mask, 25 random fields each (40x50 grids) plus one
     184x184 field (the Talequah crop size): the argmax anchor (y, x) equals the brute-force one EXACTLY; the returned
     maximum and the whole FFT sum map agree with the float64 direct sums within float32 rounding (max errors printed);
     the brute-force gap between the best and second-best anchor is printed (no near-ties).
  2. maxima planted at every edge and corner anchor (0 / H-mh, 0 / W-mw): found exactly (the old kernel never searched
     some of them and stored others one cell off).
  3. the valid-anchor map is (H-mh+1, W-mw+1) = (ylen, xlen) of the driver: the same range as the rectangular uniform
     draws (randint(0, W-mw+1)) and the KDE grid (kx, ky).
  4. the old call (catalogNumba with the driver's xlen = W-mw+1, ylen = H-mh+1) for the record: a numba broadcast
     ValueError for masks >= 3 cells wide or tall; for 1-2 cell masks it runs but returns wrong anchors.
  5. both driver call sites use the new search (source check).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Source"))
import RainyDay_utilities_Py3.RainyDay_functions as RDF   # noqa: E402

RES = []


def report(ok, msg):
    RES.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True)


def brute(temparray, trimmask):
    """every anchor, plain numpy, float64; first maximum in row-major order on exact ties"""
    H, W = temparray.shape
    mh, mw = trimmask.shape
    t = temparray.astype(np.float64)
    m = trimmask.astype(np.float64)
    out = np.empty((H - mh + 1, W - mw + 1))
    for y in range(H - mh + 1):
        for x in range(W - mw + 1):
            out[y, x] = np.sum(t[y:y + mh, x:x + mw] * m)
    k = int(np.argmax(out))
    return out, k // out.shape[1], k % out.shape[1]


def fft_map(temparray, trimmask):
    """the sum map exactly as catalogFFT_irregular_domainaware computes it (all placements valid on an all-ones domain)"""
    rain = np.ascontiguousarray(np.nan_to_num(temparray), dtype=np.float32)
    storm = np.ascontiguousarray(np.nan_to_num(trimmask), dtype=np.float32)
    return fftconvolve(rain, storm[::-1, ::-1], mode="valid")


def new_search(temparray, trimmask):
    """the driver's call for DOMAINTYPE rectangular (domainmask = np.ones(catmask.shape, float32))"""
    return RDF.catalogFFT_irregular_domainaware(temparray, trimmask, np.ones(temparray.shape, dtype="float32"))


def masks():
    basin = np.array([[0.0, 0.3, 1.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0],
                      [0.2, 1.0, 1.0, 1.0, 1.0, 0.9, 0.1, 0.0, 0.0],
                      [0.7, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.4, 0.0],      # a hole
                      [0.1, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3],
                      [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.9, 0.05],
                      [0.0, 0.0, 0.0, 0.2, 0.6, 1.0, 0.7, 0.1, 0.0]], dtype=np.float32)
    return [("1x1", np.ones((1, 1), np.float32)), ("2x3", np.ones((2, 3), np.float32)),
            ("5x7", np.ones((5, 7), np.float32)), ("basin 6x9 (fractional, holed)", basin)]


def field(rng, H, W):
    f = rng.gamma(0.6, 4.0, (H, W)).astype(np.float32)
    f[rng.random((H, W)) < 0.3] = 0.                       # dry cells
    return f


def check(name, temparray, trimmask, stats):
    ref, by, bx = brute(temparray, trimmask)
    rmax, y, x = new_search(temparray, trimmask)
    m = fft_map(temparray, trimmask).astype(np.float64)
    err = np.max(np.abs(m - ref))
    rel = err / max(np.max(np.abs(ref)), 1e-30)
    srt = np.sort(ref.ravel())
    gap = (srt[-1] - srt[-2]) / srt[-1] if srt.size > 1 and srt[-1] > 0 else np.inf
    stats["abs"] = max(stats["abs"], err)
    stats["rel"] = max(stats["rel"], rel)
    stats["maxerr"] = max(stats["maxerr"], abs(rmax - ref[by, bx]) / ref[by, bx])
    stats["gap"] = min(stats["gap"], gap)
    stats["n"] += 1
    ok = (y, x) == (by, bx) and m.shape == ref.shape
    if not ok:
        print("   mismatch %s: new (%d,%d) brute (%d,%d) gap %.2e" % (name, y, x, by, bx, gap))
    return ok


def test_random_fields():
    rng = np.random.default_rng(20261026)
    for name, tm in masks():
        stats = dict(abs=0., rel=0., maxerr=0., gap=np.inf, n=0)
        ok = all(check(name, field(rng, 40, 50), tm, stats) for _ in range(25))
        ok = ok and check(name, field(rng, 184, 184), tm, stats)
        report(ok and stats["rel"] < 1e-5,
               "mask %-30s %2d fields: argmax anchors identical to brute force; sum map max abs err %.2e (rel %.2e), "
               "returned max rel err %.2e; smallest top-2 gap %.1e" % (name, stats["n"], stats["abs"], stats["rel"],
                                                                          stats["maxerr"], stats["gap"]))


def test_edges():
    rng = np.random.default_rng(7)
    for name, tm in masks():
        mh, mw = tm.shape
        H, W = 30, 37
        ok, n = True, 0
        for ty in (0, (H - mh) // 2, H - mh):
            for tx in (0, (W - mw) // 2, W - mw):
                if ty not in (0, H - mh) and tx not in (0, W - mw):
                    continue                                  # edges and corners only
                f = field(rng, H, W)
                f[ty:ty + mh, tx:tx + mw][tm > 0] += 200.
                ref, by, bx = brute(f, tm)
                rmax, y, x = new_search(f, tm)
                ok = ok and (by, bx) == (ty, tx) and (y, x) == (ty, tx) and abs(rmax - ref[by, bx]) <= 1e-5 * ref[by, bx]
                n += 1
        report(ok, "mask %-30s maximum planted at %d edge/corner anchors (incl. (H-mh, W-mw)): found exactly" % (name, n))


def test_ranges():
    H, W = 30, 37
    ok = True
    for name, tm in masks():
        mh, mw = tm.shape
        xlen, ylen = W - mw + 1, H - mh + 1                   # driver: xlen = subdimensions[1]-maskwidth+1, ylen likewise
        kx, ky = np.meshgrid(np.arange(0, W - mw + 1), np.arange(0, H - mh + 1))   # driver KDE grid
        ok = ok and fft_map(np.ones((H, W), np.float32), tm).shape == (ylen, xlen) == kx.shape
    report(ok, "search anchors = (H-mh+1, W-mw+1) = (ylen, xlen) = KDE grid = rectangular uniform draw range")


def test_old_kernel():
    """for the record: the old call of the driver (catalogNumba with xlen = W-mw+1, ylen = H-mh+1)"""
    rng = np.random.default_rng(3)
    H, W = 30, 37
    for name, tm in masks() + [("2x2", np.ones((2, 2), np.float32))]:
        mh, mw = tm.shape
        xlen, ylen = W - mw + 1, H - mh + 1
        xloop = (W - mw - 1) / 2 if (W - mw) % 2 != 0 else (W - mw) / 2
        yloop = (H - mh - 1) / 2 if (H - mh) % 2 != 0 else (H - mh) / 2
        wrong, err = 0, None
        for ty, tx in ((H - mh, W - mw), (0, W - mw), (H - mh, 0), (3, 4)):
            f = field(rng, H, W)
            f[ty:ty + mh, tx:tx + mw][tm > 0] += 200.
            try:
                _, y, x = RDF.catalogNumba(f, tm, xlen, ylen, xloop, yloop, mh, mw, np.zeros((ylen, xlen), np.float32))
                wrong += (y, x) != (ty, tx)
            except Exception as e:                            # numba: "unable to broadcast"
                err = e
                break
        if mh >= 3 or mw >= 3:
            report(isinstance(err, ValueError), "old catalogNumba, mask %-30s -> %s: %s"
                   % (name, type(err).__name__, str(err).splitlines()[0] if err else None))
        else:
            report(err is None and wrong > 0, "old catalogNumba, mask %-30s runs but returns a wrong anchor for %d of 4 "
                                              "planted maxima" % (name, wrong))


def test_driver_hooks():
    src = (REPO / "Source" / "RainyDay_Py3.py").read_text().split("\n")
    live = [s for s in src if "catalogNumba(" in s and not s.strip().startswith("#")]
    hooks = [s for s in src if "catalogFFT_irregular_domainaware(" in s and "B17" in s and not s.strip().startswith("#")]
    report(len(live) == 0 and len(hooks) == 2, "driver: no live catalogNumba call; 2 rectangular call sites use "
                                               "catalogFFT_irregular_domainaware (catalog scan, duration rewrite)")


def main():
    test_random_fields()
    test_edges()
    test_ranges()
    test_old_kernel()
    test_driver_hooks()
    print()
    print("ALL PASS (%d checks)" % len(RES) if RES and all(RES) else "SOME TESTS FAILED (%d of %d)" % (RES.count(False), len(RES)))
    sys.exit(0 if RES and all(RES) else 1)


if __name__ == "__main__":
    main()
