"""
GP 2026-10-22: RainyDay_fast fix 3: transposition engine MULTIDURATIONENGINE "table".

Bit-identical to the "fast" engine (SSTalt_multidur) and to "exact" (SSTalt). Per storm:

  1. the exact float32 window fields of every sub-duration, as RainyDay builds them (rolling_sum per duration: float64
     nancumsum differences cast to float32); one shared nancumsum replaces the per-duration rolling_sum calls (same
     arithmetic, checked bitwise on the first storm and in verify mode);
  2. the zero rule of SSTalt / SSTalt_multidur per duration and anchor (every cell of the mh x mw box < 0.5 in every
     window of the duration -> 0), exactly: per-cell max over the windows of the duration, indicator "not < 0.5",
     sliding integer box counts -> zero[j, y, x] = (count == 0);
  3. approximate basin sums of every window at every anchor, A[w] = P[s+d] - P[s], with P the running time sum
     (float64) of the hourly basin sums (float64 FFT correlation of the hourly fields with trimmask; direct product for
     a 1x1 mask), spot-checked against direct float64 sums at a few anchors of every storm;
  4. the hours that are 0/NaN on the whole box of an anchor (zero_hours), which make consecutive windows bitwise equal.

  Per draw and duration: zero rule -> 0; else candidates = windows with A >= Amax - tol,
      tol = 4 (gamma_n + u) max(Amax, 0) + 4 eta,   u = 2^-24, n = mh*mw, gamma_n = n u / (1 - n u),
      eta = 1e-8 * (largest storm-total basin sum of the storm),
  (all windows when more than half qualify). The exact float32 sum of a window differs from its A by at most
  (gamma_n + u) Amax + eta (float32 products and summation in any order, float32 rounding of the window field, float64
  FFT and cumulative sums), so the window that holds the exact maximum is always a candidate. The candidates are
  evaluated with RainyDay's own numba_multimask_calc (the same compiled specialization SSTalt_multidur uses: float32
  2-D C window from the same 3-D C stack, float32 2-D C mask, int32 anchor, int64 mask sizes), in the loop of
  SSTalt_multidur (storesum = 0. float64, tempsum = ...*1., strict >, ascending windows, float32 store), except
  candidates bitwise identical over the box to the previous evaluated window (4.). Skipping windows that cannot hold
  the maximum, or whose sum is already in storesum, does not change the value.

  Parent duration (no duration correction): SSTalt_multidur(passrain[None], [0, 1], ...)[0], the compiled kernel with
  one window, instead of SSTalt in numba object mode (same zero rule, same sum, max(0, sum) = sum for passrain >= 0).

Verification: env RAINYDAY_VERIFY_TABLE=<fraction> brute-forces a random subset of the draws of every storm
(np.random.default_rng, the global np.random stream is never touched) with SSTalt_multidur (sub-durations) and SSTalt
(parent) and aborts on any difference. Offline: tools/verify_fast_sst.py, tools/bench_fast_sst.py.
"""
import os
import sys
import time

import numpy as np
import scipy.fft as spfft
from numba import jit

import RainyDay_utilities_Py3.RainyDay_functions as RDF   # same module object as the driver's "RainyDay"

U32 = 2.0 ** -24
ETA_REL = 1e-8
# candidate-count histogram bins (per draw and duration with a nonzero result): 1, 2, 3-4, 5-8, 9-16, >16, all windows
HBINS = ("1", "2", "3-4", "5-8", "9-16", ">16", "all")

STATS = dict(storms=0, draws=0, cells=0, zero=0, exact_evals=0, dedup_skips=0, nodedup_storms=0, brute_evals=0,
             hist=np.zeros(len(HBINS), np.int64), verified=0, verified_parent=0, seconds_prep=0., seconds_kernel=0.,
             fft_err=0.)
_FIRST = {"stack_checked": False}


def verify_fraction():
    v = os.environ.get("RAINYDAY_VERIFY_TABLE", "").strip()
    return float(v) if v not in ("", "0") else 0.


def fft_workers():
    return int(os.environ.get("RAINYDAY_TABLE_WORKERS", "1"))


# ---------------------------------------------------------------------------------------------------------------------
# 1. exact window fields
# ---------------------------------------------------------------------------------------------------------------------
def window_stack(catrain, substeps, check=False):
    """float32 C stack of the rolling sums of every duration in substeps (window counts T-s+1) and offsets (int64).
    Same values as np.concatenate([np.array(RainyDay.rolling_sum(catrain, s), dtype='float32') for s in substeps]):
    rolling_sum = nancumsum(float64) differences, here from one shared nancumsum."""
    T = catrain.shape[0]
    nwin = [T - s + 1 for s in substeps]
    offsets = np.cumsum([0] + nwin).astype(np.int64)
    # c = np.nancumsum(catrain, axis=0, dtype=float)               # exactly what rolling_sum computes, once
    c = np.empty(catrain.shape, dtype=np.float64)                   # same IEEE operations, compiled (checked below)
    _nancumsum64(catrain, c)
    allwin = np.empty((int(offsets[-1]),) + catrain.shape[1:], dtype=np.float32)
    for j, s in enumerate(substeps):
        o = int(offsets[j])
        allwin[o] = c[s - 1]                                        # window 0 = ret[s-1]
        if T > s:                                                   # windows 1.. = ret[s:] - ret[:-s] (float64), to float32
            np.subtract(c[s:], c[:-s], out=allwin[o + 1:o + 1 + T - s], casting="same_kind")
    if check:
        ref = np.ascontiguousarray(np.concatenate(
            [np.array(RDF.rolling_sum(catrain, s), dtype="float32") for s in substeps], axis=0))
        if ref.shape != allwin.shape or ref.dtype != allwin.dtype or not np.array_equal(ref.view(np.uint32),
                                                                                       allwin.view(np.uint32)):
            sys.exit("MULTIDURATIONENGINE 'table': the shared-cumsum window stack differs from rolling_sum (bug)")
    return allwin, offsets, c


@jit(nopython=True)          # no fastmath
def _nancumsum64(a, c):
    # np.nancumsum(a, axis=0, dtype=float): NaN -> +0, float64(a[t]) added sequentially along time
    T, H, W = a.shape
    for y in range(H):
        for x in range(W):
            v = a[0, y, x]
            c[0, y, x] = np.float64(v) if v == v else 0.
    for t in range(1, T):
        for y in range(H):
            for x in range(W):
                v = a[t, y, x]
                c[t, y, x] = c[t - 1, y, x] + (np.float64(v) if v == v else 0.)


# ---------------------------------------------------------------------------------------------------------------------
# 2. exact zero rule
# ---------------------------------------------------------------------------------------------------------------------
def zero_maps(allwin, offsets, mh, mw):
    """zero[j, y, x] = np.all(allwin[offsets[j]:offsets[j+1], y:y+mh, x:x+mw] < 0.5) for every anchor (y, x) of the
    array ((H-mh+1) x (W-mw+1)); exact (integer counts of cells that are not < 0.5 in some window)."""
    H, W = allwin.shape[1:]
    zero = np.empty((len(offsets) - 1, H - mh + 1, W - mw + 1), dtype=np.bool_)
    for j in range(len(offsets) - 1):
        hit = ~(np.max(allwin[offsets[j]:offsets[j + 1]], axis=0) < 0.5)   # NaN -> hit, as np.all(np.less(...))
        _box_empty_nb(hit, mh, mw, zero[j])
    return zero


@jit(nopython=True)
def _box_empty_nb(hit, mh, mw, out):
    # out[y, x] = no True in hit[y:y+mh, x:x+mw] (sliding integer counts: rows of width mw, then columns of height mh)
    H, W = hit.shape
    oh = H - mh + 1
    ow = W - mw + 1
    rowcnt = np.empty((H, ow), dtype=np.int64)
    for r in range(H):
        c = 0
        for q in range(mw):
            c += hit[r, q]
        rowcnt[r, 0] = c
        for x in range(1, ow):
            c += np.int64(hit[r, x + mw - 1]) - np.int64(hit[r, x - 1])
            rowcnt[r, x] = c
    for x in range(ow):
        c = 0
        for r in range(mh):
            c += rowcnt[r, x]
        out[0, x] = c == 0
        for y in range(1, oh):
            c += rowcnt[y + mh - 1, x] - rowcnt[y - 1, x]
            out[y, x] = c == 0


def zero_maps_ref(allwin, offsets, mh, mw):
    """numpy reference of zero_maps (summed-area tables), for tools/verify_fast_sst.py"""
    ndur = len(offsets) - 1
    H, W = allwin.shape[1:]
    zero = np.empty((ndur, H - mh + 1, W - mw + 1), dtype=np.bool_)
    sat = np.zeros((H + 1, W + 1), dtype=np.int64)
    for j in range(ndur):
        hit = ~(np.max(allwin[offsets[j]:offsets[j + 1]], axis=0) < 0.5)   # NaN -> hit, as np.all(np.less(...))
        np.cumsum(np.cumsum(hit, axis=0, dtype=np.int64), axis=1, out=sat[1:, 1:])
        cnt = sat[mh:, mw:] - sat[:-mh, mw:] - sat[mh:, :-mw] + sat[:-mh, :-mw]
        zero[j] = cnt == 0
    return zero


def zero_hours(catrain, mh, mw):
    """zh[y, x, t] = every cell of the mh x mw box anchored at (y, x) is 0 or NaN in hour t (C order, (H-mh+1) x
    (W-mw+1) x T), and ok = the storm has no -0.0. Window s of a duration d is then bitwise identical to window s-1
    over the box when zh[y, x, s-1] and zh[y, x, s-1+d]: rolling_sum builds window s >= 1 as C[s+d-1] - C[s-1] and
    window 0 as C[d-1] from the float64 nancumsum C (NaN -> +0, sequential along time), and C[t] = C[t-1] + 0 is exactly
    C[t-1] (C >= +0 without -0.0 values), so both windows are fl32(C[s+d-2] - C[s-2]) (or fl32(C[d-1] - +0) =
    fl32(C[d-1]) for s = 1) at every cell of the box."""
    T, H, W = catrain.shape
    ok = not bool(np.any((catrain == 0) & np.signbit(catrain)))
    zh = np.empty((H - mh + 1, W - mw + 1, T), dtype=np.bool_)
    _zero_hours_nb(catrain, mh, mw, zh)
    return zh, ok


@jit(nopython=True)          # no fastmath: the NaN tests must stay
def _zero_hours_nb(catrain, mh, mw, zh):
    # sliding counts (rows of width mw, then columns of height mh) of the cells that are neither 0 nor NaN
    T, H, W = catrain.shape
    oh = H - mh + 1
    ow = W - mw + 1
    nz = np.empty(W, dtype=np.int64)
    rowcnt = np.empty((H, ow), dtype=np.int64)
    cnt = np.empty((oh, ow), dtype=np.int64)
    for t in range(T):
        for r in range(H):
            for q in range(W):
                v = catrain[t, r, q]
                nz[q] = (v != 0) & (v == v)
            c = 0
            for q in range(mw):
                c += nz[q]
            rowcnt[r, 0] = c
            for x in range(1, ow):
                c += nz[x + mw - 1] - nz[x - 1]
                rowcnt[r, x] = c
        for x in range(ow):
            cnt[0, x] = 0
        for r in range(mh):
            for x in range(ow):
                cnt[0, x] += rowcnt[r, x]
        for y in range(1, oh):
            for x in range(ow):
                cnt[y, x] = cnt[y - 1, x] + rowcnt[y + mh - 1, x] - rowcnt[y - 1, x]
        for y in range(oh):
            for x in range(ow):
                zh[y, x, t] = cnt[y, x] == 0


def zero_hours_ref(catrain, mh, mw):
    """numpy reference of zero_hours (summed-area tables), for tools/verify_fast_sst.py"""
    T, H, W = catrain.shape
    nz = (catrain != 0) & ~np.isnan(catrain)
    sat = np.zeros((T, H + 1, W + 1), dtype=np.int64)
    np.cumsum(np.cumsum(nz, axis=1, dtype=np.int64), axis=2, out=sat[:, 1:, 1:])
    cnt = sat[:, mh:, mw:] - sat[:, :-mh, mw:] - sat[:, mh:, :-mw] + sat[:, :-mh, :-mw]
    return np.ascontiguousarray(np.moveaxis(cnt == 0, 0, -1))


# ---------------------------------------------------------------------------------------------------------------------
# 3. approximate basin sums (running time sums of the hourly basin sums)
# ---------------------------------------------------------------------------------------------------------------------
def anchor_cumsums(catrain, trimmask, chunk=24):
    """P[y, x, t] = sum over hours < t of the basin sum (trimmask, float64) of the hourly field anchored at (y, x),
    t = 0..T; shape (H-mh+1, W-mw+1, T+1), C order (the T+1 values of one anchor are contiguous)."""
    T, H, W = catrain.shape
    mh, mw = trimmask.shape
    oh, ow = H - mh + 1, W - mw + 1
    m = np.asarray(trimmask, dtype=np.float64)
    P = np.empty((oh, ow, T + 1), dtype=np.float64)
    P[:, :, 0] = 0.
    if mh == 1 and mw == 1:
        _cumsum_1x1(catrain, float(m[0, 0]), P)
        return P
    # circular cross-correlation on an FFT grid >= (H, W): the anchors that keep the box inside the array (the only
    # ones RainyDay can draw) never wrap, so no padding to H+mh-1 x W+mw-1 is needed
    fshape = (spfft.next_fast_len(H, True), spfft.next_fast_len(W, True))
    kfft = np.conj(spfft.rfft2(m, fshape))
    run = np.zeros((oh, ow), dtype=np.float64)
    wk = fft_workers()
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        h = np.asarray(catrain[t0:t1], dtype=np.float64)
        h[np.isnan(h)] = 0.
        full = spfft.irfft2(spfft.rfft2(h, fshape, axes=(1, 2), workers=wk) * kfft, fshape, axes=(1, 2), workers=wk)
        corr = full[:, :oh, :ow]
        for t in range(t1 - t0):
            run += corr[t]
            P[:, :, t0 + t + 1] = run
    return P


@jit(nopython=True)
def _cumsum_1x1(catrain, m, P):
    T, H, W = catrain.shape
    for y in range(H):
        for x in range(W):
            P[y, x, 0] = 0.
    for t in range(T):
        for y in range(H):
            for x in range(W):
                v = catrain[t, y, x]
                P[y, x, t + 1] = P[y, x, t] + (v * m if v == v else 0.)


# ---------------------------------------------------------------------------------------------------------------------
# 4. per-draw kernel (flags of SSTalt_multidur: nopython, fastmath; RainyDay's numba_multimask_calc for every exact sum)
# ---------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, fastmath=True)
def table_kernel(allwin, offsets, steps, zero, P, zh, dedup, sstx, ssty, trimmask, maskheight, maskwidth, tolrel,
                 tolabs, hist):
    """rainsum[j, k] = SSTalt_multidur(allwin, offsets, sstx, ssty, trimmask, maskheight, maskwidth)[j, k].
    Candidates: windows with A >= Amax - tol (all windows when more than half qualify). Among them, a window that is
    bitwise identical over the box to the previous window (dedup, see zero_hours) is skipped when the previous one is
    evaluated or itself skipped that way: its exact sum is the same value, so the maximum does not change."""
    ndur = len(offsets) - 1
    nreals = len(sstx)
    rainsum = np.zeros((ndur, nreals), dtype=np.float32)
    maxw = 0
    for j in range(0, ndur):
        if offsets[j + 1] - offsets[j] > maxw:
            maxw = offsets[j + 1] - offsets[j]
    A = np.empty(maxw, dtype=np.float64)
    nexact = 0
    nzero = 0
    nskip = 0
    for k in range(0, nreals):
        y = int(ssty[k])
        x = int(sstx[k])
        for j in range(0, ndur):
            if zero[j, y, x]:
                rainsum[j, k] = 0.
                nzero += 1
            else:
                d = steps[j]
                nw = offsets[j + 1] - offsets[j]
                A[0] = P[y, x, d] - P[y, x, 0]
                amax = A[0]
                for s in range(1, nw):
                    A[s] = P[y, x, s + d] - P[y, x, s]
                    if A[s] > amax:
                        amax = A[s]
                thr = amax - (tolrel * (amax if amax > 0. else 0.) + tolabs)
                ncand = 0
                for s in range(0, nw):
                    if A[s] >= thr:
                        ncand += 1
                doall = 2 * ncand > nw
                if doall:
                    hist[6] += 1
                elif ncand == 1:
                    hist[0] += 1
                elif ncand == 2:
                    hist[1] += 1
                elif ncand <= 4:
                    hist[2] += 1
                elif ncand <= 8:
                    hist[3] += 1
                elif ncand <= 16:
                    hist[4] += 1
                else:
                    hist[5] += 1
                # the loop of SSTalt_multidur over the candidate windows only (ascending)
                storesum = 0.
                covered = False
                for kk in range(offsets[j], offsets[j + 1]):
                    s = kk - offsets[j]
                    if dedup and covered and s > 0 and zh[y, x, s - 1] and zh[y, x, s - 1 + d]:
                        if doall or A[s] >= thr:
                            nskip += 1                  # a candidate whose sum is already in storesum
                        continue
                    if doall or A[s] >= thr:
                        tempsum = RDF.numba_multimask_calc(allwin[kk], trimmask, y, x, maskheight, maskwidth) * 1.
                        if tempsum > storesum:
                            storesum = tempsum
                        nexact += 1
                        covered = True
                    else:
                        covered = False
                rainsum[j, k] = storesum
    return rainsum, nexact, nzero, nskip


def tolerances(mh, mw, P):
    n = mh * mw
    gamma = n * U32 / (1. - n * U32)
    eta = ETA_REL * max(float(np.max(P[:, :, -1])), 0.)
    return 4. * (gamma + U32), 4. * eta


def spot_check(catrain, trimmask, P, eta, nanch=4):
    """direct float64 running basin sums at a few random anchors (default_rng) must match P to within eta / 1000
    (the FFT error is ~1e-8 eta on the validation storms; the candidate tolerance assumes <= eta)"""
    mh, mw = trimmask.shape
    oh, ow = P.shape[:2]
    rng = _rng()
    m = np.asarray(trimmask, dtype=np.float64)
    worst = 0.
    for _ in range(nanch):
        y, x = int(rng.integers(0, oh)), int(rng.integers(0, ow))
        h = np.asarray(catrain[:, y:y + mh, x:x + mw], dtype=np.float64)
        h[np.isnan(h)] = 0.
        direct = np.cumsum(np.einsum("tij,ij->t", h, m))
        worst = max(worst, float(np.max(np.abs(P[y, x, 1:] - direct))))
    STATS["fft_err"] = max(STATS["fft_err"], worst / eta if eta > 0. else (0. if worst == 0. else np.inf))
    if worst > 1e-3 * eta:
        sys.exit("MULTIDURATIONENGINE 'table': FFT basin sums off by %.3g (eta %.3g); use 'fast'" % (worst, eta))


def table_sst(catrain, allwin, offsets, steps, sstx, ssty, trimmask, maskheight, maskwidth, rng=None, verify=0.):
    """rainsum[j, k] (float32) for draws (sstx, ssty) and the durations of the window stack allwin/offsets (steps =
    hours per window, offsets[j+1]-offsets[j] = T-steps[j]+1); bit-identical to
    RainyDay.SSTalt_multidur(allwin, offsets, sstx, ssty, trimmask, maskheight, maskwidth)."""
    t0 = time.perf_counter()
    if allwin.dtype != np.float32 or not allwin.flags.c_contiguous or trimmask.dtype != np.float32 \
            or not trimmask.flags.c_contiguous or sstx.dtype != np.int32 or ssty.dtype != np.int32:
        sys.exit("MULTIDURATIONENGINE 'table': unexpected array types (needs float32 C stack/mask, int32 locations)")
    steps = np.asarray(steps, dtype=np.int64)
    if np.any(offsets[1:] - offsets[:-1] != catrain.shape[0] - steps + 1):
        sys.exit("MULTIDURATIONENGINE 'table': window counts do not match the storm length")
    zero = zero_maps(allwin, offsets, maskheight, maskwidth)
    zh, dedup = zero_hours(catrain, maskheight, maskwidth)
    P = anchor_cumsums(catrain, trimmask)
    if not np.all(np.isfinite(P[:, :, -1])):
        sys.exit("MULTIDURATIONENGINE 'table': non-finite basin sums (check the storm catalog)")
    tolrel, tolabs = tolerances(maskheight, maskwidth, P)
    spot_check(catrain, trimmask, P, tolabs / 4.)
    t1 = time.perf_counter()
    hist = np.zeros(len(HBINS), np.int64)
    rainsum, nexact, nzero, nskip = table_kernel(allwin, offsets, steps, zero, P, zh, dedup, sstx, ssty, trimmask,
                                                 int(maskheight), int(maskwidth), tolrel, tolabs, hist)
    t2 = time.perf_counter()
    check_signatures()
    STATS["seconds_prep"] += t1 - t0
    STATS["seconds_kernel"] += t2 - t1
    STATS["storms"] += 1
    STATS["draws"] += len(sstx)
    STATS["cells"] += len(sstx) * (len(offsets) - 1)
    STATS["zero"] += nzero
    STATS["exact_evals"] += nexact
    STATS["dedup_skips"] += nskip
    STATS["nodedup_storms"] += 0 if dedup else 1
    STATS["brute_evals"] += len(sstx) * int(offsets[-1])
    STATS["hist"] += hist
    if verify > 0. and len(sstx) > 0:
        sel = np.sort(rng.choice(len(sstx), size=max(1, int(np.ceil(verify * len(sstx)))), replace=False))
        ref = RDF.SSTalt_multidur(allwin, offsets, np.ascontiguousarray(sstx[sel]), np.ascontiguousarray(ssty[sel]),
                                  trimmask, maskheight, maskwidth)
        if not np.array_equal(ref.view(np.uint32), rainsum[:, sel].view(np.uint32)):
            bad = np.argwhere(ref.view(np.uint32) != rainsum[:, sel].view(np.uint32))
            j, kk = bad[0]
            sys.exit("RAINYDAY_VERIFY_TABLE: MISMATCH duration index %d draw %d (x %d, y %d): table %r, brute force %r"
                     % (j, sel[kk], sstx[sel[kk]], ssty[sel[kk]], rainsum[j, sel[kk]], ref[j, kk]))
        STATS["verified"] += len(sel)
    return rainsum


def table_multidur(catrain, substeps, sstx, ssty, trimmask, maskheight, maskwidth):
    """MULTIPLEDURATIONS sub-durations of one storm: rainsum[j, k], bit-identical to the "fast" engine
    (SSTalt_multidur on the concatenated float32 rolling sums)."""
    verify = verify_fraction()
    allwin, offsets, _ = window_stack(catrain, substeps, check=(verify > 0. or not _FIRST["stack_checked"]))
    _FIRST["stack_checked"] = True
    return table_sst(catrain, allwin, offsets, substeps, sstx, ssty, trimmask, maskheight, maskwidth,
                     rng=_rng(), verify=verify)


def table_durcorrection(catrain, passrain, nstep, sstx, ssty, trimmask, maskheight, maskwidth):
    """Single run with DURATIONCORRECTION true: passrain = float32 rolling sums of nstep steps (as the driver builds
    them); bit-identical to SSTalt_multidur(passrain, [0, nwin], ...)[0]."""
    allwin = np.ascontiguousarray(passrain, dtype="float32")
    offsets = np.array([0, allwin.shape[0]], dtype=np.int64)
    return table_sst(catrain, allwin, offsets, [nstep], sstx, ssty, trimmask, maskheight, maskwidth,
                     rng=_rng(), verify=verify_fraction())[0]


def parent_sst(passrain, sstx, ssty, trimmask, maskheight, maskwidth):
    """Parent duration without duration correction: SSTalt(passrain, ..., durcheck=False)[0] computed by the compiled
    SSTalt_multidur with one window (no numba object mode). Needs float32 passrain/trimmask (as RainyDay builds them:
    nansum of the float32 catalog); returns None otherwise (caller keeps SSTalt)."""
    if passrain.dtype != np.float32 or trimmask.dtype != np.float32 or sstx.dtype != np.int32:
        return None
    out = RDF.SSTalt_multidur(np.ascontiguousarray(passrain[None]), np.array([0, 1], dtype=np.int64), sstx, ssty,
                              trimmask, maskheight, maskwidth)[0]
    check_signatures()
    v = verify_fraction()
    if v > 0. and len(sstx) > 0:
        sel = np.sort(_rng().choice(len(sstx), size=max(1, int(np.ceil(v * len(sstx)))), replace=False))
        ref, _ = RDF.SSTalt(passrain, np.ascontiguousarray(sstx[sel]), np.ascontiguousarray(ssty[sel]), trimmask,
                            maskheight, maskwidth, durcheck=False)
        if not np.array_equal(np.asarray(ref, np.float32).view(np.uint32), out[sel].view(np.uint32)):
            k = int(np.argmax(np.asarray(ref, np.float32).view(np.uint32) != out[sel].view(np.uint32)))
            sys.exit("RAINYDAY_VERIFY_TABLE: PARENT MISMATCH draw %d: compiled %r, SSTalt %r" % (sel[k], out[sel[k]], ref[k]))
        STATS["verified_parent"] += len(sel)
    return out


class StormSelector:
    """sel(i) = np.nonzero(whichstorms == i) (C order) for every storm i, from one stable argsort of whichstorms:
    whichx[sel(i) + (pt,)] is whichx[whichstorms == i, pt] (same elements, same order) without a full boolean scan of
    whichstorms at every use. Checked against np.nonzero in verify mode (RAINYDAY_VERIFY_TABLE)."""

    def __init__(self, whichstorms, nstorms):
        flat = whichstorms.ravel()
        self.shape = whichstorms.shape
        self.order = np.argsort(flat, kind="stable")          # stable: ascending flat index (C order) within a storm
        self.bounds = np.searchsorted(flat[self.order], np.arange(int(nstorms) + 1), side="left")
        self.whichstorms = whichstorms

    def __call__(self, i):
        i = int(i)
        sel = np.unravel_index(self.order[self.bounds[i]:self.bounds[i + 1]], self.shape)
        if verify_fraction() > 0.:
            ref = np.nonzero(self.whichstorms == i)
            if len(ref) != len(sel) or not all(np.array_equal(a, b) for a, b in zip(ref, sel)):
                sys.exit("RAINYDAY_VERIFY_TABLE: StormSelector differs from np.nonzero(whichstorms==%d) (bug)" % i)
        return sel


_RNG = []


def _rng():
    if not _RNG:
        _RNG.append(np.random.default_rng(int(os.environ.get("RAINYDAY_VERIFY_SEED", "20261022"))))
    return _RNG[0]


# numba_multimask_calc specialization used by SSTalt_multidur and table_kernel (float32 2-D C window and mask, int32
# anchor, int64 sizes). Verify mode (and tools/verify_fast_sst.py, which sets ALLOW_OBJECTMODE_SIG) also calls SSTalt
# (object mode: int64 anchor), which adds a second one.
ALLOW_OBJECTMODE_SIG = [False]


def check_signatures():
    sigs = [tuple(str(a) for a in s) for s in RDF.numba_multimask_calc.signatures]
    want = ("array(float32, 2d, C)", "array(float32, 2d, C)", "int32", "int32", "int64", "int64")
    allow64 = verify_fraction() > 0. or ALLOW_OBJECTMODE_SIG[0]
    extra = [s for s in sigs if s != want and not (allow64 and s[2:] == ("int64",) * 4)]
    if want not in sigs or extra:
        sys.exit("MULTIDURATIONENGINE 'table': unexpected numba_multimask_calc specializations %r" % (sigs,))


def report():
    s = STATS
    if s["storms"] == 0:
        return
    nz = max(s["cells"] - s["zero"], 1)
    print("MULTIDURATIONENGINE 'table': %d storms, %d draws x durations, %d zero-rule (%.1f%%); exact window sums %d "
          "(%.2f per nonzero draw x duration) vs %d brute force (%.1fx fewer); prep %.1f s, kernel %.1f s"
          % (s["storms"], s["cells"], s["zero"], 100. * s["zero"] / max(s["cells"], 1), s["exact_evals"],
             s["exact_evals"] / nz, s["brute_evals"], s["brute_evals"] / max(s["exact_evals"], 1),
             s["seconds_prep"], s["seconds_kernel"]))
    print("  identical-window candidates skipped: %d; storms without that shortcut (-0.0 values): %d"
          % (s["dedup_skips"], s["nodedup_storms"]))
    print("  candidates per nonzero draw x duration: " + ", ".join(
        "%s: %d (%.2f%%)" % (b, c, 100. * c / nz) for b, c in zip(HBINS, s["hist"])))
    print("  FFT spot check: max |P_fft - P_direct| / eta = %.3g (must be < 1e-3)" % s["fft_err"])
    if s["verified"] or s["verified_parent"]:
        print("  RAINYDAY_VERIFY_TABLE: %d sub-duration draws and %d parent draws brute-forced, 0 mismatches"
              % (s["verified"], s["verified_parent"]))
