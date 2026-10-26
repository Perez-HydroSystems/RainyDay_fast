#!/usr/bin/env python
"""
GP 2026-10-26: unit tests of RECORDCI (Source/RainyDay_utilities_Py3/fast_record.py), record (sampling) uncertainty.

    python tools/test_record_ci.py            (RainyDay_Env; ~1 min; prints PASS/FAIL per test and ALL PASS at the end)

 1. pool values vs RainyDay's own transposition (SSTalt_multidur on the rolling_sum windows, the parent through
    np.nansum) at EVERY anchor of small synthetic storms (rain blobs, drizzle near the 0.5 zero-rule threshold, NaN),
    timeres 60 and 30 min, fractional and 1x1 masks, rectangular (all anchors) and irregular (a subset) domains,
    with and without duration correction: zero-rule anchors identical, other depths within 2e-6 (float32 sums)
 2. Pool.levels vs np.partition of the expanded (weight-repeated) values, bitwise (PDS and AMS rules), including the
    pool-extension path
 3. Pool.expected vs a direct simulation of RainyDay's PDS (Poisson counts, top-k) and AMS (annual maxima, k-th
    largest, -9999 for years without storms) processes, weighted, within 4.5 Monte Carlo standard errors
 4. Pool.levels / Pool.expected vs a verbatim copy of the prototype record_boot.Pool (storm weights), bitwise
 5. year draws: block 1 == np.random.default_rng(seed).integers(0, NY, (B, NY)) (the prototype's draws); moving
    blocks are runs of consecutive record years; counts and jackknife weights
 6. helpers: THR32 (float32 rounding at 0.5), cum_counts vs brute force, option parsing (strings and JSON booleans)
GP 2026-10-26: FREQENGINE (exact engine, fast_record.ExactFA and the new Pool methods):
 7. Pool._values_at vs the sorted expanded values (weights, pool extension)
 8. Pool.cdf / Pool.quantiles / Pool.expected_minmax / Pool.expected vs the direct simulation simulate(), PDS and AMS,
    all-ones and sparse weights (including rows where P(fewer than k events) is large: -9999), on gamma pools with
    zeros and ties and on pools of synthetic storms (storm_values) at timeres 60 and 30 min: quantile definition
    (P(X <= Q) >= q >= P(X < Q)) exactly on the pool and within the binomial error of the simulation, CDF within
    4.5 SE, expected minimum / maximum of N realizations within 4.5 SE; F(-9999) = P(fewer than k events)
 9. rows: selected_rows and ExactFA.exceedp/returnperiod == RainyDay.rank_frequency's, bitwise (RETURNLEVELS lists,
    'all', NYEARS 100 and 1000)
10. ExactFA mean / level == RecordCI sst_expected / sst_level bitwise (one group vs the record years, shared pool)
11. FREQENGINE parsing and the check_options message
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import gammainc, gammaincc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Source"))
import RainyDay_utilities_Py3.RainyDay_functions as RDF   # noqa: E402
import RainyDay_utilities_Py3.fast_record as FR           # noqa: E402

RES = []


def report(ok, msg):
    RES.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# 1. pool values vs RainyDay
# ---------------------------------------------------------------------------------------------------------------------
def synthetic_storm(rng, T, H, W):
    r = np.zeros((T, H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(3):                                       # moving rain cells
        y0, x0 = rng.uniform(0, H), rng.uniform(0, W)
        vy, vx = rng.normal(0, 0.4, 2)
        amp, sig = rng.uniform(3, 25), rng.uniform(1.5, 4)
        t0, t1 = sorted(rng.integers(0, T, 2))
        for t in range(t0, t1 + 1):
            r[t] += (amp * np.exp(-((yy - y0 - vy * t) ** 2 + (xx - x0 - vx * t) ** 2) / (2 * sig ** 2))).astype(np.float32)
    drz = rng.random((H, W)) < 0.3                           # drizzle: cells near the 0.5 threshold of the zero rule
    for t in rng.integers(0, T, 6):
        r[t][drz] += rng.uniform(0.05, 0.45, drz.sum()).astype(np.float32)
    r[r < 0.02] = 0.
    r[rng.random(r.shape) < 0.01] = -9999.                    # missing values
    r[np.less(r, 0.)] = np.nan                                # as RainyDay: catrain[catrain < 0] = nan
    return r


def rainyday_depths(catrain, steps, fullsum, trim, mnorm, timeres, sstx, ssty):
    """RainyDay's transposed depths of every duration at the anchors (sstx, ssty), as RainyDay_Py3.py computes them:
    parent without duration correction: passrain = np.nansum(catrain, axis=0), one window (SSTalt_multidur with one
    window = SSTalt(durcheck=False), see fast_sst.parent_sst); every other duration: rolling_sum windows, SSTalt_multidur
    (= SSTalt(durcheck=True)); then * timeres/60./mnorm in float32."""
    mh, mw = trim.shape
    out = []
    for s, full in zip(steps, fullsum):
        if full:
            win = np.nansum(catrain, axis=0)[None]
        else:
            win = np.array(RDF.rolling_sum(catrain, s), dtype="float32")
        d = RDF.SSTalt_multidur(np.ascontiguousarray(win, dtype=np.float32), np.array([0, win.shape[0]], dtype=np.int64),
                                sstx, ssty, trim, mh, mw)[0]
        out.append(d * timeres / 60. / mnorm)
    return out


def test_pool_vs_rainyday():
    rng = np.random.default_rng(7)
    masks = {"4x5 fractional": np.array([[0, .3, 1, 1, .2], [.5, 1, 1, 1, .8], [.6, 1, 1, 1, 1], [0, .1, .9, .4, 0]],
                                         dtype=np.float32),
             "1x1": np.ones((1, 1), dtype=np.float32)}
    H, W = 22, 27
    for timeres in (60, 30):
        T = 24 if timeres == 60 else 36
        for mname, trim in masks.items():
            mnorm = np.sum(trim)
            mh, mw = trim.shape
            for durcorr in (False, True):
                if durcorr:      # single duration-corrected run: rolling windows of DURATION over a longer catalog
                    steps, fullsum = [T // 3], [False]
                else:            # parent = whole catalog + MULTIPLEDURATIONS sub-durations
                    steps = [T, 1, 2, 3, 6, T // 2]
                    fullsum = [True] + [False] * 5
                worst, nzero, nz_ref, nanch, mism = 0., 0, 0, 0, 0
                for k in range(4):
                    cat = synthetic_storm(rng, T, H, W)
                    cat0 = cat.copy()
                    # rectangular domain: every anchor, row-major
                    yy, xx = np.mgrid[0:H - mh + 1, 0:W - mw + 1]
                    ymask, xmask = yy.ravel().astype(np.int32), xx.ravel().astype(np.int32)
                    mine = FR.storm_values(cat, trim, mnorm, timeres / 60., steps, fullsum, None)
                    # irregular domain: a random subset of anchors (row-major order), passed as (ymask, xmask)
                    keep = rng.random(len(ymask)) < 0.6
                    sub = FR.storm_values(cat, trim, mnorm, timeres / 60., steps, fullsum,
                                          (ymask[keep].astype(np.int64), xmask[keep].astype(np.int64)))
                    assert np.array_equal(cat, cat0, equal_nan=True), "catrain modified"
                    ref = rainyday_depths(cat, steps, fullsum, trim, mnorm, timeres, xmask, ymask)
                    for j in range(len(steps)):
                        a, b = mine[j], ref[j]
                        if not np.array_equal(sub[j], a[keep]):
                            mism += 1
                        z1, z2 = a == 0, b == 0
                        mism += int(np.sum(z1 != z2))
                        nzero += int(z1.sum())
                        nz_ref += int(z2.sum())
                        nanch += a.size
                        ok = ~z2
                        if ok.any():
                            worst = max(worst, float(np.max(np.abs(a[ok].astype(np.float64) / b[ok] - 1))))
                report(mism == 0 and worst < 2e-6,
                       "pool vs RainyDay SSTalt_multidur: timeres %d, %s mask, %s: %d anchor x duration values, "
                       "%d zero-rule (RainyDay %d), max rel diff %.2e, mismatches %d"
                       % (timeres, mname, "duration-corrected" if durcorr else "parent + 5 sub-durations", nanch,
                          nzero, nz_ref, worst, mism))


# ---------------------------------------------------------------------------------------------------------------------
# 2. levels vs np.partition, 4. vs the prototype Pool
# ---------------------------------------------------------------------------------------------------------------------
def expanded(vals, groups, w):
    """every value of storm i repeated w[groups[i]] times"""
    return np.repeat(vals, w[groups], axis=0).ravel()


def partition_level(v, Wt, sw, T, nyears, calctype):
    lam = sw / nyears
    p = (-np.log1p(-1.0 / T) / lam) if calctype == "ams" else 1.0 / (lam * T)
    x = Wt * (1 - p)
    k = int(np.clip(round(x), 0, Wt - 1)) if np.isfinite(x) else 0
    return float(np.partition(v, k)[k])


def test_levels():
    rng = np.random.default_rng(11)
    n, nv, G = 40, 300, 20
    vals = rng.gamma(0.6, 12., (n, nv)).astype(np.float32)
    vals[rng.random(vals.shape) < 0.2] = 0.                    # zeros and ties
    vals[3, :50] = vals[7, :50]                                 # exact ties across storms
    groups = np.arange(n) // 2
    nsim = 1000
    ks = np.array([1000, 500, 200, 100, 40, 10, 4, 1])
    Ts = nsim / ks
    W = np.vstack([np.ones(G, np.int64), FR.draw_counts(FR.year_draws(30, G, 5), G)])
    for calctype in ("pds", "ams"):
        for mreq, label in ((FR.m_needed(nv, ks, G, nsim, calctype), "m_needed"), (5., "tiny pool, extended")):
            pool = FR.Pool(vals, groups, G, mreq)
            M0 = pool.M
            lv = pool.levels(W, Ts, G, calctype)
            bad = 0
            for b, w in enumerate(W):
                v = expanded(vals, groups, w)
                sw = int(np.sum(w[groups]))
                for t, T in enumerate(Ts):
                    bad += partition_level(v, v.size, sw, T, G, calctype) != lv[b, t]
            report(bad == 0, "levels vs np.partition (%s, %s; pool %d -> %d values, %d extensions): %d draws x %d T, "
                   "%d differ" % (calctype.upper(), label, M0, pool.M, pool.extended, len(W), len(Ts), bad))


class ProtoPool:
    """Verbatim copy of record_boot.Pool (24_record_uncertainty/scripts/record_boot.py, GP 2026-10-26), the reference
    of the port (storm weights; np.trapz as the prototype)."""

    def __init__(self, vals, m_needed, factor=4.0):
        self.vals = vals
        self.n, self.nv = vals.shape
        self._build(int(factor * m_needed) + 1000)

    def _build(self, M):
        flat = self.vals.ravel()
        N = flat.size
        M = min(N, M)
        if M < N:
            idx = np.argpartition(flat, N - M)[N - M:]
        else:
            idx = np.arange(N)
        order = np.argsort(-flat[idx], kind="stable")
        idx = idx[order]
        self.v = flat[idx]
        self.s = (idx // self.nv).astype(np.int32)
        self.M = M
        self.full = M == N

    def extend(self):
        self._build(4 * self.M)

    def targets(self, w, Ts, nyears):
        sw = int(w.sum())
        Wt = sw * self.nv
        lam = sw / nyears
        ms = []
        for T in Ts:
            p = 1.0 / (lam * T)
            k = int(np.clip(round(Wt * (1 - p)), 0, Wt - 1))
            ms.append(Wt - k)
        return np.array(ms)

    def levels(self, W, Ts, nyears):
        W = np.atleast_2d(W)
        out = np.empty((W.shape[0], len(Ts)))
        b = 0
        while b < W.shape[0]:
            w = W[b]
            ms = self.targets(w, Ts, nyears)
            D = np.cumsum(w[self.s], dtype=np.int64)
            e = np.searchsorted(D, ms, side="left")
            if (e >= self.M).any():
                assert not self.full
                self.extend()
                continue
            out[b] = self.v[e]
            b += 1
        return out

    def expected(self, W, Ts, nyears, nsim, ngrid=20000, tol=1e-9):
        W = np.atleast_2d(W)
        ks = np.array([int(round(nsim / T)) for T in Ts])
        while True:
            lo, hi = float(self.v[-1]), float(self.v[0])
            grid = np.geomspace(max(lo, 1e-3), hi * (1 + 1e-7), ngrid)
            pos = np.searchsorted(-self.v, -grid, side="right")
            out = np.empty((W.shape[0], len(Ts)))
            ok = True
            for b in range(W.shape[0]):
                D = np.concatenate([[0], np.cumsum(W[b][self.s], dtype=np.int64)])
                mu = nsim * D[pos] / (nyears * self.nv)
                P = gammainc(ks[:, None].astype(float), mu[None, :])
                if (P[:, 0] < 1 - tol).any():
                    ok = False
                    break
                out[b] = grid[0] + np.trapz(P, grid, axis=1)
            if ok:
                return out
            assert not self.full
            self.extend()


def test_vs_prototype():
    rng = np.random.default_rng(3)
    n, nv, NY, nsim = 120, 800, 22, 1000
    vals = rng.gamma(0.5, 15., (n, nv)).astype(np.float32)
    groups = rng.integers(0, NY, n)
    Ts = [2, 5, 10, 25, 50, 100, 200, 250, 500, 1000]
    ks = np.array([nsim // T for T in Ts])
    C = FR.draw_counts(FR.year_draws(40, NY, 20261026), NY)
    C = np.vstack([np.ones(NY, np.int64), C, FR.jackknife_counts(NY)[:5]])
    Wst = C[:, groups].astype(np.int32)                         # storm weights of the prototype
    for nyears, label in ((NY, "bootstrap"), (NY - 1, "jackknife nyears")):
        mreq = (ks.max() + 10 * np.sqrt(float(ks.max())) + 10) * NY * nv / nsim     # prototype m_needed (nyears = NY)
        pp = ProtoPool(vals, mreq)
        mp = FR.Pool(vals, groups, NY, FR.m_needed(nv, ks, NY, nsim))
        a1, b1 = pp.levels(Wst, Ts, nyears), mp.levels(C, nsim / ks, nyears)
        a2, b2 = pp.expected(Wst, Ts, nyears, nsim), mp.expected(C, ks, nyears, nsim)
        report(np.array_equal(a1, b1) and np.array_equal(a2, b2) and pp.M == mp.M,
               "Pool vs the prototype record_boot.Pool (%s, %d weight vectors, pool %d / %d): levels bitwise %s, "
               "expected bitwise %s (max rel diff %.1e)" % (label, len(C), pp.M, mp.M, np.array_equal(a1, b1),
                                                             np.array_equal(a2, b2), np.max(np.abs(b2 / a2 - 1))))


# ---------------------------------------------------------------------------------------------------------------------
# 3. expected vs direct simulation
# ---------------------------------------------------------------------------------------------------------------------
def simulate(vals, groups, w, nyears, nsim, ks, calctype, R, rng):
    """R realizations of RainyDay's resampling with storm weights w[groups] and lambda* = sum(w)/nyears; returns the
    k-th largest event (PDS) or annual maximum (AMS) of each realization, -9999 where there is none."""
    n, nv = vals.shape
    sw = w[groups].astype(np.float64)
    lam = sw.sum() / nyears
    prob = sw / sw.sum()
    out = np.empty((R, len(ks)))
    if calctype == "pds":
        N = rng.poisson(nsim * lam, R)
        tot = N.sum()
        st = rng.choice(n, tot, p=prob)
        an = rng.integers(0, nv, tot)
        v = vals[st, an].astype(np.float64)
        ofs = np.r_[0, np.cumsum(N)]
        for r in range(R):
            x = np.sort(v[ofs[r]:ofs[r + 1]])[::-1]
            for j, k in enumerate(ks):
                out[r, j] = x[k - 1] if len(x) >= k else -9999.
    else:
        N = rng.poisson(lam, (R, nsim))
        tot = N.sum()
        st = rng.choice(n, tot, p=prob)
        an = rng.integers(0, nv, tot)
        v = vals[st, an].astype(np.float64)
        yid = np.repeat(np.arange(R * nsim), N.ravel())
        amax = np.full(R * nsim, -9999.)
        np.maximum.at(amax, yid, v)
        amax = np.sort(amax.reshape(R, nsim), axis=1)[:, ::-1]
        for j, k in enumerate(ks):
            out[:, j] = amax[:, k - 1]
    return out


def test_expected_vs_simulation():
    rng = np.random.default_rng(2026)
    n, nv, NY, nsim = 12, 40, 6, 40
    vals = rng.gamma(0.8, 10., (n, nv)).astype(np.float32)
    vals[rng.random(vals.shape) < 0.1] = 0.
    groups = np.arange(n) // 2
    ks = np.array([40, 20, 10, 4, 1])
    for calctype in ("pds", "ams"):
        for w in (np.ones(NY, np.int64), np.array([2, 0, 1, 1, 0, 2])):
            pool = FR.Pool(vals, groups, NY, FR.m_needed(nv, ks, NY, nsim, calctype))
            ex = pool.expected(w[None], ks, NY, nsim, calctype)[0]
            sim = simulate(vals, groups, w, NY, nsim, ks, calctype, 40000, rng)
            m, se = sim.mean(axis=0), sim.std(axis=0, ddof=1) / np.sqrt(sim.shape[0])
            zs = np.abs(m - ex) / np.maximum(se, 1e-12)
            report(np.all(zs < 4.5), "expected vs simulation (%s, weights %s, %s pool): T = %s: E %s, simulated %s, "
                   "|z| max %.2f" % (calctype.upper(), w.tolist(), "full" if pool.full else "top-M",
                                     ", ".join("%g" % (nsim / k) for k in ks), np.round(ex, 3).tolist(),
                                     np.round(m, 3).tolist(), zs.max()))
    # a top-M (not full) pool and the full pool of the same values vs the exact integral of the step function
    # P(X_(k) > x) (sum over the sorted values; no grid): the difference is the trapezoid quadrature error of the
    # prototype's 20000-point geometric grid (~1e-5 here, where the values span 10 decades), not a pool error
    vals = rng.gamma(0.5, 12., (60, 3000)).astype(np.float32)
    groups = np.arange(60) // 3
    C = FR.draw_counts(FR.year_draws(5, 20, 1), 20)
    for calctype, ks in (("pds", np.array([500, 100, 10, 1])), ("ams", np.array([100, 10, 1]))):
        top = FR.Pool(vals, groups, 20, FR.m_needed(3000, ks, 20, 1000, calctype))
        full = FR.Pool(vals, groups, 20, np.inf)
        a, b = top.expected(C, ks, 20, 1000, calctype), full.expected(C, ks, 20, 1000, calctype)
        v = full.v.astype(np.float64)
        dv = v - np.r_[v[1:], 0.]
        ex = np.array([[np.sum(dv * FR.Pool._prob(float(k), np.cumsum(w[full.g]) / (20 * 3000.) *
                                                  (1000. if calctype == "pds" else 1.), 1000, calctype)) for k in ks]
                       for w in C])
        ea, eb = np.max(np.abs(a / ex - 1)), np.max(np.abs(b / ex - 1))
        report(not top.full and full.full and ea < 1e-4 and eb < 1e-4,
               "expected (%s), top-M pool (%d of %d values) and full pool vs the exact step integral: max rel diff "
               "%.1e and %.1e (quadrature)" % (calctype.upper(), top.M, full.M, ea, eb))


# ---------------------------------------------------------------------------------------------------------------------
# 5. draws, 6. helpers
# ---------------------------------------------------------------------------------------------------------------------
def test_draws():
    for seed, B, NY in ((20261026, 1000, 22), (1, 7, 5)):
        a = FR.year_draws(B, NY, seed, 1)
        b = np.random.default_rng(seed).integers(0, NY, (B, NY))
        report(np.array_equal(a, b) and a.dtype == b.dtype,
               "block 1 == np.random.default_rng(%d).integers(0, %d, (%d, %d)) (the prototype's draws)"
               % (seed, NY, B, NY))
    NY, L, B = 22, 3, 200
    d = FR.year_draws(B, NY, 9, L)
    st = np.random.default_rng(9).integers(0, NY, (B, -(-NY // L)))
    ok = d.shape == (B, NY)
    for b in range(B):
        for j in range(0, NY, L):
            seg = d[b, j:j + L]
            ok &= bool(np.all(seg == (st[b, j // L] + np.arange(len(seg))) % NY))
    report(ok, "moving blocks (L = 3): every row = %d circular runs of consecutive years from the block starts, "
           "truncated to NY = 22" % -(-NY // L))
    c = FR.draw_counts(d, NY)
    report(np.all(c.sum(1) == NY) and np.array_equal(c[0], np.bincount(d[0], minlength=NY)), "draw counts")
    j = FR.jackknife_counts(5)
    report(np.array_equal(j, 1 - np.eye(5, dtype=int)), "jackknife weights")


def test_helpers():
    t = FR.THR32
    report(np.float32(t) == 0.5 and np.float32(np.nextafter(t, 0.)) < 0.5,
           "THR32: float32(a) >= 0.5 <=> a >= 0.5 - 2^-26")
    rng = np.random.default_rng(4)
    g = rng.integers(0, 7, 5000)
    pos = rng.integers(0, 5001, 300)
    cc = FR.cum_counts(g, 7, pos)
    ref = np.array([np.bincount(g[:p], minlength=7) for p in pos])
    report(np.array_equal(cc, ref), "cum_counts vs bincount of every prefix")
    P = FR.parse_options
    ok = P({}) is None and P({"RECORDCI": False}) is None and P({"RECORDCI": "false"}) is None
    o = P({"RECORDCI": True, "RECORDSAVE": False, "RECORDNBOOT": "200"})
    ok &= o["method"] == "bootstrap" and o["save"] is False and o["nboot"] == 200
    o = P({"RECORDCI": "Jackknife", "RECORDSAVE": "TRUE", "RECORDLEVEL": 95})
    ok &= o["method"] == "jackknife" and o["save"] is True and o["level"] == 95.
    for bad in ({"RECORDCI": "yes"}, {"RECORDCI": "bootstrap", "RECORDNBOOT": 1.5},
                {"RECORDCI": "bootstrap", "RECORDLEVEL": 100}, {"RECORDCI": "bootstrap", "RECORDSAVE": "maybe"}):
        try:
            P(bad)
            ok = False
        except SystemExit:
            pass
    report(ok, "option parsing (strings, JSON booleans, numeric strings; invalid values stop)")


# ---------------------------------------------------------------------------------------------------------------------
# GP 2026-10-26: FREQENGINE (exact engine)
# ---------------------------------------------------------------------------------------------------------------------
def test_values_at():
    rng = np.random.default_rng(21)
    n, nv, G = 30, 200, 10
    vals = rng.gamma(0.6, 12., (n, nv)).astype(np.float32)
    vals[rng.random(vals.shape) < 0.2] = 0.
    groups = np.arange(n) // 3
    bad, tot = 0, 0
    for w in (np.ones(G, np.int64), np.array([2, 0, 1, 3, 0, 1, 1, 0, 2, 1])):
        pool = FR.Pool(vals, groups, G, 50.)                   # tiny pool: forces extensions
        v = np.sort(expanded(vals, groups, w))[::-1]
        ms = np.r_[rng.integers(1, v.size + 1, 300), 1, v.size, v.size + 1, 0]
        got = pool._values_at(w, ms)
        ref = np.array([v[m - 1] if 1 <= m <= v.size else FR.MISSING for m in ms])
        bad += int(np.sum(got != ref))
        tot += ms.size
    report(bad == 0, "_values_at vs the sorted expanded values (weights, pool extensions, m = 0 / > Wt -> -9999): "
           "%d of %d differ" % (bad, tot))


def storm_pool(timeres, n, rng):
    """(n, nv) pool of synthetic storms (storm_values, parent 24 h without duration correction) at timeres."""
    trim = np.array([[0, .3, 1, 1, .2], [.5, 1, 1, 1, .8], [.6, 1, 1, 1, 1], [0, .1, .9, .4, 0]], dtype=np.float32)
    T = 24 if timeres == 60 else 48
    out = [FR.storm_values(synthetic_storm(rng, T, 22, 27), trim, np.sum(trim), timeres / 60., [T], [True], None)[0]
           for _ in range(n)]
    return np.vstack(out)


def test_exact_vs_simulation():
    """quantiles, CDF, expected min/max (and expected) vs RainyDay's resampling simulated directly (simulate)."""
    rng = np.random.default_rng(2027)
    R, Nr = 40000, 20                       # simulated realizations; realizations per min/max group
    probs = np.array([0.05, 0.25, 0.5, 0.75, 0.95, 0.01, 0.99])
    pools = []
    v = rng.gamma(0.8, 10., (12, 40)).astype(np.float32)
    v[rng.random(v.shape) < 0.1] = 0.
    v[2, :10] = v[5, :10]                   # ties
    pools.append(("gamma pool", v, np.arange(12) // 2, 6, 40, np.array([40, 20, 10, 4, 1])))
    for tr in (60, 30):
        sp = storm_pool(tr, 8, rng)
        pools.append(("storms timeres %d" % tr, sp, np.arange(8) // 2, 4, 40, np.array([40, 20, 10, 4, 1])))
    for label, vals, groups, NY, nsim, ks in pools:
        for calctype in ("pds", "ams"):
            for w in (np.ones(NY, np.int64), np.r_[1, np.zeros(NY - 1, np.int64)]):   # sparse: many -9999 rows
                pool = FR.Pool(vals, groups, NY, FR.m_needed(vals.shape[1], ks, NY, nsim, calctype))
                sim = simulate(vals, groups, w, NY, nsim, ks, calctype, R, rng)
                Q = pool.quantiles(w[None], ks, probs, NY, nsim, calctype)[0]                # (nk, nq)
                F, S = pool.cdf(w, ks, np.unique(np.r_[Q.ravel(), -9999.]), NY, nsim, calctype)
                xs = np.unique(np.r_[Q.ravel(), -9999.])
                # 1. definition, exactly on the pool: F(Q) >= q >= F(Q-) (Q- = the next float32 below Q)
                defin, simq, worst = 0, 0, 0.
                for a in range(len(ks)):
                    for b, q in enumerate(probs):
                        x = Q[a, b]
                        Fq = F[a, np.searchsorted(xs, x)]
                        xm = np.nextafter(np.float32(x), np.float32(-np.inf)) if x > 0 else -1e30
                        Fm = pool.cdf(w, ks[a:a + 1], [xm], NY, nsim, calctype)[0][0, 0] if x != FR.MISSING else 0.
                        defin += not (Fq >= q - 1e-12 and Fm <= q + 1e-12)
                        # 2. simulation: P(X <= Q) >= q >= P(X < Q) within the binomial error
                        se = np.sqrt(q * (1 - q) / R)
                        le, lt = np.mean(sim[:, a] <= x), np.mean(sim[:, a] < x)
                        simq += not (le >= q - 4.5 * se and lt <= q + 4.5 * se)
                        worst = max(worst, (q - le) / se, (lt - q) / se)
                # 3. CDF vs the empirical CDF at the quantiles and at -9999 (exact binomial test of the counts)
                cnt = np.sum(sim[:, :, None] <= xs[None, None, :], axis=0)                 # (nk, nx)
                pc = binom_p(cnt, R, F)
                Pall = pool._prob(ks.astype(float), nsim * pool._totals(w)[1] / (NY * pool.nv) if calctype == "pds"
                                  else pool._totals(w)[1] / (NY * pool.nv), nsim, calctype)
                miss = np.allclose(F[:, 0], 1. - Pall, rtol=1e-12, atol=1e-14)
                # 4. expected minimum / maximum of Nr realizations, and the mean. A -9999 row (fewer than k events)
                # can have a tiny probability (e.g. 1e-7) that still moves the mean by ~1e-3 x 9999: the simulation
                # cannot resolve it, so each expectation is checked as two parts: P(-9999) (binomial test of the
                # count of -9999) and the rest, E[X; X >= 0] = E[X] + 9999 P(-9999) (|z| of the simulated mean)
                emin, emax = pool.expected_minmax(w[None], ks, NY, nsim, Nr, calctype)
                ex = pool.expected(w[None], ks, NY, nsim, calctype)[0]
                g = sim[:R // Nr * Nr].reshape(R // Nr, Nr, len(ks))
                mn, mx = g.min(axis=1), g.max(axis=1)
                pmiss = (-np.expm1(Nr * np.log(Pall)), np.exp(Nr * np.log1p(-Pall)), 1. - Pall)   # min, max, X
                z, pm = [], []
                for e, s, p in zip((emin[0], emax[0], ex), (mn, mx, sim), pmiss):
                    pm.append(binom_p(np.sum(s == FR.MISSING, axis=0), s.shape[0], p).min())
                    cont = e - FR.MISSING * p                                    # E[X; X >= 0]
                    sp = np.where(s == FR.MISSING, 0., s)
                    se = sp.std(axis=0, ddof=1) / np.sqrt(s.shape[0])
                    dz = np.abs(sp.mean(axis=0) - cont)
                    zz = np.where(se > 0, dz / np.maximum(se, 1e-300), np.where(dz < 1e-6, 0., np.inf))
                    # the rest is tested where the simulation holds >= 500 non-missing values in expectation
                    z.append(np.where((1. - p) * s.shape[0] >= 500, zz, 0.))
                nmiss = int(np.sum(Q == FR.MISSING))
                ok = defin == 0 and simq == 0 and pc.min() > 1e-5 and miss and max(zz.max() for zz in z) < 4.5 \
                    and min(pm) > 1e-5
                report(ok, "exact vs simulation (%s, %s, weights %s; %d -9999 quantiles): quantile definition on the "
                       "pool %d bad, vs simulation %d bad (worst %.2f SE); CDF binomial p min %.1e; F(-9999) = P(N < k) "
                       "%s; E[min], E[max] of %d, E[X]: |z| max %.2f, %.2f, %.2f, P(-9999) binomial p min %.1e"
                       % (label, calctype.upper(), "ones" if w.sum() == NY else w.tolist(), nmiss, defin, simq, worst,
                          pc.min(), miss, Nr, z[0].max(), z[1].max(), z[2].max(), min(pm)))


def binom_p(c, n, p):
    """two-sided exact binomial p-value of the counts c ~ Binomial(n, p) (elementwise)"""
    from scipy.stats import binom
    c, p = np.broadcast_arrays(np.asarray(c, float), np.clip(np.asarray(p, float), 0., 1.))
    out = np.minimum(1., 2. * np.minimum(binom.cdf(c, n, p), binom.sf(c - 1, n, p)))
    return np.where((p == 0) & (c == 0) | (p == 1) & (c == n), 1., out)


def test_rows():
    bad = []
    for nsim in (100, 1000):
        for spec in ([2, 5, 10, 25, 50, 100, 200, 250, 500, 1000], [1, 1.5, 2, 7, 33.3, 100], "all"):
            alll = spec == "all"
            sl = None
            if not alll:
                sl = np.float32(spec)
                sl = sl[sl <= nsim + 0.00001]
                sl = sl[sl >= 0.99999]
            wr = np.random.default_rng(1).gamma(1., 10., (4, nsim, 3, 1)).astype(np.float32)
            _, ex, rp = RDF.rank_frequency(wr, "pds", nsim, alll, sl)
            rows, exf = FR.selected_rows(nsim, alll, sl)
            efa = FR.ExactFA("exact", [24.], [24], True, np.ones((1, 1), np.float32), 1., 60, (5, 6), None, 2, 3, "pds",
                             nsim, 4, alll, sl, "ensemble", None)
            if not (np.array_equal(exf[rows], ex) and np.array_equal(efa.exceedp, ex) and
                    np.array_equal(efa.returnperiod, rp) and np.array_equal(efa.ks, nsim - rows)):
                bad.append((nsim, str(spec)))
    report(not bad, "rows: selected_rows and ExactFA exceedp/returnperiod == RainyDay.rank_frequency bitwise "
           "(NYEARS 100/1000, RETURNLEVELS lists and 'all')%s" % ("" if not bad else ": " + str(bad)))


def test_exact_vs_recordci():
    """ExactFA (one group, shared pool) vs RecordCI (record-year groups): mean == sst_expected, level == sst_level,
    bitwise, for PDS and AMS; ExactFA on its own pool gives the same numbers."""
    rng = np.random.default_rng(5)
    trim = np.array([[0, .3, 1, 1], [.5, 1, 1, .8], [.6, 1, 1, 1]], dtype=np.float32)
    H, W, T, n = 20, 24, 24, 12
    storms = [synthetic_storm(rng, T, H, W) for _ in range(n)]
    years = np.repeat(np.arange(2002, 2006), 3)
    cattime = np.array([[np.datetime64("%d-07-01T00:00" % y) + np.timedelta64(h, "h") for h in range(T)] for y in years])
    steps, durs = [T, 1, 6], [24., 1., 6.]
    ok, msg = True, []
    for calctype in ("pds", "ams"):
        with tempfile.TemporaryDirectory() as d:
            opts = FR.parse_options({"RECORDCI": "bootstrap", "RECORDNBOOT": 10})
            rc = FR.RecordCI(opts, durs, steps, True, trim, np.sum(trim), 60, (H, W), None, cattime, list(range(2002, 2006)),
                             4, calctype, 100, False, np.float32([2, 5, 10, 25, 50, 100]), d, "t")
            ex = FR.ExactFA("both", durs, steps, True, trim, np.sum(trim), 60, (H, W), None, n, 4, calctype, 100, 10,
                            False, np.float32([2, 5, 10, 25, 50, 100]), "ensemble", None, record=rc)
            alone = FR.ExactFA("exact", durs, steps, True, trim, np.sum(trim), 60, (H, W), None, n, 4, calctype, 100,
                               10, False, np.float32([2, 5, 10, 25, 50, 100]), "ensemble", None)
            for i, s in enumerate(storms):
                rc.add_storm(i, s)
                alone.add_storm(i, s)
            for j in range(len(steps)):
                a = ex.compute(j)
                b = rc.estimates(j, ex.ks)
                c = alone.compute(j)
                same = (np.array_equal(a["mean"], b["ex_pt"]) and np.array_equal(a["level"], b["lv_pt"]) and
                        all(np.array_equal(a[k], c[k]) for k in ("mean", "level", "q", "exp_min", "exp_max")))
                ok &= same and ex.shared and rc.write_level and ex.pool is rc.pool
                msg.append("%s %gh %s" % (calctype.upper(), durs[j], "ok" if same else "DIFFER"))
    report(ok, "ExactFA mean/level == RecordCI sst_expected/sst_level bitwise (shared pool, one group vs record years), "
           "== ExactFA on its own pool: " + ", ".join(msg))


def test_engine_options():
    P = FR.parse_engine
    ok = P({}) == "montecarlo" and P({"FREQENGINE": "Exact"}) == "exact" and P({"FREQENGINE": " both "}) == "both"
    for bad in ({"FREQENGINE": "mc"}, {"FREQENGINE": True}, {"FREQENGINE": 1}):
        try:
            P(bad)
            ok = False
        except SystemExit:
            pass
    try:
        FR.check_options(None, True, "uniform", "poisson", "pds", "basin", "none", False, False, True, False,
                         who="FREQENGINE 'exact'")
        ok = False
    except SystemExit as e:
        ok &= "FREQENGINE 'exact'" in str(e) and "SCENARIOS" in str(e)
    report(ok, "FREQENGINE parsing (montecarlo default, case, invalid values stop) and the check_options message")


def main():
    test_helpers()
    test_draws()
    test_levels()
    test_vs_prototype()
    test_expected_vs_simulation()
    test_pool_vs_rainyday()
    test_engine_options()                    # GP 2026-10-26: FREQENGINE
    test_values_at()
    test_rows()
    test_exact_vs_recordci()
    test_exact_vs_simulation()
    print("ALL PASS" if RES and all(RES) else "SOME TESTS FAILED (%d of %d)" % (RES.count(False), len(RES)))
    sys.exit(0 if RES and all(RES) else 1)


if __name__ == "__main__":
    main()
