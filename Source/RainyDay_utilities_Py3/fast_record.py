"""
GP 2026-10-26: RainyDay_fast: record (sampling) uncertainty of the SST frequency estimate, JSON "RECORDCI".

RainyDay's ensemble spread (the realizations) holds the storm catalog fixed: it is the Monte Carlo variability of the
synthetic records, not the uncertainty that comes from having only NY years of rainfall. This module adds the latter.
It only adds outputs: every existing output stays bit-identical (it reads copies of RainyDay's arrays and uses its own
random generator, np.random.default_rng(RECORDSEED); the global np.random stream is never touched).

1. Pool (per storm, inside RainyDay's resampling loop, from the storm rain RainyDay has already read): the basin-average
   d-hour maximum of the storm at every location RainyDay can draw (the valid anchors of uniform transposition), for
   every duration of the run:
       v_d(storm, anchor) = max over the d-hour windows of sum_ij trimmask_ij P(t, y+i, x+j) * timeres/60 / mnorm,
   with RainyDay's own trimmask, mnorm and windows (float64 FFT correlation of the hourly fields, one cumulative sum
   per storm), and RainyDay's zero rule (0 when every cell of the box is < 0.5 in every window of the duration).
   It equals RainyDay's own transposed depths (float32 sums) to ~1e-6 (tools/test_record_ci.py).
2. Noise-free estimators of the SST model RainyDay samples (Poisson storm counts at lambda = n_storms/NY per year,
   storms equally likely, anchors uniform), for every return-period row of the FreqAnalysis CSV: row i of NYEARS
   synthetic years is the k-th largest event (PDS) or annual maximum (AMS), k = NYEARS - i, T = NYEARS/k.
       Lambda(x) = sum_i w_i #{a: v(i, a) >= x} / (NY n_anchors)       storms (or years) >= x per year, w_i = weights
       expected  E[X_(k)] = int_0^inf P(X_(k) > x) dx: the noise-free value of the RainyDay realization mean,
                 PDS  P(Poisson(NYEARS Lambda(x)) >= k)                 scipy.special.gammainc
                 AMS  P(Binomial(NYEARS, 1 - exp(-Lambda(x))) >= k)     scipy.special.betainc
                 (trapezoid on a geometric grid of 20000 points above the pool floor; -9999 when fewer than k events,
                 as RainyDay writes)
       level     the weighted quantile with Lambda(x_T) = 1/T (PDS) or -ln(1 - 1/T) (AMS)
3. Year bootstrap: B records of NY years drawn with replacement from the record years (moving blocks of RECORDBLOCK
   consecutive years, circular), or leave-one-year-out (jackknife); the weight of a storm is how many times its start
   year was drawn; lambda* = sum(w)/NY.
4. Outputs per duration: <name>_RecordBootstrap.csv (every bootstrap record, like "all_realizations"),
   <name>_RecordCI.csv (bands); once per run <scenario>_RecordDraws.csv (the drawn years).

Port of the validated prototype (Basin_IDF_Orientation_Elongation, PUBLIC/ANALYSIS/OUTPUT):
22_huc_ellipse_diagnostic/scripts/04_correlation_emulator.py (basin_series, dmax, levels) and
24_record_uncertainty/scripts/record_boot.py (Pool, year_draws, storm_weights, loyo_weights, m_needed). Validation:
22 leave-one-year-out RainyDay runs of the Iowa pixel, median difference -0.17 % (24_record_uncertainty/REPORT.md).

GP 2026-10-26: exact frequency engine, JSON "FREQENGINE" = "montecarlo" (default) | "exact" | "both" (class ExactFA).
From the same pool (StormPool, shared with RECORDCI: one copy), the frequency analysis of the SST model RainyDay samples,
without Monte Carlo noise, per FreqAnalysis row (X_(k): k-th largest event of NYEARS synthetic years, PDS, or k-th
largest annual maximum, AMS; k = NYEARS/T):
    mean        E[X_(k)] (Pool.expected, RainyDay's own number of record years as divisor of Lambda): the noise-free
                value of RainyDay's estimate (the mean of the realizations); depends on NYEARS
    level       Lambda(x_T) = 1/T (PDS) or -ln(1 - 1/T) (AMS) (Pool.levels); independent of NYEARS
    qNN         quantiles of X_(k), the T-yr value of ONE realization (Pool.quantiles):
                    P(X_(k) <= x) = P(Poisson(NYEARS Lambda_>(x)) < k)                       PDS
                                  = P(Binomial(NYEARS, 1 - exp(-Lambda_>(x))) < k)           AMS
                Lambda_>(x) = #{pool values > x} / (NY n_anchors) (strict count: the CDF is right-continuous and
                Q(q) = min{x: P(X_(k) <= x) >= q} is a pool value; ties and the atom at 0 are exact); -9999 where
                P(fewer than k events) >= q
    exp_min/max expected minimum / maximum of NREALIZATIONS realizations (Pool.expected_minmax): survival P^N and
                1 - (1 - P)^N in the integral of expected
"exact" draws no synthetic years (no np.random call after the seed; README_CHANGES B8) and writes <name>_FreqExact.csv
and .png instead of the realizations; "both" adds them to RainyDay's unchanged Monte Carlo outputs.
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import fftconvolve
from scipy.special import betainc, betaincinv, gammainc, gammaincc, gammaincinv, ndtri

import RainyDay_utilities_Py3.RainyDay_functions as RDF   # same module object as the driver's "RainyDay"
import RainyDay_utilities_Py3.fast_prof as FP

KEYS = ("RECORDCI", "RECORDNBOOT", "RECORDLEVEL", "RECORDSEED", "RECORDBLOCK", "RECORDSAVE")
MISSING = -9999.                 # RainyDay's value of a row without an event
THR32 = 0.5 - 2.0 ** -26         # float32(a) >= 0.5  <=>  a >= THR32 (a float64; round to nearest even)
NGRID = 20000                    # integration grid of expected() (as the prototype)
TOL = 1e-9                       # P(X_(k) > pool floor) must be > 1 - TOL, else the pool grows (as the prototype)
NZ = 1000                        # equally probable normal deviates of the Monte Carlo error (est band, jackknife)
ENGINES = ("montecarlo", "exact", "both")    # GP 2026-10-26: FREQENGINE
PROBS = (5., 25., 50., 75., 95.)             # GP 2026-10-26: quantiles of X_(k) in _FreqExact.csv (%)


# =====================================================================================================================
# options
# =====================================================================================================================
def _boolean(v, key):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    sys.exit(key + " must be 'true' or 'false' (got " + repr(v) + ")")


def _integer(v, key, lo):
    try:
        if isinstance(v, (bool, np.bool_)):
            raise ValueError
        x = float(v)
        if not x.is_integer():
            raise ValueError
        x = int(x)
    except (TypeError, ValueError):
        sys.exit(key + " must be an integer (got " + repr(v) + ")")
    if x < lo:
        sys.exit(key + " must be >= " + str(lo) + " (got " + str(x) + ")")
    return x


def parse_options(cardinfo):
    """RECORD* keys of the JSON -> dict, or None when RECORDCI is off (default). Strings or JSON booleans."""
    v = cardinfo.get("RECORDCI", "false")
    if v is None or isinstance(v, (bool, np.bool_)):
        method = "bootstrap" if v else None
    elif isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "false", "none", "no", "off"):
            method = None
        elif s in ("bootstrap", "true"):
            method = "bootstrap"
        elif s == "jackknife":
            method = "jackknife"
        else:
            sys.exit("RECORDCI must be 'false', 'bootstrap' or 'jackknife' (got " + repr(v) + ")")
    else:
        sys.exit("RECORDCI must be 'false', 'bootstrap' or 'jackknife' (got " + repr(v) + ")")
    if method is None:
        other = [k for k in KEYS[1:] if k in cardinfo]
        if other:
            print("NOTE: " + ", ".join(other) + " ignored: RECORDCI is off")
        return None
    try:
        level = float(cardinfo.get("RECORDLEVEL", 90))
    except (TypeError, ValueError):
        sys.exit("RECORDLEVEL must be a number in (0, 100) (got " + repr(cardinfo.get("RECORDLEVEL")) + ")")
    if not 0. < level < 100.:
        sys.exit("RECORDLEVEL must be in (0, 100) (got " + str(level) + ")")
    opts = dict(method=method, nboot=_integer(cardinfo.get("RECORDNBOOT", 1000), "RECORDNBOOT", 2), level=level,
                seed=_integer(cardinfo.get("RECORDSEED", 20261026), "RECORDSEED", 0),
                block=_integer(cardinfo.get("RECORDBLOCK", 1), "RECORDBLOCK", 1),
                save=_boolean(cardinfo.get("RECORDSAVE", "true"), "RECORDSAVE"))
    if method == "jackknife":
        for k in ("RECORDNBOOT", "RECORDSEED", "RECORDBLOCK"):
            if k in cardinfo:
                print("NOTE: " + k + " is not used by RECORDCI 'jackknife'")
    print("RECORDCI '%s': %s%d %% level, RECORDSAVE %s (record uncertainty, RainyDay_fast fast_record.py)"
          % (method, "B = %d, seed %d, block %d year(s), " % (opts["nboot"], opts["seed"], opts["block"])
             if method == "bootstrap" else "", level, str(opts["save"]).lower()))
    return opts


def parse_engine(cardinfo):
    """GP 2026-10-26: FREQENGINE of the JSON: 'montecarlo' (default, RainyDay's realizations), 'exact' (the frequency
    analysis from the complete storm x location pool, no realizations) or 'both'."""
    v = cardinfo.get("FREQENGINE", "montecarlo")
    s = v.strip().lower() if isinstance(v, str) else None
    if s == "":
        s = "montecarlo"
    if s not in ENGINES:
        sys.exit("FREQENGINE must be 'montecarlo', 'exact' or 'both' (got " + repr(v) + ")")
    if s != "montecarlo":
        print("FREQENGINE '%s': exact frequency analysis from the storm x location pool (_FreqExact.csv/.png)%s "
              "(RainyDay_fast fast_record.py)" % (s, "; no Monte Carlo realizations, no random draws" if s == "exact"
                                                  else ", next to RainyDay's unchanged Monte Carlo outputs"))
    return s


def check_options(opts, freqanalysis, transpotype, samplingtype, calctype, areatype, rescaletype, rotation,
                  intensdistr, scenarios, seasonalsampling, who="RECORDCI"):
    """Stop runs the emulator does not model (validated scope: uniform transposition, Poisson counts, PDS/AMS,
    watershed/box, single duration or MULTIPLEDURATIONS). GP 2026-10-26: who = the option named in the message
    (RECORDCI or FREQENGINE: same scope)."""
    bad = []
    if not freqanalysis:
        bad.append("FREQANALYSIS must be 'true'")
    if transpotype != "uniform":
        bad.append("TRANSPOSITION must be 'uniform' (got '" + str(transpotype) + "')")
    if str(samplingtype).lower() != "poisson":
        bad.append("RESAMPLING must be 'poisson' (got '" + str(samplingtype) + "')")
    if str(calctype).lower() not in ("pds", "ams"):
        bad.append("CALCTYPE must be 'pds' or 'ams'")
    if str(areatype).lower() not in ("basin", "box"):
        bad.append("POINTAREA must be 'watershed' or 'box' (not point/pointlist)")
    if rescaletype != "none":
        bad.append("ENHANCEDSST is not supported")
    if rotation:
        bad.append("ROTATIONANGLE is not supported")
    if intensdistr:
        bad.append("INTENSDISTR is not supported")
    if scenarios:
        bad.append("SCENARIOS is not supported")
    if seasonalsampling:
        bad.append("SEASONALSAMPLING is not supported")
    if bad:
        sys.exit(who + " doesn't support this run:\n  - " + "\n  - ".join(bad))


def check_duration(durationcheck, durcorrection, who="RECORDCI"):
    if durationcheck == False and durcorrection == False:   # noqa: E712 (RainyDay's flags may be numpy bools)
        sys.exit(who + ": the storm catalog is longer than DURATION and DURATIONCORRECTION is 'false' (RainyDay's "
                 "duration-rewrite path) is not supported; use DURATIONCORRECTION 'true', or MULTIPLEDURATIONS with a "
                 "DURATION-long catalog")


# =====================================================================================================================
# record years and draws
# =====================================================================================================================
def storm_years(cattime):
    """Start year of every storm (cattime[:, 0]), as validated (a storm across New Year counts in its start year)."""
    return np.asarray(cattime)[:, 0].astype("datetime64[Y]").astype(np.int64) + 1970


def record_years(syears, includeyears):
    """The record years: INCLUDEYEARS when given, else the full span min..max of the storm start years."""
    if includeyears is not False and includeyears is not None and len(includeyears) > 0:
        years = np.array(sorted(set(int(y) for y in includeyears)), dtype=np.int64)
        rule = "INCLUDEYEARS"
    else:
        years = np.arange(int(syears.min()), int(syears.max()) + 1, dtype=np.int64)
        rule = "span of the storm start years"
    out = np.unique(syears[~np.isin(syears, years)])
    if out.size:
        sys.exit("RECORDCI: storms start in years outside the record years (" + rule + "): " + ", ".join(
            str(y) for y in out) + "; set INCLUDEYEARS to the years of the rainfall record")
    return years, rule


def year_draws(B, NY, seed, block=1):
    """(B, NY) indices into the record years. block = 1: np.random.default_rng(seed).integers(0, NY, (B, NY)), the
    prototype's draws (record_boot.year_draws). block L > 1: moving blocks, ceil(NY/L) uniform block starts per record
    (same call), each followed by its L - 1 next years (circular), concatenated and truncated to NY."""
    nb = -(-NY // block)
    starts = np.random.default_rng(seed).integers(0, NY, (B, nb))
    if block == 1:
        return starts
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % NY
    return idx.reshape(B, nb * block)[:, :NY]


def draw_counts(draws, NY):
    """(B, NY) int64: how many times each record year was drawn (the weight of every storm of that year)."""
    cnt = np.zeros((draws.shape[0], NY), dtype=np.int64)
    for j in range(NY):
        cnt[:, j] = (draws == j).sum(axis=1)
    return cnt


def jackknife_counts(NY):
    """(NY, NY): row j leaves record year j out (weights 0/1; NYEARS of the record = NY - 1)."""
    return 1 - np.eye(NY, dtype=np.int64)


# =====================================================================================================================
# per-storm values (inside RainyDay's resampling loop)
# =====================================================================================================================
def basin_windows(rain0, trim, steps):
    """[dm_j] (H-mh+1, W-mw+1) float64: max over the windows of steps[j] time steps of the basin sums
    sum_ij trim_ij rain0(t, y+i, x+j); rain0 has no NaN. Same operations as the prototype (04_correlation_emulator
    basin_series + dmax), with one cumulative sum for all durations."""
    S = fftconvolve(rain0.astype(np.float64), trim[::-1, ::-1].astype(np.float64)[None], mode="valid", axes=(1, 2))
    cs = np.concatenate([np.zeros((1,) + S.shape[1:]), np.cumsum(S, axis=0)], axis=0)
    del S
    return [(cs[s:] - cs[:-s]).max(axis=0) for s in steps]


def box_empty(hit, mh, mw):
    """out[y, x] = no True in hit[y:y+mh, x:x+mw] (summed-area table, exact integers)."""
    H, W = hit.shape
    sat = np.zeros((H + 1, W + 1), dtype=np.int64)
    np.cumsum(np.cumsum(hit, axis=0, dtype=np.int64), axis=1, out=sat[1:, 1:])
    return (sat[mh:, mw:] - sat[:-mh, mw:] - sat[mh:, :-mw] + sat[:-mh, :-mw]) == 0


def zero_anchors(rain0, steps, fullsum, mh, mw):
    """[zero_j] (H-mh+1, W-mw+1) bool: RainyDay's zero rule (SSTalt, SSTalt_multidur): every cell of the mh x mw box
    is < 0.5 in every window of the duration, with the float32 window fields RainyDay builds: np.nansum over time for
    the parent duration without duration correction (fullsum[j]), else rolling_sum (float64 nancumsum differences
    cast to float32). Exact and cheap: a cell whose largest hourly value is >= 0.5 is >= 0.5 in some window of every
    duration; a cell whose storm total c[T-1] < THR32 is < 0.5 in every window (windows <= total, rain >= 0); only the
    cells in between get their windows computed."""
    T = rain0.shape[0]
    hmax = rain0.max(axis=0)
    tot = np.zeros(rain0.shape[1:], dtype=np.float64)      # c[T-1] of the float64 nancumsum (sequential, as cumsum)
    for t in range(T):
        tot += rain0[t]
    out = []
    for s, full in zip(steps, fullsum):
        if full:
            hit = ~(np.sum(rain0, axis=0) < 0.5)              # passrain = np.nansum(catrain, axis=0) (float32)
        else:
            hit = hmax >= 0.5
            amb = ~hit & (tot >= THR32)
            if amb.any():
                c = np.cumsum(rain0[:, amb], axis=0, dtype=np.float64)
                c = np.concatenate([np.zeros((1, c.shape[1])), c], axis=0)
                hit[amb] = ~((c[s:] - c[:-s]).astype(np.float32).max(axis=0) < 0.5)
        out.append(box_empty(hit, mh, mw))
    return out


def dry_support(rain0, trim):
    """(H-mh+1, W-mw+1) bool: no rain (> 0) in any hour on the cells of the box where trim > 0; RainyDay's weighted
    sums are then exactly 0 in every window (its maximum starts from 0.), while the FFT sums carry ~1e-18 noise."""
    wet = (rain0 > 0).any(axis=0)
    sup = trim > 0
    if sup.all():
        return box_empty(wet, trim.shape[0], trim.shape[1])
    cnt = fftconvolve(wet.astype(np.float64), sup[::-1, ::-1].astype(np.float64), mode="valid")
    return np.rint(cnt) == 0                                # integer counts (FFT error << 0.5)


def storm_values(rain, trim, mnorm, scale, steps, fullsum, anchors):
    """Basin-average depths of one storm at every valid anchor, per duration: [float32 (n_anchors,)].
    rain: RainyDay's catrain (T, H, W), negatives already NaN (not modified: NaN -> 0 in a copy);
    trim, mnorm: RainyDay's trimmask and mnorm; scale = timeres/60; steps: window length (time steps) per duration;
    anchors: (ymask, xmask) of irregular domains, or None (rectangular: every anchor, row-major)."""
    rain0 = np.where(np.isnan(rain), np.float32(0.), rain).astype(np.float32, copy=False)
    dms = basin_windows(rain0, trim, steps)
    zeros = zero_anchors(rain0, steps, fullsum, trim.shape[0], trim.shape[1])
    dry = dry_support(rain0, trim)
    dry = dry.ravel() if anchors is None else dry[anchors]
    out = []
    for dm, z in zip(dms, zeros):
        if anchors is None:
            v, zz = dm.ravel(), z.ravel()
        else:
            v, zz = dm[anchors], z[anchors]
        v = (v * scale / mnorm).astype(np.float32)          # prototype: (dmax[valid] / W).astype(float32)
        v[zz | dry] = 0.
        np.maximum(v, 0., out=v)                            # FFT noise of dry anchors (RainyDay: max(0, sums) >= 0)
        out.append(v)
    return out


class StormPool:
    """GP 2026-10-26: the per-storm pool of basin-average depths, vals[j] (n_storms, n_anchors) float32 for every
    duration j (0 = the parent DURATION, then the MULTIPLEDURATIONS sub-durations), filled inside RainyDay's resampling
    loop (add_storm, storm_values on a copy of catrain). One object per run, shared by RECORDCI and FREQENGINE (one copy
    in memory); free(j) releases a duration once its outputs are written. (Moved unchanged from RecordCI.)"""

    def __init__(self, durations, steps, parent_fullsum, trimmask, mnorm, timeres, shape, anchors, nstorms,
                 label="record: per-storm pool"):
        self.durations = [float(d) for d in durations]
        self.steps = [int(s) for s in steps]
        self.fullsum = [bool(parent_fullsum)] + [False] * (len(steps) - 1)
        self.trim = np.asarray(trimmask, dtype=np.float32)
        self.mnorm = mnorm
        self.scale = timeres / 60.
        self.shape = (int(shape[0]), int(shape[1]))
        mh, mw = self.trim.shape
        if anchors is None:
            self.anchors = None
            self.nv = (self.shape[0] - mh + 1) * (self.shape[1] - mw + 1)
        else:
            self.anchors = (np.asarray(anchors[0]), np.asarray(anchors[1]))
            self.nv = len(self.anchors[0])
        self.nstorms = int(nstorms)
        self.vals = [np.empty((self.nstorms, self.nv), dtype=np.float32) for _ in self.steps]
        self.filled = np.zeros(self.nstorms, dtype=bool)
        self.seconds = 0.
        self.label = label

    def megabytes(self):
        return len(self.steps) * self.nstorms * self.nv * 4 / 2 ** 20

    def add_storm(self, i, catrain):
        """Pool values of storm i (RainyDay's catrain: copied, never modified)."""
        FP.tic(self.label)
        t0 = time.time()
        if tuple(catrain.shape[1:]) != self.shape:
            sys.exit("pool: storm %d has shape %s, expected %s" % (i, catrain.shape[1:], self.shape))
        vals = storm_values(catrain, self.trim, self.mnorm, self.scale, self.steps, self.fullsum, self.anchors)
        for j, v in enumerate(vals):
            self.vals[j][i] = v
        self.filled[i] = True
        self.seconds += time.time() - t0
        FP.toc(self.label)

    def check(self, j, who):
        if not self.filled.all():
            sys.exit("%s: %d storms have no pool values (bug)" % (who, int((~self.filled).sum())))
        if self.vals[j] is None:
            sys.exit("%s: the pool of duration %s h was already freed (bug)" % (who, RDF.durtag(self.durations[j])))

    def free(self, j):
        self.vals[j] = None


# =====================================================================================================================
# pool and estimators
# =====================================================================================================================
def cum_counts(g, G, pos):
    """(len(pos), G) int64: cum[i, c] = #{p < pos[i]: g[p] = c}; pos in any order, 0 <= pos <= len(g)."""
    pos = np.asarray(pos, dtype=np.int64)
    order = np.argsort(pos, kind="stable")
    ps = pos[order]
    seg = np.searchsorted(ps, np.arange(len(g)), side="right")      # element p counts for every ps[j] > p
    h = np.bincount(seg * G + g, minlength=(len(ps) + 1) * G).reshape(len(ps) + 1, G)
    out = np.empty((len(ps), G), dtype=np.int64)
    out[order] = np.cumsum(h[:-1], axis=0)
    return out


def m_needed(nv, ks, nyears, nsim, calctype="pds"):
    """Pool depth (expanded values) that the levels and expectations of the rows ks can reach with all weights 1
    (prototype record_boot.m_needed for PDS: k + 10 sqrt(k) + 10 events). inf = the whole pool."""
    k = float(max(ks))
    if calctype == "ams":
        q = (k + 10 * np.sqrt(k) + 10) / nsim
        lam = -np.log1p(-q) if q < 1. else np.inf
        lam = max(lam, -np.log1p(-k / nsim) if k < nsim else np.inf)
        return lam * nyears * nv
    return (k + 10 * np.sqrt(k) + 10) * nyears * nv / nsim


class Pool:
    """Top-M pooled (storm x anchor) values of one duration, sorted descending, with the group (record year) of each
    value; port of record_boot.Pool with storm weights given per group (all storms of a year share their weight).
    A weight vector w (G,) gives every value of a storm of group c the weight w[c]; D(p) = cumulative weight of the
    first p + 1 values. Cumulative group counts at checkpoints turn D into a small matrix product (exact integers)."""

    def __init__(self, vals, groups, G, mreq, factor=4.0, block=1024):
        self.vals = vals                                        # (n_storms, n_anchors) float32
        self.n, self.nv = vals.shape
        self.groups = np.asarray(groups, dtype=np.int64)
        self.G = int(G)
        self.gsize = np.bincount(self.groups, minlength=self.G).astype(np.int64)
        self.block = block
        self.extended = 0
        self._build(int(factor * mreq) + 1000 if np.isfinite(mreq) else self.n * self.nv)

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
        self.g = self.groups[idx // self.nv]
        self.M = M
        self.full = M == N
        self.qpos = np.unique(np.r_[np.arange(self.block, M, self.block), M]).astype(np.int64)
        self.qcnt = cum_counts(self.g, self.G, self.qpos).astype(np.float64)

    def extend(self):
        self.extended += 1
        self._build(4 * self.M)

    def _totals(self, w):
        sw = int(np.dot(w, self.gsize))
        return sw, sw * self.nv

    def targets(self, w, Ts, nyears, calctype):
        """m_T = number of expanded values >= the T-yr level (prototype index rule, weighted)."""
        sw, Wt = self._totals(w)
        lam = sw / nyears
        ms = []
        for T in Ts:
            p = (-np.log1p(-1.0 / T) / lam) if calctype == "ams" else 1.0 / (lam * T)
            x = Wt * (1 - p)
            k = int(np.clip(round(x), 0, Wt - 1)) if np.isfinite(x) else 0
            ms.append(Wt - k)
        return np.array(ms, dtype=np.int64)

    def levels(self, W, Ts, nyears, calctype="pds"):
        """(n_draws, len(Ts)) weighted quantiles with Lambda*(x_T) = 1/T (PDS) or -ln(1-1/T) (AMS)."""
        W = np.atleast_2d(np.asarray(W, dtype=np.int64))
        out = np.full((W.shape[0], len(Ts)), np.nan)
        b = 0
        while b < W.shape[0]:
            w = W[b]
            if self._totals(w)[0] == 0:
                b += 1
                continue
            ms = self.targets(w, Ts, nyears, calctype)
            Dq = self.qcnt @ w.astype(np.float64)               # D at the checkpoints (exact integers)
            jq = np.searchsorted(Dq, ms, side="left")
            if (jq >= len(self.qpos)).any():
                assert not self.full
                self.extend()
                continue
            for t in range(len(Ts)):
                j = jq[t]
                p0 = int(self.qpos[j - 1]) if j > 0 else 0
                base = Dq[j - 1] if j > 0 else 0.
                loc = base + np.cumsum(w[self.g[p0:self.qpos[j]]])
                out[b, t] = self.v[p0 + np.searchsorted(loc, ms[t], side="left")]
            b += 1
        return out

    @staticmethod
    def _prob(kf, x, nsim, calctype):
        """P(X_(k) > the value whose exceedance count is x): PDS x = NYEARS Lambda (Poisson mean), AMS x = Lambda."""
        if calctype == "ams":
            return betainc(kf, nsim - kf + 1., -np.expm1(-x))
        return gammainc(kf, x)

    def expected(self, W, ks, nyears, nsim, calctype="pds", ngrid=NGRID, tol=TOL, chunk=64):
        """(n_draws, len(ks)) E[X_(k)] = int_0^inf P(X_(k) > x) dx of the weighted pool (prototype Pool.expected:
        trapezoid on a geometric grid above the pool floor, P = 1 assumed below it, checked P(floor) > 1 - tol; a
        full pool uses the exact P below its floor). Rows with fewer than k events count as -9999, as in RainyDay."""
        W = np.atleast_2d(np.asarray(W, dtype=np.int64))
        kf = np.asarray(ks, dtype=np.float64)[:, None]
        while True:
            lo, hi = float(self.v[-1]), float(self.v[0])
            grid = np.geomspace(max(lo, 1e-3), hi * (1 + 1e-7), ngrid)
            pos = np.searchsorted(-self.v, -grid, side="right")             # #(v >= grid)
            cnt = cum_counts(self.g, self.G, pos).astype(np.float64)
            out = np.empty((W.shape[0], len(ks)))
            ok = True
            for b0 in range(0, W.shape[0], chunk):
                Wb = W[b0:b0 + chunk]
                Dg = cnt @ Wb.T.astype(np.float64)                         # (ngrid, nb) exact integers
                for jb in range(Wb.shape[0]):
                    sw, Wt = self._totals(Wb[jb])
                    if calctype == "ams":
                        x, xall = Dg[:, jb] / (nyears * self.nv), Wt / (nyears * self.nv)
                    else:
                        x, xall = nsim * Dg[:, jb] / (nyears * self.nv), nsim * Wt / (nyears * self.nv)
                    u, inv = np.unique(x, return_inverse=True)              # same P values, fewer evaluations;
                    P = np.ascontiguousarray(self._prob(kf, u[None, :], nsim, calctype)[:, inv])   # C order: same sums
                    Pall = self._prob(kf[:, 0], xall, nsim, calctype)
                    if not self.full and (P[:, 0] < 1 - tol).any():
                        ok = False
                        break
                    base = grid[0] * Pall if self.full else grid[0]
                    out[b0 + jb] = base + trapezoid(P, grid, axis=1) + MISSING * (1. - Pall)
                if not ok:
                    break
            if ok:
                return out
            assert not self.full
            self.extend()

    # -----------------------------------------------------------------------------------------------------------------
    # GP 2026-10-26: FREQENGINE (exact engine): the distribution of X_(k), not only its mean. New methods; levels,
    # expected and _prob above are unchanged (RECORDCI outputs stay byte-identical).
    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _cprob(kf, x, nsim, calctype):
        """1 - _prob = P(X_(k) <= the value whose exceedance count is x), computed directly (accurate when small)."""
        if calctype == "ams":
            return betainc(nsim - kf + 1., kf, np.exp(-x))
        return gammaincc(kf, x)

    def _values_at(self, w, ms):
        """The m-th largest expanded (weight-repeated) pool value for every m of ms (1-based), -9999 where m exceeds the
        number of expanded values Wt (or m < 1). Same search as levels (checkpoint totals, then a cumulative sum inside
        one block of the sorted pool); the pool grows when an m lies below its floor."""
        w = np.asarray(w, dtype=np.int64)
        ms = np.asarray(ms, dtype=np.int64)
        out = np.full(ms.shape, MISSING)
        Wt = self._totals(w)[1]
        inside = (ms >= 1) & (ms <= Wt)
        mi = ms[inside]
        if mi.size == 0:
            return out
        while True:
            Dq = self.qcnt @ w.astype(np.float64)               # D at the checkpoints (exact integers)
            jq = np.searchsorted(Dq, mi, side="left")
            if (jq >= len(self.qpos)).any():
                assert not self.full
                self.extend()
                continue
            break
        res = np.empty(mi.size)
        for j in np.unique(jq):
            sel = jq == j
            p0 = int(self.qpos[j - 1]) if j > 0 else 0
            base = Dq[j - 1] if j > 0 else 0.
            loc = base + np.cumsum(w[self.g[p0:self.qpos[j]]])
            res[sel] = self.v[p0 + np.searchsorted(loc, mi[sel], side="left")]
        out[inside] = res
        return out

    def critical_counts(self, ks, qs, nyears, nsim, calctype="pds"):
        """(len(ks), len(qs)) c*: P(X_(k) <= x) >= q  <=>  (weighted count of pool values > x) <= c*.
        PDS: P(X_(k) <= x) = P(Poisson(mu) < k) = 1 - gammainc(k, mu), mu = nsim D / (nyears nv), so
        c* = gammaincinv(k, 1 - q) nyears nv / nsim. AMS: P(Binomial(nsim, p) < k) = 1 - betainc(k, nsim - k + 1, p),
        p = 1 - exp(-D / (nyears nv)), so c* = -ln(1 - betaincinv(k, nsim - k + 1, 1 - q)) nyears nv."""
        kf = np.asarray(ks, dtype=np.float64)[:, None]
        qf = np.asarray(qs, dtype=np.float64)[None, :]
        if calctype == "ams":
            p = betaincinv(kf, nsim - kf + 1., 1. - qf)
            with np.errstate(divide="ignore"):
                return -np.log1p(-p) * nyears * self.nv
        return gammaincinv(kf, 1. - qf) * nyears * self.nv / nsim

    def quantiles(self, W, ks, qs, nyears, nsim, calctype="pds"):
        """(n_draws, len(ks), len(qs)) exact quantiles of X_(k) at probabilities qs (0-1): Q(q) = min{x: P(X_(k) <= x)
        >= q} with the strict count #{v > x} (right-continuous CDF, so Q(q) is a pool value and ties / the atom at 0 are
        exact): the m-th largest expanded value, m = floor(c*) + 1 (critical_counts); -9999 where m > Wt, i.e. where
        P(fewer than k events in all NYEARS) >= q (RainyDay writes -9999 for such a realization)."""
        W = np.atleast_2d(np.asarray(W, dtype=np.int64))
        c = self.critical_counts(ks, qs, nyears, nsim, calctype)
        c = np.where(np.isfinite(c), c, 4e18)
        m = (np.floor(np.minimum(c, 4e18)) + 1).astype(np.int64)
        out = np.empty((W.shape[0],) + m.shape)
        for b in range(W.shape[0]):
            out[b] = self._values_at(W[b], m.ravel()).reshape(m.shape)
        return out

    def cdf(self, w, ks, xs, nyears, nsim, calctype="pds"):
        """(F, S), each (len(ks), len(xs)): F = P(X_(k) <= x), S = 1 - F = P(X_(k) > x) for one weight vector w,
        both computed directly (accurate tails); x < 0 (e.g. -9999) counts every value (F = P(fewer than k events)).
        The pool grows until its floor is <= every x >= 0 (or it is full)."""
        w = np.asarray(w, dtype=np.int64)
        xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
        while not self.full and np.any((xs >= 0) & (xs < float(self.v[-1]))):
            self.extend()
        Wt = self._totals(w)[1]
        pos = np.searchsorted(-self.v.astype(np.float64), -xs, side="left")      # #(v > x)
        D = cum_counts(self.g, self.G, pos).astype(np.float64) @ w.astype(np.float64)
        D[xs < 0] = Wt
        x = D / (nyears * self.nv) if calctype == "ams" else nsim * D / (nyears * self.nv)
        kf = np.asarray(ks, dtype=np.float64)[:, None]
        return self._cprob(kf, x[None, :], nsim, calctype), self._prob(kf, x[None, :], nsim, calctype)

    def expected_minmax(self, W, ks, nyears, nsim, N, calctype="pds", ngrid=NGRID, tol=TOL, chunk=64):
        """(emin, emax), each (n_draws, len(ks)): the expected minimum and maximum of N independent realizations of
        X_(k) (N = NREALIZATIONS: the noise-free values of RainyDay's minrain/maxrain). The integral of expected with the
        survival P = P(X_(k) > x) replaced by P^N (minimum) and 1 - (1 - P)^N (maximum, computed as
        -expm1(N log1p(-P))); the -9999 terms have probabilities 1 - Pall^N (minimum: any realization without a k-th
        event) and (1 - Pall)^N (maximum: none has one). The pool grows until P(floor)^N > 1 - tol."""
        W = np.atleast_2d(np.asarray(W, dtype=np.int64))
        kf = np.asarray(ks, dtype=np.float64)[:, None]
        N = float(N)
        while True:
            lo, hi = float(self.v[-1]), float(self.v[0])
            grid = np.geomspace(max(lo, 1e-3), hi * (1 + 1e-7), ngrid)
            pos = np.searchsorted(-self.v, -grid, side="right")             # #(v >= grid), as expected
            cnt = cum_counts(self.g, self.G, pos).astype(np.float64)
            emin, emax = np.empty((W.shape[0], len(ks))), np.empty((W.shape[0], len(ks)))
            ok = True
            for b0 in range(0, W.shape[0], chunk):
                Wb = W[b0:b0 + chunk]
                Dg = cnt @ Wb.T.astype(np.float64)
                for jb in range(Wb.shape[0]):
                    sw, Wt = self._totals(Wb[jb])
                    if calctype == "ams":
                        x, xall = Dg[:, jb] / (nyears * self.nv), Wt / (nyears * self.nv)
                    else:
                        x, xall = nsim * Dg[:, jb] / (nyears * self.nv), nsim * Wt / (nyears * self.nv)
                    u, inv = np.unique(x, return_inverse=True)
                    Pu = self._prob(kf, u[None, :], nsim, calctype)
                    with np.errstate(divide="ignore"):
                        smin = np.ascontiguousarray(np.power(Pu, N)[:, inv])
                        smax = np.ascontiguousarray((-np.expm1(N * np.log1p(-Pu)))[:, inv])
                        Pall = self._prob(kf[:, 0], xall, nsim, calctype)
                        amin = np.power(Pall, N)                            # P(min > -9999) = P(every rlz has X_(k))
                        miss_min = -np.expm1(N * np.log(Pall))              # 1 - Pall^N (log(0) = -inf -> 1)
                        miss_max = np.exp(N * np.log1p(-Pall))              # P(no rlz has a k-th event)
                    if not self.full and (smin[:, 0] < 1 - tol).any():
                        ok = False
                        break
                    bmin = grid[0] * amin if self.full else grid[0]
                    bmax = grid[0] * (1. - miss_max) if self.full else grid[0]
                    emin[b0 + jb] = bmin + trapezoid(smin, grid, axis=1) + MISSING * miss_min
                    emax[b0 + jb] = bmax + trapezoid(smax, grid, axis=1) + MISSING * miss_max
                if not ok:
                    break
            if ok:
                return emin, emax
            assert not self.full
            self.extend()


# =====================================================================================================================
# the run object (driver hooks: RainyDay_Py3.py, "GP 2026-10-26")
# =====================================================================================================================
def selected_rows(nsim, alllevels, speclevels):
    """Row indices of RainyDay's frequency table: all NYEARS rows, or RETURNLEVELS through find_nearest (the same
    operations as RainyDay_Py3.py and RainyDay.rank_frequency)."""
    exceedp = np.linspace(1, 1. / nsim, nsim)
    returnperiod = 1 / exceedp
    if alllevels == False:   # noqa: E712
        rows = np.array([RDF.find_nearest(returnperiod, speclevels[i]) for i in range(0, len(speclevels))],
                        dtype=np.int64)
    else:
        rows = np.arange(nsim, dtype=np.int64)
    return rows, exceedp


def _pct(a, q, axis=None):
    return np.percentile(a, q, axis=axis)


class RecordCI:
    """Record uncertainty of one RainyDay run: the per-storm pool (add_storm) and the outputs per duration (write)."""

    def __init__(self, opts, durations, steps, parent_fullsum, trimmask, mnorm, timeres, shape, anchors, cattime,
                 includeyears, nyears_rd, calctype, nsim, alllevels, speclevels, folder, basename):
        t0 = time.time()
        self.opts = opts
        self.method = opts["method"]
        self.level = float(opts["level"])
        # GP 2026-10-26: the per-storm pool moved to StormPool (unchanged arithmetic), shared with FREQENGINE (ExactFA)
        self.pool = StormPool(durations, steps, parent_fullsum, trimmask, mnorm, timeres, shape, anchors,
                              len(np.asarray(cattime)), label="record: per-storm pool")
        self.durations, self.steps, self.nv = self.pool.durations, self.pool.steps, self.pool.nv
        self.vals, self.filled = self.pool.vals, self.pool.filled  # the same objects: vals[j] = None frees both users
        self.write_level = False         # GP 2026-10-26: _RecordBootstrapLevel.csv (set by ExactFA: FREQENGINE on)
        self.calctype = str(calctype).lower()
        self.nsim = int(nsim)
        self.folder, self.basename = folder, basename
        self.rows, self.exceedp_full = selected_rows(self.nsim, alllevels, speclevels)
        # record years, storm groups and the draws
        self.syears = storm_years(cattime)
        self.nstorms = len(self.syears)
        self.years, self.yrule = record_years(self.syears, includeyears)
        self.NY = len(self.years)
        self.sgroup = np.searchsorted(self.years, self.syears).astype(np.int64)
        if int(nyears_rd) != self.NY:
            print("WARNING: RECORDCI uses %d record years (%s), RainyDay's resampling uses NYEARS of the record = %d "
                  "(distinct end years of the storms): sst_expected is then not the noise-free value of rd_mean; set "
                  "INCLUDEYEARS to the years of the rainfall record" % (self.NY, self.yrule, int(nyears_rd)))
        if self.NY < 2:
            if self.method == "jackknife":
                sys.exit("RECORDCI 'jackknife' needs at least 2 record years (got %d)" % self.NY)
            print("WARNING: RECORDCI: one record year only; every bootstrap record equals the record (zero record CI)")
        if self.method == "bootstrap":
            if opts["block"] > self.NY:
                sys.exit("RECORDBLOCK (%d) must be <= the number of record years (%d)" % (opts["block"], self.NY))
            self.C = draw_counts(year_draws(opts["nboot"], self.NY, opts["seed"], opts["block"]), self.NY)
            self.ny_draw = self.NY
            self.colnames = ["rec%d" % (b + 1) for b in range(self.C.shape[0])]
        else:
            self.C = jackknife_counts(self.NY)
            self.ny_draw = self.NY - 1
            self.colnames = ["drop%d" % y for y in self.years]
        if self.nstorms != self.pool.nstorms:
            sys.exit("RECORDCI: %d storm years for %d storms (bug)" % (self.nstorms, self.pool.nstorms))
        self.draws_written = False
        self.seconds_fa = 0.
        per_year = np.bincount(self.sgroup, minlength=self.NY)
        print("RECORDCI: %d storms, record years %d-%d (NY = %d, %s; storm year = start year; storms per year %d-%d), "
              "%d anchors, durations %s h, %d %s records; pool arrays %.0f MB (%.1f s)"
              % (self.nstorms, self.years[0], self.years[-1], self.NY, self.yrule, per_year.min(), per_year.max(),
                 self.nv, ", ".join(RDF.durtag(d) for d in self.durations), self.C.shape[0], self.method,
                 self.pool.megabytes(), time.time() - t0))

    # -----------------------------------------------------------------------------------------------------------------
    def add_storm(self, i, catrain):
        """Pool values of storm i (RainyDay's catrain: copied, never modified). GP 2026-10-26: StormPool.add_storm."""
        self.pool.add_storm(i, catrain)

    # -----------------------------------------------------------------------------------------------------------------
    def _mreq(self, ks):
        return m_needed(self.nv, ks, self.NY, self.nsim, self.calctype)

    def estimates(self, j, ks):
        """Point and record values of duration j: dict of arrays (point: (nT,), bootstrap: (n_draws, nT))."""
        Ts = self.nsim / ks
        pool = Pool(self.vals[j], self.sgroup, self.NY, self._mreq(ks))
        ones = np.ones((1, self.NY), dtype=np.int64)
        r = dict(lv_pt=pool.levels(ones, Ts, self.NY, self.calctype)[0],
                 ex_pt=pool.expected(ones, ks, self.NY, self.nsim, self.calctype)[0])
        r["lv_bt"] = pool.levels(self.C, Ts, self.ny_draw, self.calctype)
        r["ex_bt"] = pool.expected(self.C, ks, self.ny_draw, self.nsim, self.calctype)
        r["M"], r["full"], r["extended"] = pool.M, pool.full, pool.extended
        return r

    def write(self, j, sortrain, exceedp, returnperiod, folder, name, spreadtype, quantilecalc, exact=None):
        """Record outputs of duration j (0 = the parent DURATION, 1 + jd = MULTIPLEDURATIONS subdurs[jd]), next to
        that duration's FreqAnalysis CSV (sortrain, exceedp, returnperiod: RainyDay's rows).
        GP 2026-10-26: FREQENGINE 'exact' (no realizations): sortrain None and exact = dict(mean, ens_lo, ens_hi,
        sample) from ExactFA: rd_mean = the exact mean, no Monte Carlo error, ens = the exact band, and the predictive
        band uses sample[t] (NZ equally probable quantiles of X_(k), deterministic) in place of the realizations."""
        FP.tic("record: estimators + write")
        t0 = time.time()
        if not self.filled.all():
            sys.exit("RECORDCI: %d storms have no pool values (bug)" % int((~self.filled).sum()))
        rows = self.rows
        if not np.array_equal(self.exceedp_full[rows], np.asarray(exceedp)):
            sys.exit("RECORDCI: the frequency rows differ from RainyDay's (bug)")
        ks = self.nsim - rows
        r = self.estimates(j, ks)
        self.vals[j] = None                                      # free the pool of this duration
        a = (100. - self.level) / 2.
        q = [a, 100. - a]
        z = ndtri(1. - a / 100.)
        zk = ndtri((np.arange(NZ) + 0.5) / NZ)                 # equally probable standard normal deviates
        if sortrain is not None:
            sortrain = np.asarray(sortrain, dtype=np.float64)
            nreal = sortrain.shape[1]
            rd_mean = np.nanmean(sortrain, axis=1)
            se_mc = np.nanstd(sortrain, axis=1, ddof=1) / np.sqrt(nreal) if nreal > 1 else np.full(len(rows), np.nan)
            if spreadtype == "ensemble" or spreadtype == "all_realizations":
                ens_lo, ens_hi = np.nanmin(sortrain, axis=1), np.nanmax(sortrain, axis=1)
            else:
                ens_lo = np.percentile(sortrain, (100 - quantilecalc) / 2, axis=1)
                ens_hi = np.percentile(sortrain, quantilecalc + (100 - quantilecalc) / 2, axis=1)
            real = sortrain                                      # values of one realization (pred band)
        else:                                                    # GP 2026-10-26: FREQENGINE 'exact'
            rd_mean = np.asarray(exact["mean"], dtype=np.float64)
            se_mc = np.zeros(len(rows))
            ens_lo, ens_hi = np.asarray(exact["ens_lo"]), np.asarray(exact["ens_hi"])
            real = np.asarray(exact["sample"], dtype=np.float64)
        pt, bt = r["ex_pt"], r["ex_bt"]
        n_out = len(rows)
        rec_lo, rec_hi, rec_se = np.empty(n_out), np.empty(n_out), np.empty(n_out)
        est_lo, est_hi, pred_lo, pred_hi = np.empty(n_out), np.empty(n_out), np.empty(n_out), np.empty(n_out)
        for t in range(n_out):
            sem = se_mc[t] if np.isfinite(se_mc[t]) else 0.
            if self.method == "bootstrap":
                dev = bt[:, t] - pt[t]                         # record deviations of the estimate
                rec_lo[t], rec_hi[t] = _pct(bt[:, t], q)
                rec_se[t] = bt[:, t].std(ddof=1)
                est = rd_mean[t] + dev[:, None] + sem * zk[None, :]
                pred = real[t][None, :] + dev[:, None]
                est_lo[t], est_hi[t] = _pct(est, q)
                pred_lo[t], pred_hi[t] = _pct(pred, q)
            else:
                n = bt.shape[0]
                se = np.sqrt((n - 1) / n * np.sum((bt[:, t] - bt[:, t].mean()) ** 2))
                rec_se[t] = se
                rec_lo[t], rec_hi[t] = pt[t] - z * se, pt[t] + z * se
                sc = np.sqrt(se ** 2 + sem ** 2)
                est_lo[t], est_hi[t] = rd_mean[t] - z * sc, rd_mean[t] + z * sc
                pred = real[t][None, :] + se * zk[:, None]
                pred_lo[t], pred_hi[t] = _pct(pred, q)
        # ---- files ----
        os.makedirs(folder, exist_ok=True)
        if self.opts["save"]:
            if not self.draws_written:
                self.write_draws()
            tab = np.column_stack((exceedp, returnperiod, pt, bt.T))
            np.savetxt(os.path.join(folder, name + "_RecordBootstrap.csv"), tab, delimiter=",",
                       header="prob.exceed,returnperiod,sst_expected," + ",".join(self.colnames),
                       fmt=["%.6f", "%.4f"] + ["%.4f"] * (tab.shape[1] - 2), comments="")
            if self.write_level:         # GP 2026-10-26: FREQENGINE on: the record bootstrap of the return level too
                tab = np.column_stack((exceedp, returnperiod, r["lv_pt"], r["lv_bt"].T))
                np.savetxt(os.path.join(folder, name + "_RecordBootstrapLevel.csv"), tab, delimiter=",",
                           header="prob.exceed,returnperiod,sst_level," + ",".join(self.colnames),
                           fmt=["%.6f", "%.4f"] + ["%.4f"] * (tab.shape[1] - 2), comments="")
        cols = ["returnperiod", "rd_mean", "sst_expected", "sst_level", "ens_lo", "ens_hi", "rec_lo", "rec_hi", "rec_se",
                "est_lo", "est_hi", "pred_lo", "pred_hi"]
        tab = np.column_stack((returnperiod, rd_mean, pt, r["lv_pt"], ens_lo, ens_hi, rec_lo, rec_hi, rec_se, est_lo,
                               est_hi, pred_lo, pred_hi))
        np.savetxt(os.path.join(folder, name + "_RecordCI.csv"), tab, delimiter=",", header=",".join(cols),
                   fmt="%.4f", comments="")
        self._prepend_definitions(os.path.join(folder, name + "_RecordCI.csv"), name, spreadtype, quantilecalc,
                                  exact=sortrain is None)
        # ---- log ----
        jt = int(np.argmin(np.abs(self.nsim / ks - 100.)))
        if sortrain is None:             # GP 2026-10-26: FREQENGINE 'exact': no Monte Carlo error to compare with
            msg = ("RECORDCI %s: T=%g yr: rd_mean (exact engine) %.2f, sst_expected %.2f mm, record %g %% CI "
                   "[%.2f, %.2f]%s; |rd_mean - sst_expected| max %.2g mm"
                   % (name, returnperiod[jt], rd_mean[jt], pt[jt], self.level, rec_lo[jt], rec_hi[jt],
                      "" if self.method == "bootstrap" else " (jackknife, normal)", np.max(np.abs(rd_mean - pt))))
            msg += "; pool %d values%s, %.1f s" % (r["M"], " (all)" if r["full"] else "", time.time() - t0)
            print(msg)
            self.seconds_fa += time.time() - t0
            FP.toc("record: estimators + write")
            return
        zmc = np.abs(rd_mean - pt) / se_mc
        msg = ("RECORDCI %s: T=%g yr: rd_mean %.2f, sst_expected %.2f mm, record %g %% CI [%.2f, %.2f]%s; "
               "|rd_mean - sst_expected| max %.2f Monte Carlo SE (T=%g)"
               % (name, returnperiod[jt], rd_mean[jt], pt[jt], self.level, rec_lo[jt], rec_hi[jt],
                  "" if self.method == "bootstrap" else " (jackknife, normal)",
                  np.nanmax(zmc) if np.isfinite(zmc).any() else np.nan,
                  returnperiod[int(np.nanargmax(zmc))] if np.isfinite(zmc).any() else np.nan))
        msg += "; pool %d values%s, %.1f s" % (r["M"], " (all)" if r["full"] else "", time.time() - t0)
        print(msg)
        if np.isfinite(zmc).any() and np.nanmax(zmc) > 4:
            print("WARNING: RECORDCI %s: rd_mean and sst_expected differ by more than 4 Monte Carlo standard errors "
                  "(T=%g yr); the noise-free model may not match this run" % (name, returnperiod[int(np.nanargmax(zmc))]))
        self.seconds_fa += time.time() - t0
        FP.toc("record: estimators + write")

    def _prepend_definitions(self, path, name, spreadtype, quantilecalc, exact=False):
        L = "%g %%" % self.level
        spread = "min-max" if spreadtype in ("ensemble", "all_realizations") else "%s %% quantile range" % quantilecalc
        if self.method == "bootstrap":
            how = ("year bootstrap, B = %d, RECORDSEED %d, RECORDBLOCK %d" % (self.C.shape[0], self.opts["seed"],
                                                                            self.opts["block"]))
            rec = "percentiles of the B bootstrap values of sst_expected (_RecordBootstrap.csv); rec_se = their sd"
            est = ("percentiles of rd_mean + record deviation (bootstrap value - sst_expected) + Monte Carlo error of "
                   "rd_mean (normal, SE = sd(realizations)/sqrt(NREALIZATIONS)), every pair of the two independent errors")
            pred = "percentiles of realization + record deviation, every pair"
        else:
            how = "leave-one-year-out jackknife (%d records of %d years)" % (self.NY, self.NY - 1)
            rec = "sst_expected -/+ z rec_se, rec_se = jackknife SE (normal approximation)"
            est = "rd_mean -/+ z sqrt(rec_se^2 + SE^2), SE = sd(realizations)/sqrt(NREALIZATIONS) (normal approximation)"
            pred = "percentiles of realization + normal record deviation (sd rec_se)"
        rdm = "# rd_mean       RainyDay estimate: mean of the realizations (_FreqAnalysis.csv, unchanged)"
        ens = "# ens_lo/hi     ensemble spread of the realizations as in _FreqAnalysis.csv (UNCERTAINTY: %s). NOT a CI" % spread
        if exact:                        # GP 2026-10-26: FREQENGINE 'exact' (no realizations)
            rdm = ("# rd_mean       RainyDay estimate from the exact engine (FREQENGINE 'exact'): the noise-free mean of the "
                   "realizations (_FreqExact.csv mean; no Monte Carlo error)")
            ens = ("# ens_lo/hi     exact ensemble spread (_FreqExact.csv: %s). NOT a CI"
                   % ("expected min-max of NREALIZATIONS realizations" if spreadtype in ("ensemble", "all_realizations")
                      else "%s %% quantile range of one realization" % quantilecalc))
            est = est.split(" + Monte Carlo error")[0] + " (no Monte Carlo error: exact engine)" \
                if self.method == "bootstrap" else "rd_mean -/+ z rec_se (no Monte Carlo error: exact engine)"
            pred = pred.replace("realization", "the exact distribution of one realization (%d equally probable "
                                               "quantiles of X_(k))" % NZ)
        head = [
            "# RainyDay_fast RECORDCI (GP 2026-10-26): record (sampling) uncertainty of %s" % name,
            "# %s; record years %d-%d (NY = %d, %s; storm year = start year); level %s; values in mm"
            % (how, self.years[0], self.years[-1], self.NY, self.yrule, L),
            rdm,
            "# sst_expected  noise-free value of rd_mean for this storm catalog (exact SST model, no Monte Carlo noise)",
            "# sst_level     quantile estimator of the same model: Lambda(x_T) = 1/T (PDS) or -ln(1 - 1/T) (AMS)",
            ens,
            "# rec_lo/hi     record CI (%s) of the estimate: %s" % (L, rec),
            "# est_lo/hi     total CI (%s) of the RainyDay estimate (%s): %s"
            % (L, "record only" if exact else "record + Monte Carlo", est),
            "# pred_lo/hi    predictive band (%s): %s. NOT a CI: it does not shrink with more realizations and"
            " depends on NYEARS" % (L, pred),
        ]
        with open(path) as f:
            body = f.read()
        with open(path, "w") as f:
            f.write("\n".join(head) + "\n" + body)

    def write_draws(self):
        """<basename>_RecordDraws.csv in the run folder: draw x record year counts (the same for every duration)."""
        path = os.path.join(self.folder, self.basename + "_RecordDraws.csv")
        head = ["# RainyDay_fast RECORDCI draws (GP 2026-10-26); compare_runs.py needs identical draws in both runs",
                "# method: %s" % self.method,
                "# seed: %s" % (self.opts["seed"] if self.method == "bootstrap" else "none"),
                "# block: %s" % (self.opts["block"] if self.method == "bootstrap" else "none"),
                "# draws: %d" % self.C.shape[0],
                "# years: %s" % " ".join(str(y) for y in self.years),
                "# rule: record years = %s; storm year = start year (cattime[:, 0]); storm weight = count of its year"
                % self.yrule,
                "draw," + ",".join(str(y) for y in self.years)]
        with open(path, "w") as f:
            f.write("\n".join(head) + "\n")
            for name, c in zip(self.colnames, self.C):
                f.write(name + "," + ",".join(str(int(x)) for x in c) + "\n")
        self.draws_written = True

    def report(self):
        print("RECORDCI: per-storm pool %.1f s, estimators + outputs %.1f s" % (self.pool.seconds, self.seconds_fa))


# =====================================================================================================================
# GP 2026-10-26: FREQENGINE, the exact frequency engine (driver hooks: RainyDay_Py3.py, "GP 2026-10-26")
# =====================================================================================================================
def _qname(p):
    """Column name of the p % quantile: q05, q25, q50, ...; q2.5, q97.5 for fractional percentages."""
    return "q%02d" % p if float(p).is_integer() else "q%g" % p


def exact_png(path, exceedp, mean, level, lo, hi, band):
    """<name>_FreqExact.png in the style of RainyDay's _FrequencyAnalysis.png (RainyDay.write_frequency_outputs, not
    used and not changed): exceedance probabilities <= 0.5, log-log axes, the same y-limit rule, dpi 250; the exact mean
    (blue line), the exact band (dodgerblue) and the return level (dashed). The mean is always computed here (the
    upstream function leaves it unset for a numeric UNCERTAINTY)."""
    import matplotlib.patches as mpatches
    from matplotlib.font_manager import FontProperties
    from matplotlib import pyplot as plt
    sel = exceedp <= 0.5
    if not sel.any():
        sel = np.ones(len(exceedp), dtype=bool)
    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(1)
    line1, = plt.plot(exceedp[sel], mean[sel], lw=1, label='Average (exact)', color='blue')
    line2, = plt.plot(exceedp[sel], level[sel], lw=0.8, ls='--', label='Return level (exact)', color='black')
    ax.fill_between(exceedp[sel], lo[sel], hi[sel], facecolor='dodgerblue', alpha=0.5, label=band)
    blue_patch = mpatches.Patch(color='dodgerblue', label=band)
    plt.legend(handles=[line1, line2, blue_patch], loc='lower right', prop=fontP)
    top = np.nanmax(hi[sel])
    upperlimit = 10. if top < 10. else 100. if top < 100. else 1000. if top < 1000. else 10000.
    bot = np.nanmin(lo[sel])
    lowerlimit = 0.1 if bot < 1. else 1 if bot < 10. else 10. if bot < 100. else 100.
    plt.ylim(lowerlimit, upperlimit)
    ax.set_xlabel('Annual Exceed. Prob. [-]\n1/(Return Period) [year]')
    ax.set_ylabel('Precip. Depth [mm]')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.gca().invert_xaxis()
    ax.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close('all')


class ExactFA:
    """FREQENGINE 'exact' / 'both': the frequency analysis of every duration of a run from the complete storm x location
    pool (StormPool; shared with RECORDCI when it is on: record = the RecordCI object), without Monte Carlo noise.
    Per duration (write): <name>_FreqExact.csv and .png in that duration's folder, the same rows (return periods) as
    _FreqAnalysis.csv (selected_rows); in 'exact' mode with RECORDCI also RecordCI.write(sortrain=None, exact=...).
    nyears_rd = RainyDay's own number of record years (its storm rate is n_storms/nyears_rd), so `mean` is the
    noise-free value of RainyDay's realization mean; it equals RECORDCI's sst_expected bitwise when RECORDCI's record
    years count the same (same pool construction and call order as RecordCI.estimates; one group G = 1 gives the same
    exact integer counts as the record years for all-ones weights)."""

    def __init__(self, engine, durations, steps, parent_fullsum, trimmask, mnorm, timeres, shape, anchors, nstorms,
                 nyears_rd, calctype, nsim, nreal, alllevels, speclevels, spreadtype, quantilecalc, record=None):
        t0 = time.time()
        self.engine = engine
        self.record = record
        self.shared = record is not None
        if self.shared:
            self.pool = record.pool
            record.write_level = True    # _RecordBootstrapLevel.csv (compare_runs.py --stat level)
        else:
            self.pool = StormPool(durations, steps, parent_fullsum, trimmask, mnorm, timeres, shape, anchors, nstorms,
                                  label="exact: per-storm pool")
        if self.pool.nstorms != int(nstorms):
            sys.exit("FREQENGINE: %d storms in the pool, %d in the catalog (bug)" % (self.pool.nstorms, int(nstorms)))
        self.nyears = int(nyears_rd)
        self.calctype = str(calctype).lower()
        self.nsim = int(nsim)
        self.nreal = max(int(nreal), 1)
        self.spreadtype = spreadtype
        self.quantilecalc = quantilecalc
        self.rows, exceedp_full = selected_rows(self.nsim, alllevels, speclevels)
        self.exceedp = exceedp_full[self.rows]
        self.returnperiod = (1 / exceedp_full)[self.rows]
        self.ks = self.nsim - self.rows
        self.probs = list(PROBS)
        self.band_q = None
        if spreadtype not in ("ensemble", "all_realizations"):       # numeric UNCERTAINTY: its quantile range
            self.band_q = ((100. - quantilecalc) / 2., quantilecalc + (100. - quantilecalc) / 2.)
            for p in self.band_q:
                if not any(np.isclose(p, x) for x in self.probs):
                    self.probs.append(float(p))
        self.seconds_fa = 0.
        lam = self.pool.nstorms / self.nyears
        print("FREQENGINE '%s': %d storms / %d record years (lambda %.3f per year), %d anchors, durations %s h, NYEARS "
              "%d, %d return periods, %s; pool %s (%.1f s)"
              % (engine, self.pool.nstorms, self.nyears, lam, self.pool.nv,
                 ", ".join(RDF.durtag(d) for d in self.pool.durations), self.nsim, len(self.rows), self.calctype.upper(),
                 "shared with RECORDCI" if self.shared else "arrays %.0f MB" % self.pool.megabytes(), time.time() - t0))

    def add_storm(self, i, catrain):
        """Pool values of storm i (into the shared pool when RECORDCI is on)."""
        self.pool.add_storm(i, catrain)

    # -----------------------------------------------------------------------------------------------------------------
    def compute(self, j, sample=False):
        """Exact quantities of duration j (arrays over the rows): level, mean, q (rows x probs), exp_min, exp_max, and
        with sample=True the NZ equally probable quantiles of X_(k) (rows x NZ, RECORDCI's predictive band)."""
        self.pool.check(j, "FREQENGINE")
        vals = self.pool.vals[j]
        ks = self.ks
        Ts = self.nsim / ks
        pool = Pool(vals, np.zeros(vals.shape[0], dtype=np.int64), 1,
                    m_needed(self.pool.nv, ks, self.nyears, self.nsim, self.calctype))
        ones = np.ones((1, 1), dtype=np.int64)
        r = dict(level=pool.levels(ones, Ts, self.nyears, self.calctype)[0],
                 mean=pool.expected(ones, ks, self.nyears, self.nsim, self.calctype)[0])
        r["q"] = pool.quantiles(ones, ks, np.array(self.probs) / 100., self.nyears, self.nsim, self.calctype)[0]
        emin, emax = pool.expected_minmax(ones, ks, self.nyears, self.nsim, self.nreal, self.calctype)
        r["exp_min"], r["exp_max"] = emin[0], emax[0]
        if sample:
            r["sample"] = pool.quantiles(ones, ks, (np.arange(NZ) + 0.5) / NZ, self.nyears, self.nsim,
                                         self.calctype)[0]
        r["M"], r["full"], r["extended"], r["N"] = pool.M, pool.full, pool.extended, pool.n * pool.nv
        dump = os.environ.get("RAINYDAY_DUMP_EXACTPOOL", "").strip()
        r["_pool"] = pool if dump != "" else None
        return r

    def write(self, j, folder, name):
        """<name>_FreqExact.csv/.png of duration j in folder; with RECORDCI in 'exact' mode also its outputs."""
        FP.tic("exact: estimators + write")
        t0 = time.time()
        withrec = self.shared and self.engine == "exact"
        r = self.compute(j, sample=withrec)
        os.makedirs(folder, exist_ok=True)
        N = self.nreal
        qcols = [_qname(p) for p in self.probs]
        cols = ["prob.exceed", "returnperiod", "mean", "level"] + qcols + ["exp_min_%d" % N, "exp_max_%d" % N]
        tab = np.column_stack((self.exceedp, self.returnperiod, r["mean"], r["level"], r["q"], r["exp_min"],
                               r["exp_max"]))
        path = os.path.join(folder, name + "_FreqExact.csv")
        np.savetxt(path, tab, delimiter=",", header=",".join(cols), fmt=["%.6f", "%.4f"] + ["%.4f"] * (tab.shape[1] - 2),
                   comments="")
        with open(path) as f:
            body = f.read()
        with open(path, "w") as f:
            f.write("\n".join(self.definitions(name, r)) + "\n" + body)
        if self.band_q is None:
            lo, hi = r["exp_min"], r["exp_max"]
            band = "Spread (exact): expected min-max of %d realizations" % N
        else:
            lo = r["q"][:, [i for i, p in enumerate(self.probs) if np.isclose(p, self.band_q[0])][0]]
            hi = r["q"][:, [i for i, p in enumerate(self.probs) if np.isclose(p, self.band_q[1])][0]]
            band = "Spread (exact): %g-%g %% of one realization" % self.band_q
        exact_png(os.path.join(folder, name + "_FreqExact.png"), self.exceedp, r["mean"], r["level"], lo, hi, band)
        dump = os.environ.get("RAINYDAY_DUMP_EXACTPOOL", "").strip()
        if dump != "":                   # debug / verification (tools/verify_exact_engine.py): the pool used here
            os.makedirs(dump, exist_ok=True)
            p = r["_pool"]
            np.savez(os.path.join(dump, name + "_exactpool.npz"), v=p.v, nall=r["N"], full=p.full, nv=self.pool.nv,
                     nstorms=self.pool.nstorms, nyears=self.nyears, nsim=self.nsim, calctype=self.calctype,
                     ks=self.ks, returnperiod=self.returnperiod, nreal=N)
        jt = int(np.argmin(np.abs(self.returnperiod - 100.)))
        iq = [self.probs.index(5.), self.probs.index(95.)]
        print("FREQENGINE %s: T=%g yr: mean %.2f, level %.2f mm, one realization 5-95 %% [%.2f, %.2f], expected "
              "min-max of %d [%.2f, %.2f]; pool %d of %d values%s, %.1f s"
              % (name, self.returnperiod[jt], r["mean"][jt], r["level"][jt], r["q"][jt, iq[0]], r["q"][jt, iq[1]], N,
                 r["exp_min"][jt], r["exp_max"][jt], r["M"], r["N"], " (all)" if r["full"] else "", time.time() - t0))
        self.seconds_fa += time.time() - t0
        FP.toc("exact: estimators + write")
        if withrec:
            exact = dict(mean=r["mean"], ens_lo=lo, ens_hi=hi, sample=r["sample"])
            self.record.write(j, None, self.exceedp, self.returnperiod, folder, name, self.spreadtype,
                              self.quantilecalc, exact=exact)       # frees the pool of duration j
        elif not self.shared:
            self.pool.free(j)

    def write_all(self, wd, basescenario, fullpath, scenarioname):
        """Every duration: the parent in fullpath/scenarioname, each MULTIPLEDURATIONS sub-duration d in
        wd/<basescenario>_<d> (RainyDay's folders). Called before the Monte Carlo writes ('both'), so RECORDCI frees the
        shared pool afterwards."""
        for j, d in enumerate(self.pool.durations):
            if j == 0:
                self.write(0, fullpath, scenarioname)
            else:
                name = basescenario + "_" + RDF.durtag(d)
                self.write(j, os.path.join(wd, name), name)
        print("FREQENGINE '%s': per-storm pool %.1f s%s, exact estimators + outputs %.1f s"
              % (self.engine, self.pool.seconds, " (shared with RECORDCI)" if self.shared else "", self.seconds_fa))

    def definitions(self, name, r):
        """'#' header of _FreqExact.csv (read it with comment='#')."""
        N = self.nreal
        lam = self.pool.nstorms / self.nyears
        what = ("X_(k) = the k-th largest storm (PDS) of NYEARS = %d synthetic years" if self.calctype == "pds" else
                "X_(k) = the k-th largest annual maximum (AMS) of NYEARS = %d synthetic years") % self.nsim
        extra = [_qname(p) for p in self.probs if p not in PROBS]
        return [
            "# RainyDay_fast FREQENGINE (GP 2026-10-26): exact frequency analysis of %s; values in mm; -9999 = no "
            "event (as RainyDay)" % name,           # no engine name: 'exact' and 'both' write identical files
            "# model: the SST resampling RainyDay samples, without Monte Carlo noise: Poisson storm counts (lambda = %d "
            "storms / %d record years = %.4f per year), storms and the %d transposition locations equally likely; %s, "
            "k = NYEARS/T; pool: every storm at every location (%d values, top %d used)"
            % (self.pool.nstorms, self.nyears, lam, self.pool.nv, what, r["N"], r["M"]),
            "# mean          E[X_(k)]: the noise-free value of RainyDay's estimate (the mean of the realizations, "
            "_FreqAnalysis.csv); depends on NYEARS",
            "# level         return level of the same model: Lambda(x_T) = 1/T (PDS) or -ln(1 - 1/T) (AMS), Lambda(x) = "
            "storms (or years) per year >= x; independent of NYEARS",
            "# q05 ... q95   quantiles of X_(k), the T-yr value of ONE realization (the distribution of the rlz "
            "columns)%s: Q(q) = min{x: P(X_(k) <= x) >= q}, P(X_(k) <= x) = P(fewer than k events > x); a pool value; "
            "-9999 where P(fewer than k events) >= q" % ((" and " + ", ".join(extra) + " (UNCERTAINTY)") if extra else ""),
            "# exp_min_%d/exp_max_%d  expected minimum / maximum of N = NREALIZATIONS = %d realizations (the noise-free "
            "values of RainyDay's minrain/maxrain)" % (N, N, N),
            "# NOT confidence intervals: q* and exp_* are the ensemble (Monte Carlo) spread for this storm catalog; the "
            "record (sampling) uncertainty is RECORDCI",
        ]
