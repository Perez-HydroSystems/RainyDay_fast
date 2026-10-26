#!/usr/bin/env python
"""
GP 2026-10-26: FREQENGINE check: the exact engine vs RainyDay's Monte Carlo realizations of the same run.

    python tools/verify_exact_engine.py <variant> [name ...] [--out FILE.csv]

For a FREQENGINE "both" run of the validation matrix (tools/make_test_jsons.py, tools/run_matrix.sh) made with
RAINYDAY_DUMP_EXACTPOOL=$FAST_TEST_ROOT/dumps/<variant>_pool (the exact engine's pool of every duration) and the usual
RAINYDAY_DUMP_WHICHRAIN dumps (the raw transposed depths: the realizations at full precision), for every shape x duration
x return period (FreqAnalysis row, X_(k) = k-th largest of NYEARS synthetic years, PDS or AMS):
  rows      _FreqExact.csv return periods == _FreqAnalysis.csv's; the realizations ranked from the dump == the CSV's
            (2 decimals); _FreqExact.csv q05..q95 == the quantiles recomputed from the dumped pool (4 decimals)
  mean      z = (mean of the N realizations - exact mean) / (sd / sqrt(N))
  quantiles for q = 5, 25, 50, 75, 95 %: the exact quantile Q(q) vs the N realizations: #{rlz < Q} and #{rlz <= Q} must
            bracket the Binomial(N, q) central 99 % (and 95 %) interval (order-statistic sampling error of the sample
            quantile; the two counts handle ties and atoms)
  min, max  the observed minimum / maximum of the N realizations vs the exact distribution of the min / max of N
            independent draws of X_(k): [P(min < obs), P(min <= obs)] must meet the central 90 % [0.05, 0.95]
            (P(min <= x) = 1 - (1 - F(x))^N, P(max <= x) = F(x)^N, F = the exact CDF of X_(k) from the pool)
Prints a summary per shape and the worst cells; --out writes every cell.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betainc, betaincinv, gammaincc, gammaincinv
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Source"))
import RainyDay_utilities_Py3.RainyDay_functions as RDF   # noqa: E402

ROOT = Path(os.environ.get("FAST_TEST_ROOT", "/Users/gjperezm/Databases/RainyDay_fast_tests"))
PROBS = (5., 25., 50., 75., 95.)
MISSING = -9999.


class DumpedPool:
    """The exact engine's pool of one duration (RAINYDAY_DUMP_EXACTPOOL): the top-M values (descending, float32), all
    weights 1."""

    def __init__(self, f):
        z = np.load(f)
        self.v = z["v"]
        self.v64 = -self.v.astype(np.float64)               # ascending, for searchsorted
        self.nall, self.full, self.nv = int(z["nall"]), bool(z["full"]), int(z["nv"])
        self.nyears, self.nsim, self.calctype = int(z["nyears"]), int(z["nsim"]), str(z["calctype"])
        self.ks, self.nreal = z["ks"].astype(np.int64), int(z["nreal"])

    def _F(self, k, count):
        """P(X_(k) <= x) when `count` pool values exceed x (fewer than k events above x)."""
        if self.calctype == "ams":
            return betainc(self.nsim - k + 1., float(k), np.exp(-count / (self.nyears * self.nv)))
        return gammaincc(float(k), self.nsim * count / (self.nyears * self.nv))

    def cdf(self, k, x, strict=False):
        """F(x) = P(X_(k) <= x); strict: P(X_(k) < x). None when x lies below the dumped pool (not full)."""
        if x < 0:
            return 0. if strict and x <= MISSING else self._F(k, self.nall)
        if not self.full and (x < float(self.v[-1]) or (strict and x == float(self.v[-1]))):
            return None
        cnt = np.searchsorted(self.v64, -x, side="right" if strict else "left")   # #(v >= x) or #(v > x)
        return self._F(k, cnt)

    def quantile(self, k, q):
        if self.calctype == "ams":
            c = -np.log1p(-betaincinv(float(k), self.nsim - k + 1., 1. - q)) * self.nyears * self.nv
        else:
            c = gammaincinv(float(k), 1. - q) * self.nyears * self.nv / self.nsim
        m = int(np.floor(min(c, 4e18))) + 1
        if m > self.nall:
            return MISSING
        if m > len(self.v):
            return None
        return float(self.v[m - 1])


def realizations(dumpdir, scen, pool):
    """sortrain of all NYEARS rows (ranked from the raw depth dump <scen>_whichrain.npy as RainyDay ranks them)."""
    wr = np.load(os.path.join(dumpdir, scen + "_whichrain.npy"))
    s, _, _ = RDF.rank_frequency(wr, pool.calctype, pool.nsim, True, None)
    return s.astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--pools", help="default $FAST_TEST_ROOT/dumps/<variant>_pool")
    ap.add_argument("--out")
    a = ap.parse_args()
    pools = Path(a.pools or ROOT / "dumps" / (a.variant + "_pool"))
    names = a.names or sorted(p.name for p in (ROOT / "dumps" / a.variant).iterdir() if p.is_dir())
    cells = []
    sanity = []
    for n in names:
        hits = glob.glob(str(ROOT / "runs" / a.variant / "*" / ("%s_72" % n)))
        if not hits:
            sys.exit("no run folder for " + n)
        top = Path(hits[0]).parent
        for fe in sorted(glob.glob(str(top / ("%s_*" % n) / ("%s_*_FreqExact.csv" % n)))):
            folder = Path(fe).parent
            scen = folder.name
            d = scen[len(n) + 1:]
            pool = DumpedPool(pools / (scen + "_exactpool.npz"))
            ex = pd.read_csv(fe, comment="#")
            fa = pd.read_csv(folder / (scen + "_FreqAnalysis.csv"), skipinitialspace=True)
            rlz = fa.filter(regex=r"^rlz\d+$").values.astype(float)
            rows = pool.nsim - pool.ks
            srt = realizations(ROOT / "dumps" / a.variant / n, scen, pool)[rows]
            N = srt.shape[1]
            ok_rows = np.array_equal(np.round(ex["returnperiod"].values, 2), fa["returnperiod"].values) and \
                np.allclose(srt, rlz, atol=0.0051, rtol=0)
            qcols = ["q%02d" % p for p in PROBS]
            for t, k in enumerate(pool.ks):
                x = srt[t]
                T = pool.nsim / k
                mean_ex = ex["mean"].values[t]
                sd = x.std(ddof=1)
                z = (x.mean() - mean_ex) / (sd / np.sqrt(N)) if sd > 0 else 0.
                r = dict(shape=n, duration=d, T=T, k=int(k), N=N, rd_mean=x.mean(), exact_mean=mean_ex,
                         level=ex["level"].values[t], z_mean=z, rows_ok=ok_rows)
                qok99, qok95, qcsv = [], [], []
                for p, c in zip(PROBS, qcols):
                    Q = pool.quantile(k, p / 100.)
                    qcsv.append(Q is not None and abs(round(Q, 4) - ex[c].values[t]) <= 1.5e-4)
                    lt, le = int(np.sum(x < Q)), int(np.sum(x <= Q))
                    for lev, lst in ((0.99, qok99), (0.95, qok95)):
                        lo, hi = binom.ppf((1 - lev) / 2, N, p / 100.), binom.ppf(1 - (1 - lev) / 2, N, p / 100.)
                        lst.append(lt <= hi and le >= lo)
                    r["Q%02d" % p], r["n_lt_%02d" % p], r["n_le_%02d" % p] = Q, lt, le
                r["q_ok99"], r["q_ok95"], r["q_csv_ok"] = int(np.sum(qok99)), int(np.sum(qok95)), all(qcsv)
                for kind, obs in (("min", x.min()), ("max", x.max())):
                    F, Fs = pool.cdf(k, obs), pool.cdf(k, obs, strict=True)
                    if F is None or Fs is None:
                        r["%s_ok" % kind], r["%s_p" % kind] = np.nan, np.nan
                        continue
                    if kind == "min":
                        plo, phi = 1 - (1 - Fs) ** N, 1 - (1 - F) ** N     # P(min < obs), P(min <= obs)
                    else:
                        plo, phi = Fs ** N, F ** N                           # P(max < obs), P(max <= obs)
                    r["%s_obs" % kind], r["%s_plo" % kind], r["%s_phi" % kind] = obs, plo, phi
                    r["%s_ok" % kind] = bool(phi >= 0.05 and plo <= 0.95)
                    r["%s_exp" % kind] = ex["exp_%s_%d" % (kind, N)].values[t]
                cells.append(r)
            sanity.append((scen, ok_rows))
    df = pd.DataFrame(cells)
    pd.set_option("display.width", 220)
    print("rows/realizations/CSV quantiles consistent: %d of %d durations; CSV q columns == pool quantiles: %d of %d "
          "cells" % (sum(o for _, o in sanity), len(sanity), int(df["q_csv_ok"].sum()), len(df)))
    summ = []
    for n, g in df.groupby("shape", sort=False):
        nq = 5 * len(g)
        summ.append(dict(shape=n, cells=len(g), z_mean_absmax=g["z_mean"].abs().max(),
                         z_mean_absmedian=g["z_mean"].abs().median(), frac_absz_lt2=(g["z_mean"].abs() < 2).mean(),
                         frac_absz_lt3=(g["z_mean"].abs() < 3).mean(),
                         quant_ok99="%d/%d" % (g["q_ok99"].sum(), nq), quant_ok95="%d/%d" % (g["q_ok95"].sum(), nq),
                         min_ok90="%d/%d" % (g["min_ok"].sum(), g["min_ok"].notna().sum()),
                         max_ok90="%d/%d" % (g["max_ok"].sum(), g["max_ok"].notna().sum())))
    nq = 5 * len(df)
    summ.append(dict(shape="ALL", cells=len(df), z_mean_absmax=df["z_mean"].abs().max(),
                     z_mean_absmedian=df["z_mean"].abs().median(), frac_absz_lt2=(df["z_mean"].abs() < 2).mean(),
                     frac_absz_lt3=(df["z_mean"].abs() < 3).mean(),
                     quant_ok99="%d/%d" % (df["q_ok99"].sum(), nq), quant_ok95="%d/%d" % (df["q_ok95"].sum(), nq),
                     min_ok90="%d/%d" % (df["min_ok"].sum(), df["min_ok"].notna().sum()),
                     max_ok90="%d/%d" % (df["max_ok"].sum(), df["max_ok"].notna().sum())))
    print(pd.DataFrame(summ).to_string(index=False, float_format=lambda v: "%.3f" % v))
    print("worst mean cells (|z|):")
    w = df.reindex(df["z_mean"].abs().sort_values(ascending=False).index).head(5)
    print(w[["shape", "duration", "T", "rd_mean", "exact_mean", "level", "z_mean"]].to_string(index=False))
    bad = df[df["q_ok99"] < 5]
    print("quantile cells outside the 99 %% band: %d" % len(bad))
    if len(bad):
        cols = ["shape", "duration", "T"] + [c for c in df.columns if c.startswith(("Q", "n_lt", "n_le"))]
        print(bad[cols].head(10).to_string(index=False))
    for kind in ("min", "max"):
        b = df[df["%s_ok" % kind] == False]   # noqa: E712
        print("%s outside the central 90 %%: %d" % (kind, len(b)))
        if len(b):
            print(b[["shape", "duration", "T", "%s_obs" % kind, "%s_exp" % kind, "%s_plo" % kind,
                     "%s_phi" % kind]].head(8).to_string(index=False))
    rel = 100 * (df["exact_mean"] - df["level"]) / df["level"]
    df["mean_minus_level_pct"] = rel
    print("mean - level (%% of level) by T: " + ", ".join(
        "T=%g: %.1f..%.1f" % (T, g.min(), g.max()) for T, g in rel.groupby(df["T"])))
    if a.out:
        df.to_csv(a.out, index=False)
        print("wrote " + a.out)


if __name__ == "__main__":
    main()
