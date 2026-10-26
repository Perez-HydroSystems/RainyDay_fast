#!/usr/bin/env python
"""
GP 2026-10-26: compare two RainyDay runs (IDF_A vs IDF_B) with the combined Monte Carlo + record CI of their difference.

    python tools/compare_runs.py RUN_A RUN_B [--stat median|mean|level] [--engine auto|montecarlo|exact] [--margin 5]
                                 [--level 90] [--seed 42] [--nboot-mc 10000] [--bh] [--out FILE.csv]
                                 [--durations 1 24 ...] [--returnperiods 100 ...]
    python tools/compare_runs.py --family PAIRS.txt [same options]       (one "RUN_A,RUN_B" per line; --bh over all)

RUN_A, RUN_B: MAINPATH/<SCENARIONAME> of a MULTIPLEDURATIONS run (its <SCENARIONAME>_<d>/ folders are found), or the
folder of a single-duration run. Both runs need RECORDCI with RECORDSAVE "true" (_RecordBootstrap.csv, _RecordDraws.csv)
and the SAME record draws (record years, RECORDCI method, RECORDSEED, RECORDNBOOT, RECORDBLOCK): _RecordDraws.csv must
match, else the tool stops. A Monte Carlo run needs UNCERTAINTY "all_realizations" (rlz columns in _FreqAnalysis.csv).
GP 2026-10-26: a run may also be a FREQENGINE "exact" run (_FreqExact.csv, no realizations) or "both" (both files):
  --engine  auto (default): the realizations of _FreqAnalysis.csv when the run has them, else _FreqExact.csv;
            montecarlo / exact: force one (a "both" run counts as Monte Carlo under auto)
  --stat    median (default; Monte Carlo runs only: the median of the realizations), mean (realizations, or the exact
            `mean`, the noise-free value of the realization mean), level (the exact return level `level` of
            _FreqExact.csv in both runs; record part from _RecordBootstrapLevel.csv, written by RECORDCI when FREQENGINE
            is "exact" or "both")
GP 2026-10-27: --stat median (the line above, old) now works with every engine and has its own record part:
            Monte Carlo run: the median of the realizations (unchanged); exact run: `q50` of _FreqExact.csv, the exact
            median of ONE realization's T-yr value X_(k), which the median of the realizations estimates (no Monte Carlo
            noise). Record part: _RecordBootstrapMedian.csv (`sst_median` = q50 and the per-record medians rec*/drop*,
            written by RECORDCI with RECORDSAVE "true" since 2026-10-27). Runs made before that file existed fall back
            to the MEAN's record bootstrap (_RecordBootstrap.csv; an approximation) with a printed WARNING; the output
            column rec_source says which: median, mean-fallback (--stat median), mean, level.
  An exact statistic has no Monte Carlo noise: its Monte Carlo bootstrap is the constant itself (both runs exact, or
  --stat level: the Monte Carlo part is delta with zero width, and the CI is the record part only). Mixed pairs (Monte
  Carlo vs exact) take the Monte Carlo part from the Monte Carlo run only.

Per duration x return period (the rows both runs share):
  delta     = 100 (stat_B - stat_A) / stat_A, stat = median (default) or mean of the realizations (or the exact values)
  Monte Carlo part: unpaired bootstrap of the realization statistic of each run (independent runs), --nboot-mc draws,
            np.random.default_rng(seed + d) (the indices of A, then of B: PUBLIC/FIGURES/sst_compare.delta_boot)
  record part: paired bootstrap records of the two runs (same drawn years): delta*_j = 100 (B*_j - A*_j) / A*_j of the
            noise-free estimates (_RecordBootstrap.csv; --stat level: _RecordBootstrapLevel.csv; GP 2026-10-27:
            --stat median: _RecordBootstrapMedian.csv); deviations delta*_j - delta*_point (jackknife: normal deviations
            with the jackknife SE of delta)
  combined: every Monte Carlo draw plus an independent record deviation (record draw j uniform,
            np.random.default_rng(seed + 1000 + d))
  classes (the paper's rule, margin m, level L = 90 -> alpha = 0.05):
            equivalent   the L % CI of the combined draws is inside (-m, +m)            (TOST at alpha)
            increase / decrease   the (100 - 100 alpha) % CI (95 %) is beyond +m / -m   (minimum-effect test at alpha;
                         with --bh: Benjamini-Hochberg q <= alpha over all rows instead)
            inconclusive otherwise
  p-values (bootstrap): p_tost (small = equivalent), p_diff (two-sided minimum-effect, small = |delta| > m), as
            PUBLIC/FIGURES/sst_compare.py; class_mc / p_*_mc: the same with the Monte Carlo part only.
Output: one CSV row per duration x return period (default ./compare_<A>_vs_<B>.csv), and a class count summary.
GP 2026-10-27: new columns go at the end (rec_source is the last column), so the existing column positions are unchanged.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------------------------------
# statistics (port of PUBLIC/FIGURES/sst_compare.py, GP 2026-10-26)
# ---------------------------------------------------------------------------------------------------------------------
def p_tost(boot, margin):
    """Bootstrap p-value of the two one-sided tests for equivalence within +-margin (small p = equivalent)."""
    B = boot.size
    return float(max((np.sum(boot >= margin) + 1) / (B + 1), (np.sum(boot <= -margin) + 1) / (B + 1)))


def p_min_effect(boot, delta, margin):
    """Two-sided bootstrap p-value of the minimum-effect test H0: |delta| <= margin (small p = |delta| > margin)."""
    B = boot.size
    p_one = (np.sum(boot <= margin) + 1) / (B + 1) if delta >= 0 else (np.sum(boot >= -margin) + 1) / (B + 1)
    return float(min(1.0, 2.0 * p_one))


def bh(p):
    """Benjamini-Hochberg adjusted p-values (NaN kept as NaN and excluded from the family)."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def classify(delta, eq_lo, eq_hi, d_lo, d_hi, margin, alpha, q_diff=None):
    if eq_lo > -margin and eq_hi < margin:
        return "equivalent"
    if q_diff is not None and np.isfinite(q_diff):
        different = q_diff <= alpha
    else:
        different = d_lo > margin or d_hi < -margin
    if different:
        return "increase" if delta > 0 else "decrease"
    return "inconclusive"


# ---------------------------------------------------------------------------------------------------------------------
# run files
# ---------------------------------------------------------------------------------------------------------------------
RESULTS = ("_FreqAnalysis.csv", "_FreqExact.csv")    # GP 2026-10-26: Monte Carlo and exact (FREQENGINE) results


def _has_results(folder, name):
    return any(os.path.exists(os.path.join(folder, name + s)) for s in RESULTS)


def durations_of(run):
    """{duration (float or None): (folder, name)} of a run stem or a single-duration run folder (GP 2026-10-26: with a
    _FreqAnalysis.csv or a _FreqExact.csv)."""
    run = os.path.abspath(run.rstrip("/"))
    base = os.path.basename(run)
    if os.path.isdir(run) and _has_results(run, base):
        m = re.search(r"_(\d+(?:\.\d+)?)$", base)
        return {float(m.group(1)) if m else None: (run, base)}
    out = {}
    for d in glob.glob(run + "_*"):
        tag = os.path.basename(d)[len(base) + 1:]
        if os.path.isdir(d) and re.fullmatch(r"\d+(?:\.\d+)?", tag) and _has_results(d, os.path.basename(d)):
            out[float(tag)] = (d, os.path.basename(d))
    if not out:
        sys.exit("compare_runs: no FreqAnalysis results found for " + run)
    return out


def read_draws(run, durs):
    files = sorted(set(f for folder, _ in durs.values() for f in glob.glob(os.path.join(folder, "*_RecordDraws.csv"))))
    if len(files) != 1:
        sys.exit("compare_runs: expected one _RecordDraws.csv for %s (RECORDCI with RECORDSAVE 'true'), found %d"
                 % (run, len(files)))
    head = {}
    with open(files[0]) as f:
        for line in f:
            m = re.match(r"#\s*(method|seed|block|draws|years):\s*(.*)$", line.strip())
            if m:
                head[m.group(1)] = m.group(2).strip()
    tab = pd.read_csv(files[0], comment="#")
    return files[0], head, tab


def read_fa(folder, name):
    df = pd.read_csv(os.path.join(folder, name + "_FreqAnalysis.csv"), skipinitialspace=True)
    r = df.filter(regex=r"^rlz\d+$")
    if r.shape[1] == 0:
        sys.exit("compare_runs: %s_FreqAnalysis.csv has no realization columns (UNCERTAINTY 'all_realizations')" % name)
    return np.round(df["returnperiod"].values.astype(float), 2), r.values.astype(float)


def read_exact(folder, name, column):
    """GP 2026-10-26: one column (mean or level) of _FreqExact.csv (FREQENGINE 'exact'/'both')."""
    f = os.path.join(folder, name + "_FreqExact.csv")
    if not os.path.exists(f):
        sys.exit("compare_runs: missing " + f + " (FREQENGINE 'exact' or 'both')")
    df = pd.read_csv(f, comment="#", skipinitialspace=True)
    return np.round(df["returnperiod"].values.astype(float), 2), df[column].values.astype(float)


def read_result(folder, name, args):
    """GP 2026-10-26: (engine, return periods, values): ('montecarlo', rp, realizations (rows x rlz)) or ('exact', rp,
    the exact statistic per row), by --engine and --stat."""
    has_rlz = False
    fa = os.path.join(folder, name + "_FreqAnalysis.csv")
    if os.path.exists(fa):
        has_rlz = any(re.fullmatch(r"rlz\d+", c.strip()) for c in open(fa).readline().split(","))
    if args.stat == "level" or args.engine == "exact" or (args.engine == "auto" and not has_rlz):
        # GP 2026-10-27: --stat median of an exact run = q50 of _FreqExact.csv (the exact median of one realization's
        # T-yr value, which the median of the realizations estimates); the old stop is kept as a comment
        # if args.stat == "median":
        #     sys.exit("compare_runs: --stat median is a Monte Carlo statistic (the median of the realizations); %s has "
        #              "no realizations (FREQENGINE 'exact') or --engine exact was given: use --stat mean (the exact "
        #              "mean = the noise-free realization mean) or --stat level (the return level); the median of ONE "
        #              "realization's T-yr value is q50 in _FreqExact.csv" % name)
        # rp, v = read_exact(folder, name, "level" if args.stat == "level" else "mean")
        rp, v = read_exact(folder, name, {"level": "level", "median": "q50"}.get(args.stat, "mean"))
        return "exact", rp, v
    rp, r = read_fa(folder, name)
    return "montecarlo", rp, r


# GP 2026-10-27: the record bootstrap file and its point column of each record statistic
REC_FILES = {"mean": ("_RecordBootstrap.csv", "sst_expected"), "level": ("_RecordBootstrapLevel.csv", "sst_level"),
             "median": ("_RecordBootstrapMedian.csv", "sst_median")}


def has_rec(folder, name, kind):
    """GP 2026-10-27: True when the run folder has the record bootstrap file of kind (mean, level, median)."""
    return os.path.exists(os.path.join(folder, name + REC_FILES[kind][0]))


# def read_rec(folder, name, level=False):     # GP 2026-10-27: old line; kind = mean | level | median
def read_rec(folder, name, kind="mean"):
    # GP 2026-10-26: level=True: _RecordBootstrapLevel.csv (record bootstrap of the return level, --stat level)
    # GP 2026-10-27: kind "median": _RecordBootstrapMedian.csv (record bootstrap of the median q50, --stat median)
    # f = os.path.join(folder, name + ("_RecordBootstrapLevel.csv" if level else "_RecordBootstrap.csv"))
    level = kind == "level"
    fname, point = REC_FILES[kind]
    f = os.path.join(folder, name + fname)
    if not os.path.exists(f):
        sys.exit("compare_runs: missing " + f + " (RECORDCI with RECORDSAVE 'true'" +
                 ("; the level bootstrap is written when FREQENGINE is 'exact' or 'both')" if level else ")"))
    df = pd.read_csv(f, skipinitialspace=True)
    cols = [c for c in df.columns if re.fullmatch(r"(rec|drop)\d+", c)]
    # return np.round(df["returnperiod"].values.astype(float), 2), \
    #     df["sst_level" if level else "sst_expected"].values.astype(float), df[cols].values.astype(float), cols
    return np.round(df["returnperiod"].values.astype(float), 2), \
        df[point].values.astype(float), df[cols].values.astype(float), cols


# ---------------------------------------------------------------------------------------------------------------------
def compare_pair(A, B, args):
    da, db = durations_of(A), durations_of(B)
    fa, ha, ta = read_draws(A, da)
    fb, hb, tb = read_draws(B, db)
    if ha != hb or not ta.equals(tb):
        diff = [k for k in sorted(set(ha) | set(hb)) if ha.get(k) != hb.get(k)] or ["the drawn years"]
        sys.exit("compare_runs: the two runs used different record draws (%s differ):\n  %s\n  %s\nRun both with the "
                 "same record years (INCLUDEYEARS), RECORDCI method, RECORDSEED, RECORDNBOOT and RECORDBLOCK."
                 % (", ".join(diff), fa, fb))
    method = ha.get("method", "bootstrap")
    common = sorted(set(da) & set(db), key=lambda x: -1 if x is None else x)
    if args.durations:
        common = [d for d in common if d is not None and any(np.isclose(d, x) for x in args.durations)]
    if not common:
        sys.exit("compare_runs: no common durations")
    stat = np.median if args.stat == "median" else np.mean
    # GP 2026-10-27: the record part of --stat median comes from _RecordBootstrapMedian.csv; when a run lacks it (made
    # before 2026-10-27), both runs fall back to the mean's bootstrap (the same statistic in both, paired by draw)
    rec_kind = {}
    for d in common:
        if args.stat == "median":
            both = has_rec(*da[d], "median") and has_rec(*db[d], "median")
            rec_kind[d] = ("median", "median") if both else ("mean", "mean-fallback")
        else:
            rec_kind[d] = ("level", "level") if args.stat == "level" else ("mean", "mean")
    lack = [(tag, run) for tag, run, dd in (("A", A, da), ("B", B, db))
            if any(not has_rec(*dd[d], "median") for d in common)] if args.stat == "median" else []
    if lack:
        nfb = sum(rec_kind[d][1] == "mean-fallback" for d in common)
        print("WARNING: compare_runs --stat median: no _RecordBootstrapMedian.csv in %s (a run made before 2026-10-27):"
              " the record part of %d of %d durations borrows the MEAN's record bootstrap (_RecordBootstrap.csv, an "
              "approximation); rec_source = mean-fallback"
              % (" and ".join("run %s %s" % (t, os.path.abspath(r)) for t, r in lack), nfb, len(common)), flush=True)
    L = args.level
    alpha = (100. - L) / 200.
    qe = [100 * alpha, 100 * (1 - alpha)]                 # L % CI (equivalence, TOST at alpha)
    qd = [100 * alpha / 2, 100 * (1 - alpha / 2)]         # (100 - 100 alpha) % CI (difference, two-sided alpha)
    le, ld = "%g" % L, "%g" % (100 - 100 * alpha)
    rows = []
    for d in common:
        (fA, nA), (fB, nB) = da[d], db[d]
        # rpA, ra = read_fa(fA, nA)                  # GP 2026-10-26: old lines; Monte Carlo or exact results
        # rpB, rb = read_fa(fB, nB)
        engA, rpA, ra = read_result(fA, nA, args)
        engB, rpB, rb = read_result(fB, nB, args)
        # qA, ptA, recA, cA = read_rec(fA, nA, level=args.stat == "level")    # GP 2026-10-27: old lines
        # qB, ptB, recB, cB = read_rec(fB, nB, level=args.stat == "level")
        qA, ptA, recA, cA = read_rec(fA, nA, rec_kind[d][0])
        qB, ptB, recB, cB = read_rec(fB, nB, rec_kind[d][0])
        if cA != cB:
            sys.exit("compare_runs: %s and %s have different record columns" % (nA, nB))
        for T in [t for t in rpA if t in set(rpB)]:
            if args.returnperiods and not any(np.isclose(T, x) for x in args.returnperiods):
                continue
            i, j = int(np.where(rpA == T)[0][0]), int(np.where(rpB == T)[0][0])
            ir, jr = int(np.where(qA == T)[0][0]), int(np.where(qB == T)[0][0])
            # GP 2026-10-26: an exact statistic (FREQENGINE) is a constant: no Monte Carlo bootstrap for that run
            # a, b = ra[i][np.isfinite(ra[i])], rb[j][np.isfinite(rb[j])]
            # sa, sb = stat(a), stat(b)
            a = ra[i][np.isfinite(ra[i])] if engA == "montecarlo" else None
            b = rb[j][np.isfinite(rb[j])] if engB == "montecarlo" else None
            sa = stat(a) if a is not None else float(ra[i])
            sb = stat(b) if b is not None else float(rb[j])
            delta = 100. * (sb - sa) / sa
            dseed = int(round(d)) if d is not None else 0
            rng = np.random.default_rng(args.seed + dseed)
            # ma = stat(a[rng.integers(0, a.size, (args.nboot_mc, a.size))], axis=1)
            # mb = stat(b[rng.integers(0, b.size, (args.nboot_mc, b.size))], axis=1)
            ma = stat(a[rng.integers(0, a.size, (args.nboot_mc, a.size))], axis=1) if a is not None else \
                np.full(args.nboot_mc, sa)
            mb = stat(b[rng.integers(0, b.size, (args.nboot_mc, b.size))], axis=1) if b is not None else \
                np.full(args.nboot_mc, sb)
            mc = 100. * (mb - ma) / ma
            rec = 100. * (recB[jr] - recA[ir]) / recA[ir]
            rec_pt = 100. * (ptB[jr] - ptA[ir]) / ptA[ir]
            rng2 = np.random.default_rng(args.seed + 1000 + dseed)
            if method == "jackknife":
                n = rec.size
                se = np.sqrt((n - 1) / n * np.sum((rec - rec.mean()) ** 2))
                dev = se * rng2.standard_normal(args.nboot_mc)
                rlo, rhi = rec_pt - se * _z(qe[1]), rec_pt + se * _z(qe[1])
            else:
                dev = rec[rng2.integers(0, rec.size, args.nboot_mc)] - rec_pt
                rlo, rhi = np.percentile(rec, qe)
            comb = mc + dev
            r = dict(run_a=nA, run_b=nB, duration_h=d, returnperiod=T, stat=args.stat, stat_a_mm=sa, stat_b_mm=sb,
                     delta_pct=delta)
            r["mc_ci%s_lo" % le], r["mc_ci%s_hi" % le] = np.percentile(mc, qe)
            r["mc_ci%s_lo" % ld], r["mc_ci%s_hi" % ld] = np.percentile(mc, qd)
            r["rec_delta_point_pct"] = rec_pt
            r["rec_ci%s_lo" % le], r["rec_ci%s_hi" % le] = rlo, rhi
            r["comb_ci%s_lo" % le], r["comb_ci%s_hi" % le] = np.percentile(comb, qe)
            r["comb_ci%s_lo" % ld], r["comb_ci%s_hi" % ld] = np.percentile(comb, qd)
            r["p_tost_mc"], r["p_diff_mc"] = p_tost(mc, args.margin), p_min_effect(mc, delta, args.margin)
            r["p_tost"], r["p_diff"] = p_tost(comb, args.margin), p_min_effect(comb, delta, args.margin)
            r["engine_a"], r["engine_b"] = engA, engB             # GP 2026-10-26: montecarlo or exact
            r["rec_source"] = rec_kind[d][1]                      # GP 2026-10-27: moved to the last column (finish)
            r["_le"], r["_ld"] = le, ld
            rows.append(r)
    return rows


def _z(q):
    from scipy.special import ndtri
    return float(ndtri(q / 100.))


def finish(rows, args):
    df = pd.DataFrame(rows)
    le, ld = df["_le"].iloc[0], df["_ld"].iloc[0]
    df = df.drop(columns=["_le", "_ld"])
    alpha = (100. - args.level) / 200.
    if args.bh:
        df["q_diff_mc"] = bh(df["p_diff_mc"].values)
        df["q_diff"] = bh(df["p_diff"].values)
    df["class_mc"] = [classify(r["delta_pct"], r["mc_ci%s_lo" % le], r["mc_ci%s_hi" % le], r["mc_ci%s_lo" % ld],
                               r["mc_ci%s_hi" % ld], args.margin, alpha, r["q_diff_mc"] if args.bh else None)
                      for _, r in df.iterrows()]
    df["class"] = [classify(r["delta_pct"], r["comb_ci%s_lo" % le], r["comb_ci%s_hi" % le], r["comb_ci%s_lo" % ld],
                            r["comb_ci%s_hi" % ld], args.margin, alpha, r["q_diff"] if args.bh else None)
                   for _, r in df.iterrows()]
    df = df[[c for c in df.columns if c != "rec_source"] + ["rec_source"]]     # GP 2026-10-27: last column
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="RUN_A RUN_B")
    ap.add_argument("--family", help="text file, one 'RUN_A,RUN_B' pair per line (BH with --bh over all rows)")
    # ap.add_argument("--stat", choices=("median", "mean"), default="median")   # GP 2026-10-26: old line
    # ap.add_argument("--stat", choices=("median", "mean", "level"), default="median")   # GP 2026-10-27: old line
    ap.add_argument("--stat", choices=("median", "mean", "level"), default="median",
                    help="median: of the realizations (exact run: q50 of _FreqExact.csv), record part from "
                         "_RecordBootstrapMedian.csv (older runs: the mean's, with a warning); mean; level "
                         "(GP 2026-10-27)")
    ap.add_argument("--engine", choices=("auto", "montecarlo", "exact"), default="auto",
                    help="result of each run: auto = realizations if present, else _FreqExact.csv (GP 2026-10-26)")
    ap.add_argument("--margin", type=float, default=5.)
    ap.add_argument("--level", type=float, default=90.)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nboot-mc", type=int, default=10000)
    ap.add_argument("--bh", action="store_true", help="Benjamini-Hochberg over all output rows decides differences")
    ap.add_argument("--durations", type=float, nargs="*")
    ap.add_argument("--returnperiods", type=float, nargs="*")
    ap.add_argument("--out")
    args = ap.parse_args()
    if not 0 < args.level < 100:
        sys.exit("--level must be in (0, 100)")
    pairs = []
    if args.family:
        for line in open(args.family):
            line = line.strip()
            if line and not line.startswith("#"):
                pairs.append([x.strip() for x in re.split(r"[,\s]+", line) if x.strip()][:2])
    if args.runs:
        if len(args.runs) != 2:
            sys.exit("give two runs (RUN_A RUN_B) or --family")
        pairs.append(args.runs)
    if not pairs:
        sys.exit("give two runs (RUN_A RUN_B) or --family")
    rows = []
    for A, B in pairs:
        rows += compare_pair(A, B, args)
    df = finish(rows, args)
    out = args.out or "compare_%s_vs_%s.csv" % (os.path.basename(pairs[0][0].rstrip("/")),
                                                os.path.basename(pairs[0][1].rstrip("/")))
    df.to_csv(out, index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    show = ["run_b", "duration_h", "returnperiod", "delta_pct"] + [c for c in df.columns if c.startswith("comb_ci")][:2] \
        + ["class_mc", "class"]
    print(df[show].to_string(index=False, float_format=lambda v: "%.2f" % v))
    print("classes (Monte Carlo only -> Monte Carlo + record):")
    print(pd.crosstab(df["class_mc"], df["class"]).to_string())
    print("wrote " + os.path.abspath(out))


if __name__ == "__main__":
    main()
