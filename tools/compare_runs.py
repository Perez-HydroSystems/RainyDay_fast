#!/usr/bin/env python
"""
GP 2026-10-26: compare two RainyDay runs (IDF_A vs IDF_B) with the combined Monte Carlo + record CI of their difference.

    python tools/compare_runs.py RUN_A RUN_B [--stat median|mean] [--margin 5] [--level 90] [--seed 42]
                                 [--nboot-mc 10000] [--bh] [--out FILE.csv] [--durations 1 24 ...]
    python tools/compare_runs.py --family PAIRS.txt [same options]       (one "RUN_A,RUN_B" per line; --bh over all)

RUN_A, RUN_B: MAINPATH/<SCENARIONAME> of a MULTIPLEDURATIONS run (its <SCENARIONAME>_<d>/ folders are found), or the
folder of a single-duration run. Both runs need UNCERTAINTY "all_realizations" and RECORDCI with RECORDSAVE "true"
(_FreqAnalysis.csv with rlz columns, _RecordBootstrap.csv, _RecordDraws.csv), and the SAME record draws (record years,
RECORDCI method, RECORDSEED, RECORDNBOOT, RECORDBLOCK): _RecordDraws.csv must match, else the tool stops.

Per duration x return period (the rows both runs share):
  delta     = 100 (stat_B - stat_A) / stat_A, stat = median (default) or mean of the realizations
  Monte Carlo part: unpaired bootstrap of the realization statistic of each run (independent runs), --nboot-mc draws,
            np.random.default_rng(seed + d) (the indices of A, then of B: PUBLIC/FIGURES/sst_compare.delta_boot)
  record part: paired bootstrap records of the two runs (same drawn years): delta*_j = 100 (B*_j - A*_j) / A*_j of the
            noise-free estimates (_RecordBootstrap.csv); deviations delta*_j - delta*_point (jackknife: normal
            deviations with the jackknife SE of delta)
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
def durations_of(run):
    """{duration (float or None): (folder, name)} of a run stem or a single-duration run folder."""
    run = os.path.abspath(run.rstrip("/"))
    base = os.path.basename(run)
    if os.path.isdir(run) and os.path.exists(os.path.join(run, base + "_FreqAnalysis.csv")):
        m = re.search(r"_(\d+(?:\.\d+)?)$", base)
        return {float(m.group(1)) if m else None: (run, base)}
    out = {}
    for d in glob.glob(run + "_*"):
        tag = os.path.basename(d)[len(base) + 1:]
        if os.path.isdir(d) and re.fullmatch(r"\d+(?:\.\d+)?", tag) and os.path.exists(
                os.path.join(d, os.path.basename(d) + "_FreqAnalysis.csv")):
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


def read_rec(folder, name):
    f = os.path.join(folder, name + "_RecordBootstrap.csv")
    if not os.path.exists(f):
        sys.exit("compare_runs: missing " + f + " (RECORDCI with RECORDSAVE 'true')")
    df = pd.read_csv(f, skipinitialspace=True)
    cols = [c for c in df.columns if re.fullmatch(r"(rec|drop)\d+", c)]
    return np.round(df["returnperiod"].values.astype(float), 2), df["sst_expected"].values.astype(float), \
        df[cols].values.astype(float), cols


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
    L = args.level
    alpha = (100. - L) / 200.
    qe = [100 * alpha, 100 * (1 - alpha)]                 # L % CI (equivalence, TOST at alpha)
    qd = [100 * alpha / 2, 100 * (1 - alpha / 2)]         # (100 - 100 alpha) % CI (difference, two-sided alpha)
    le, ld = "%g" % L, "%g" % (100 - 100 * alpha)
    rows = []
    for d in common:
        (fA, nA), (fB, nB) = da[d], db[d]
        rpA, ra = read_fa(fA, nA)
        rpB, rb = read_fa(fB, nB)
        qA, ptA, recA, cA = read_rec(fA, nA)
        qB, ptB, recB, cB = read_rec(fB, nB)
        if cA != cB:
            sys.exit("compare_runs: %s and %s have different record columns" % (nA, nB))
        for T in [t for t in rpA if t in set(rpB)]:
            if args.returnperiods and not any(np.isclose(T, x) for x in args.returnperiods):
                continue
            i, j = int(np.where(rpA == T)[0][0]), int(np.where(rpB == T)[0][0])
            ir, jr = int(np.where(qA == T)[0][0]), int(np.where(qB == T)[0][0])
            a, b = ra[i][np.isfinite(ra[i])], rb[j][np.isfinite(rb[j])]
            sa, sb = stat(a), stat(b)
            delta = 100. * (sb - sa) / sa
            dseed = int(round(d)) if d is not None else 0
            rng = np.random.default_rng(args.seed + dseed)
            ma = stat(a[rng.integers(0, a.size, (args.nboot_mc, a.size))], axis=1)
            mb = stat(b[rng.integers(0, b.size, (args.nboot_mc, b.size))], axis=1)
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
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="RUN_A RUN_B")
    ap.add_argument("--family", help="text file, one 'RUN_A,RUN_B' pair per line (BH with --bh over all rows)")
    ap.add_argument("--stat", choices=("median", "mean"), default="median")
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
