"""
GP 2026-10-22: RainyDay_fast acceptance check: a test variant must reproduce a reference variant BIT FOR BIT.

    python tools/compare_dumps.py <ref_variant> <test_variant> [name ...] [--catalogs] [--csv-ref DIR]

For every shape (default: every dump folder of the reference) under $FAST_TEST_ROOT
(default /Users/gjperezm/Databases/RainyDay_fast_tests):
  1. dumps/<ref>/<name>/*_whichrain.npy vs dumps/<test>/<name>/: same files, np.array_equal (raw transposed depths
     of every duration, before ranking; written with RAINYDAY_DUMP_WHICHRAIN)
  2. runs/<ref>/<set>/<name>_<d>/<name>_<d>_FreqAnalysis.csv vs the test run: byte-identical
  3. --catalogs: tools/compare_catalogs.py on the two StormCatalog folders (only when both runs built a full catalog)
  4. --csv-ref DIR: the test CSVs also byte-identical to DIR/<set>/<name>_<d>[h]/..._FreqAnalysis.csv
     (e.g. the Mac dry run of 2026-10-21 in ubuntu_kit/reference_md)
Prints PASS/FAIL per item and ALL PASS at the end; exit 0 = ALL PASS.
"""
import argparse
import filecmp
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("FAST_TEST_ROOT", "/Users/gjperezm/Databases/RainyDay_fast_tests"))
TOOLS = Path(__file__).resolve().parent


def run_dir(variant, name):
    hits = glob.glob(str(ROOT / "runs" / variant / "*" / f"{name}_72"))
    return Path(hits[0]).parent if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("test"); ap.add_argument("names", nargs="*")
    ap.add_argument("--catalogs", action="store_true"); ap.add_argument("--csv-ref")
    a = ap.parse_args()
    names = a.names or sorted(p.name for p in (ROOT / "dumps" / a.ref).iterdir() if p.is_dir())
    res = []

    def report(ok, msg):
        res.append(ok); print(f"{'PASS' if ok else 'FAIL'}  {msg}", flush=True)

    for n in names:
        rd, td = ROOT / "dumps" / a.ref / n, ROOT / "dumps" / a.test / n
        rf = sorted(p.name for p in rd.glob("*_whichrain.npy")); tf = sorted(p.name for p in td.glob("*_whichrain.npy"))
        report(rf == tf and len(rf) > 0, f"{n}: dump files {len(rf)} ref / {len(tf)} test")
        for f in rf:
            if f in tf:
                x, y = np.load(rd / f), np.load(td / f)
                same = x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)
                extra = "" if same or x.shape != y.shape else \
                    f" ({int(np.sum(x != y))} values differ, max |diff| {float(np.max(np.abs(x - y))):.3g})"
                report(same, f"{n}: {f} bitwise{extra}")
        rr, tr = run_dir(a.ref, n), run_dir(a.test, n)
        if rr is None or tr is None:
            report(False, f"{n}: run folder missing (ref {rr}, test {tr})"); continue
        csvs = sorted(glob.glob(str(rr / f"{n}_*" / f"{n}_*_FreqAnalysis.csv")))
        for c in csvs:
            t = tr / Path(c).parent.name / Path(c).name
            report(t.exists() and filecmp.cmp(c, t, shallow=False), f"{n}: {Path(c).name} byte-identical")
        if a.csv_ref:
            for c in sorted(glob.glob(str(tr / f"{n}_*" / f"{n}_*_FreqAnalysis.csv"))):
                d = Path(c).parent.name.rsplit("_", 1)[1]
                # GP 2026-10-22: exact duration folder (<name>_<d> or <name>_<d>h): "_1*" also matched "_12h"
                # cands = glob.glob(os.path.join(a.csv_ref, "*", f"{n}_{d}*", f"{n}_{d}*_FreqAnalysis.csv"))
                cands = [p for p in glob.glob(os.path.join(a.csv_ref, "*", f"{n}_{d}*", f"{n}_{d}*_FreqAnalysis.csv"))
                         if Path(p).parent.name in (f"{n}_{d}", f"{n}_{d}h")]
                if cands:
                    report(filecmp.cmp(c, cands[0], shallow=False), f"{n}: {Path(c).name} == {cands[0]}")
        if a.catalogs:
            rc, tc = rr / f"{n}_72" / "StormCatalog", tr / f"{n}_72" / "StormCatalog"
            if rc.is_dir() and tc.is_dir() and any(tc.glob("*_storm_*.nc")):
                p = subprocess.run([sys.executable, str(TOOLS / "compare_catalogs.py"), str(rc), str(tc), "--rtol", "0"],
                                   capture_output=True, text=True)
                report(p.returncode == 0, f"{n}: compare_catalogs.py " + (p.stdout.strip().splitlines() or ["?"])[-1])
    print("ALL PASS" if res and all(res) else f"SOME CHECKS FAILED ({res.count(False)} of {len(res)})")
    sys.exit(0 if res and all(res) else 1)


if __name__ == "__main__":
    main()
