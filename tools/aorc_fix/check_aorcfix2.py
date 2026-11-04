"""
GP 2026-10-21: verify a corrected AORC view built by make_aorcfix2.py.

    python check_aorcfix2.py <src_dir> <prev_fix_dir> <dst_dir> [--years 2002 2023] [--sample 20] [--only-days ...]

Checks (prints PASS/FAIL per item and ALL PASS at the end; exit 0/1):
  C1 completeness: every source file has an entry in dst (file or symlink); patched days are real files with `aorcfix2`
  C2 every patched day up to 2020: RAINRATE 23:00 is no longer identical to 22:00 (when 22:00 has rain)
  C3 sample of days: patched hour == NOAA v1.1 at the RainyDay cells (|diff| <= 0.035 mm, quantisation), all other
     hours of RAINRATE and the whole TMP_2maboveground byte-identical to the source file
  C4 2021-12-31 differs from 2021-12-30 and equals v1.1; 2011-07-23 00:00 not all missing; 2022-05-29 20:00 not uniform
  C5 manifest of every real (patched) file -> dst/_aorcfix2_manifest.tsv: name, md5 of the file, md5 of the patched
     RAINRATE data (hour 23:00; whole RAINRATE for the 2011-07-22, 2021-12-31 and 2022-05-29 files). The file md5 depends
     on the HDF5 build; the data md5 must be the same on every machine (compare the 3rd column Mac vs Ubuntu)
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_aorcfix2 import file_times, open_v11, v11_fields  # noqa: E402

RES = []


def report(name, ok, msg=""):
    RES.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {msg}", flush=True)


def read(path, var="RAINRATE"):
    with nc.Dataset(path) as f:
        v = f.variables[var]; v.set_auto_mask(False)
        return np.asarray(v[:], np.float32), ("aorcfix2" in f.ncattrs())


def read_hours(path, k0, k1):
    """RAINRATE[k0:k1] only (C2 reads 2 of 24 hours: one netCDF time chunk instead of the whole file)."""
    with nc.Dataset(path) as f:
        v = f.variables["RAINRATE"]; v.set_auto_mask(False)
        return np.asarray(v[k0:k1], np.float32), ("aorcfix2" in f.ncattrs())


def source_of(name, src, prev):
    p = prev / name
    return p if p.exists() and not p.is_symlink() else src / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("prev"); ap.add_argument("dst")
    ap.add_argument("--years", nargs=2, type=int, default=[2002, 2023])
    ap.add_argument("--last-dup-year", type=int, default=2020)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--only-days", nargs="*", default=None)
    a = ap.parse_args()
    src, prev, dst = Path(a.src), Path(a.prev), Path(a.dst)
    rng = np.random.default_rng(20261021)
    if a.only_days:
        days = [pd.Timestamp(d) for d in a.only_days]
        sample = [d for d in days if d.year <= a.last_dup_year]
    else:
        missing = [f.name for f in src.glob("AORC.*.preciptemp.nc") if not (dst / f.name).exists()]
        report("C1 completeness", len(missing) == 0, f"{len(missing)} source files without an entry in dst")
        days = list(pd.date_range(f"{a.years[0]}-01-01", f"{min(a.years[1], a.last_dup_year)}-12-31"))
        days = [d for d in days if (src / f"AORC.{d:%Y%m%d}.preciptemp.nc").exists()]
        sample = list(rng.choice(days, size=min(a.sample, len(days)), replace=False))
    bad_attr, still_dup, data_md5 = [], [], {}
    for d in days:
        if d.year > a.last_dup_year:
            continue
        p = dst / f"AORC.{d:%Y%m%d}.preciptemp.nc"
        if p.is_symlink() or not p.exists():
            bad_attr.append(p.name); continue
        t = file_times(p)
        k23 = int(np.where(t.hour == 23)[0][0])
        r, has = read_hours(p, k23 - 1, k23 + 1)                  # 22:00, 23:00
        if not has:
            bad_attr.append(p.name)
        if (r[0] > 0).any() and np.array_equal(r[1], r[0], equal_nan=True):
            still_dup.append(p.name)
        data_md5[p.name] = hashlib.md5(r[1].tobytes()).hexdigest()
    report("C1b patched days are real files with the aorcfix2 attribute", len(bad_attr) == 0, f"{len(bad_attr)} bad {bad_attr[:3]}")
    report("C2 23:00 no longer a copy of 22:00", len(still_dup) == 0, f"{len(still_dup)} days still identical {still_dup[:3]}")
    worst, other_bad = 0.0, []
    for d in sample:
        d = pd.Timestamp(d)
        name = f"AORC.{d:%Y%m%d}.preciptemp.nc"
        p, s = dst / name, source_of(name, src, prev)
        r, _ = read(p); r0, _ = read(s)
        tmp, _ = read(p, "TMP_2maboveground"); tmp0, _ = read(s, "TMP_2maboveground")
        t = file_times(p)
        k23 = int(np.where(t.hour == 23)[0][0])
        st = t[k23]
        ref = v11_fields(open_v11(st.year), [st])[0]
        ok = np.isfinite(ref) & (r[k23] >= 0)
        worst = max(worst, float(np.max(np.abs(r[k23][ok] - ref[ok]))) if ok.any() else 0.0)
        others = [k for k in range(len(t)) if k != k23 and not (name == "AORC.20110722.preciptemp.nc" and t[k] == pd.Timestamp("2011-07-23 00:00"))]
        if not all(np.array_equal(r[k], r0[k], equal_nan=True) for k in others) or not np.array_equal(tmp, tmp0, equal_nan=True):
            other_bad.append(name)
    report("C3 patched hour == NOAA v1.1", worst <= 0.035, f"max |diff| {worst:.4f} mm over {len(sample)} sampled days")
    report("C3b other hours and TMP byte-identical to the source", len(other_bad) == 0, f"{other_bad[:3]}")
    if a.years[0] <= 2021 <= a.years[1] and (a.only_days is None or "20211231" in a.only_days):
        r31, _ = read(dst / "AORC.20211231.preciptemp.nc"); r30, _ = read(source_of("AORC.20211230.preciptemp.nc", src, prev))
        t = file_times(dst / "AORC.20211231.preciptemp.nc")
        ref = np.concatenate([v11_fields(open_v11(2021), list(t[:-1])), v11_fields(open_v11(2022), [t[-1]])])
        ok = np.isfinite(ref) & np.isfinite(r31) & (r31 >= 0)
        report("C4a 2021-12-31 != 2021-12-30 and == v1.1", (not np.array_equal(r31, r30, equal_nan=True))
               and float(np.max(np.abs(r31[ok] - ref[ok]))) <= 0.01, f"max |diff| vs v1.1 {float(np.max(np.abs(r31[ok] - ref[ok]))):.4f}")
    if a.years[0] <= 2011 <= a.years[1] and (a.only_days is None or "20110722" in a.only_days):
        r, _ = read(dst / "AORC.20110722.preciptemp.nc"); t = file_times(dst / "AORC.20110722.preciptemp.nc")
        k = int(np.where(t == pd.Timestamp("2011-07-23 00:00"))[0][0])
        report("C4b 2011-07-23 00:00 not all missing", (r[k] >= 0).mean() > 0.3, f"valid fraction {(r[k] >= 0).mean():.3f}")
    if a.years[0] <= 2022 <= a.years[1] and a.only_days is None:
        r, _ = read(dst / "AORC.20220529.preciptemp.nc"); t = file_times(dst / "AORC.20220529.preciptemp.nc")
        k = int(np.where(t == pd.Timestamp("2022-05-29 20:00"))[0][0]); v = r[k][r[k] >= 0]
        report("C4c 2022-05-29 20:00 not uniform", float(np.std(v)) > 0.01, f"std {np.std(v):.3f}")
    if a.only_days is None:
        rows = []
        for p in sorted(dst.glob("AORC.*.preciptemp.nc")):
            if not p.is_symlink():
                h = data_md5.get(p.name)
                if h is None or p.name in ("AORC.20110722.preciptemp.nc", "AORC.20211231.preciptemp.nc"):
                    h = hashlib.md5(read(p)[0].tobytes()).hexdigest()
                rows.append(f"{p.name}\t{hashlib.md5(p.read_bytes()).hexdigest()}\t{h}")
        (dst / "_aorcfix2_manifest.tsv").write_text("\n".join(rows) + "\n")
        print(f"C5 manifest: {len(rows)} patched files -> {dst / '_aorcfix2_manifest.tsv'}")
    print("ALL PASS" if all(RES) else "SOME CHECKS FAILED")
    sys.exit(0 if all(RES) else 1)


if __name__ == "__main__":
    main()
