"""
Check RainyDay-format AORC daily files over a date range. Instructions: AORC_DOWNLOAD_INSTRUCTIONS.md in this folder.

    python qa_daily.py <dir> <YYYYMMDD start> <YYYYMMDD end> <out.csv> [--mask MASK] [--workers 8]

Per day: file present; 24 hourly stamps d+1h .. d+24h; lat/lon equal to the first file; RAINRATE NaN exactly on the
static coverage mask (hours with NaN everywhere reported separately); no -9999, negative or inf values; values
multiples of 1/16; hours whose valid cells are all equal (uniform field, e.g. the 2022-05-29 20:00 defect); TMP range.
Over the whole range: hours whose RAINRATE field is byte-identical to another hour (e.g. the 23:00 = 22:00 copy of
2002-2020), excluding all-dry hours.
Writes <out.csv> (one row per day) and <out>_duplicated_hours.txt, and prints a summary. It only reads the files.
"""
import argparse
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

MASK = None
DEFAULT_MASK = Path(__file__).resolve().parent / "coverage_mask_nan.npz"


def init(mask_path):
    global MASK
    m = np.load(mask_path)
    MASK = m["mask"] if hasattr(m, "files") else m


def one(args):
    path, day = args
    r = dict(day=day.strftime("%Y%m%d"), present=path.exists())
    if not r["present"]:
        return r, [], None
    with nc.Dataset(path) as f:
        f.set_auto_mask(False)
        tv = f["time"]
        t = pd.to_datetime([x.isoformat() for x in nc.num2date(tv[:], tv.units, only_use_cftime_datetimes=False)])
        R = f["RAINRATE"][:]; T = f["TMP_2maboveground"][:]
        lat = f["latitude"][:]; lon = f["longitude"][:]
    exp = pd.date_range(day + pd.Timedelta(hours=1), periods=24, freq="h")
    r["n_steps"] = len(t)
    r["times_ok"] = bool(len(t) == 24 and (t == exp).all())
    nan = np.isnan(R)
    allnan = [h for h in range(R.shape[0]) if nan[h].all()]
    r["allnan_hours"] = ";".join(str(exp[h]) for h in allnan)
    r["nan_off_mask_hours"] = sum(1 for h in range(R.shape[0]) if h not in allnan and not np.array_equal(nan[h], MASK))
    v = R[~nan]
    r["n_neg"] = int((v < 0).sum()); r["n_m9999"] = int((v == -9999).sum()); r["n_inf"] = int(np.isinf(v).sum())
    r["not_1_16"] = int((np.abs(v * 16 - np.round(v * 16)) > 1e-4).sum())
    r["rain_max"] = float(v.max()) if v.size else np.nan
    r["rain_mean"] = float(v.mean()) if v.size else np.nan
    uni = []
    for h in range(R.shape[0]):
        x = R[h][~nan[h]]
        if x.size and x.max() > 0 and x.min() == x.max():
            uni.append(f"{exp[h]}={x[0]}")
    r["uniform_hours"] = ";".join(uni)
    tv_ = T[~np.isnan(T)]
    r["tmp_min"] = float(tv_.min()) if tv_.size else np.nan
    r["tmp_max"] = float(tv_.max()) if tv_.size else np.nan
    r["tmp_nan_eq_mask"] = all(np.array_equal(np.isnan(T[h]), MASK) for h in range(T.shape[0]) if h not in allnan)
    hashes = []
    for h in range(R.shape[0]):
        x = np.nan_to_num(R[h], nan=-1.0)
        dry = not (x > 0).any()
        hashes.append((str(exp[h]), hashlib.md5(x.tobytes()).hexdigest(), dry))
    return r, hashes, (lat, lon)


def main():
    ap = argparse.ArgumentParser(description="Check RainyDay-format AORC daily files")
    ap.add_argument("dir"); ap.add_argument("start"); ap.add_argument("end"); ap.add_argument("out")
    ap.add_argument("--mask", default=str(DEFAULT_MASK), help="static coverage mask (default: the one in this folder)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    d = Path(a.dir)
    days = pd.date_range(a.start, a.end, freq="D")
    jobs = [(d / f"AORC.{x:%Y%m%d}.preciptemp.nc", x) for x in days]
    with Pool(a.workers, initializer=init, initargs=(a.mask,)) as pool:
        res = pool.map(one, jobs, chunksize=4)
    rows = [r for r, _, _ in res]
    df = pd.DataFrame(rows)
    grids = [g for _, _, g in res if g is not None]
    if not grids:
        sys.exit(f"No AORC.YYYYMMDD.preciptemp.nc file of {a.start}-{a.end} in {a.dir}")
    grid_ok = all(np.array_equal(g[0], grids[0][0]) and np.array_equal(g[1], grids[0][1]) for g in grids)
    hashes = [h for _, hs, _ in res for h in hs]
    seen, dups, prev = {}, [], None
    for stamp, md, dry in hashes:
        if not dry and md in seen:
            dups.append(f"{stamp}=={seen[md]}")
        seen.setdefault(md, stamp)
    consec = [f"{hashes[i][0]}=={hashes[i - 1][0]}" for i in range(1, len(hashes))
              if hashes[i][1] == hashes[i - 1][1] and not hashes[i][2]]
    df.to_csv(a.out, index=False)
    with open(a.out.replace(".csv", "_duplicated_hours.txt"), "w") as f:
        f.write("# non-dry RAINRATE hours byte-identical to an earlier hour (any)\n" + "\n".join(dups) +
                "\n# consecutive\n" + "\n".join(consec) + "\n")
    n = len(df)
    print(f"QA {a.dir} {a.start}-{a.end}: {n} days expected, {int(df.present.sum())} present")
    p = df[df.present]
    print(f"  times ok: {int(p.times_ok.sum())}/{len(p)}; grid identical across files: {grid_ok}")
    print(f"  all-NaN hours: {[x for x in p.allnan_hours if x]}")
    print(f"  hours with NaN off the static mask: {int(p.nan_off_mask_hours.sum())}; TMP NaN == mask: {bool(p.tmp_nan_eq_mask.all())}")
    print(f"  negative: {int(p.n_neg.sum())}, -9999: {int(p.n_m9999.sum())}, inf: {int(p.n_inf.sum())}, not multiple of 1/16: {int(p.not_1_16.sum())}")
    print(f"  rain max {p.rain_max.max():.3f} mm/h on {p.loc[p.rain_max.idxmax(), 'day']}; daily-file mean of valid cells {p.rain_mean.min():.4f}-{p.rain_mean.max():.4f}")
    print(f"  uniform-field hours: {[x for x in p.uniform_hours if x]}")
    print(f"  TMP range {p.tmp_min.min():.2f}-{p.tmp_max.max():.2f} K")
    print(f"  duplicated non-dry hours (any): {len(dups)} {dups[:10]}; consecutive: {len(consec)} {consec[:10]}")


if __name__ == "__main__":
    main()
