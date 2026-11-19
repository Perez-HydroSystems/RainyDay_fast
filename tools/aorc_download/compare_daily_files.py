"""
Compare the RainyDay-format AORC daily files of <new_dir> with the files of the same name in <ref_dir>: arrays, NaN
pattern, encoding and variable attributes. Used to test make_aorc_from_v11.py against days that already exist
(AORC_DOWNLOAD_INSTRUCTIONS.md, step B2). It only reads the files.

    python compare_daily_files.py <new_dir> <ref_dir>
"""
import sys
from pathlib import Path
import numpy as np, netCDF4 as nc
new, ref = Path(sys.argv[1]), Path(sys.argv[2])
allok = True
for p in sorted(new.glob("AORC.*.preciptemp.nc")):
    a, b = nc.Dataset(p), nc.Dataset(ref / p.name)
    a.set_auto_mask(False); b.set_auto_mask(False)
    res = []
    ok = a.data_model == b.data_model
    for v in ["time", "latitude", "longitude", "RAINRATE", "TMP_2maboveground"]:
        x, y = a[v][:], b[v][:]
        same = x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y, equal_nan=True)
        enc = (a[v].filters() == b[v].filters() and a[v].chunking() == b[v].chunking() and
               {k: str(a[v].getncattr(k)) for k in a[v].ncattrs()} == {k: str(b[v].getncattr(k)) for k in b[v].ncattrs()})
        if v in ("RAINRATE", "TMP_2maboveground"):
            m = np.isfinite(x) & np.isfinite(y)
            d = float(np.abs(x[m] - y[m]).max()) if m.any() else 0.0
            res.append(f"{v}: identical={same} max|d|={d:.4g} nan_new={int(np.isnan(x).sum())} nan_ref={int(np.isnan(y).sum())} "
                       f"ndiff={int((x[m] != y[m]).sum())} enc_same={enc}")
        else:
            res.append(f"{v}: {same}/{enc}")
        ok &= same and enc
    allok &= ok
    print(p.name, "OK" if ok else "DIFF", " | ".join(res), flush=True)
print("ALL IDENTICAL" if allok else "SOME DIFFER")
