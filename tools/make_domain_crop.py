#!/usr/bin/env python
"""
GP 2026-10-22: RainyDay_fast Fix 1 -- cropped copy of the daily input files for one transposition domain.

    python tools/make_domain_crop.py <src_dir> <domain.shp> <dst_dir> [--years 2002 2023] [--margin 2]
        [--deflate 0] [--verify] [--watershed SHP ...] [--only-days YYYYMMDD ...] [--workers 4] [--dry-run]
        [--bench N]

Writes <dst_dir>/AORC.YYYYMMDD.preciptemp.nc (same names as the source) so that UNMODIFIED RainyDay, with
RAINPATH = <dst_dir>/AORC.*.preciptemp.nc, gives bit-identical results to a run on the CONUS files:
  * the crop window is the cell range RainyDay selects for the domain (find_indices on the float32 shapefile bbox,
    D:452) plus a margin of at least --margin cells per side. rainprop_setup takes the grid resolution as the MEAN
    of the coordinate differences over the whole file (F:2805-2806) and that value goes into rastermask/from_origin,
    so the margins are searched (smallest first) until both means equal the full-file values bitwise (asserted);
  * every file keeps its 24 time steps, its own time variable (raw doubles + attributes), the exact subset of its
    own latitude/longitude values (+ attributes), RAINRATE as raw float32 (-9999 kept) with the same attributes and
    NO _FillValue added (least_significant_digit is copied as an attribute only: passed to createVariable it would
    re-quantise the data); other data variables (TMP_2maboveground) are dropped -- RainyDay reads only
    VARIABLES["precip"];
  * written with netCDF4 (not xarray), to <name>.tmp then renamed (an interrupted run leaves no partial file that
    matches the RAINPATH glob); existing files are skipped unless --overwrite.
  * --deflate L: RAINRATE as one deflate-L + shuffle chunk per file (lossless); default 0 = contiguous.
  * --verify: for EVERY file, RainyDay's readnetcdf (cropped, crop indices) == readnetcdf (source, domain indices):
    raw values, mask, dtype, times; and once: rainprop_setup resolution/timeres/calendar/units, findsubbox extent,
    dimensions and coordinates, find_indices (shifted by the crop origin), rastermask of the domain ('simple') and
    of each --watershed ('fraction') bitwise equal. Differences expected and harmless are printed (dimensions,
    bndbox, droplist, nodata of the first file).
  * --bench N: time readnetcdf on N files, source vs crop.
A manifest (_crop_manifest.json, not matched by the RAINPATH glob) records the source, domain and windows.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Source"))
DATE_RE = r'\d{4}(?:\d{2})?(?:\d{2}|\-\d{2}\-\d{2}|\/\d{2}/\d{2})'          # createfilelist (F:2752)


def file_date(path):
    m = re.search(DATE_RE, os.path.basename(path))
    return datetime.strptime(m.group().replace("-", "").replace("/", ""), '%Y%m%d')


def mean_res(a):
    return np.abs((a[1:] - a[:-1])).mean()                                  # rainprop_setup F:2805-2806


def search_window(coord, i0, i1, full_res, mmin, mmax=40):
    """Smallest margins (a below i0, b above i1), both >= mmin, with mean_res of the window == full_res bitwise."""
    n = len(coord)
    for tot in range(2 * mmin, 2 * mmax + 1):
        cands = sorted(((a, tot - a) for a in range(mmin, tot - mmin + 1)), key=lambda ab: abs(ab[0] - ab[1]))
        for a, b in cands:
            lo, hi = i0 - a, i1 + b
            if lo < 0 or hi > n - 1:
                continue
            if mean_res(coord[lo:hi + 1]) == full_res:
                return lo, hi, a, b
    # the domain touches the file edge: allow smaller margins there rather than fail
    for a in range(0, i0 + 1):
        for b in range(0, n - i1):
            if mean_res(coord[i0 - a:i1 + b + 1]) == full_res:
                return i0 - a, i1 + b, a, b
    return None


def bitwise_equal(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.dtype.kind == 'f':
        return np.array_equal(x.view(np.dtype('u%d' % x.dtype.itemsize)), y.view(np.dtype('u%d' % y.dtype.itemsize)))
    return np.array_equal(x, y)


# ---------------------------------------------------------------------------------------------------------------------
# worker: crop one file (and verify it)
# ---------------------------------------------------------------------------------------------------------------------
_W = {}


def _init_worker(cfg):
    _W.update(cfg)
    if cfg["verify"]:
        import RainyDay_utilities_Py3.RainyDay_functions as RD
        _W["RD"] = RD


def _crop_one(src):
    import netCDF4 as nc
    c = _W
    dst = os.path.join(c["dst_dir"], os.path.basename(src))
    pre, lat, lon = c["vars"]["precip"], c["vars"]["latname"], c["vars"]["longname"]
    (la0, la1), (lo0, lo1) = c["latwin"], c["lonwin"]
    out = dict(file=os.path.basename(src), wrote=False, verified=None, msg="")
    t0 = time.perf_counter()
    if c["overwrite"] or not os.path.exists(dst):
        with nc.Dataset(src) as s:
            s.set_auto_maskandscale(False)
            flat, flon = s[lat][:], s[lon][:]
            if not (bitwise_equal(flat, c["reflat"]) and bitwise_equal(flon, c["reflon"])):
                return dict(out, msg="ERROR: latitude/longitude differ from the reference file")
            tmp = dst + ".tmp"
            with nc.Dataset(tmp, "w", format=s.data_model) as d:
                d.setncatts({k: s.getncattr(k) for k in s.ncattrs()})
                d.setncattr("rainyday_fast_crop", "GP 2026-10-22 tools/make_domain_crop.py: %s[%d:%d], %s[%d:%d] of %s"
                            % (lat, la0, la1 + 1, lon, lo0, lo1 + 1, os.path.realpath(src)))
                for name, dim in s.dimensions.items():
                    size = (la1 - la0 + 1) if name == lat else (lo1 - lo0 + 1) if name == lon else len(dim)
                    d.createDimension(name, None if dim.isunlimited() else size)
                for name, v in s.variables.items():
                    if name not in ("time", lat, lon, pre):
                        continue                                             # e.g. TMP_2maboveground (never read)
                    kw = {}
                    if name == pre:
                        if c["deflate"] > 0:                                 # one chunk per file
                            kw = dict(zlib=True, complevel=c["deflate"], shuffle=True,
                                      chunksizes=tuple(len(d.dimensions[dn]) for dn in v.dimensions))
                        else:
                            kw = dict(contiguous=True)
                    attrs = {k: v.getncattr(k) for k in v.ncattrs()}
                    fill = attrs.pop("_FillValue", None)                     # AORC has none: none is added
                    nv = d.createVariable(name, v.datatype, v.dimensions, fill_value=fill, **kw)
                    nv.set_auto_maskandscale(False)
                    nv.setncatts(attrs)                                      # least_significant_digit: attribute only
                    sl = tuple(slice(la0, la1 + 1) if dn == lat else slice(lo0, lo1 + 1) if dn == lon else slice(None)
                               for dn in v.dimensions)
                    nv[:] = v[sl]
            os.replace(tmp, dst)
        out["wrote"] = True
    out["t_write"] = time.perf_counter() - t0
    if c["verify"]:
        RD = _W["RD"]
        t1 = time.perf_counter()
        r1, t_1 = RD.readnetcdf(src, c["vars"], c["idx_full"], dropvars=c["droplist_full"],
                                calendar=c["calendar"], time_units=c["time_units"])
        t2 = time.perf_counter()
        r2, t_2 = RD.readnetcdf(dst, c["vars"], c["idx_crop"], dropvars=c["droplist_crop"],
                                calendar=c["calendar"], time_units=c["time_units"])
        t3 = time.perf_counter()
        ok = (type(r1) is type(r2) and r1.shape == r2.shape and r1.dtype == r2.dtype
              and bitwise_equal(np.ma.getdata(r1), np.ma.getdata(r2))
              and np.array_equal(np.ma.getmaskarray(r1), np.ma.getmaskarray(r2))
              and t_1.dtype == t_2.dtype and np.array_equal(t_1, t_2))
        out.update(verified=bool(ok), t_read_src=t2 - t1, t_read_crop=t3 - t2,
                   nmasked=int(np.ma.count_masked(r1)) if np.ma.isMaskedArray(r1) else 0)
        if not ok:
            out["msg"] = "VERIFY FAIL: readnetcdf differs"
    return out


# ---------------------------------------------------------------------------------------------------------------------
def global_checks(RD, src0, dst0, variables, inarea, idx_full, latwin, lonwin, domainshp, watersheds):
    """rainprop_setup / findsubbox / find_indices / rastermask on the first source file vs the first cropped file."""
    res, ok = [], True

    def rep(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(("  PASS  " if good else "  FAIL  ") + msg, flush=True)

    a = RD.rainprop_setup(src0, None, dict(variables))
    b = RD.rainprop_setup(dst0, None, dict(variables))
    rep(bitwise_equal(np.float64(a[0][0]), np.float64(b[0][0])) and bitwise_equal(np.float64(a[0][1]), np.float64(b[0][1])),
        "rainprop_setup spatialres bitwise %r %r" % (b[0][0], b[0][1]))
    rep(a[3] == b[3] and type(a[3]) is type(b[3]), "rainprop_setup timeres %r" % (b[3],))
    rep(a[6] == b[6] and a[7] == b[7], "rainprop_setup calendar/time_units %r %r" % (b[6], b[7]))
    print("  info  rainprop_setup dimensions %s -> %s, bndbox %s -> %s, droplist %s -> %s, nodata %r -> %r "
          "(not used by the run)" % (a[1], b[1], [float(x) for x in a[2]], [float(x) for x in b[2]], a[5], b[5], a[4], b[4]))

    fa = RD.findsubbox(inarea, variables, src0)
    fb = RD.findsubbox(inarea, variables, dst0)
    rep(bitwise_equal(fa[0], fb[0]), "findsubbox subextent %s" % (fb[0],))
    rep(np.array_equal(fa[1], fb[1]), "findsubbox subdimensions %s" % (fb[1],))
    rep(bitwise_equal(fa[2].values, fb[2].values) and bitwise_equal(fa[3].values, fb[3].values)
        and fa[2].attrs == fb[2].attrs and fa[3].attrs == fb[3].attrs, "findsubbox latrange/lonrange values + attrs")
    off = np.array([latwin[0], latwin[0], lonwin[0], lonwin[0]])
    rep(np.array_equal(np.array(fa[4]) - off, np.array(fb[4])), "findsubbox indices shifted by the crop origin")
    ib = RD.find_indices(dst0, inarea, variables)
    rep(np.array_equal(np.array(idx_full) - off, np.array(ib)), "find_indices %s -> %s" % (list(idx_full), list(ib)))

    for (shp, kind) in [(domainshp, 'simple')] + [(w, 'fraction') for w in watersheds]:
        ma, mb = [RD.rastermask(shp, SimpleNamespace(subextent=f[0], subdimensions=f[1], spatialres=s[0]), kind)
                  for f, s in ((fa, a), (fb, b))]
        rep(bitwise_equal(ma, mb), "rastermask %s %s (%d cells > 0)" % (kind, os.path.basename(shp), int(np.sum(mb > 0))))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_dir"); ap.add_argument("domain_shp"); ap.add_argument("dst_dir")
    ap.add_argument("--pattern", default="AORC.*.preciptemp.nc")
    ap.add_argument("--years", nargs=2, type=int, default=[2002, 2023])
    ap.add_argument("--only-days", nargs="*", default=None)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--deflate", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--watershed", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--precip", default="RAINRATE"); ap.add_argument("--lat", default="latitude")
    ap.add_argument("--lon", default="longitude")
    a = ap.parse_args()

    import netCDF4 as nc
    import shapefile
    variables = {"precip": a.precip, "latname": a.lat, "longname": a.lon}

    files = sorted(glob.glob(os.path.join(a.src_dir, a.pattern)))
    files = [f for f in files if a.years[0] <= file_date(f).year <= a.years[1]]
    if a.only_days:
        files = [f for f in files if file_date(f).strftime('%Y%m%d') in set(a.only_days)]
    if not files:
        sys.exit("no source files")
    ref = files[0]

    # domain window exactly as RainyDay: float32 shapefile bbox (D:452-454) -> find_indices (F:2202)
    bb = shapefile.Reader(a.domain_shp).bbox
    inarea = np.array([bb[0], bb[2], bb[1], bb[3]], dtype='float32')
    with nc.Dataset(ref) as s:
        s.set_auto_maskandscale(False)
        reflat, reflon = s[a.lat][:], s[a.lon][:]
        nt = len(s["time"])
    lonv = reflon - 360 if max(reflon) > 180 else reflon
    lat_inds = np.where((reflat >= inarea[2]) & (reflat <= inarea[3]))[0]
    lon_inds = np.where((lonv >= inarea[0]) & (lonv <= inarea[1]))[0]
    idx_full = [lat_inds.min(), lat_inds.max(), lon_inds.min(), lon_inds.max()]
    yres, xres = mean_res(reflat), mean_res(reflon)
    wl = search_window(reflat, idx_full[0], idx_full[1], yres, a.margin)
    wo = search_window(reflon, idx_full[2], idx_full[3], xres, a.margin)
    if wl is None or wo is None:
        sys.exit("no crop window reproduces the full-file mean resolution bitwise (lat %s, lon %s)" % (wl, wo))
    latwin, lonwin = (wl[0], wl[1]), (wo[0], wo[1])
    assert mean_res(reflat[wl[0]:wl[1] + 1]) == yres and mean_res(reflon[wo[0]:wo[1] + 1]) == xres
    H, W = wl[1] - wl[0] + 1, wo[1] - wo[0] + 1
    bytes_per = nt * H * W * 4 + 8 * (H + W + nt) + 20000
    total = bytes_per * len(files)
    probe = Path(a.dst_dir).resolve()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    print("source       %s (%d files %d-%d)" % (a.src_dir, len(files), a.years[0], a.years[1]))
    print("domain       %s  inarea(float32) %s" % (a.domain_shp, inarea))
    print("domain idx   lat %d..%d, lon %d..%d (%d x %d cells)" % (idx_full[0], idx_full[1], idx_full[2], idx_full[3],
                                                                 idx_full[1] - idx_full[0] + 1, idx_full[3] - idx_full[2] + 1))
    print("crop window  lat %d..%d (margins %d/%d), lon %d..%d (margins %d/%d) -> %d x %d x %d steps"
          % (wl[0], wl[1], wl[2], wl[3], wo[0], wo[1], wo[2], wo[3], H, W, nt))
    print("resolution   yres %r xres %r (bitwise equal to the full file)" % (yres, xres))
    print("size         %.2f MB per file, %.2f GB total uncompressed%s; free disk %.1f GB"
          % (bytes_per / 1e6, total / 1e9, " (deflate %d: smaller)" % a.deflate if a.deflate else "", free / 1e9))
    if a.dry_run:
        return
    if a.deflate == 0 and total > free - 20e9:
        sys.exit("not enough free disk (keeping 20 GB spare)")
    os.makedirs(a.dst_dir, exist_ok=True)

    RD = None
    cfg = dict(dst_dir=a.dst_dir, vars=variables, latwin=latwin, lonwin=lonwin, deflate=a.deflate,
               overwrite=a.overwrite, verify=a.verify, reflat=reflat, reflon=reflon)
    if a.verify or a.bench:
        import RainyDay_utilities_Py3.RainyDay_functions as RD
        # the first file must exist in the crop before the crop-side setup values can be computed
        _init_worker(dict(cfg, verify=False)); r0 = _crop_one(ref)
        if r0["msg"]:
            sys.exit(r0["msg"])
        dst0 = os.path.join(a.dst_dir, os.path.basename(ref))
        sa = RD.rainprop_setup(ref, None, dict(variables)); sb = RD.rainprop_setup(dst0, None, dict(variables))
        cfg.update(idx_full=idx_full, idx_crop=RD.find_indices(dst0, inarea, variables),
                   droplist_full=sa[5], droplist_crop=sb[5], calendar=sa[6], time_units=sa[7])

    manifest = dict(created=str(datetime.now()), tool="tools/make_domain_crop.py", source=os.path.abspath(a.src_dir),
                    pattern=a.pattern, years=a.years, nfiles=len(files), domain_shp=os.path.abspath(a.domain_shp),
                    inarea_float32=[float(x) for x in inarea], domain_idx=[int(x) for x in idx_full],
                    lat_window=[int(x) for x in latwin], lon_window=[int(x) for x in lonwin],
                    margins=dict(lat=[int(wl[2]), int(wl[3])], lon=[int(wo[2]), int(wo[3])]),
                    shape=[int(nt), int(H), int(W)], yres=repr(yres), xres=repr(xres), deflate=a.deflate, variables=variables)
    json.dump(manifest, open(os.path.join(a.dst_dir, "_crop_manifest.json"), "w"), indent=2)

    t0 = time.perf_counter()
    nbad, nver, nwrote, tsrc, tcrop = 0, 0, 0, 0., 0.
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for k, r in enumerate(pool.imap(_crop_one, files, chunksize=4)):
            nwrote += r["wrote"]
            if r["verified"] is not None:
                nver += r["verified"]; tsrc += r["t_read_src"]; tcrop += r["t_read_crop"]
            if r["msg"] or r["verified"] is False:
                nbad += 1
                print("  %s: %s" % (r["file"], r["msg"]), flush=True)
            if (k + 1) % 200 == 0 or k + 1 == len(files):
                el = time.perf_counter() - t0
                print("  %d/%d files (%d written, %d verified, %d bad) %.0f s, eta %.0f s"
                      % (k + 1, len(files), nwrote, nver, nbad, el, el / (k + 1) * (len(files) - k - 1)), flush=True)
    print("done: %d files, %d written, %d bad, %.1f s" % (len(files), nwrote, nbad, time.perf_counter() - t0))
    if a.verify:
        print("readnetcdf per file (worker time): source %.3f s, crop %.4f s (%d files verified bitwise, %d failed)"
              % (tsrc / max(1, nver), tcrop / max(1, nver), nver, len(files) - nver))
        dst0 = os.path.join(a.dst_dir, os.path.basename(ref))
        gok = global_checks(RD, ref, dst0, variables, inarea, idx_full, latwin, lonwin, a.domain_shp, a.watershed)
        allok = gok and nbad == 0 and nver == len(files)
        print("VERIFY ALL PASS" if allok else "VERIFY FAILED")
        manifest.update(verified=bool(allok), verified_at=str(datetime.now()))
        json.dump(manifest, open(os.path.join(a.dst_dir, "_crop_manifest.json"), "w"), indent=2)
    if a.bench:
        bench(RD, files, a.dst_dir, variables, cfg, a.bench)
    if nbad:
        sys.exit(1)


def bench(RD, files, dst_dir, variables, cfg, n):
    """Sequential readnetcdf timing (cold-ish: files spread over the record), source vs crop."""
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(files), size=min(n, len(files)), replace=False))
    for label, paths, idx, dv in (("source", [files[i] for i in pick], cfg["idx_full"], cfg["droplist_full"]),
                                  ("crop", [os.path.join(dst_dir, os.path.basename(files[i])) for i in pick],
                                   cfg["idx_crop"], cfg["droplist_crop"])):
        t = time.perf_counter()
        for p in paths:
            RD.readnetcdf(p, variables, idx, dropvars=dv, calendar=cfg["calendar"], time_units=cfg["time_units"])
        dt = time.perf_counter() - t
        print("bench readnetcdf %-6s %d files: %.4f s/file" % (label, len(paths), dt / len(paths)))


if __name__ == "__main__":
    main()
