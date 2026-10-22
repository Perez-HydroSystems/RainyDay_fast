#!/usr/bin/env python
"""
GP 2026-10-22: RainyDay_fast Fix 1 -- cropped copy of the daily input files for one transposition domain (or a region
that holds several domains), so that RainyDay gives bit-identical results reading the crop.

From a RainyDay JSON (RAINPATH, VARIABLES, DOMAINTYPE/DOMAINSHP or AREA_EXTENT, INCLUDEYEARS, EXCLUDEMONTHS and, for
--verify, WATERSHEDSHP are taken from it):
    python tools/make_domain_crop.py --json RUN.json [RUN2.json ...] [--dst DIR] [--json-out OUT.json]
                                     [--deflate 1] [--verify] [--margin 2] [--workers 4] [--dry-run] [--bench N]
  -> the crop in DIR (default: <parent of the RAINPATH folder>/crops/<DOMAINSHP name>) and a ready-to-run copy of
     each JSON with RAINPATH pointing to the crop: <json folder>/<json stem>_crop.json (or --json-out, one JSON).
     Several JSONs (same RAINPATH and VARIABLES): one crop covering all their domains and years.
     DIR already holds a crop: the domains must be inside it (else it stops); only missing files are added (same
     window and compression); --verify checks this run's domain on it. That is how a region crop serves every
     domain inside it.
Positional form (no JSON):
    python tools/make_domain_crop.py <src_dir> <domain.shp> <dst_dir> [--pattern AORC.*.preciptemp.nc]
        [--years 2002 2023] [--only-days YYYYMMDD ...] [--watershed SHP ...] [--precip RAINRATE --lat latitude
        --lon longitude] [same options]

What the input must satisfy (RainyDay's own requirements):
  * one file per day, the date in the file name (YYYYMMDD, YYYY-MM-DD or YYYY/MM/DD), all time steps of the day in
    the file (steps x time resolution = 1440 min), a variable named 'time' with units (and calendar);
  * a regular grid with 1-D latitude and longitude variables (names = VARIABLES latname/longname), the same in every
    file; longitudes in -180..180 or 0..360 (RainyDay subtracts 360 when max > 180);
  * the precipitation variable (VARIABLES precip) with dimensions (time, latitude, longitude).

How the crop stays bit-identical:
  * window = RainyDay's selection of the domain cells (find_indices on the run's inarea: the float32 bounding box of
    DOMAINSHP, or AREA_EXTENT) plus margins of at least --margin cells, searched until the MEAN coordinate spacing of
    the cropped file equals that of the full file bitwise (rainprop_setup takes the resolution as that mean and it
    feeds rastermask); checked with RainyDay's own rainprop_setup/findsubbox on the first file before the rest is cut;
  * each file keeps all its time steps, its own time variable and the exact subset of its own latitude/longitude
    (raw values + attributes), the precipitation as raw values with the same dtype and attributes, no _FillValue added
    (least_significant_digit is copied as an attribute only); other data variables are dropped (RainyDay reads only
    VARIABLES precip); same file names; written with netCDF4 to <name>.tmp then renamed;
  * --deflate 1 (default): one zlib level-1 + shuffle chunk per file (lossless); --deflate 0: contiguous, uncompressed
    (larger, a little faster to read);
  * _crop_manifest.json (not matched by the RAINPATH glob) records the source, window, margins, the source cells just
    outside the window (used by RainyDay's coverage check when a run starts) and the domains made/verified with it.
  * --verify: for EVERY file of each domain's list, RainyDay's readnetcdf on the crop equals readnetcdf on the source
    (raw values, mask, dtype, times); per domain: rainprop_setup resolution/timeres/calendar/units, findsubbox,
    find_indices (shifted by the crop origin), rastermask of the domain ('simple') and of the watersheds ('fraction').
  * When the source files change (e.g. a corrected day), crop again into a new folder (or delete the changed days
    from the crop and run the same command: missing files are added).
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
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Source"))
DATE_RE = r'\d{4}(?:\d{2})?(?:\d{2}|\-\d{2}\-\d{2}|\/\d{2}/\d{2})'          # createfilelist (F:2752)
MANIFEST = "_crop_manifest.json"


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
    for a in range(0, i0 + 1):                       # the domain touches the file edge: smaller margins there
        for b in range(0, n - i1):
            if mean_res(coord[i0 - a:i1 + b + 1]) == full_res:
                return i0 - a, i1 + b, a, b
    return None


def bitwise_equal(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.dtype.kind == 'f':
        u = np.dtype('u%d' % x.dtype.itemsize)
        return np.array_equal(np.ascontiguousarray(x).view(u), np.ascontiguousarray(y).view(u))
    return np.array_equal(x, y)


# ---------------------------------------------------------------------------------------------------------------------
# worker: crop one file (and verify it for every domain that uses it)
# ---------------------------------------------------------------------------------------------------------------------
_W = {}


def _init_worker(cfg):
    _W.clear()
    _W.update(cfg)
    if cfg["verify"]:
        import RainyDay_utilities_Py3.RainyDay_functions as RD
        _W["RD"] = RD


def _crop_one(task):
    import netCDF4 as nc
    src, pairs = task
    c = _W
    dst = os.path.join(c["dst_dir"], os.path.basename(src))
    pre, lat, lon = list(c["vars"].values())
    (la0, la1), (lo0, lo1) = c["latwin"], c["lonwin"]
    out = dict(file=os.path.basename(src), wrote=False, verified=None, msg="", t_read_src=0., t_read_crop=0.)
    t0 = time.perf_counter()
    if not os.path.exists(dst):
        with nc.Dataset(src) as s:
            s.set_auto_maskandscale(False)
            flat, flon = s[lat][:], s[lon][:]
            if not (bitwise_equal(flat, c["reflat"]) and bitwise_equal(flon, c["reflon"])):
                return dict(out, msg="ERROR: latitude/longitude differ from the reference file")
            if s[pre].dimensions[1:] != (lat, lon):
                return dict(out, msg="ERROR: %s has dimensions %s, RainyDay needs (time, %s, %s)"
                            % (pre, s[pre].dimensions, lat, lon))
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
    if c["verify"] and pairs:
        RD = _W["RD"]
        ok = True
        for idx_full, idx_crop in pairs:
            t1 = time.perf_counter()
            r1, t_1 = RD.readnetcdf(src, c["vars"], idx_full, dropvars=False, calendar=c["calendar"],
                                    time_units=c["time_units"])
            t2 = time.perf_counter()
            r2, t_2 = RD.readnetcdf(dst, c["vars"], idx_crop, dropvars=False, calendar=c["calendar"],
                                    time_units=c["time_units"])
            t3 = time.perf_counter()
            ok &= (type(r1) is type(r2) and r1.shape == r2.shape and r1.dtype == r2.dtype
                   and bitwise_equal(np.ma.getdata(r1), np.ma.getdata(r2))
                   and np.array_equal(np.ma.getmaskarray(r1), np.ma.getmaskarray(r2))
                   and t_1.dtype == t_2.dtype and np.array_equal(t_1, t_2))
            out["t_read_src"] += t2 - t1; out["t_read_crop"] += t3 - t2
        out["verified"] = bool(ok)
        if not ok:
            out["msg"] = "VERIFY FAIL: readnetcdf differs"
    return out


# ---------------------------------------------------------------------------------------------------------------------
# run settings
# ---------------------------------------------------------------------------------------------------------------------
def domains_from_json(paths):
    import RainyDay_utilities_Py3.RainyDay_functions as RD
    import RainyDay_utilities_Py3.fast_io as FIO
    out = []
    for p in paths:
        p = os.path.abspath(p)
        j = json.load(open(p), object_pairs_hook=OrderedDict)
        wd = j.get("MAINPATH", os.path.dirname(p))
        inpath = j["RAINPATH"]
        inpath = inpath if os.path.isabs(inpath) else os.path.join(wd, inpath)   # RainyDay globs after chdir(MAINPATH)
        includeyears, excludemonths = FIO.parse_years_months(j)
        files, _ = RD.createfilelist(inpath, includeyears, excludemonths)
        inarea, desc = FIO.run_inarea(j)
        irregular = str(j.get("DOMAINTYPE", "rectangular")).lower() == "irregular"
        ws = [j["WATERSHEDSHP"]] if str(j.get("POINTAREA", "")).lower() == "watershed" and j.get("WATERSHEDSHP") else []
        out.append(dict(json=p, cardinfo=j, inpath=os.path.abspath(inpath), variables=dict(j["VARIABLES"]),
                        inarea=inarea, description=desc, domainshp=j["DOMAINSHP"] if irregular else None,
                        watersheds=ws, files=[os.path.abspath(f) for f in files],
                        name=Path(j["DOMAINSHP"]).stem if irregular else "area_" + "_".join("%g" % x for x in inarea)))
    return out


def domains_from_args(a):
    import shapefile
    files = sorted(glob.glob(os.path.join(a.src_dir, a.pattern)))
    files = [f for f in files if a.years[0] <= file_date(f).year <= a.years[1]]
    if a.only_days:
        files = [f for f in files if file_date(f).strftime('%Y%m%d') in set(a.only_days)]
    bb = shapefile.Reader(a.domain_shp).bbox
    inarea = np.array([bb[0], bb[2], bb[1], bb[3]], dtype='float32')           # D:452-454
    return [dict(json=None, cardinfo=None, inpath=os.path.abspath(os.path.join(a.src_dir, a.pattern)),
                 variables={"precip": a.precip, "latname": a.lat, "longname": a.lon}, inarea=inarea,
                 description="DOMAINSHP " + os.path.abspath(a.domain_shp), domainshp=os.path.abspath(a.domain_shp),
                 watersheds=a.watershed, files=[os.path.abspath(f) for f in files], name=Path(a.domain_shp).stem)]


# ---------------------------------------------------------------------------------------------------------------------
def domain_checks(RD, src0, dst0, dom, idx_full, latwin, lonwin, full=True):
    """rainprop_setup / findsubbox (always) and find_indices / rastermask (full) on the first source vs cropped file."""
    ok = True
    variables, inarea = dom["variables"], dom["inarea"]

    def rep(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(("  PASS  " if good else "  FAIL  ") + msg, flush=True)

    a = RD.rainprop_setup(src0, None, dict(variables))
    b = RD.rainprop_setup(dst0, None, dict(variables))
    rep(bitwise_equal(np.float64(a[0][0]), np.float64(b[0][0])) and bitwise_equal(np.float64(a[0][1]), np.float64(b[0][1])),
        "rainprop_setup spatialres bitwise %r %r" % (b[0][0], b[0][1]))
    rep(a[3] == b[3] and type(a[3]) is type(b[3]) and a[6] == b[6] and a[7] == b[7],
        "rainprop_setup timeres/calendar/time_units %r %r %r" % (b[3], b[6], b[7]))
    if full:
        print("  info  rainprop_setup dimensions %s -> %s, droplist %s -> %s, nodata %r -> %r (not used by the run)"
              % (a[1], b[1], a[5], b[5], a[4], b[4]))
    fa = RD.findsubbox(inarea, variables, src0)
    fb = RD.findsubbox(inarea, variables, dst0)
    rep(bitwise_equal(fa[0], fb[0]) and np.array_equal(fa[1], fb[1]) and bitwise_equal(fa[2].values, fb[2].values)
        and bitwise_equal(fa[3].values, fb[3].values) and fa[2].attrs == fb[2].attrs and fa[3].attrs == fb[3].attrs,
        "findsubbox extent, dimensions %s, latitude/longitude values + attributes" % (fb[1],))
    if full:
        off = np.array([latwin[0], latwin[0], lonwin[0], lonwin[0]])
        rep(np.array_equal(np.array(fa[4]) - off, np.array(fb[4])), "findsubbox indices shifted by the crop origin")
        ib = RD.find_indices(dst0, inarea, variables)
        rep(np.array_equal(np.array(idx_full) - off, np.array(ib)), "find_indices %s -> %s" % (list(idx_full), list(ib)))
        shapes = ([(dom["domainshp"], 'simple')] if dom["domainshp"] else []) + [(w, 'fraction') for w in dom["watersheds"]]
        for (shp, kind) in shapes:
            ma, mb = [RD.rastermask(shp, SimpleNamespace(subextent=f[0], subdimensions=f[1], spatialres=s[0]), kind)
                      for f, s in ((fa, a), (fb, b))]
            rep(bitwise_equal(ma, mb), "rastermask %s %s (%d cells > 0)" % (kind, os.path.basename(shp), int(np.sum(mb > 0))))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_dir", nargs="?"); ap.add_argument("domain_shp", nargs="?"); ap.add_argument("dst_dir", nargs="?")
    ap.add_argument("--json", nargs="+")
    ap.add_argument("--dst"); ap.add_argument("--json-out")
    ap.add_argument("--pattern", default="AORC.*.preciptemp.nc")
    ap.add_argument("--years", nargs=2, type=int, default=[2002, 2023])
    ap.add_argument("--only-days", nargs="*", default=None)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--deflate", type=int, default=1)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--watershed", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--precip", default="RAINRATE"); ap.add_argument("--lat", default="latitude")
    ap.add_argument("--lon", default="longitude")
    a = ap.parse_args()

    import netCDF4 as nc
    import RainyDay_utilities_Py3.RainyDay_functions as RD
    import RainyDay_utilities_Py3.fast_io as FIO

    if a.json:
        doms = domains_from_json(a.json)
        if len({d["inpath"] for d in doms}) > 1 or len({tuple(d["variables"].values()) for d in doms}) > 1:
            sys.exit("all JSONs must have the same RAINPATH and VARIABLES")
        if a.json_out and len(doms) > 1:
            sys.exit("--json-out needs exactly one --json")
    else:
        if not (a.src_dir and a.domain_shp and a.dst_dir):
            sys.exit("give --json RUN.json, or <src_dir> <domain.shp> <dst_dir>")
        doms = domains_from_args(a)
    variables = doms[0]["variables"]
    pre, latn, lonn = list(variables.values())
    inpath = doms[0]["inpath"]
    src_dir, pattern = os.path.dirname(inpath), os.path.basename(inpath)
    if os.path.exists(os.path.join(src_dir, MANIFEST)):
        sys.exit("RAINPATH " + src_dir + " is already a crop: use the JSON whose RAINPATH points to the original files")
    files = sorted(set(f for d in doms for f in d["files"]))
    if not files:
        sys.exit("no input files (RAINPATH %s, INCLUDEYEARS/EXCLUDEMONTHS)" % inpath)
    if len({os.path.basename(f) for f in files}) != len(files):
        sys.exit("several input files share a file name: the crop keeps file names, so it can't hold them")
    dst = a.dst or a.dst_dir
    if dst is None:
        if len(doms) > 1:
            sys.exit("several JSONs: give --dst")
        dst = os.path.join(os.path.dirname(os.path.abspath(src_dir)), "crops", doms[0]["name"])
    dst = os.path.abspath(dst)
    ref = files[0]

    # reference grid (find_indices conventions) and the domain windows
    with nc.Dataset(ref) as s:
        s.set_auto_maskandscale(False)
        reflat, reflon = s[latn][:], s[lonn][:]
        nt = len(s["time"])
        if reflat.ndim != 1 or reflon.ndim != 1:
            sys.exit("RainyDay needs 1-D latitude/longitude")
        if s[pre].dimensions[1:] != (latn, lonn):
            sys.exit("%s has dimensions %s; RainyDay needs (time, %s, %s)" % (pre, s[pre].dimensions, latn, lonn))
    shift = max(reflon) > 180
    lonv = reflon - 360 if shift else reflon
    for d in doms:
        d["idx_full"] = [int(x) for x in RD.find_indices(ref, d["inarea"], variables)]
    union = [min(d["idx_full"][0] for d in doms), max(d["idx_full"][1] for d in doms),
             min(d["idx_full"][2] for d in doms), max(d["idx_full"][3] for d in doms)]
    yres, xres = mean_res(reflat), mean_res(lonv)

    man = json.load(open(os.path.join(dst, MANIFEST))) if os.path.exists(os.path.join(dst, MANIFEST)) else None
    if man is not None:                                      # existing crop: same source, domains inside it
        if os.path.realpath(man["source"]) != os.path.realpath(src_dir) or man["pattern"] != pattern:
            sys.exit("%s holds a crop of %s/%s, not of %s" % (dst, man["source"], man["pattern"], inpath))
        latwin, lonwin = tuple(man["lat_window"]), tuple(man["lon_window"])
        if a.deflate != man.get("deflate", 0):
            print("NOTE: the existing crop uses --deflate %s; new files use it too" % man.get("deflate", 0))
        deflate = man.get("deflate", 0)
        have = [os.path.join(dst, os.path.basename(f)) for f in files if os.path.exists(os.path.join(dst, os.path.basename(f)))]
        for d in doms:
            probs, _ = FIO.crop_coverage_problems(dst, d["inarea"], variables, have[0] if have else ref)
            if probs:
                sys.exit("the domain of %s (%s) is not inside the crop %s: %s. Use another --dst (a new crop for this "
                         "domain, or a region crop that holds all the domains)."
                         % (d["json"] or "", d["description"], dst, "; ".join(probs)))
        margins = man.get("margins")
        print("existing crop %s: window lat %s, lon %s; the domains are inside it" % (dst, latwin, lonwin))
    else:
        deflate = a.deflate
        wl = search_window(reflat, union[0], union[1], yres, a.margin)
        wo = search_window(lonv, union[2], union[3], xres, a.margin)
        if wl is None or wo is None:
            sys.exit("no crop window reproduces the full-file mean resolution bitwise (lat %s, lon %s)" % (wl, wo))
        latwin, lonwin = (int(wl[0]), int(wl[1])), (int(wo[0]), int(wo[1]))
        assert mean_res(reflat[wl[0]:wl[1] + 1]) == yres and mean_res(lonv[wo[0]:wo[1] + 1]) == xres
        if shift and not max(reflon[wo[0]:wo[1] + 1]) > 180:
            sys.exit("the source longitudes are 0..360 but the crop's would all be <= 180 (RainyDay would treat them "
                     "differently)")
        margins = dict(lat=[int(wl[2]), int(wl[3])], lon=[int(wo[2]), int(wo[3])])
    H, W = latwin[1] - latwin[0] + 1, lonwin[1] - lonwin[0] + 1
    todo = [f for f in files if not os.path.exists(os.path.join(dst, os.path.basename(f)))]
    bytes_per = nt * H * W * 4 + 8 * (H + W + nt) + 20000
    probe = Path(dst)
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    print("source       %s (%d files, %s .. %s)" % (inpath, len(files), os.path.basename(files[0]), os.path.basename(files[-1])))
    for d in doms:
        print("domain       %s  inarea %s  idx %s%s" % (d["description"], d["inarea"], d["idx_full"],
                                                         ("  (" + d["json"] + ")") if d["json"] else ""))
    print("crop         %s: lat %d..%d, lon %d..%d (margins %s) -> %d x %d x %d steps, deflate %d"
          % (dst, latwin[0], latwin[1], lonwin[0], lonwin[1], margins, H, W, nt, deflate))
    print("resolution   yres %r xres %r (bitwise equal to the full file)" % (yres, xres))
    print("size         %d files to write, %.2f MB per file uncompressed = %.2f GB%s; free disk %.1f GB"
          % (len(todo), bytes_per / 1e6, bytes_per * len(todo) / 1e9,
             " (deflate: AORC is typically 10-30x smaller)" if deflate else "", free / 1e9))
    if a.dry_run:
        return
    need = bytes_per * len(todo) * (0.3 if deflate else 1.0)
    if need > free - 20e9:
        sys.exit("not enough free disk (keeping 20 GB spare)")
    os.makedirs(dst, exist_ok=True)

    # first file + RainyDay's own setup functions on it before cutting the rest
    cfg = dict(dst_dir=dst, vars=variables, latwin=latwin, lonwin=lonwin, deflate=deflate, verify=False,
               reflat=reflat, reflon=reflon)
    _init_worker(cfg)
    r0 = _crop_one((ref, ()))
    if r0["msg"]:
        sys.exit(r0["msg"])
    dst0 = os.path.join(dst, os.path.basename(ref))
    print("setup check on %s:" % os.path.basename(ref))
    sa = RD.rainprop_setup(ref, None, dict(variables))
    for d in doms:
        if not domain_checks(RD, ref, dst0, d, d["idx_full"], latwin, lonwin, full=False):
            sys.exit("the crop doesn't reproduce RainyDay's setup values; nothing else was written")
        d["idx_crop"] = [int(x) for x in RD.find_indices(dst0, d["inarea"], variables)]
    cfg.update(verify=a.verify, calendar=sa[6], time_units=sa[7])

    nb = dict(lat_before=float(reflat[latwin[0] - 1]) if latwin[0] > 0 else None,
              lat_after=float(reflat[latwin[1] + 1]) if latwin[1] < len(reflat) - 1 else None,
              lon_before=float(lonv[lonwin[0] - 1]) if lonwin[0] > 0 else None,
              lon_after=float(lonv[lonwin[1] + 1]) if lonwin[1] < len(lonv) - 1 else None)
    man = man or dict(created=str(datetime.now()), tool="tools/make_domain_crop.py", source=os.path.abspath(src_dir),
                      pattern=pattern, variables=variables, source_shape=[int(len(reflat)), int(len(reflon))],
                      coord_dtype=str(reflat.dtype), lon_minus_360=bool(shift), lat_window=list(latwin),
                      lon_window=list(lonwin), margins=margins, neighbors=nb, shape=[int(nt), int(H), int(W)],
                      yres=repr(yres), xres=repr(xres), deflate=deflate, domains=[])
    man.setdefault("neighbors", nb)
    man.setdefault("domains", [])

    t0 = time.perf_counter()
    nbad, nver, nwrote, tsrc, tcrop, nver_exp = 0, 0, 0, 0., 0., 0
    domfiles = [set(d["files"]) for d in doms]
    tasks = []
    for f in files:
        pairs = tuple((d["idx_full"], d["idx_crop"]) for d, fs in zip(doms, domfiles) if f in fs) if a.verify else ()
        if f in todo or pairs:
            tasks.append((f, pairs))
            nver_exp += bool(pairs)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for k, r in enumerate(pool.imap(_crop_one, tasks, chunksize=4)):
            nwrote += r["wrote"]
            if r["verified"] is not None:
                nver += r["verified"]; tsrc += r["t_read_src"]; tcrop += r["t_read_crop"]
            if r["msg"]:
                nbad += 1
                print("  %s: %s" % (r["file"], r["msg"]), flush=True)
            if (k + 1) % 500 == 0 or k + 1 == len(tasks):
                el = time.perf_counter() - t0
                print("  %d/%d files (%d written, %d verified, %d bad) %.0f s, eta %.0f s"
                      % (k + 1, len(tasks), nwrote, nver, nbad, el, el / (k + 1) * (len(tasks) - k - 1)), flush=True)
    cropfiles = glob.glob(os.path.join(dst, pattern))
    print("done: %d files in the crop, %d written now (+ the setup file), %d bad, %.1f s, crop size %.2f GB"
          % (len(cropfiles), nwrote, nbad, time.perf_counter() - t0, sum(os.path.getsize(p) for p in cropfiles) / 1e9))
    allok = nbad == 0
    if a.verify:
        print("readnetcdf per file and domain (worker time): source %.4f s, crop %.4f s (%d files verified bitwise, "
              "%d expected)" % (tsrc / max(1, nver), tcrop / max(1, nver), nver, nver_exp))
        for d in doms:
            print("domain checks: %s" % d["description"])
            d["verified"] = domain_checks(RD, ref, dst0, d, d["idx_full"], latwin, lonwin, full=True)
            allok &= d["verified"]
        allok &= nver == nver_exp
        print("VERIFY ALL PASS" if allok else "VERIFY FAILED")
    yrs = sorted(file_date(p).year for p in cropfiles)
    man.update(updated=str(datetime.now()), nfiles=len(cropfiles), years=[yrs[0], yrs[-1]] if yrs else [])
    for d in doms:
        entry = dict(description=d["description"], inarea=[float(x) for x in d["inarea"]], source_idx=d["idx_full"],
                     json=d["json"], nfiles=len(d["files"]))
        if a.verify:
            entry.update(verified=bool(d.get("verified")) and nbad == 0, verified_at=str(datetime.now()))
        man["domains"] = [x for x in man["domains"] if x.get("description") != d["description"]] + [entry]
    json.dump(man, open(os.path.join(dst, MANIFEST), "w"), indent=2)
    if a.bench:
        bench(RD, files, dst, variables, doms[0], cfg, a.bench)
    if not allok:
        sys.exit(1)
    for d in doms:
        if d["json"]:
            outp = a.json_out or os.path.join(os.path.dirname(d["json"]), Path(d["json"]).stem + "_crop.json")
            j = OrderedDict(d["cardinfo"])
            j["RAINPATH"] = os.path.join(dst, pattern)
            json.dump(j, open(outp, "w"), indent=4)
            print("ready-to-run JSON: %s (RAINPATH %s)" % (outp, j["RAINPATH"]))


def bench(RD, files, dst_dir, variables, dom, cfg, n):
    """Sequential readnetcdf timing, source vs crop (files spread over the record)."""
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(files), size=min(n, len(files)), replace=False))
    for label, paths, idx in (("source", [files[i] for i in pick], dom["idx_full"]),
                              ("crop", [os.path.join(dst_dir, os.path.basename(files[i])) for i in pick], dom["idx_crop"])):
        t = time.perf_counter()
        for p in paths:
            RD.readnetcdf(p, variables, idx, dropvars=False, calendar=cfg["calendar"], time_units=cfg["time_units"])
        dt = time.perf_counter() - t
        print("bench readnetcdf %-6s %d files: %.4f s/file" % (label, len(paths), dt / len(paths)))


if __name__ == "__main__":
    main()
