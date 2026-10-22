"""
GP 2026-10-22: RainyDay_fast Fix 2 -- light storm catalog (JSON key CATALOGSTORAGE: "full" (default) | "light").

A full catalog writes one file per storm, <StormCatalog>/<CATALOGNAME>_storm_<i+1>_<YYYYMMDD>.nc, with the storm's
rain(time, latitude, longitude) plus the arrays of ALL storms (basinrainfall, xlocation, ylocation, cattime), the
masks and the time resolution (RainyDay_functions.writecatalog). For a large domain that is GBs per basin, and most of
it is a copy of the input files.

A light catalog is ONE small netCDF, <StormCatalog>/<CATALOGNAME>_lightcatalog.nc, with every catalog variable except
rain (written like writecatalog, so it decodes to the same values and dtypes), plus:
  storm_file(storm_dim)  the storm file names a full catalog would have (D: storm_name), so every name-based filter of
                         RainyDay (extract_storm_number / extract_date: EXCLUDESTORMS, INCLUDEYEARS, EXCLUDEMONTHS,
                         NSTORMS trimming) works on "virtual" paths <StormCatalog>/<storm_file>;
  rain_crc32(storm_dim)  CRC32 of each storm's rain array exactly as the full storm file holds it;
  rain_file(rain_file)   the year/month-filtered input file list of the run (absolute paths, normally a domain crop);
  attributes             the readnetcdf arguments (variables, crop indices, droplist, calendar, units) and the
                         date rule of the catalog write loop (check_time of the first input time).
The rain of storm i is rebuilt by get_storm_rain(), a copy of the catalog write loop of RainyDay_Py3.py (search
"This part saves each storm as single file"): date of each step (previous day for hour-ending data), the FIRST
file of the list with that date, the previous ("sticky") file when a date has no file, the first matching time
step or zeros, raw values (readnetcdf), NaN -> -9999 (writecatalog), float32 C-contiguous. Every rebuild is checked
against the stored CRC32 (any change of the input files stops the run).

Re-rooting: the rain is read from THIS run's RAINPATH files, matched by file name (set_rainpath; the stored paths
are kept for information and as a fallback), at the cells of the catalog grid found by exact coordinate matching in
those files (a crop with another window, deflated or not, or the full files all work). The CRC32 is checked on every
rebuild, so wrong or changed data stop the run.

Crops (tools/make_domain_crop.py): check_crop_coverage() stops a run whose domain is not inside the crop given as
RAINPATH (the crop folder holds _crop_manifest.json).

Hooks in RainyDay_Py3.py ("GP 2026-10-22"): CATALOGSTORAGE parsing, list_storms() instead of the storm-file globs,
LightCatalogWriter in the catalog write loop, readcatalog() instead of RainyDay.readcatalog at the per-storm reads,
attrs_path() for the catalog description check; SCENARIOS stops with a light catalog; set_rainpath() after RAINPATH
is read; check_crop_coverage() after the input file list is built.
"""
import glob
import json
import os
import re
import sys
import zlib
from datetime import datetime

import numpy as np
import xarray as xr

from . import RainyDay_functions as RainyDay

LIGHT_SUFFIX = '_lightcatalog.nc'
VIRTUAL_RE = re.compile(r'^(.*)_storm_(\d+)_(\d{8})\.nc$')
DATE_RE = r'\d{4}(?:\d{2})?(?:\d{2}|\-\d{2}\-\d{2}|\/\d{2}/\d{2})'           # catalog write loop / createfilelist
_CACHE = {}
_RAINPATH = dict(glob=None, byname=None, version=0)
MANIFEST = '_crop_manifest.json'


def parse_catalogstorage(value):
    v = str(value if value is not None else 'full').strip().lower()
    if v not in ('full', 'light'):
        sys.exit("CATALOGSTORAGE must be 'full' or 'light'")
    return v


def light_path(prefix):
    """prefix = <StormCatalog folder>/<CATALOGNAME> (RainyDay's catalogpath)."""
    return prefix + LIGHT_SUFFIX


def rain_crc32(rain):
    return zlib.crc32(np.ascontiguousarray(rain, dtype=np.float32)) & 0xffffffff


# ---------------------------------------------------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------------------------------------------------
class LightCatalogWriter(object):
    """Collects the storms of the catalog write loop and writes the light catalog (instead of writecatalog)."""

    def __init__(self, prefix, scenarioname, parameterfile, catmax, catx, caty, cattime, latrange, lonrange, gridmask,
                 dmask, nstorms, timeresolution, flist, variables, idxes, droplist, calendar, time_units, first_time):
        self.prefix = prefix
        self.args = dict(scenarioname=scenarioname, parameterfile=parameterfile, catmax=catmax, catx=catx, caty=caty,
                         cattime=cattime, latrange=latrange, lonrange=lonrange, gridmask=gridmask, dmask=dmask,
                         nstorms=nstorms, timeresolution=timeresolution)
        self.flist = [os.path.abspath(f) for f in flist]
        self.meta = dict(variables=dict(variables), idxes=[int(x) for x in idxes], droplist=list(droplist or []),
                         calendar=calendar, time_units=time_units,
                         first_time_is_day_boundary=bool(RainyDay.check_time(first_time)))
        self.names = [None] * nstorms
        self.crcs = np.zeros(nstorms, dtype=np.uint32)
        self.shape = None

    def add(self, i, catrain, storm_name):
        catrain[np.isnan(catrain)] = -9999.          # in place, as writecatalog does (F: writecatalog)
        self.names[i] = os.path.basename(storm_name)
        self.crcs[i] = rain_crc32(catrain)
        self.shape = catrain.shape

    def close(self):
        a = self.args
        nstorms = a['nstorms']
        if any(n is None for n in self.names):
            sys.exit("light catalog: some storms were not collected")
        with open(a['parameterfile'], 'r') as f:
            params = json.loads(f.read())
        timeresolution = a['timeresolution'] if a['timeresolution'] != False else "None"
        # same variables, attributes and encodings as writecatalog, without rain and the per-storm time coordinate
        data_vars = dict(
            basinrainfall=(("storm_dim"), a['catmax'].reshape(nstorms),
                           {'units': 'mm', 'long_name': 'storm total basin averaged precipitation'}),
            xlocation=(("storm_dim"), a['catx'].reshape(nstorms), {'units': 'dimensionless', 'long_name': 'x index of storm'}),
            ylocation=(("storm_dim"), a['caty'].reshape(nstorms), {'units': 'dimensionless', 'long_name': 'y index of storm'}),
            cattime=(("storm_dim", "time"), a['cattime']),
            gridmask=(("latitude", "longitude"), a['gridmask'],
                      {'units': 'dimensionless', 'long_name': 'mask for Aw (control volume)'}),
            domainmask=(("latitude", "longitude"), a['dmask'],
                        {'units': 'dimensionless', 'long_name': 'mask for transposition domain'}),
            timeresolution=((), timeresolution),
            storm_file=(("storm_dim"), np.array(self.names, dtype=object),
                        {'long_name': 'storm file name of the equivalent full catalog'}),
            rain_crc32=(("storm_dim"), self.crcs,
                        {'long_name': 'CRC32 of the float32 C-order rain array of the equivalent storm file'}),
            rain_file=(("rain_file"), np.array(self.flist, dtype=object),
                       {'long_name': 'input files of the catalog (year/month filtered list of the run)'}))
        coords = dict(longitude=(("longitude"), a['lonrange'].data, {'units': 'degrees_east', 'long_name': 'longitude'}),
                      latitude=(("latitude"), a['latrange'].data, {'units': 'degrees_north', 'long_name': 'latitude'}))
        attrs = dict(history='Created ' + str(datetime.now()),
                     source='RainyDay Storm Catalog for scenario ' + a['scenarioname'] + '. See description for JSON file contents.',
                     missing='-9999.', description=str(params), calendar='gregorian',
                     catalogstorage='light (RainyDay_fast fast_io.py): rain rebuilt from rain_file',
                     rain_meta=json.dumps(dict(self.meta, shape=[int(x) for x in self.shape])))
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        out = light_path(self.prefix)
        tmp = out + '.tmp'
        ds.to_netcdf(tmp)
        ds.close()
        os.replace(tmp, out)
        return out


# ---------------------------------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------------------------------
def _load(lpath):
    st = os.stat(lpath)
    key = (st.st_mtime_ns, st.st_size)
    c = _CACHE.get(lpath)
    if c is None or c['key'] != key:
        with xr.open_dataset(lpath) as ds:
            ds = ds.load()
        meta = json.loads(ds.attrs['rain_meta'])
        names = [str(x) for x in ds['storm_file'].values]
        flist = [str(x) for x in ds['rain_file'].values]
        first = first_file_by_date(flist)
        c = dict(key=key, lpath=lpath, ds=ds, meta=meta, names=names, index={n: k for k, n in enumerate(names)},
                 flist=flist, first=first, cattime=np.array(ds['cattime'], dtype='datetime64[m]'),
                 crc=np.array(ds['rain_crc32'], dtype=np.uint32), resolved={}, grididx={}, notes=set(),
                 rainpath_version=_RAINPATH['version'])
        _CACHE[lpath] = c
    if c['rainpath_version'] != _RAINPATH['version']:
        c.update(resolved={}, grididx={}, notes=set(), rainpath_version=_RAINPATH['version'])
    return c


# ---------------------------------------------------------------------------------------------------------------------
# re-rooting: this run's RAINPATH files instead of the stored absolute paths
# ---------------------------------------------------------------------------------------------------------------------
def set_rainpath(inpath):
    """RAINPATH of this run (glob, relative to MAINPATH like createfilelist). Light catalogs rebuild their rain from
    these files, matched by file name; without RAINPATH the stored paths are used."""
    if isinstance(inpath, str) and inpath.strip() != '' and inpath.strip().lower() != 'none':
        g = os.path.abspath(os.path.expanduser(inpath.strip()))
        byname = {}
        for f in sorted(glob.glob(g)):
            byname.setdefault(os.path.basename(f), f)
        _RAINPATH.update(glob=g, byname=byname)
    else:
        _RAINPATH.update(glob=None, byname=None)
    _RAINPATH['version'] += 1


def _resolve(c, stored):
    """This run's file for a stored input file (same file name; the date is part of the name)."""
    bn = os.path.basename(stored)
    by = _RAINPATH['byname']
    if by is not None and bn in by:
        return by[bn]
    if os.path.exists(stored):
        if 'fallback' not in c['notes']:
            c['notes'].add('fallback')
            print("NOTE: light catalog: " + bn + " is not in this run's RAINPATH (" + str(_RAINPATH['glob'])
                  + "); using the file the catalog was built from, " + os.path.dirname(stored))
        return stored
    sys.exit("light catalog " + c['lpath'] + ": input file " + bn + " is neither in this run's RAINPATH ("
             + str(_RAINPATH['glob']) + ") nor at its stored path " + stored + ". Point RAINPATH to the files the "
             "catalog was built from (the same dataset: its full files or a crop that covers the catalog domain).")


class _ResolvedFirst(object):
    """first_file_by_date of the stored list, each file resolved to this run's RAINPATH when first needed."""

    def __init__(self, c):
        self.c = c

    def get(self, date, default=None):
        r = self.c['resolved']
        if date not in r:
            if date not in self.c['first']:
                return default
            r[date] = _resolve(self.c, self.c['first'][date])
        return r[date]


def _grid_idx(c, path):
    """readnetcdf indices of the catalog grid in the files of path's folder: exact match of the catalog latitude and
    longitude values (find_indices conventions: raw values, longitude - 360 when > 180)."""
    d = os.path.dirname(path)
    if d not in c['grididx']:
        import netCDF4
        meta = c['meta']
        _, latn, lonn = meta['variables'].values()               # RainyDay: rain_name,lat_name,lon_name=values()
        with netCDF4.Dataset(path) as f:
            lat = np.array(f.variables[latn][:])
            lon = np.array(f.variables[lonn][:])
            if max(f.variables[lonn]) > 180:
                lon = lon - 360
        clat, clon = np.array(c['ds']['latitude'].values), np.array(c['ds']['longitude'].values)

        def find(a, sub):
            for k in np.where(a == sub[0])[0]:
                if k + len(sub) <= len(a) and a[k:k + len(sub)].dtype == sub.dtype and np.array_equal(a[k:k + len(sub)], sub):
                    return int(k)
            return None

        i0, j0 = find(lat, clat), find(lon, clon)
        if i0 is None or j0 is None:
            if d == os.path.dirname(c['flist'][0]):
                idx = meta['idxes']                          # the catalog's own files: stored indices
            else:
                sys.exit("light catalog " + c['lpath'] + ": the files in " + d + " don't contain the catalog grid "
                         "(latitude " + str(clat[0]) + ".." + str(clat[-1]) + ", longitude " + str(clon[0]) + ".."
                         + str(clon[-1]) + "). Point RAINPATH to the same dataset (full files or a crop that covers "
                         "the catalog domain).")
        else:
            idx = [i0, i0 + len(clat) - 1, j0, j0 + len(clon) - 1]
        c['grididx'][d] = idx
    return c['grididx'][d]


def _split_virtual(path):
    d, b = os.path.split(path)
    m = VIRTUAL_RE.match(b)
    if not m:
        return None
    lpath = os.path.join(d, m.group(1) + LIGHT_SUFFIX)
    return lpath, b


def is_virtual(path):
    """True for a storm path of a light catalog (the storm file itself does not exist)."""
    if os.path.exists(path):
        return False
    s = _split_virtual(path)
    return s is not None and os.path.exists(s[0])


def list_storms(prefix, catalogname, storage):
    """Replaces glob.glob(prefix+'_storm_*.nc'): the storm files of a full catalog, or the virtual storm paths of the
    light catalog prefix+LIGHT_SUFFIX (same names, same order after RainyDay's sort by storm number). A full catalog
    is used when its storm files exist, unless CATALOGSTORAGE is 'light' and a light catalog exists."""
    import glob
    full = glob.glob(prefix + '_storm_' + '*' + '.nc')
    lpath = light_path(prefix)
    has_light = os.path.exists(lpath)
    if has_light and (storage == 'light' or len(full) == 0):
        c = _load(lpath)
        d = os.path.dirname(prefix)
        if storage != 'light':
            print("NOTE: no storm files found; using the light storm catalog " + lpath)
        else:
            print("Using the light storm catalog " + lpath + " (" + str(len(c['names'])) + " storms, rain rebuilt from "
                  + (os.path.dirname(_RAINPATH['glob']) if _RAINPATH['glob'] else os.path.dirname(c['flist'][0])) + ")")
        return [os.path.join(d, n) for n in c['names']]
    if storage == 'light' and len(full) > 0:
        print("NOTE: CATALOGSTORAGE is 'light' but " + os.path.dirname(prefix) + " holds a full catalog; using it")
    return full


def attrs_path(path):
    """File holding the catalog attributes (description) of a storm path."""
    return _split_virtual(path)[0] if is_virtual(path) else path


def first_file_by_date(flist):
    """{YYYYMMDD: first file of flist with that date in its name} (the write loop's search, which breaks at the first
    match)."""
    first = {}
    for f in flist:
        m = re.search(DATE_RE, os.path.basename(f))
        first.setdefault(m.group().replace("-", "").replace("/", ""), f)
    return first


def rebuild_storm_rain(storm_cattime, shape, first, day_boundary, timeres, variables, idxes, droplist, calendar,
                       time_units):
    """Copy of the catalog write loop of RainyDay_Py3.py ("This part saves each storm as single file") for one storm:
    storm_cattime = cattime[i,:] (datetime64[m]), shape = (nsteps, ny, nx), first = first_file_by_date(flist),
    day_boundary = RainyDay.check_time(first input time), idxes = readnetcdf indices or a function file -> indices.
    Returns the rain as writecatalog stores it."""
    start_time = storm_cattime[0]
    end_time = storm_cattime[-1]
    current_datetime = start_time
    dataset_date = None
    catrain = np.zeros(tuple(shape), dtype='float32')
    k = 0
    stm_file = None
    while current_datetime <= end_time:
        if day_boundary:
            current_date = np.datetime_as_string(current_datetime, unit='D')
        else:
            current_date = np.datetime_as_string(current_datetime - timeres, unit='D')
        if current_date != dataset_date:
            dataset_date = current_date
            stm_file = first.get(current_date.replace("-", ""), stm_file)      # no file for the date: sticky
            if stm_file is None:
                sys.exit("light catalog: no input file for " + current_date)
            stm_rain, stm_time = RainyDay.readnetcdf(stm_file, variables, idxes(stm_file) if callable(idxes) else idxes,
                                                     dropvars=droplist, calendar=calendar, time_units=time_units)
        match = np.where(stm_time == current_datetime)[0] if len(stm_time) > 0 else []
        if len(match) == 0:
            catrain[k, :] = 0.0
        else:
            catrain[k, :] = stm_rain[match[0], :]
        current_datetime += timeres
        k += 1
    catrain[np.isnan(catrain)] = -9999.              # writecatalog
    return catrain


def get_storm_rain(lpath, i, check=True):
    """Rain array of storm index i (0-based) exactly as the full catalog's storm file holds it (see module doc), from
    this run's RAINPATH files; the CRC32 is always checked unless check=False."""
    c = _load(lpath)
    meta = c['meta']
    timeres = np.int32(np.int16(c['ds'].timeresolution))
    catrain = rebuild_storm_rain(c['cattime'][i, :], meta['shape'], _ResolvedFirst(c), meta['first_time_is_day_boundary'],
                                 timeres, meta['variables'], lambda f: _grid_idx(c, f), meta['droplist'],
                                 meta['calendar'], meta['time_units'])
    if check and rain_crc32(catrain) != int(c['crc'][i]):
        used = sorted(set(os.path.dirname(v) for v in c['resolved'].values()))
        sys.exit("light catalog " + lpath + ": the rain of storm " + str(i + 1) + " (" + c['names'][i] + ") rebuilt "
                 "from " + ", ".join(used) + " doesn't match the catalog (CRC32): wrong or changed input data. Use the "
                 "data the catalog was built from, or rebuild the catalog.")
    return catrain


def readstorm(path):
    """Same tuple as RainyDay.readcatalog(storm file) for a virtual storm path of a light catalog."""
    lpath, base = _split_virtual(path)
    c = _load(lpath)
    if base not in c['index']:
        sys.exit("light catalog " + lpath + " has no storm " + base)
    i = c['index'][base]
    ds = c['ds']
    rain = get_storm_rain(lpath, i)
    stormtime = c['cattime'][i, :].copy()
    outlatitude = ds['latitude'].copy(deep=True)
    outlongitude = ds['longitude'].copy(deep=True)
    outrain = xr.DataArray(rain, dims=("time", "latitude", "longitude"), name='rain',
                           coords=dict(time=("time", stormtime.astype('datetime64[ns]')),
                                       longitude=outlongitude, latitude=outlatitude),
                           attrs={'units': 'mm hr^-1', 'long_name': 'precipitation rate'})
    outmask = ds['gridmask'].copy(deep=True)
    domainmask = np.array(ds['domainmask'])
    outlocx = np.array(ds['xlocation'])
    outlocy = np.array(ds['ylocation'])
    outmax = np.array(ds['basinrainfall'])
    cattime = c['cattime'].copy()
    try:
        timeresolution = np.int16(ds.timeresolution)
        resexists = True
    except Exception:
        resexists = False
    if resexists:
        return outrain, stormtime, outlatitude, outlongitude, outlocx, outlocy, outmax, outmask, domainmask, cattime, timeresolution
    return outrain, stormtime, outlatitude, outlongitude, outlocx, outlocy, outmax, outmask, domainmask, cattime


def readcatalog(path):
    """RainyDay.readcatalog for a storm file of a full catalog, readstorm for a virtual storm of a light catalog."""
    return readstorm(path) if is_virtual(path) else RainyDay.readcatalog(path)


# ---------------------------------------------------------------------------------------------------------------------
# crops (tools/make_domain_crop.py): run settings as RainyDay reads them, coverage check
# ---------------------------------------------------------------------------------------------------------------------
def parse_years_months(cardinfo):
    """(includeyears, excludemonths) exactly as RainyDay_Py3.py parses INCLUDEYEARS / EXCLUDEMONTHS."""
    try:
        excludemonths = cardinfo["EXCLUDEMONTHS"]
        if isinstance(excludemonths, str):
            if ',' in excludemonths:
                excludemonths = [np.int32(exstorm) for exstorm in excludemonths.split(",")]
            elif '-' in excludemonths:
                start_year, end_year = map(int, excludemonths.split('-'))
                excludemonths = [exstorm for exstorm in range(start_year, end_year + 1)]
            elif excludemonths.lower() == 'none':
                excludemonths = []
            else:
                excludemonths = []
        elif isinstance(excludemonths, list):
            excludemonths = excludemonths
        else:
            excludemonths = []
    except Exception:
        excludemonths = []
    try:
        includeyr = cardinfo["INCLUDEYEARS"]
        if isinstance(includeyr, str):
            if includeyr.lower == "all":                 # (sic, as in RainyDay: "all" ends in the except below)
                includeyears = False
            elif ',' in includeyr:
                includeyears = [np.int32(year) for year in includeyr.split(",")]
            elif '-' in includeyr:
                start_year, end_year = map(np.int32, includeyr.split('-'))
                includeyears = [year for year in range(start_year, end_year + 1)]
            else:
                includeyears = [np.int32(includeyr)]
        elif isinstance(includeyr, list):
            if len(includeyr) == 0:
                includeyears = False
            else:
                includeyears = includeyr
        else:
            includeyears = False
    except Exception:
        includeyears = False
    return includeyears, excludemonths


def run_inarea(cardinfo):
    """(inarea, description) of a new-catalog run, as RainyDay_Py3.py computes it: DOMAINTYPE irregular -> float32
    bounding box of DOMAINSHP; otherwise (rectangular) -> AREA_EXTENT values in JSON order."""
    try:
        domain_type = cardinfo["DOMAINTYPE"]
    except Exception:
        domain_type = 'rectangular'
    if domain_type.lower() == 'irregular':
        import shapefile
        domainshp = cardinfo["DOMAINSHP"]
        tempbox = shapefile.Reader(domainshp, 'rb').bbox
        return np.array([tempbox[0], tempbox[2], tempbox[1], tempbox[3]], dtype='float32'), "DOMAINSHP " + domainshp
    lim1, lim2, lim3, lim4 = cardinfo["AREA_EXTENT"].values()
    return np.array([lim3, lim4, lim1, lim2]), "AREA_EXTENT " + json.dumps(cardinfo["AREA_EXTENT"])


def _coords(path, latname, lonname):
    import netCDF4
    with netCDF4.Dataset(path) as f:
        lat = np.array(f.variables[latname][:])
        lon = np.array(f.variables[lonname][:])
        if max(f.variables[lonname]) > 180:                  # find_indices
            lon = lon - 360
    return lat, lon


def crop_coverage_problems(crop_dir, inarea, variables, first_file):
    """Reasons why the domain selection of a run (find_indices on inarea) would differ between the crop in crop_dir
    and the files it was cut from ([] = covered). Uses the source cells just outside the crop window recorded in the
    manifest: if one of them is inside the domain bounds, the crop lacks cells the run needs."""
    man = json.load(open(os.path.join(crop_dir, MANIFEST)))
    names = list(variables.values())                          # RainyDay: rain_name,lat_name,lon_name=values()
    if 'variables' in man and list(man['variables'].values()) != names:
        return ["VARIABLES %s differ from the crop's %s" % (names, list(man['variables'].values()))], man
    lat, lon = _coords(first_file, names[1], names[2])
    sel_lat = (lat >= inarea[2]) & (lat <= inarea[3])
    sel_lon = (lon >= inarea[0]) & (lon <= inarea[1])
    if not sel_lat.any() or not sel_lon.any():
        return ["the domain does not overlap the crop at all"], man
    probs = []
    nb = man.get('neighbors')
    if nb is None:                                           # manifest without neighbours: crop edges must stay free
        for name, sel in (('latitude', sel_lat), ('longitude', sel_lon)):
            if sel[0] or sel[-1]:
                probs.append("the domain reaches the %s edge of the crop" % name)
        return probs, man
    for key, cname, lo, hi in (('lat_before', 'latitude', inarea[2], inarea[3]), ('lat_after', 'latitude', inarea[2], inarea[3]),
                               ('lon_before', 'longitude', inarea[0], inarea[1]), ('lon_after', 'longitude', inarea[0], inarea[1])):
        v = nb.get(key)
        if v is None:                                        # the crop reaches the edge of the source files
            continue
        a = np.array([v], dtype=man.get('coord_dtype', 'float64'))
        if bool(((a >= lo) & (a <= hi))[0]):
            probs.append("source %s %r, just outside the crop, is inside the domain bounds [%s, %s]"
                         % (cname, float(v), lo, hi))
    return probs, man


def check_crop_coverage(inpath, inarea, variables, flist, parameterfile=None):
    """Hook after createfilelist: if the RAINPATH folder is a crop of tools/make_domain_crop.py, stop when this run's
    domain is not inside it (the run would silently use a smaller domain)."""
    if not flist:
        return
    for d in sorted(set(os.path.dirname(os.path.abspath(f)) for f in flist)):
        if not os.path.exists(os.path.join(d, MANIFEST)):
            continue
        first = [f for f in flist if os.path.dirname(os.path.abspath(f)) == d][0]
        probs, man = crop_coverage_problems(d, inarea, variables, first)
        if probs:
            lat, lon = _coords(first, list(variables.values())[1], list(variables.values())[2])
            made = "; ".join(x.get('description', '') for x in man.get('domains', [])) or str(man.get('domain_shp', '?'))
            try:
                with open(parameterfile, 'r') as f:
                    dom = run_inarea(json.loads(f.read()))[1]
            except Exception:
                dom = "this run's domain"
            sys.exit("RAINPATH " + d + " is a crop made by tools/make_domain_crop.py (for " + made + "; latitude "
                     + "%.4f..%.4f, longitude %.4f..%.4f" % (lat.min(), lat.max(), lon.min(), lon.max()) + "), but "
                     "this run's domain (" + dom + ", inarea " + str([float(x) for x in inarea]) + ") is not inside "
                     "it: " + "; ".join(probs) + ". On this crop the run would use a smaller domain than on the full "
                     "files. Make a crop for this domain (python tools/make_domain_crop.py --json " + str(parameterfile)
                     + ") or point RAINPATH to the full files.")
        print("RAINPATH is a crop (" + d + "): the domain is inside it")
