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

Hooks in RainyDay_Py3.py ("GP 2026-10-22"): CATALOGSTORAGE parsing, list_storms() instead of the storm-file globs,
LightCatalogWriter in the catalog write loop, readcatalog() instead of RainyDay.readcatalog at the per-storm reads,
attrs_path() for the catalog description check; SCENARIOS stops with a light catalog.
"""
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
        first = {}
        for f in flist:                              # first file of the list for every date (write-loop search)
            m = re.search(DATE_RE, os.path.basename(f))
            first.setdefault(m.group().replace("-", "").replace("/", ""), f)
        c = dict(key=key, ds=ds, meta=meta, names=names, index={n: k for k, n in enumerate(names)}, flist=flist,
                 first=first, cattime=np.array(ds['cattime'], dtype='datetime64[m]'),
                 crc=np.array(ds['rain_crc32'], dtype=np.uint32))
        _CACHE[lpath] = c
    return c


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
                  + os.path.dirname(c['flist'][0]) + ")")
        return [os.path.join(d, n) for n in c['names']]
    if storage == 'light' and len(full) > 0:
        print("NOTE: CATALOGSTORAGE is 'light' but " + os.path.dirname(prefix) + " holds a full catalog; using it")
    return full


def attrs_path(path):
    """File holding the catalog attributes (description) of a storm path."""
    return _split_virtual(path)[0] if is_virtual(path) else path


def get_storm_rain(lpath, i, check=True):
    """Rain array of storm index i (0-based) exactly as the full catalog's storm file holds it (see module doc)."""
    c = _load(lpath)
    meta = c['meta']
    variables, idxes, droplist = meta['variables'], meta['idxes'], meta['droplist']
    calendar, time_units = meta['calendar'], meta['time_units']
    timeres = np.int32(np.int16(c['ds'].timeresolution))
    cattime = c['cattime']
    # ---- copy of the catalog write loop (RainyDay_Py3.py, "This part saves each storm as single file") ----
    start_time = cattime[i, 0]
    end_time = cattime[i, -1]
    current_datetime = start_time
    dataset_date = None
    catrain = np.zeros([cattime.shape[1]] + list(meta['shape'][1:]), dtype='float32')
    k = 0
    stm_file = None
    while current_datetime <= end_time:
        if meta['first_time_is_day_boundary']:
            current_date = np.datetime_as_string(current_datetime, unit='D')
        else:
            current_date = np.datetime_as_string(current_datetime - timeres, unit='D')
        if current_date != dataset_date:
            dataset_date = current_date
            stm_file = c['first'].get(current_date.replace("-", ""), stm_file)     # no file for the date: sticky
            if stm_file is None:
                sys.exit("light catalog: no input file for " + current_date)
            stm_rain, stm_time = RainyDay.readnetcdf(stm_file, variables, idxes, dropvars=droplist,
                                                     calendar=calendar, time_units=time_units)
        match = np.where(stm_time == current_datetime)[0] if len(stm_time) > 0 else []
        if len(match) == 0:
            catrain[k, :] = 0.0
        else:
            catrain[k, :] = stm_rain[match[0], :]
        current_datetime += timeres
        k += 1
    catrain[np.isnan(catrain)] = -9999.              # writecatalog
    if list(catrain.shape) != list(meta['shape']):
        sys.exit("light catalog: storm " + str(i + 1) + " rebuilt with shape " + str(catrain.shape))
    if check and rain_crc32(catrain) != int(c['crc'][i]):
        sys.exit("light catalog: the rain of storm " + str(i + 1) + " rebuilt from " + os.path.dirname(stm_file)
                 + " doesn't match the catalog (CRC32). The input files changed since the catalog was written.")
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
