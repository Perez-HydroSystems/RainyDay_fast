"""Dimensionless ("normalized") SST for a watershed: ENHANCEDSST = "dimensionless".

Port of the watershed path of RainyDay (HydroclimateExtremesGroup/RainyDay, commit ff2b809, 2026-09-28:
RainyDay_functions.SSTalt_normalized, read_quantilefile, Normalized_SST_write, and the preparation block of
RainyDay_Py3.py), after Nathan et al. (2016).

A storm is transposed as a fraction of the local rainfall climatology, not as an absolute depth. With R a
reference rainfall field on the catalog grid (for example the 10-year depth of the analysis duration), a storm
moved so that basin pixel (i, j) lies over the origin pixel (y+i, x+j) has its rain multiplied, pixel by pixel, by

    multiplier[i, j] = R[home pixel (i, j)] / R[origin pixel (y+i, x+j)]

Storms from places with less extreme rain than the basin are scaled up, storms from wetter places scaled down.
The same multiplier is applied to the basin rainfall of the frequency analysis and to the scenario files.

Reference field (RAINDISTRIBUTIONFILE), log-transformed here:
  *.nc, RainyDay format: variable `precrate`(year, duration, latitude, longitude) of annual maxima on the SAME grid
        as the storm catalog; the reference is the empirical quantile of return period DIMENSIONLESSRETURNPERIOD
        (default 10, the value RainyDay hard-codes) of the analysis duration;
  *.nc, a 2-d field: variable `reference`(latitude, longitude), any regular grid covering the domain; bilinear;
  *.asc, an ArcASCII grid (the format of the NOAA Atlas 14 GIS grids), any regular grid covering the domain; bilinear.
Differences from RainyDay, all in the reference reader: the two regridded forms (RainyDay wires the .asc file only
to the point modes) and the return period being a card instead of a constant.
"""
import sys

import numpy as np
from numba import njit


def _bilinear(field, lat, lon, latrange, lonrange):
    from scipy.interpolate import RegularGridInterpolator
    lat, lon, field = np.asarray(lat, float), np.asarray(lon, float), np.asarray(field, float)
    if lat[0] > lat[-1]:
        lat, field = lat[::-1], field[::-1]
    if lon[0] > lon[-1]:
        lon, field = lon[::-1], field[:, ::-1]
    f = RegularGridInterpolator((lat, lon), field, method='linear', bounds_error=False, fill_value=np.nan)
    yy, xx = np.meshgrid(np.asarray(latrange, float), np.asarray(lonrange, float), indexing='ij')
    return f(np.column_stack((yy.ravel(), xx.ravel()))).reshape(yy.shape)


def read_reference(rescalingfile, duration, latrange, lonrange, returnperiod=10.):
    """The reference field on the catalog grid (latrange, lonrange as the catalog stores them), NOT yet in logs."""
    if rescalingfile.endswith('.nc'):
        import xarray as xr
        with xr.open_dataset(rescalingfile) as ds:
            if 'precrate' in ds and 'duration' in ds['precrate'].dims:           # RainyDay's annual-maxima file
                if duration not in ds['duration'].values:
                    sys.exit(f"The duration {duration} is not in RAINDISTRIBUTIONFILE; available: {ds['duration'].values}")
                d = ds.sel(duration=duration)
                lat, lon = np.asarray(d['latitude']), np.asarray(d['longitude'])
                if not (lat.shape == np.shape(latrange) and lon.shape == np.shape(lonrange)
                        and np.allclose(lat, latrange) and np.allclose(lon, lonrange)):
                    sys.exit("The annual-maxima file of RAINDISTRIBUTIONFILE is not on the storm catalog grid")
                return np.quantile(d['precrate'].values, 1. - 1. / returnperiod, axis=0, method='linear')
            if 'reference' in ds:
                return _bilinear(ds['reference'].values, ds['latitude'].values, ds['longitude'].values, latrange, lonrange)
        sys.exit("RAINDISTRIBUTIONFILE (.nc) needs `precrate`(year, duration, latitude, longitude) or `reference`(latitude, longitude)")
    if rescalingfile.endswith('.asc'):
        hdr = {}
        with open(rescalingfile) as f:
            for _ in range(6):
                k, v = f.readline().split()
                hdr[k.lower()] = float(v)
        grid = np.loadtxt(rescalingfile, skiprows=6, dtype='float32')
        grid[grid == hdr.get('nodata_value', -9999.)] = np.nan
        cs = hdr['cellsize']
        x0 = hdr['xllcorner'] + cs / 2. if 'xllcorner' in hdr else hdr['xllcenter']
        y0 = hdr['yllcorner'] + cs / 2. if 'yllcorner' in hdr else hdr['yllcenter']
        lon = x0 + cs * np.arange(grid.shape[1])
        lat = y0 + cs * np.arange(grid.shape[0])[::-1]                             # first row = north
        return _bilinear(grid, lat, lon, latrange, lonrange)
    sys.exit('Unrecognized file format provided for dimensionless SST (use .nc or .asc)')


def prepare(rescalingfile, duration, latrange, lonrange, catmask, domainmask, returnperiod=10.):
    """(intensegrid, homegrid): the log reference on the catalog grid and its part under the basin box."""
    ref = np.asarray(read_reference(rescalingfile, duration, latrange, lonrange, returnperiod), dtype='float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        intensegrid = np.log(ref)
    intensegrid = intensegrid[:domainmask.shape[0], :domainmask.shape[1]]
    yy, xx = np.nonzero(catmask != 0)
    homegrid = intensegrid[yy.min():yy.max() + 1, xx.min():xx.max() + 1].copy()
    # every cell a transposed basin box can touch must have a reference value: the cells of the domain and of the
    # basin box at home (RainyDay checks the domain cells; a missing value would silently zero that rain)
    if np.any(~np.isfinite(intensegrid[domainmask > 0])) or np.any(~np.isfinite(homegrid)):
        sys.exit("Design field is missing inside the transposition domain")
    return np.ascontiguousarray(intensegrid, dtype='float64'), np.ascontiguousarray(homegrid, dtype='float64')


def box_multiplier(intensegrid, homegrid, y, x):
    """The multiplier field of a storm transposed to anchor (y, x); non-finite values become 0 (as RainyDay)."""
    mh, mw = homegrid.shape
    with np.errstate(over='ignore', invalid='ignore'):
        m = np.exp(homegrid - intensegrid[y:y + mh, x:x + mw])
    return np.where(np.isfinite(m), m, 0.0).astype('float32')


@njit(cache=True)
def _kernel(exprain, sstx, ssty, trimmask, mh, mw, intensegrid, homegrid, durcheck):
    n = len(sstx); nsteps = exprain.shape[0]
    rainsum = np.zeros(n, dtype=np.float32); mult = np.full(n, -9999., dtype=np.float32); step = np.zeros(n, dtype=np.int32)
    for k in range(n):
        y = int(ssty[k]); x = int(sstx[k])
        wet = False
        for s in range(nsteps):
            for i in range(mh):
                for j in range(mw):
                    if not (exprain[s, y + i, x + j] < 0.5):
                        wet = True
        if not wet:                                           # RainyDay: all cells below 0.5 -> no storm here
            continue
        best = 0.0; beststep = 0; bestraw = 0.0
        for s in range(nsteps):
            tot = 0.0; raw = 0.0
            for i in range(mh):
                for j in range(mw):
                    m = np.exp(homegrid[i, j] - intensegrid[y + i, x + j])
                    if not np.isfinite(m):
                        m = 0.0
                    r = exprain[s, y + i, x + j] * trimmask[i, j]
                    tot += r * m; raw += r
            if (not durcheck) or tot > best:
                best = tot; beststep = s; bestraw = raw
        rainsum[k] = best; step[k] = beststep
        mult[k] = best / bestraw if bestraw > 0 else np.nan     # effective multiplier of the basin rainfall (diagnostic)
    return rainsum, mult, step


def sst_dimensionless(passrain, sstx, ssty, trimmask, maskheight, maskwidth, intensegrid, homegrid, durcheck=False):
    """Basin rainfall sums of one parent storm at the anchors (sstx, ssty), with the dimensionless multiplier.

    passrain: (ny, nx) storm total, or (nsteps, ny, nx) rolling sums with durcheck=True (the largest step is kept).
    Returns (rainsum float32, effective multiplier float32 [rescaled / unrescaled basin sum; -9999 = no storm],
    whichstep int32). rainsum is what RainyDay's SSTalt_normalized returns."""
    exprain = passrain if durcheck else passrain[None]
    return _kernel(np.ascontiguousarray(exprain, dtype='float32'), np.asarray(sstx, dtype=np.int64), np.asarray(ssty, dtype=np.int64),
                   np.ascontiguousarray(trimmask, dtype='float32'), int(maskheight), int(maskwidth), intensegrid, homegrid, bool(durcheck))
