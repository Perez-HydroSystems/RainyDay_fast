#==============================================================================
# WELCOME
#==============================================================================


#    Welcome to RainyDay, a framework for coupling gridded precipitation
#    fields with Stochastic Storm Transposition for assessment of rainfall-driven hazards.
#    Copyright (C) 2017  Daniel Benjamin Wright (danielb.wright@gmail.com)
#

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.#




#==============================================================================
# THIS DOCUMENT CONTAINS VARIOUS FUNCTIONS NEEDED TO RUN RainyDay
#==============================================================================
#%%                                               
import os
# Disable CuPy's CUB/cuTENSOR accelerators before cupy is ever imported (the
# batchcuda engine imports it lazily). CUB compiles reductions (argmax/max) with
# nvcc, which needs an MSVC host compiler (cl.exe) absent on this Windows box, so
# argmax would raise NVCCException; the empty value forces the NVRTC path. Set only
# if the user hasn't chosen accelerators themselves (e.g. on Linux with a working nvcc).
os.environ.setdefault('CUPY_ACCELERATORS', '')
import sys
import numpy as np
import scipy as sp
import glob
import re     
from datetime import datetime, date    
import time
import fiona
import copy
#import nctoolkit

from netCDF4 import Dataset, num2date
#import h5netcdf
import rasterio
from rasterio.transform import from_origin
from rasterio.shutil import delete
from rasterio.mask import mask
from rasterio.io import MemoryFile
import pandas as pd
from numba import prange,jit

import pyproj
from shapely.ops import transform

import shapely
from shapely.ops import unary_union
from shapely.geometry import shape
import json
import xarray as xr

import geopandas as gp


from scipy.stats import norm
from scipy.stats import lognorm

# plotting stuff, really only needed for diagnostic plots
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm 

import subprocess
try:
    os.environ.pop('PYTHONIOENCODING')
except KeyError:
    pass

import warnings
warnings.filterwarnings("ignore")

from numba.types import int32,int64,float32,uint32
import linecache
import dask
from dask.diagnostics import ProgressBar
from numba import njit, prange


# =============================================================================
# Smoother that is compatible with nan values. Adapted from https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
# =============================================================================

def mysmoother(inarray,sigma=[3,3]):
    if len(sigma)!=len(inarray.shape):
        sys.exit("there seems to be a mismatch between the sigma dimension and the dimension of the array you are trying to smooth")
    V=inarray.copy()
    V[np.isnan(inarray)]=0.
    VV=sp.ndimage.gaussian_filter(V,sigma=sigma)

    W=0.*inarray.copy()+1.
    W[np.isnan(inarray)]=0.
    WW=sp.ndimage.gaussian_filter(W,sigma=sigma)
    outarray=VV/WW
    outarray[np.isnan(inarray)]=np.nan
    return outarray

def my_kde_bandwidth(obj, fac=1):     # this 1.5 choice is completely subjective :(
    #We use Scott's Rule, multiplied by a constant factor
    return np.power(obj.n, -1./(obj.d+4)) * fac

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def convert_3D_2D(geometry):
    '''
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
    '''
    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == 'Polygon':
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = shapely.geometry.Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == 'MultiPolygon':
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = shapely.geometry.Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(shapely.geometry.MultiPolygon(new_multi_p))
    return new_geo



# adapted from https://pythonadventures.wordpress.com/2016/03/06/detect-duplicate-keys-in-a-json-file/
def dict_raise_on_duplicates(ordered_pairs):
    """Reject duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
           sys.exit("duplicate key: %r" % (k,))
        else:
           d[k] = v
    return d

#==============================================================================
# LOOP TO DO SPATIAL SEARCHING FOR MAXIMUM RAINFALL LOCATION AT EACH TIME STEP
# THIS IS THE CORE OF THE STORM CATALOG CREATION TECHNIQUE
#==============================================================================
    
#def catalogweave(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum):
#    rainsum[:]=0.
#    code= """
#        #include <stdio.h>
#        int i,j,x,y;
#        for (x=0;x<xlen;x++) {
#            for (y=0;y<ylen;y++) {
#                for (j=0;j<maskheight;j++) {
#                    for (i=0;i<maskwidth;i++) {
#                        rainsum(y,x)=rainsum(y,x)+temparray(y+j,x+i)*trimmask(j,i);                     
#                    }                               
#                }
#            }                      
#        }
#    """
#    vars=['temparray','trimmask','xlen','ylen','maskheight','maskwidth','rainsum']
#    sp.weave.inline(code,vars,type_converters=converters.blitz,compiler='gcc')
#    rmax=np.nanmax(rainsum)
#    wheremax=np.where(rainsum==rmax)
#    return rmax, wheremax[0][0], wheremax[1][0]
#    


def catalogAlt(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask):
    rainsum[:]=0.
    for i in range(0,(ylen)*(xlen)):
        y=i//xlen
        x=i-y*xlen
        #print x,
        rainsum[y,x]=np.nansum(np.multiply(temparray[(y):(y+maskheight),(x):(x+maskwidth)],trimmask))
    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(rainsum==rmax)
    return rmax, wheremax[0][0], wheremax[1][0]

def catalogAlt_irregular(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask):
    rainsum[:]=0.
    for i in range(0,(ylen)*(xlen)):
        y=i//xlen
        x=i-y*xlen
        #print x,y
        if np.any(np.equal(domainmask[y+maskheight/2,x:x+maskwidth],1.)) and np.any(np.equal(domainmask[y:y+maskheight,x+maskwidth/2],1.)):
            rainsum[y,x]=np.nansum(np.multiply(temparray[(y):(y+maskheight),(x):(x+maskwidth)],trimmask))
        else:
            rainsum[y,x]=0.
    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(rainsum==rmax)
    
    return rmax, wheremax[0][0], wheremax[1][0]



# @jit(nopython=True, fastmath =  True)
# def catalogNumba_irregular(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask,stride=1):
#     rainsum[:]=0.
#     halfheight=int32(np.ceil(maskheight/2))
#     halfwidth=int32(np.ceil(maskwidth/2))
#     for i in range(0,(ylen)*(xlen),stride):
#         y=i//xlen
#         x=i-y*xlen
#         # Ensure that the slice does not exceed the bounds of temparray

    #     if np.any(np.equal(domainmask[y+halfheight, x:x+maskwidth], 1.)) and np.any(np.equal(domainmask[y:y+maskheight, x+halfwidth], 1.)):
    #         rainsum[y, x] = np.nansum(np.multiply(temparray[y:(y+maskheight), x:(x+maskwidth)], trimmask))
    #
    #
    #     else:
    #         rainsum[y, x] = 0
    #
    # #wheremax=np.argmax(rainsum)
    # rmax=np.nanmax(rainsum)
    # wheremax=np.where(np.equal(rainsum,rmax))
    # return rmax, wheremax[0][0], wheremax[1][0]

# @jit(nopython=True, fastmath =  True)
# def catalogNumba_irregular(temparray, trimmask, xlen, ylen, xloop, yloop, maskheight, maskwidth, rainsum, stride=1):
#     rainsum[:, :] = 0.0  # clear output array

#     rmax = -1e30
#     ymax = -1
#     xmax = -1

#     for y in range(0, yloop):
#         for x in range(0, xloop, stride):
#             # Define all 4 symmetric locations
#             positions = [
#                 (y, x),  # original
#                 (y, xlen - x - 1),  # x-flipped
#                 (ylen - y - 1, x),  # y-flipped
#                 (ylen - y - 1, xlen - x - 1)  # x & y flipped
#             ]

#             for yy, xx in positions:
#                 # Ensure we’re within bounds
#                 if yy + maskheight <= temparray.shape[0] and xx + maskwidth <= temparray.shape[1]:
#                     patch = temparray[yy:yy + maskheight, xx:xx + maskwidth]
#                     value = np.nansum(patch * trimmask)
#                     rainsum[yy, xx] = value

#                     if not np.isnan(value) and value > rmax:
#                         rmax = value
#                         ymax = yy
#                         xmax = xx

#     return rmax, ymax, xmax

# # Testing (parallel, including symmetric positions)
# @njit(parallel=True, fastmath=True)
# def catalogNumba_irregular(temparray, trimmask, xlen, ylen, xloop, yloop,
#                                      maskheight, maskwidth, rainsum, stride=1):
#     # Ensure loop bounds are integers
#     xloop = int(xloop)
#     yloop = int(yloop)
#     xlen = int(xlen)
#     ylen = int(ylen)

#     # Clear rainsum
#     rainsum[:, :] = 0.0

#     # Parallel loop over y
#     for y in prange(0, yloop):
#         for x in range(0, xloop, stride):
#             # Define symmetric positions
#             positions = [
#                 (int(y), int(x)),
#                 (int(y), int(xlen - x - 1)),
#                 (int(ylen - y - 1), int(x)),
#                 (int(ylen - y - 1), int(xlen - x - 1))
#             ]

#             for yy, xx in positions:
#                 if yy + maskheight > temparray.shape[0] or xx + maskwidth > temparray.shape[1]:
#                     continue

#                 val = 0.0
#                 for i in range(maskheight):
#                     for j in range(maskwidth):
#                         a = temparray[yy + i, xx + j]
#                         b = trimmask[i, j]
#                         if not np.isnan(a) and not np.isnan(b):
#                             val += a * b

#                 rainsum[yy, xx] = val

#     # Serial reduction to find max
#     rmax = -1e30
#     ymax = -1
#     xmax = -1
#     for y in range(rainsum.shape[0]):
#         for x in range(rainsum.shape[1]):
#             val = rainsum[y, x]
#             if val > rmax:
#                 rmax = val
#                 ymax = y
#                 xmax = x

#     return rmax, ymax, xmax

# Testing (parallel, without symmetric positions). FASTEST for large domains
@njit(parallel=True, fastmath=True)
def catalogNumba_irregular(temparray, trimmask, xlen, ylen, xloop, yloop,
                           maskheight, maskwidth, rainsum, stride=1):
    # Since we are not doing symmetric positions, we can simplify the logic
    xloop = int(xlen)
    yloop = int(ylen)

    rainsum[:, :] = 0.0

    # Parallel storm scan
    for y in prange(0, yloop):
        for x in range(0, xloop, stride):
            if y + maskheight > temparray.shape[0] or x + maskwidth > temparray.shape[1]:
                continue

            val = 0.0
            for i in range(maskheight):
                for j in range(maskwidth):
                    a = temparray[y + i, x + j]
                    b = trimmask[i, j]
                    if not np.isnan(a) and not np.isnan(b):
                        val += a * b

            rainsum[y, x] = val

    # --- Parallel row-wise reduction ---
    row_max = np.full(rainsum.shape[0], -1e30)
    row_x = np.full(rainsum.shape[0], -1, dtype=np.int32)

    for y in prange(rainsum.shape[0]):
        max_val = -1e30
        max_x = -1
        for x in range(rainsum.shape[1]):
            val = rainsum[y, x]
            if val > max_val:
                max_val = val
                max_x = x
        row_max[y] = max_val
        row_x[y] = max_x

    # --- Serial reduction over rows ---
    rmax = -1e30
    ymax = -1
    xmax = -1
    for y in range(rainsum.shape[0]):
        if row_max[y] > rmax:
            rmax = row_max[y]
            ymax = y
            xmax = row_x[y]

    return rmax, ymax, xmax

import numpy as np
from scipy.signal import fftconvolve, oaconvolve

def catalogFFT_irregular(temparray, trimmask):
    """
    CPU version using FFT-based convolution (cross-correlation equivalent to manual loop).

    Parameters:
    -----------
    temparray : np.ndarray
        2D rainfall field (float32 or float64)
    trimmask : np.ndarray
        2D storm mask kernel (float32 or float64)

    Returns:
    --------
    rmax : float
        Maximum value of convolution
    ymax, xmax : int
        Location (row, col) of maximum alignment
    """
    # Clean NaNs
    temparray_clean = np.nan_to_num(temparray)
    trimmask_clean = np.nan_to_num(trimmask)

    # Flip kernel so fftconvolve behaves like cross-correlation
    trimmask_clean = trimmask_clean[::-1, ::-1]  # flip both axes
    result = fftconvolve(temparray_clean, trimmask_clean, mode='valid')

    # result = oaconvolve(temparray_clean, trimmask_clean, mode='valid')

    # Find max value and its location
    rmax = np.max(result)
    ymax, xmax = np.unravel_index(np.argmax(result), result.shape)

    return float(rmax), int(ymax), int(xmax)

## New version preserving domain mask. Added by GP Jan-29 2026

from scipy.signal import fftconvolve
import numpy as np

def catalogFFT_irregular_domainaware(temparray, trimmask, domainmask):
    """
    Finds the storm placement that produces the largest total rainfall,
    while ensuring the storm lies fully inside the domain.

    Equivalent to a manual sliding-window dot product/cross-correlation.
    Assumes temparray was already multiplied by domainmask upstream.
    """

    # --- Clean inputs ---
    rain = np.ascontiguousarray(np.nan_to_num(temparray), dtype=np.float32)
    storm = np.ascontiguousarray(np.nan_to_num(trimmask), dtype=np.float32)
    domain = np.ascontiguousarray((domainmask > 0).astype(np.float32))

    # --- Flip kernels so fftconvolve behaves like cross-correlation ---
    storm_flip = storm[::-1, ::-1]

    # storm_footprint = (storm > 1e-6).astype(np.float32)
    storm_footprint = np.ones_like(storm, dtype=np.float32)
    footprint_flip = storm_footprint[::-1, ::-1]

    # Total rainfall captured by storm at each valid placement
    storm_rainfall_map = fftconvolve(
        rain,
        storm_flip,
        mode="valid"
    )

    # Count how many storm-footprint pixels fall inside the domain
    inside_count_map = fftconvolve(
        domain,
        footprint_flip,
        mode="valid"
    )

    required_pixels = storm_footprint.sum()

    # Invalidate placements where the full storm footprint is not inside domain
    storm_rainfall_map[
        inside_count_map < (required_pixels - 0.5)
    ] = -np.inf

    # Maximum rainfall and location
    flat_idx = np.argmax(storm_rainfall_map)
    rmax = storm_rainfall_map.flat[flat_idx]

    ymax, xmax = np.unravel_index(
        flat_idx,
        storm_rainfall_map.shape
    )

    return float(rmax), int(ymax), int(xmax)


## Batched FFT catalog engine. Added by GP Jul-12 2026
# Same math as catalogFFT_irregular_domainaware, but everything that is
# constant across timesteps (kernel FFT, domain-validity map) is computed once,
# and whole blocks of timesteps are processed with batched multithreaded FFTs.

import scipy.fft as spfft

def catalogFFT_precompute(trimmask, domainmask):
    """
    One-time precomputation for the batched catalog engine.

    Returns a dict with:
      kfft      : rfft2 of the flipped storm kernel, padded to the FFT shape
      valid     : boolean map of placements whose full rectangular footprint
                  lies inside the transposition domain
      fshape    : padded FFT shape (scipy next_fast_len, matching fftconvolve)
      maskshape : (maskheight, maskwidth)
      outshape  : shape of the 'valid'-mode correlation output
    """
    storm = np.ascontiguousarray(np.nan_to_num(trimmask), dtype=np.float32)
    domain = np.ascontiguousarray((np.nan_to_num(domainmask) > 0).astype(np.float32))
    mh, mw = storm.shape
    fh, fw = domain.shape
    outh, outw = fh - mh + 1, fw - mw + 1
    fshape = (spfft.next_fast_len(fh + mh - 1, True),
              spfft.next_fast_len(fw + mw - 1, True))
    kfft = spfft.rfft2(storm[::-1, ::-1].copy(), fshape)

    # placements where the full (rectangular) storm footprint is inside the domain,
    # identical to the inside_count_map test in catalogFFT_irregular_domainaware
    footprint = np.ones_like(storm)
    inside_count = fftconvolve(domain, footprint[::-1, ::-1], mode='valid')
    valid = inside_count >= (footprint.sum() - 0.5)

    return dict(kfft=kfft, valid=valid, fshape=fshape, storm=storm,
                maskshape=(mh, mw), outshape=(outh, outw))


def catalogFFT_batch(rainblock, pre, workers=-1, chunksize=256):
    """
    Batched equivalent of catalogFFT_irregular_domainaware.

    rainblock : (T,H,W) float32 windowed rainfall sums, NaN-free and already
                multiplied by the domain mask
    pre       : dict from catalogFFT_precompute

    Returns rmax[T], ymax[T], xmax[T] — the best placement per timestep.
    chunksize bounds peak memory (the padded complex spectra of one chunk).
    """
    mh, mw = pre['maskshape']
    outh, outw = pre['outshape']
    fshape = pre['fshape']
    kfft = pre['kfft']
    invalid = ~pre['valid']

    nt = rainblock.shape[0]
    rmax = np.empty(nt, dtype=np.float64)
    ymax = np.empty(nt, dtype=np.int64)
    xmax = np.empty(nt, dtype=np.int64)

    for s in range(0, nt, chunksize):
        blk = rainblock[s:s + chunksize]
        spec = spfft.rfft2(blk, fshape, axes=(-2, -1), workers=workers)
        spec *= kfft[None, :, :]
        conv = spfft.irfft2(spec, fshape, axes=(-2, -1), workers=workers)
        conv = conv[:, mh - 1:mh - 1 + outh, mw - 1:mw - 1 + outw]
        conv[:, invalid] = -np.inf
        flat = conv.reshape(conv.shape[0], -1)
        idx = np.argmax(flat, axis=1)
        rmax[s:s + chunksize] = flat[np.arange(flat.shape[0]), idx]
        ymax[s:s + chunksize], xmax[s:s + chunksize] = np.unravel_index(idx, (outh, outw))

    return rmax, ymax, xmax


def read_file_block(bfiles, variables, idxes, dropvars, calendar, time_units, timestep):
    """
    Read a block of chronologically sorted files and return the fields on a
    contiguous time grid (missing timesteps zero-filled), so that count-based
    rolling windows equal the original time-based window logic.

    Returns (rainblock (B,H,W) float32, blocktimes (B,) datetime64[m],
    present (B,) bool, fileidx (B,) int64 — index into bfiles per timestep,
    -1 for zero-filled gap steps). Negative and NaN values are set to 0,
    matching inrain[inrain<0]=nan + nansum in the per-timestep path.
    """
    rains = []
    times = []
    for f in bfiles:
        r, t = readnetcdf(f, variables, idxes, dropvars=dropvars,
                          calendar=calendar, time_units=time_units)
        r = np.asarray(r, dtype=np.float32)
        r[~np.isfinite(r)] = 0.
        r[r < 0.] = 0.
        rains.append(r)
        times.append(np.asarray(t, dtype='datetime64[m]'))
    times_all = np.concatenate(times)
    rain_all = np.concatenate(rains, axis=0)
    fileidx_all = np.repeat(np.arange(len(bfiles), dtype=np.int64),
                            [len(t) for t in times])

    grid_times = np.arange(times_all[0], times_all[-1] + timestep, timestep)
    if len(grid_times) == len(times_all):
        present = np.ones(len(times_all), dtype=bool)
        return rain_all, times_all, present, fileidx_all

    pos = ((times_all - grid_times[0]) / timestep).astype(np.int64)
    rainblock = np.zeros((len(grid_times),) + rain_all.shape[1:], dtype=np.float32)
    rainblock[pos] = rain_all
    present = np.zeros(len(grid_times), dtype=bool)
    present[pos] = True
    fileidx = np.full(len(grid_times), -1, dtype=np.int64)
    fileidx[pos] = fileidx_all
    return rainblock, grid_times, present, fileidx


def build_windowed_block(rainblock, context, nwindow, domainmask):
    """
    Rolling nwindow-step rainfall totals for every timestep of a block, via
    cumulative sums (float64 accumulator to avoid float32 drift).

    context holds the nwindow-1 fields immediately preceding the block on the
    contiguous grid (zeros where no data exists). Returns (windowed (B,H,W)
    float32 multiplied by domainmask, new context for the next block).
    """
    full = np.concatenate([context, rainblock], axis=0)
    csum = np.cumsum(full, axis=0, dtype=np.float64)
    nb = rainblock.shape[0]
    windowed = np.empty_like(rainblock)
    windowed[0] = csum[nwindow - 1]
    if nb > 1:
        windowed[1:] = csum[nwindow:nwindow - 1 + nb] - csum[0:nb - 1]
    windowed *= np.nan_to_num(domainmask).astype(np.float32)
    return windowed, full[-(nwindow - 1):].copy()


# ---------------------------------------------------------------------------
# Execution-config resolution for the storm-catalog scan.
#
# The JSON parameter file may carry an optional "COMPUTE" block, e.g.
#     "COMPUTE": {"engine": "auto", "nworkers": "auto", "blockfiles": "auto",
#                 "gpu_chunk": "auto", "gpu_delta": 1e-3, "prefetch": "auto"}
# Every field is optional; "auto" (or a missing field / missing block) triggers
# machine auto-detection. Precedence per field: environment variable > JSON value
# > auto. resolve_compute_config() is the single entry point used by RainyDay_Py3;
# it injects the resolved values into the RAINYDAY_* env vars the engine code reads,
# so nothing downstream needs to change. Irregular domains only (batched engines).
# ---------------------------------------------------------------------------

def _free_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return None                       # unknown (psutil missing) -> generous default

def _free_vram_bytes():
    try:
        os.environ.setdefault('CUPY_ACCELERATORS', '')
        import cupy
        free, _ = cupy.cuda.runtime.memGetInfo()
        return free
    except Exception:
        return None

def _fork_available():
    try:
        import multiprocessing as mp
        return 'fork' in mp.get_all_start_methods()
    except Exception:
        return False

def _cuda_available():
    try:
        import importlib
        if importlib.util.find_spec('cupy') is None:
            return False
        os.environ.setdefault('CUPY_ACCELERATORS', '')
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

def _auto_engine(ncores):
    # batchfft_mp (parallel CPU) is fastest where it can run, but it needs the 'fork'
    # start method (this script has no __main__ guard, so spawn would re-run it). Fall
    # back to the CUDA GPU engine, then to single-process batchfft.
    if _fork_available() and (ncores or 1) >= 4:
        return 'batchfft_mp'
    if _cuda_available():
        return 'batchcuda'
    return 'batchfft'

def _validate_engine(engine, ncores):
    if engine == 'batchfft_mp' and not _fork_available():
        alt = 'batchcuda' if _cuda_available() else 'batchfft'
        return alt, "engine 'batchfft_mp' needs the 'fork' start method (unavailable here) -> using '" + alt + "'"
    if engine == 'batchcuda' and not _cuda_available():
        alt = 'batchfft_mp' if _fork_available() else 'batchfft'
        return alt, "engine 'batchcuda' needs CuPy + a CUDA device (unavailable here) -> using '" + alt + "'"
    return engine, None

def _auto_workers_blocks(engine, ncores, free_ram, steps_per_file, H, W):
    def block_gb(B):
        return B * steps_per_file * H * W * (4 + 8) / 1e9   # float32 raw + float64 cumsum
    budget = (0.6 * free_ram) if free_ram else 1e9          # unknown RAM -> effectively uncapped
    if engine == 'batchfft_mp':
        nw = min(max(1, (ncores or 2) - 2), 8)              # single disk saturates ~8 readers (measured)
        while nw > 1 and block_gb(1) * nw > budget:
            nw -= 1
        B = 16
        while B > 1 and block_gb(B) * nw > budget:
            B //= 2
        return nw, max(1, B)
    nw = min(max(1, (ncores or 2) - 2), 8)                  # informational for single-process engines
    B = 16
    while B > 1 and block_gb(B) > budget:
        B //= 2
    return nw, max(1, B)

def _auto_gpu_chunk(H, W):
    free = _free_vram_bytes()
    if not free:
        return 64
    per_ts = 6 * H * W * 4                 # ~working set per timestep (measured ~33 MB at CONUS scale)
    return max(8, min(256, int(0.30 * free / max(1, per_ts))))

def _resolve_knob(env_name, json_val, auto_val):
    """Precedence: existing env var > JSON value (unless 'auto') > auto_val. Injects the
    chosen value into the env so downstream engine code reads it. auto_val=None leaves
    it unset (the engine keeps its own per-call default). Returns (value_str, source)."""
    if env_name in os.environ:
        return os.environ[env_name], 'env'
    if json_val is not None and str(json_val).strip().lower() != 'auto':
        os.environ[env_name] = str(json_val)
        return str(json_val), 'JSON'
    if auto_val is not None:
        os.environ[env_name] = str(auto_val)
        return str(auto_val), 'auto'
    return None, 'default'

def resolve_compute_config(compute, domainshape, steps_per_file, nwindow, verbose=True):
    """
    Resolve the catalog-scan execution config from the JSON 'COMPUTE' block, environment
    variables, and machine auto-detection (precedence: env > JSON > auto). Sets the
    RAINYDAY_* env vars the engine code reads and returns the chosen engine string.
    Auto-sizing uses CPU count + free RAM (workers, blockfiles) and free VRAM (gpu_chunk)
    so a user need not know their machine's limits. See the block comment above.
    """
    compute = compute or {}
    H, W = int(domainshape[0]), int(domainshape[1])
    ncores = os.cpu_count() or 2
    free_ram = _free_ram_gb()

    env_engine = os.environ.get('RAINYDAY_CATALOG_ENGINE')
    json_engine = compute.get('engine')
    if env_engine:
        engine, esrc = env_engine, 'env'
    elif json_engine and str(json_engine).strip().lower() != 'auto':
        engine, esrc = str(json_engine), 'JSON'
    else:
        engine, esrc = _auto_engine(ncores), 'auto'
    engine, warn = _validate_engine(engine, ncores)
    if warn and verbose:
        print('  [compute] ' + warn)
        esrc += '->fallback'

    nworkers_auto, blockfiles_auto = _auto_workers_blocks(engine, ncores, free_ram, steps_per_file, H, W)
    bf, bf_src = _resolve_knob('RAINYDAY_BLOCKFILES', compute.get('blockfiles'), blockfiles_auto)
    nw, nw_src = _resolve_knob('RAINYDAY_NWORKERS', compute.get('nworkers'),
                               nworkers_auto if engine == 'batchfft_mp' else None)
    gc, gc_src = _resolve_knob('RAINYDAY_GPU_CHUNK', compute.get('gpu_chunk'),
                               _auto_gpu_chunk(H, W) if engine == 'batchcuda' else None)
    gd, gd_src = _resolve_knob('RAINYDAY_GPU_DELTA', compute.get('gpu_delta'), '1e-3')
    pf, pf_src = _resolve_knob('RAINYDAY_SCAN_PREFETCH', compute.get('prefetch'), None)

    if verbose:
        msg = 'Compute config: engine=' + engine + ' (' + esrc + '), blockfiles=' + str(bf) + ' (' + bf_src + ')'
        if engine == 'batchfft_mp':
            msg += ', nworkers=' + str(nw) + ' (' + nw_src + ')'
        if engine == 'batchcuda':
            msg += ', gpu_chunk=' + str(gc) + ' (' + gc_src + ')'
            msg += ', prefetch=' + (pf if pf else '2') + ' (' + (pf_src if pf else 'auto') + ')'
        if engine in ('batchtf', 'batchcuda'):
            msg += ', gpu_delta=' + str(gd) + ' (' + gd_src + ')'
        ramtxt = '{0:.0f} GB free RAM'.format(free_ram) if free_ram else 'RAM unknown (no psutil)'
        print(msg + '  [' + str(ncores) + ' cores, ' + ramtxt + ']')
    return engine


def _iter_read_blocks(files, blockfiles, variables, idxes, dropvars, calendar,
                      time_units, timestep, prefetch=0):
    """
    Yield (bstart, bfiles, blockrain, blocktimes, present, fileidx) for each block
    of `blockfiles` consecutive files. With prefetch>0 a daemon thread reads the
    next block(s) while the caller processes the current one, overlapping file I/O
    (netCDF reads release the GIL) with the caller's windowing + GPU compute. Block
    order and contents are identical to a plain sequential read, so scan results are
    unchanged — only the read latency is hidden. This is the in-process alternative
    to parallel reader processes (threads can't parallelize the HDF5 reads
    themselves, but one reader thread does overlap I/O with the GPU); shipping
    windowed blocks between processes is infeasible (~tens of GB of IPC).
    """
    starts = list(range(0, len(files), blockfiles))

    def readblk(bstart):
        bfiles = files[bstart:bstart + blockfiles]
        return (bstart, bfiles) + read_file_block(bfiles, variables, idxes,
                                                  dropvars, calendar, time_units, timestep)

    if prefetch <= 0:
        for bstart in starts:
            yield readblk(bstart)
        return

    import threading
    try:
        import queue
    except ImportError:
        import Queue as queue
    q = queue.Queue(maxsize=prefetch)
    _DONE = object()
    err = []

    def worker():
        try:
            for bstart in starts:
                q.put(readblk(bstart))
        except BaseException as e:      # surface read errors to the consumer thread
            err.append(e)
        finally:
            q.put(_DONE)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is _DONE:
            break
        yield item
    if err:
        raise err[0]


def catalog_scan_files(files, ctxfiles, variables, idxes, dropvars, calendar,
                       time_units, timestep, nwindow, trimmask, domainmask,
                       blockfiles=16, verbose=True, engine='batchfft',
                       return_fileidx=False):
    """
    Scan a chronologically sorted list of files with the batched catalog engine
    and return the best storm placement for every present timestep:
    (times, rainmax, ycat, xcat), each 1-D and in chronological order.
    With return_fileidx=True, also returns the index into 'files' that each
    timestep came from (used by the batchtf exact-confirm selection).

    ctxfiles: files immediately preceding 'files' (may be empty) — read only to
    seed the rolling window context, no results are emitted for them. Used by
    the multiprocessing engine so each worker sees the full window history.
    """
    fftpre = catalogFFT_precompute(trimmask, domainmask)
    fh, fw = domainmask.shape
    context = np.zeros((nwindow - 1, fh, fw), dtype=np.float32)
    context_endtime = None
    if len(ctxfiles) > 0:
        ctxrain, ctxtimes, _, _ = read_file_block(ctxfiles, variables, idxes,
                                                  dropvars, calendar, time_units, timestep)
        m = min(nwindow - 1, ctxrain.shape[0])
        if m > 0:
            context[nwindow - 1 - m:] = ctxrain[-m:]
            context_endtime = ctxtimes[-1]

    # Overlap file reads with GPU compute for the CUDA engine: a reader thread
    # fetches the next block(s) while this thread windows + runs the GPU. Off by
    # default for the CPU engines (their scipy FFT already interleaves I/O and is
    # verified bit-identical); override with RAINYDAY_SCAN_PREFETCH.
    prefetch = int(os.environ.get('RAINYDAY_SCAN_PREFETCH', '2' if engine == 'batchcuda' else '0'))

    all_times, all_rmax, all_y, all_x, all_fidx = [], [], [], [], []
    for bstart, bfiles, blockrain, blocktimes, present, fileidx in _iter_read_blocks(
            files, blockfiles, variables, idxes, dropvars, calendar, time_units, timestep, prefetch):
        if verbose:
            print('Processing files ' + str(bstart + 1) + '-' + str(bstart + len(bfiles))
                  + ' out of ' + str(len(files))
                  + ' (' + "{0:0.0f}".format(100 * (bstart + len(bfiles)) / len(files)) + '%): '
                  + bfiles[0].split('/')[-1] + ' ...')

        # zero-fill any gap between the previous block and this one so the
        # rolling window stays aligned with wall-clock time
        if context_endtime is not None:
            gapsteps = int((blocktimes[0] - (context_endtime + timestep)) / timestep)
            if gapsteps < 0:
                sys.exit("The batched catalog engines need chronologically sorted, non-overlapping input files!")
            elif gapsteps >= nwindow - 1:
                context[:] = 0.
            elif gapsteps > 0:
                context = np.concatenate([context[gapsteps:],
                                          np.zeros((gapsteps,) + context.shape[1:], dtype='float32')], axis=0)

        windowed, context = build_windowed_block(blockrain, context, nwindow, domainmask)
        context_endtime = blocktimes[-1]

        if engine == 'batchtf':
            rmax_blk, y_blk, x_blk = catalogTF_batch(windowed, fftpre)
        elif engine == 'batchcuda':
            rmax_blk, y_blk, x_blk = catalogCUDA_batch(windowed, fftpre)
        else:
            rmax_blk, y_blk, x_blk = catalogFFT_batch(windowed, fftpre)

        keep = np.where(present)[0]
        all_times.append(blocktimes[keep])
        all_rmax.append(rmax_blk[keep])
        all_y.append(y_blk[keep])
        all_x.append(x_blk[keep])
        all_fidx.append(bstart + fileidx[keep])

    if return_fileidx:
        return (np.concatenate(all_times), np.concatenate(all_rmax),
                np.concatenate(all_y), np.concatenate(all_x),
                np.concatenate(all_fidx))
    return (np.concatenate(all_times), np.concatenate(all_rmax),
            np.concatenate(all_y), np.concatenate(all_x))


def catalog_scan_worker(task):
    """Unpacks one catalog_scan_files task tuple; module-level so it can be
    pickled by ProcessPoolExecutor (macOS spawn/forkserver-safe)."""
    return catalog_scan_files(*task)


def catalogTF_batch(rainblock, pre, chunksize=256):
    """
    TensorFlow (Metal GPU) equivalent of catalogFFT_batch. tf.nn.conv2d is a
    cross-correlation (no kernel flip), i.e. exactly the sliding storm-mask dot
    product. GPU constants are uploaded once per scan and cached on 'pre'.
    Requires an environment with tensorflow(-metal), e.g. RainyDay_Env_GPU.
    """
    import tensorflow as tf   # local import: only needed for engine 'batchtf'
    mh, mw = pre['maskshape']
    outh, outw = pre['outshape']
    if '_tf' not in pre:
        pre['_tf'] = (tf.constant(pre['storm'].reshape(mh, mw, 1, 1), dtype=tf.float32),
                      tf.constant(pre['valid'].reshape(1, outh, outw), dtype=tf.bool))
    kernel, valid = pre['_tf']
    neginf = tf.constant(-np.inf, dtype=tf.float32)

    nt = rainblock.shape[0]
    rmax = np.empty(nt, dtype=np.float64)
    ymax = np.empty(nt, dtype=np.int64)
    xmax = np.empty(nt, dtype=np.int64)
    for s in range(0, nt, chunksize):
        blk = tf.constant(rainblock[s:s + chunksize, :, :, None], dtype=tf.float32)
        conv = tf.nn.conv2d(blk, kernel, strides=1, padding='VALID')[:, :, :, 0]
        conv = tf.where(valid, conv, neginf)
        flat = tf.reshape(conv, (conv.shape[0], -1))
        idx = tf.argmax(flat, axis=1).numpy()
        rmax[s:s + chunksize] = tf.reduce_max(flat, axis=1).numpy()
        ymax[s:s + chunksize] = idx // outw
        xmax[s:s + chunksize] = idx % outw
    return rmax, ymax, xmax


def catalogCUDA_batch(rainblock, pre, chunksize=256):
    """
    CuPy/cuFFT equivalent of catalogFFT_batch. Unlike catalogTF_batch (direct
    conv2d, ~30x the arithmetic), this keeps the same rfft2 x conj-kernel-spectrum
    / irfft2 correlation the CPU engine uses, so it does O(N log N) work on the
    GPU. GPU constants (kernel spectrum, invalid-placement mask) are uploaded once
    per scan and cached on 'pre'. Returns numpy rmax/ymax/xmax.

    Requires an environment with CuPy + a working cuFFT (e.g. RainyDay_Env_GPU).
    Two Windows-specific notes:
      - CUPY_ACCELERATORS is forced empty so argmax/max compile via NVRTC; the
        default CUB path invokes nvcc, which needs an MSVC host compiler (cl.exe).
      - chunksize bounds peak VRAM (~33 MB/timestep at CONUS scale, complex64
        spectra). Override with RAINYDAY_GPU_CHUNK on small-memory GPUs; on an
        8 GB card CONUS needs ~64-128 while the default 256 is fine for small
        domains like Talequah.
    """
    os.environ.setdefault('CUPY_ACCELERATORS', '')  # must precede first cupy import
    import cupy as cp   # local import: only needed for engine 'batchcuda'
    chunksize = int(os.environ.get('RAINYDAY_GPU_CHUNK', chunksize))
    mh, mw = pre['maskshape']
    outh, outw = pre['outshape']
    fshape = pre['fshape']
    if '_cuda' not in pre:
        # kfft is complex64 (scipy rfft2 of a float32 kernel); cupy rfft2 of a
        # float32 block is also complex64, so the whole pipeline stays single
        # precision and the spectra are half the size of the complex128 default.
        pre['_cuda'] = (cp.asarray(pre['kfft']), cp.asarray(~pre['valid']))
    kfft, invalid = pre['_cuda']

    nt = rainblock.shape[0]
    rmax = np.empty(nt, dtype=np.float64)
    ymax = np.empty(nt, dtype=np.int64)
    xmax = np.empty(nt, dtype=np.int64)
    for s in range(0, nt, chunksize):
        blk = cp.asarray(rainblock[s:s + chunksize], dtype=cp.float32)
        spec = cp.fft.rfft2(blk, s=fshape, axes=(-2, -1))
        spec *= kfft[None, :, :]
        conv = cp.fft.irfft2(spec, s=fshape, axes=(-2, -1))
        conv = conv[:, mh - 1:mh - 1 + outh, mw - 1:mw - 1 + outw]
        conv[:, invalid] = -cp.inf
        flat = conv.reshape(conv.shape[0], -1)
        idx = cp.asnumpy(cp.argmax(flat, axis=1))
        rmax[s:s + chunksize] = cp.asnumpy(cp.max(flat, axis=1))
        ymax[s:s + chunksize] = idx // outw
        xmax[s:s + chunksize] = idx % outw
        del blk, spec, conv, flat   # let the pool reuse the block's buffers
    return rmax, ymax, xmax


def catalog_select_confirmed(scantimes, scanmax, scany, scanx, scanfidx, files,
                             nctx, catmax, cattime, catx, caty, catduration,
                             timestep, timeseparation, variables, idxes,
                             dropvars, calendar, time_units, nwindow, trimmask,
                             domainmask, blockfiles=16, delta=1e-3):
    """
    Storm-catalog selection for approximate (GPU) scan values that produces a
    catalog bit-identical to the exact CPU engine ('batchfft').

    GPU values are trusted only as a screen with relative error <= delta. Any
    timestep whose value comes within delta of the running catalog minimum —
    which includes every timestep that gets accepted — has its (rainmax,y,x)
    recomputed with the exact CPU FFT, and the selection restarts. A pass that
    completes without new recomputes is exact: every decision was either
    exact-vs-exact (catalog slots only ever hold values from accepted, i.e.
    exact, timesteps) or a GPU value below the running minimum by more than
    the error bound, which the exact value therefore is too.

    Fills catmax/cattime/catx/caty in place (they must hold initial values).
    """
    nscan = len(scantimes)
    exact = {}              # scan index -> (rainmax, ycat, xcat), CPU-exact
    donefiles = set()
    maxerr = 0.
    catmax0, cattime0 = catmax.copy(), cattime.copy()
    catx0, caty0 = catx.copy(), caty.copy()

    for iteration in range(1, 21):
        wmax, wtime = catmax0.copy(), cattime0.copy()
        wx, wy = catx0.copy(), caty0.copy()
        marked = set()
        for k in range(nscan):
            if k in exact:
                rainmax, ycat, xcat = exact[k]
                isexact = True
            else:
                rainmax, ycat, xcat = float(scanmax[k]), int(scany[k]), int(scanx[k])
                isexact = False
            minind = np.argmin(wmax)
            tempmin = wmax[minind]
            if (not isexact) and rainmax > 0. and rainmax >= tempmin - delta * max(rainmax, tempmin):
                marked.add(k)
            # legacy bookkeeping (optimistic where not exact; the pass only
            # counts when nothing was marked, and then every accepted step
            # was exact)
            if rainmax > tempmin:
                checksep = scantimes[k] - wtime[:, -1]
                starttime = scantimes[k] - np.timedelta64(int(catduration * 60.), 'm')
                subtime = np.arange(scantimes[k], starttime, -timestep)[::-1]
                if (checksep < timeseparation).any():
                    checkind = np.where(checksep < timeseparation)
                    if rainmax >= wmax[checkind]:
                        wmax[checkind] = rainmax
                        wtime[checkind, :] = subtime
                        wx[checkind] = xcat
                        wy[checkind] = ycat
                else:
                    wmax[minind] = rainmax
                    wtime[minind, :] = subtime
                    wx[minind] = xcat
                    wy[minind] = ycat

        if len(marked) == 0:
            catmax[:], cattime[:], catx[:], caty[:] = wmax, wtime, wx, wy
            print('GPU exact-confirm selection: converged after ' + str(iteration)
                  + ' pass(es), ' + str(len(donefiles)) + ' file(s) recomputed exactly, '
                  + 'max GPU-vs-CPU relative error {0:0.2e}'.format(maxerr))
            return

        # exact recompute, grouped into runs of consecutive source files so
        # context files are read once per run instead of once per file; every
        # timestep of a touched file becomes exact (they come for free and
        # speed up convergence)
        needfiles = sorted(set(int(scanfidx[k]) for k in marked) - donefiles)
        if len(needfiles) == 0:
            sys.exit("exact-confirm selection: marked timesteps remained after recomputing their files — this should not happen")
        runs = [[needfiles[0], needfiles[0]]]
        for fi in needfiles[1:]:
            if fi == runs[-1][1] + 1:
                runs[-1][1] = fi
            else:
                runs.append([fi, fi])
        for lo, hi in runs:
            ctxf = files[max(0, lo - nctx):lo]
            etimes, ermax, ey, ex = catalog_scan_files(files[lo:hi + 1], ctxf, variables,
                                                       idxes, dropvars, calendar,
                                                       time_units, timestep, nwindow,
                                                       trimmask, domainmask,
                                                       blockfiles, False, 'batchfft')
            kidx = np.searchsorted(scantimes, etimes)
            if kidx[-1] >= nscan or not np.array_equal(scantimes[kidx], etimes):
                sys.exit("exact-confirm selection: recomputed timesteps do not line up with the scan — check the input files")
            for j, k in enumerate(kidx):
                k = int(k)
                if k not in exact:
                    if ermax[j] > 0.:
                        err = abs(float(scanmax[k]) - float(ermax[j])) / float(ermax[j])
                        if err > maxerr:
                            maxerr = err
                    exact[k] = (float(ermax[j]), int(ey[j]), int(ex[j]))
            donefiles.update(range(lo, hi + 1))

    sys.exit("exact-confirm selection did not converge in 20 passes — use catalog engine 'batchfft' instead")


# ## New version using TensorFlow (For Macs). Added by GP Jul-7 2025
# import tensorflow as tf
# import numpy as np

# def catalogTF_irregular(temparray, trimmask):
#     """
#     GPU version of catalogNumba_irregular using TensorFlow with Metal backend.

#     Parameters:
#     -----------
#     temparray : np.ndarray
#         2D rainfall field (float32)
#     trimmask : np.ndarray
#         2D storm mask kernel (float32)

#     Returns:
#     --------
#     rmax : float
#         Maximum value of convolution
#     ymax, xmax : int
#         Location (row, col) of maximum alignment
#     """

#     # Step 1: Preprocess input (reshape for TF)
#     h, w = temparray.shape
#     mh, mw = trimmask.shape
#     rain_tf = tf.convert_to_tensor(np.nan_to_num(temparray).reshape(1, h, w, 1), dtype=tf.float32)
#     mask_tf = tf.convert_to_tensor(trimmask.reshape(mh, mw, 1, 1), dtype=tf.float32)

#     # Step 2: Perform valid 2D convolution
#     conv_out = tf.nn.conv2d(rain_tf, mask_tf, strides=1, padding='VALID')

#     # Step 3: Extract result
#     conv_np = conv_out.numpy().squeeze()
#     rmax = np.max(conv_np)
#     ymax, xmax = np.unravel_index(np.argmax(conv_np), conv_np.shape)

#     return float(rmax), int(ymax), int(xmax)


# ## New version using CUDA (NVIDIA GPUs). Added by GP Jul-7 2025
# import cupy as cp
# from cupyx.scipy.signal import convolve2d # Small kernels (masks)
# from cupyx.scipy.signal import fftconvolve # Large kernels (masks)  

# def catalog_gpu_cupy(temparray_np, trimmask_np):
#     # Move data to GPU
#     temparray = cp.asarray(temparray_np, dtype=cp.float32)
#     trimmask = cp.asarray(trimmask_np, dtype=cp.float32)

#     h, w = temparray.shape
#     mh, mw = trimmask.shape
#     yh = h - mh + 1
#     xw = w - mw + 1

#     # Create output matrix
#     rainsum = cp.zeros((h, w), dtype=cp.float32)

#     # Slide mask using 2D convolution
#     valmat = cp.nan_to_num(temparray)
#     # rainsum[:yh, :xw] = convolve2d(valmat, trimmask[::-1, ::-1], mode='valid')
#     rainsum[:yh, :xw] = fftconvolve(valmat, trimmask[::-1, ::-1], mode='valid')

#     # Find max
#     max_val = cp.max(rainsum)
#     max_idx = cp.argmax(rainsum)
#     ymax, xmax = cp.unravel_index(max_idx, rainsum.shape)

#     return float(max_val), int(ymax), int(xmax)

# # Note: This version requires xlen and ylen defined in RainyDay_Py3.py defined as follow:
# # xlen =rainprop.subdimensions[1]-maskwidth
# # ylen =rainprop.subdimensions[0]-maskheight-2 
# # That definition is inconsistent since it removes some edges in the padding.
# @jit(nopython=True, fastmath =  True)
# def catalogNumba_irregular(temparray,trimmask,xlen,ylen,xloop,yloop,maskheight,maskwidth,rainsum,stride=1):
#     for y in range(0, int32(yloop)):
#         for x in range(0, int32(xloop),stride):
#             rainsum[y, x] = np.nansum(np.multiply(temparray[y:(y+maskheight), x:(x+maskwidth)], trimmask))
#             rainsum[y, xlen-x-1] = np.nansum(np.multiply(temparray[y:(y+maskheight), xlen-x:(xlen-x+maskwidth)], trimmask))           
#             rainsum[ylen-y-1, x] = np.nansum(np.multiply(temparray[ylen-y-1:(ylen-y-1+maskheight), x:(x+maskwidth)], trimmask))
#             rainsum[ylen-y-1, xlen-x-1] = np.nansum(np.multiply(temparray[ylen-y:(ylen-y+maskheight), xlen-x:(xlen-x+maskwidth)], trimmask))

#     #wheremax=np.argmax(rainsum)
#     rmax=np.nanmax(rainsum)
#     wheremax=np.where(np.equal(rainsum,rmax))
    
#     return rmax, wheremax[0][0], wheremax[1][0]

@jit(nopython=True,  fastmath =  True)
def catalogNumba(temparray,trimmask,xlen,ylen,xloop,yloop,maskheight,maskwidth,rainsum,stride=1):
    for y in range(0, int32(yloop)):
        for x in range(0, int32(xloop),stride):

            rainsum[y, x] = np.nansum(np.multiply(temparray[y:(y+maskheight), x:(x+maskwidth)], trimmask))

            rainsum[y, xlen-x-1] = np.nansum(np.multiply(temparray[y:(y+maskheight), xlen-x:(xlen-x+maskwidth)], trimmask))
            rainsum[ylen-y-1, x] = np.nansum(np.multiply(temparray[ylen-y-1:(ylen-y-1+maskheight), x:(x+maskwidth)], trimmask))


            rainsum[ylen-y-1, xlen-x-1] = np.nansum(np.multiply(temparray[ylen-y:(ylen-y+maskheight), xlen-x:(xlen-x+maskwidth)], trimmask))

    #wheremax=np.argmax(rainsum)
    rmax=np.nanmax(rainsum)
    wheremax=np.where(np.equal(rainsum,rmax))
    return rmax, wheremax[0][0], wheremax[1][0]

# @jit(nopython=True)
# def catalogNumba(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,stride=1):
#     rainsum[:]=0.
#     for i in range(0,(ylen)*(xlen),stride):
#         y=i//xlen
#         x=i-y*xlen
#         #print x,y
#         rainsum[y,x]=np.nansum(np.multiply(temparray[(y):(y+maskheight),(x):(x+maskwidth)],trimmask))

#     #wheremax=np.argmax(rainsum)
#     rmax=np.nanmax(rainsum)
#     wheremax=np.where(np.equal(rainsum,rmax))
#     return rmax, wheremax[0][0], wheremax[1][0]


@jit(nopython=True)
def DistributionBuilder(intenserain,tempmax,xlen,ylen,checksep):
    for y in np.arange(0,ylen):
        for x in np.arange(0,xlen):
            if np.any(checksep[:,y,x]):
                #fixind=np.where(checksep[:,y,x]==True)
                for i in np.arange(0,checksep.shape[0]):
                    if checksep[i,y,x]==True:
                        fixind=i
                        break
                if tempmax[y,x]>intenserain[fixind,y,x]:
                    intenserain[fixind,y,x]=tempmax[y,x]
                    checksep[:,y,x]=False
                    checksep[fixind,y,x]=True
                else:
                    checksep[fixind,y,x]=False
            elif tempmax[y,x]>np.min(intenserain[:,y,x]):
                fixind=np.argmin(intenserain[:,y,x])
                intenserain[fixind,y,x]=tempmax[y,x]
                checksep[fixind,y,x]=True
    return intenserain,checksep

# slightly faster numpy-based version of above
def DistributionBuilderFast(intenserain,tempmax,xlen,ylen,checksep):
    minrain=np.min(intenserain,axis=0)
    if np.any(checksep):
        
        flatsep=np.any(checksep,axis=0)
        minsep=np.argmax(checksep[:,flatsep],axis=0)
        
        islarger=np.greater(tempmax[flatsep],intenserain[minsep,flatsep])
        if np.any(islarger):
            intenserain[minsep,flatsep][islarger]=tempmax[flatsep][islarger]
            checksep[:]=False
            checksep[minsep,flatsep]=True
        else:
            checksep[minsep,flatsep]=False
    elif np.any(np.greater(tempmax,minrain)):
        #else:
        fixind=np.greater(tempmax,minrain)
        minrainind=np.argmin(intenserain,axis=0)
        
        intenserain[minrainind[fixind],fixind]=tempmax[fixind]
        checksep[minrainind[fixind],fixind]=True
    return intenserain,checksep



#def SSTalt(passrain,sstx,ssty,trimmask,maskheight,maskwidth,intense_data=False):
#    rainsum=np.zeros((len(sstx)),dtype='float32')
#   nreals=len(rainsum)
#
#    for i in range(0,nreals):
#        rainsum[i]=np.nansum(np.multiply(passrain[(ssty[i]) : (ssty[i]+maskheight) , (sstx[i]) : (sstx[i]+maskwidth)],trimmask))
#    return rainsum


@jit(fastmath=True)
def SSTalt(passrain,sstx,ssty,trimmask,maskheight,maskwidth,intensemean=None,intensestd=None,intensecorr=None,homemean=None,homestd=None,durcheck=False):
    maxmultiplier=1.5
    rainsum=np.zeros((len(sstx)),dtype='float32')
    whichstep=np.zeros((len(sstx)),dtype='int32')
    nreals=len(rainsum)
    nsteps=passrain.shape[0]
    multiout=np.empty_like(rainsum)
    if (intensemean is not None) and (homemean is not None):
        domean=True
    else:
        domean=False

    if (intensestd is not None) and (intensecorr is not None) and (homestd is not None):
        #rquant=np.random.random_integers(5,high=95,size=nreals)/100.
        rquant=np.random.random_sample(size=nreals)
        doall=True
    else:
        doall=False
        rquant=np.nan
        
    
    if durcheck==False:
        exprain=np.expand_dims(passrain,0)
    else:
        exprain=passrain
        
    for k in range(0,nreals):
        y=int(ssty[k])
        x=int(sstx[k])
        if np.all(np.less(exprain[:,y:y+maskheight,x:x+maskwidth],0.5)):
            rainsum[k]=0.
            multiout[k]=-999.
        else:
            if domean:
                #sys.exit('need to fix short duration part')
                muR=homemean-intensemean[y,x]
                if doall:
                    stdR=np.sqrt(np.power(homestd,2)+np.power(intensestd[y,x],2)-2.*intensecorr[y,x]*homestd*intensestd[y,x])
                   # multiplier=sp.stats.lognorm.ppf(rquant[k],stdR,loc=0,scale=np.exp(muR))     
                    #multiplier=10.
                    #while multiplier>maxmultiplier:       # who knows what the right number is to use here...
                    inverrf=sp.special.erfinv(2.*rquant-1.)
                    multiplier=np.exp(muR+np.sqrt(2.*np.power(stdR,2))*inverrf[k])
                    
                    #multiplier=np.random.lognormal(muR,stdR)
                    if multiplier>maxmultiplier:
                        multiplier=1.    
                else:
                    multiplier=np.exp(muR)
                    if multiplier>maxmultiplier:
                        multiplier=1.
            else:
                multiplier=1.
#            print("still going!")
            if multiplier>maxmultiplier:
                sys.exit("Something seems to be going horribly wrong in the multiplier scheme!")
            else:
                multiout[k]=multiplier
        
            if durcheck==True:            
                storesum=0.
                storestep=0
                for kk in range(0,nsteps):
                    #tempsum=numba_multimask_calc(passrain[kk,:],rsum,train,trimmask,ssty[k],maskheight,sstx[k],maskwidth)*multiplier
                    tempsum=numba_multimask_calc(passrain[kk,:],trimmask,y,x,maskheight,maskwidth)*multiplier
                    if tempsum>storesum:
                        storesum=tempsum
                        storestep=kk
                rainsum[k]=storesum
                whichstep[k]=storestep
            else:
                rainsum[k]=numba_multimask_calc(passrain,trimmask,y,x,maskheight,maskwidth)*multiplier
    
    if domean:
        return rainsum,multiout,whichstep
    else:
        return rainsum,whichstep


#@jit(nopython=True,fastmath=True,parallel=True)
@jit(nopython=True,fastmath=True)
def numba_multimask_calc(passrain,trimmask,ssty,sstx,maskheight,maskwidth):
    train=np.multiply(passrain[ssty : ssty+maskheight , sstx : sstx+maskwidth],trimmask)
    rainsum=np.sum(train)       
    return rainsum


@jit(fastmath=True)
def SSTalt_singlecell(passrain,sstx,ssty,trimmask,maskheight,maskwidth,intensemean=None,intensestd=None,intensecorr=None,homemean=None,homestd=None,durcheck=False):
    rainsum=np.zeros((len(sstx)),dtype='float32')
    whichstep=np.zeros((len(sstx)),dtype='int32')
    nreals=len(rainsum)
    nsteps=passrain.shape[0]
    multiout=np.empty_like(rainsum)

    # do we do deterministic or dimensionless rescaling?
    if (intensemean is not None) and (homemean is not None):
        domean=True
    else:
        domean=False       

    # do we do stochastic rescaling?    
    if (intensestd is not None) and (intensecorr is not None) and (homestd is not None):
        rquant=np.random.random_sample(size=nreals)
        inverrf=sp.special.erfinv(2.*rquant-1.)
        doall=True
    else:
        doall=False
        #rquant=np.nan

    if durcheck==False:
        passrain=np.expand_dims(passrain,0)
       
    # deterministic or dimensionless:
    if domean and doall==False:
        rain,multi,step=killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=durcheck,intensemean=intensemean,homemean=homemean,multiout=multiout)
        return rain,multi,step
    
    # stochastic:
    elif doall:
        rain,multi,step=killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=durcheck,intensemean=intensemean,intensestd=intensestd,intensecorr=intensecorr,homemean=homemean,homestd=homestd,multiout=multiout,inverrf=inverrf)
        return rain,multi,step
    
    # no rescaling:
    else:
        rain,_,step=killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=durcheck,multiout=multiout)
        return rain,step
    


#@jit(nopython=True,fastmath=True,parallel=True)
@jit(nopython=True,fastmath=True)
def killerloop_singlecell(passrain,rainsum,whichstep,nreals,ssty,sstx,nsteps,durcheck=False,intensemean=None,homemean=None,homestd=None,multiout=None,rquant=None,intensestd=None,intensecorr=None,inverrf=None):
    maxmultiplier=1.5  # who knows what the right number is to use here...
    for k in prange(nreals):
        y=int(ssty[k])
        x=int(sstx[k])
        
        # deterministic or dimensionless:
        if (intensemean is not None) and (homemean is not None) and (homestd is None):
            if np.less(homemean,0.001) or np.less(intensemean[y,x],0.001):
                multiplier=1.           # or maybe this should be zero     
            else:
                multiplier=np.exp(homemean-intensemean[y,x])
                if multiplier>maxmultiplier:           
                    multiplier=1.        # or maybe this should be zero
                    
        # stochastic:
        elif (intensemean is not None) and (homemean is not None) and (homestd is not None):
            if np.less(homemean,0.001) or np.less(intensemean[y,x],0.001):
                multiplier=1.          # or maybe this should be zero
            else:
                muR=homemean-intensemean[y,x]
                stdR=np.sqrt(np.power(homestd,2)+np.power(intensestd[y,x],2)-2*intensecorr[y,x]*homestd*intensestd[y,x])

                multiplier=np.exp(muR+np.sqrt(2.*np.power(stdR,2))*inverrf[k])
                if multiplier>maxmultiplier:
                    multiplier=1.        # or maybe this should be zero
        
        # no rescaling:
        else:
            multiplier=1.
            
        if durcheck==False:
            rainsum[k]=np.nansum(passrain[:,y, x])
        else:
            storesum=0.
            storestep=0
            for kk in range(nsteps):
                tempsum=passrain[kk,y,x]
                if tempsum>storesum:
                    storesum=tempsum
                    storestep=kk
            rainsum[k]=storesum*multiplier
            multiout[k]=multiplier
            whichstep[k]=storestep
            
    return rainsum,multiout,whichstep



#@jit(nopython=True,fastmath=True,parallel=True)
#def killerloop(passrain,rainsum,nreals,ssty,sstx,maskheight,maskwidth,trimmask,nsteps,durcheck):
#    for k in prange(nreals):
#        spanx=int64(sstx[k]+maskwidth)
#        spany=int64(ssty[k]+maskheight)
#        if np.all(np.less(passrain[:,ssty[k]:spany,sstx[k]:spanx],0.5)):
#            rainsum[k]=0.
#        else:
#            if durcheck==False:
#                rainsum[k]=np.nansum(np.multiply(passrain[ssty[k] : spany , sstx[k] : spanx],trimmask))
#            else:
#                storesum=float32(0.)
#                for kk in range(nsteps):
#                    tempsum=np.nansum(np.multiply(passrain[kk,ssty[k]:spany,sstx[k]:spanx],trimmask))
#                    if tempsum>storesum:
#                        storesum=tempsum
#                rainsum[k]=storesum
#    return rainsum
    
    
                    #whichstep[k]=storestep
#return rainsum,whichstep



# this function below never worked for some unknown Numba problem-error messages indicated that it wasn't my fault!!! Some problem in tempsum
#@jit(nopython=True,fastmath=True,parallel=True)
#def killerloop(passrain,rainsum,nreals,ssty,sstx,maskheight,maskwidth,masktile,nsteps,durcheck):
#    for k in prange(nreals):
#        spanx=sstx[k]+maskwidth
#        spany=ssty[k]+maskheight
#        if np.all(np.less(passrain[:,ssty[k]:spany,sstx[k]:spanx],0.5)):
#            rainsum[k]=0.
#        else:
#            if durcheck==False:
#                #tempstep=np.multiply(passrain[:,ssty[k] : spany , sstx[k] : spanx],trimmask)
#                #xnum=int64(sstx[k])
#                #ynum=int64(ssty[k])
#                #rainsum[k]=np.nansum(passrain[:,ssty[k], sstx[k]])
#                rainsum[k]=np.nansum(np.multiply(passrain[:,ssty[k] : spany , sstx[k] : spanx],masktile))
#            else:
#                storesum=float32(0.)
#                for kk in range(nsteps):
#                    #tempsum=0.
#                    #tempsum=np.multiply(passrain[kk,ssty[k]:spany,sstx[k]:spanx],masktile[0,:,:])
#                    tempsum=np.nansum(np.multiply(passrain[kk,ssty[k]:spany,sstx[k]:spanx],masktile[0,:,:]))
#    return rainsum


#==============================================================================
# THIS VARIANT IS SIMPLER AND UNLIKE SSTWRITE, IT ACTUALLY WORKS RELIABLY!
#==============================================================================
#def SSTwriteAlt(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth):
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)
#    #ctr=0
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        for j in unqwhere:
#            #ctr=ctr+1
#            #print ctr
#            outrain[j,:]=np.multiply(catrain[unqstm[i],:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],trimmask)
#    return outrain
       

#==============================================================================
# THIS VARIANT IS SAME AS ABOVE, BUT HAS A MORE INTERESTING RAINFALL PREPENDING PROCEDURE
#==============================================================================

#def SSTwriteAltPreCat(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime):    
#    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
#    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1]+precat.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)
#
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        unqmonth=ptime[unqstm[i]]
#        pretimeind=np.where(np.logical_and(ptime>unqmonth-2,ptime<unqmonth+2))[0]
#        for j in unqwhere:
#            temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),catrain[unqstm[i],:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#            outrain[j,:]=np.multiply(temprain,trimmask)
#    return outrain
#    

#==============================================================================
# SAME AS ABOVE, BUT HANDLES STORM ROTATION
#==============================================================================    
    
#def SSTwriteAltPreCatRotation(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,delarray,rlzanglebin,rainprop):
##def SSTwriteAltPreCatRotation(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,delarray,rlzanglebin):
#    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
#    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1]+precat.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)      # unqstm is the storm number
#
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        unqmonth=ptime[unqstm[i]]
#        pretimeind=np.where(np.logical_and(ptime>unqmonth-2,ptime<unqmonth+2))[0]
#        for j in unqwhere:
#            inrain=catrain[unqstm[i],:].copy()
#            
#            xctr=rlzx[j]+maskwidth/2.
#            yctr=rlzy[j]+maskheight/2.
#            xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
#            ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
#    
#            ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
#            ingridx=ingridx.flatten()
#            ingridy=ingridy.flatten()
#            outgrid=np.column_stack((ingridx,ingridy))  
#            
#            for k in range(0,inrain.shape[0]):
#                interp=sp.interpolate.LinearNDInterpolator(delarray[unqstm[i]][rlzanglebin[j]-1],inrain[k,:].flatten(),fill_value=0.)
#                inrain[k,:]=np.reshape(interp(outgrid),rainprop.subdimensions)
#                #inrain[k,:]=temprain
#            
#            temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#
#            outrain[j,:]=np.multiply(temprain,trimmask)
#    return outrain
       
@jit(fastmath=True)
def SSTspin_write_v2(catrain,rlzx,rlzy,rlzstm,trimmask,maskheight,maskwidth,precat,ptime,rainprop,rlzanglebin=None,delarray=None,spin=False,flexspin=True,samptype='uniform',cumkernel=None,rotation=False,domaintype='rectangular'):
    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
    nyrs=np.int16(rlzx.shape[0])
    raindur=np.int16(catrain.shape[1]+precat.shape[1])
    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)      # unqstm is the storm number
    
    for i in range(0,len(unqstm)):
        unqwhere=np.where(unqstm[i]==rlzstm)[0]
        unqmonth=ptime[unqstm[i]]
        pretimeind=np.where(np.logical_and(ptime>unqmonth-1,ptime<unqmonth+1))[0]
        
        # flexspin allows you to use spinup rainfall from anywhere in transposition domain, rather than just storm locations, but it doesn't seem to be very useful based on initial testing
        if spin==True and flexspin==True:       
            if samptype=='kernel' or domaintype=='irregular':
                rndloc=np.random.random_sample(len(unqwhere))
                shiftprex,shiftprey=numbakernel(rndloc,cumkernel)
            else:
                shiftprex=np.random.random_integers(0,np.int16(rainprop.subdimensions[1])-maskwidth-1,len(unqwhere))
                shiftprey=np.random.random_integers(0,np.int16(rainprop.subdimensions[0])-maskheight-1,len(unqwhere))
            
        ctr=0   
        for j in unqwhere:
            inrain=catrain[unqstm[i],:].copy()
                        
            # this doesn't rotate the prepended rainfall
            if rotation==True:
                xctr=rlzx[j]+maskwidth/2.
                yctr=rlzy[j]+maskheight/2.
                xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
                ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
        
                ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
                ingridx=ingridx.flatten()
                ingridy=ingridy.flatten()
                outgrid=np.column_stack((ingridx,ingridy))  
                
                for k in range(0,inrain.shape[0]):
                    interp=sp.interpolate.LinearNDInterpolator(delarray[unqstm[i]][rlzanglebin[j]-1],inrain[k,:].flatten(),fill_value=0.)
                    inrain[k,:]=np.reshape(interp(outgrid),rainprop.subdimensions)
                    
            if spin==True and flexspin==True:
                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(shiftprey[ctr]) : (shiftprey[ctr]+maskheight) , (shiftprex[ctr]) : (shiftprex[ctr]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
            elif spin==True and flexspin==False:
                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
            elif spin==False:
                temprain=inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]
            else:
                sys.exit("what else is there?")
            ctr=ctr+1

            outrain[j,:]=np.multiply(temprain,trimmask)
    return outrain


##==============================================================================
## SAME AS ABOVE, BUT A BIT MORE DYNAMIC IN TERMS OF SPINUP
##==============================================================================    
#def SSTspin_write_v2(catrain,rlzx,rlzy,rlzstm,trimmask,xmin,xmax,ymin,ymax,maskheight,maskwidth,precat,ptime,rainprop,rlzanglebin=None,delarray=None,spin=False,flexspin=True,samptype='uniform',cumkernel=None,rotation=False,domaintype='rectangular',intense_data=False):
#    catyears=ptime.astype('datetime64[Y]').astype(int)+1970
#    ptime=ptime.astype('datetime64[M]').astype(int)-(catyears-1970)*12+1
#    nyrs=np.int(rlzx.shape[0])
#    raindur=np.int(catrain.shape[1]+precat.shape[1])
#    outrain=np.zeros((nyrs,raindur,maskheight,maskwidth),dtype='float32')
#    unqstm,unqind,unqcnts=np.unique(rlzstm,return_inverse=True,return_counts=True)      # unqstm is the storm number
#    
#    if intense_data!=False:
#        sys.exit("Scenario writing for intensity-based resampling not tested!")
#        intquant=intense_data[0]
#        fullmu=intense_data[1]
#        fullstd=intense_data[2]
#        muorig=intense_data[3]
#        stdorig=intense_data[4]
#    
#    for i in range(0,len(unqstm)):
#        unqwhere=np.where(unqstm[i]==rlzstm)[0]
#        unqmonth=ptime[unqstm[i]]
#        pretimeind=np.where(np.logical_and(ptime>unqmonth-1,ptime<unqmonth+1))[0]
#        
#        if transpotype=='intensity':
#            origmu=np.multiply(murain[caty[i]:caty[i]+maskheight,catx[i]:catx[i]+maskwidth],trimmask)
#            origstd=np.multiply(stdrain[caty[i]:caty[i]+maskheight,catx[i]:catx[i]+maskwidth],trimmask)
#            #intense_dat=[intquant[],murain,stdrain,origmu,origstd]
#        
#        # flexspin allows you to use spinup rainfall from anywhere in transposition domain, rather than just storm locations, but it doesn't seem to be very useful based on initial testing
#        if spin==True and flexspin==True:       
#            if samptype=='kernel' or domaintype=='irregular':
#                rndloc=np.random.random_sample(len(unqwhere))
#                shiftprex,shiftprey=numbakernel(rndloc,cumkernel)
#            else:
#                shiftprex=np.random.random_integers(0,np.int(rainprop.subdimensions[1])-maskwidth-1,len(unqwhere))
#                shiftprey=np.random.random_integers(0,np.int(rainprop.subdimensions[0])-maskheight-1,len(unqwhere))
#            
#        ctr=0   
#        for j in unqwhere:
#            inrain=catrain[unqstm[i],:].copy()
#            
#            if intense_data!=False:
#                transmu=np.multiply(fullmu[(rlzy[i]) : (rlzy[i]+maskheight) , (rlzx[i]) : (rlzx[i]+maskwidth)],trimmask)
#                transtd=np.multiply(fullstd[(rlzy[i]) : (rlzy[i]+maskheight) , (rlzx[i]) : (rlzx[i]+maskwidth)],trimmask)
#                mu_multi=transmu/muorig
#                std_multi=np.abs(transtd-stdorig)/stdorig
#                multipliermask=norm.ppf(intquant[i],loc=mu_multi,scale=std_multi)
#                multipliermask[multipliermask<0.]=0.
#                multipliermask[np.isnan(multipliermask)]=0.
#            
#            # this doesn't rotate the prepended rainfall
#            if rotation==True:
#                xctr=rlzx[j]+maskwidth/2.
#                yctr=rlzy[j]+maskheight/2.
#                xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
#                ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
#        
#                ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
#                ingridx=ingridx.flatten()
#                ingridy=ingridy.flatten()
#                outgrid=np.column_stack((ingridx,ingridy))  
#                
#                for k in range(0,inrain.shape[0]):
#                    interp=sp.interpolate.LinearNDInterpolator(delarray[unqstm[i]][rlzanglebin[j]-1],inrain[k,:].flatten(),fill_value=0.)
#                    inrain[k,:]=np.reshape(interp(outgrid),rainprop.subdimensions)
#                    
#            if spin==True and flexspin==True:
#                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(shiftprey[ctr]) : (shiftprey[ctr]+maskheight) , (shiftprex[ctr]) : (shiftprex[ctr]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#            elif spin==True and flexspin==False:
#                temprain=np.concatenate((np.squeeze(precat[np.random.choice(pretimeind, 1),:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)],axis=0),inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]),axis=0)
#            elif spin==False:
#                temprain=inrain[:,(rlzy[j]) : (rlzy[j]+maskheight) , (rlzx[j]) : (rlzx[j]+maskwidth)]
#            else:
#                sys.exit("what else is there?")
#            ctr=ctr+1
#            if intense_data!=False:
#                outrain[j,:]=np.multiply(temprain,multipliermask)
#            else:
#                outrain[j,:]=np.multiply(temprain,trimmask)
#    return outrain
    
    
#==============================================================================
# LOOP FOR KERNEL BASED STORM TRANSPOSITION
# THIS FINDS THE TRANSPOSITION LOCATION FOR EACH REALIZATION IF YOU ARE USING THE KERNEL-BASED RESAMPLER
# IF I CONFIGURE THE SCRIPT SO THE USER CAN PROVIDE A CUSTOM RESAMPLING SCHEME, THIS WOULD PROBABLY WORK FOR THAT AS WELL
#==============================================================================    
#def weavekernel(rndloc,cumkernel):
#    nlocs=len(rndloc)
#    nrows=cumkernel.shape[0]
#    ncols=cumkernel.shape[1]
#    tempx=np.empty((len(rndloc)),dtype="int32")
#    tempy=np.empty((len(rndloc)),dtype="int32")
#    code= """
#        #include <stdio.h>
#        int i,x,y,brklp;
#        double prevprob;
#        for (i=0;i<nlocs;i++) {
#            prevprob=0.0;
#            brklp=0;
#            for (y=0; y<nrows; y++) {
#                for (x=0; x<ncols; x++) {
#                    if ( (rndloc(i)<=cumkernel(y,x)) && (rndloc(i)>prevprob) ) {
#                        tempx(i)=x;
#                        tempy(i)=y;
#                        prevprob=cumkernel(y,x);
#                        brklp=1;
#                        break;
#                    }                     
#                }
#                if (brklp==1) {
#                    break;                    
#                }                         
#            }   
#        }
#    """
#    vars=['rndloc','cumkernel','nlocs','nrows','ncols','tempx','tempy']
#    sp.weave.inline(code,vars,type_converters=converters.blitz,compiler='gcc')
#    return tempx,tempy
    
    
def pykernel(rndloc,cumkernel):
    nlocs=len(rndloc)
    ncols=cumkernel.shape[1]
    tempx=np.empty((len(rndloc)),dtype="int32")
    tempy=np.empty((len(rndloc)),dtype="int32")
    flatkern=np.append(0.,cumkernel.flatten())
    
    for i in range(0,nlocs):
        x=rndloc[i]-flatkern
        x[np.less(x,0.)]=1000.
        whereind = np.argmin(x)
        y=whereind//ncols
        x=whereind-y*ncols        
        tempx[i]=x
        tempy[i]=y
    return tempx,tempy

@jit 
def numbakernel(rndloc,cumkernel,tempx,tempy,ncols):
    nlocs=len(rndloc)
    #ncols=xdim
    flatkern=np.append(0.,cumkernel.flatten())
    #x=np.zeros_like(rndloc,dtype='float64')
    for i in np.arange(0,nlocs):
        x=rndloc[i]-flatkern
        x[np.less(x,0.)]=10.
        whereind=np.argmin(x)
        y=whereind//ncols
        x=whereind-y*ncols 
        tempx[i]=x
        tempy[i]=y
    return tempx,tempy


@jit 
def numbakernel_fast(rndloc,cumkernel,tempx,tempy,ncols):
    nlocs=int32(len(rndloc))
    ncols=int32(cumkernel.shape[1])
    flatkern=np.append(0.,cumkernel.flatten()) 
    return kernelloop(nlocs,rndloc,flatkern,ncols,tempx,tempy)

#@jit(nopython=True,fastmath=True,parallel=True)
@jit(nopython=True,fastmath=True)
def kernelloop(nlocs,rndloc,flatkern,ncols,tempx,tempy):
    for i in prange(nlocs):
        diff=rndloc[i]-flatkern
        diff[np.less(diff,0.)]=10.
        whereind=np.argmin(diff)
        y=whereind//ncols
        x=whereind-y*ncols 
        tempx[i]=x
        tempy[i]=y
    return tempx,tempy



#==============================================================================
# FIND THE BOUNDARY INDICIES AND COORDINATES FOR THE USER-DEFINED SUBAREA
# NOTE THAT subind ARE THE MATRIX INDICIES OF THE SUBBOX, STARTING FROM UPPER LEFT CORNER OF DOMAIN AS (0,0)
# NOTE THAT subcoord ARE THE COORDINATES OF THE OUTSIDE BORDER OF THE SUBBOX
# THEREFORE THE DISTANCE FROM THE WESTERN (SOUTHERN) BOUNDARY TO THE EASTERN (NORTHERN) BOUNDARY IS NCOLS (NROWS) +1 TIMES THE EAST-WEST (NORTH-SOUTH) RESOLUTION
#============================================================================== 


def findsubbox(inarea,variables,fname):
    outextent = np.empty([4])
    outdim=np.empty([2], dtype= 'int')
    infile=xr.open_dataset(fname)
    latmin,latmax,longmin,longmax = inarea[2],inarea[3],inarea[0],inarea[1]
    rain_name,lat_name,lon_name = variables.values()
    if max(infile[lon_name].values) > 180: # convert from positive degrees west to negative degrees west
        infile[lon_name] = infile[lon_name] - 360 
        # inarea[:2] += 360
        # latmin,latmax,longmin,longmax = inarea[2],inarea[3],inarea[0],inarea[1]
    outrain=infile[rain_name].sel(**{lat_name:slice(latmin,latmax)},\
                                              **{lon_name:slice(longmin,longmax)})
    outextent[2], outextent[3],outextent[0], outextent[1]=outrain[lat_name][0],outrain[lat_name][-1],\
                                outrain[lon_name][0], outrain[lon_name][-1]       
    outdim[0], outdim[1] = len(outrain[lat_name]), len(outrain[lon_name])
    lat = infile[lat_name]; lon = infile[lon_name] 
    min_latidx = np.where((lat >= latmin) & (lat <= latmax))[0][0]
    max_latidx = np.where((lat >= latmin) & (lat <= latmax))[0][-1]
    min_lonidx = np.where((lon >= longmin) & (lon <= longmax))[0][0]
    max_lonidx = np.where((lon >= longmin) & (lon <= longmax))[0][-1]
    indices = np.array([min_latidx, max_latidx, min_lonidx, max_lonidx])
    infile.close()
    return outextent, outdim, outrain[lat_name], outrain[lon_name], indices
    
    
    

#==============================================================================
# THIS RETURNS A LOGICAL GRID THAT CAN THEN BE APPLIED TO THE GLOBAL GRID TO EXTRACT
# A USER-DEFINED SUBGRID
# THIS HELPS TO KEEP ARRAY SIZES SMALL
#==============================================================================
def creategrids(rainprop):
    globrangex=np.arange(0,rainprop.dimensions[1],1)
    globrangey=np.arange(0,rainprop.dimensions[0],1)
    subrangex=np.arange(rainprop.subind[0],rainprop.subind[1]+1,1)
    subrangey=np.arange(rainprop.subind[3],rainprop.subind[2]+1,1)
    subindx=np.logical_and(globrangex>=subrangex[0],globrangex<=subrangex[-1])
    subindy=np.logical_and(globrangey>=subrangey[0],globrangey<=subrangey[-1])
    gx,gy=np.meshgrid(subindx,subindy)
    outgrid=np.logical_and(gx==True,gy==True)
    return outgrid,subindx,subindy


#==============================================================================
# FUNCTION TO CREATE A MASK ACCORDING TO A USER-DEFINED POLYGON SHAPEFILE AND PROJECTION
#==============================================================================
def rastermask(shpname,rainprop,masktype='simple',dissolve=True,ngenfile=False):            
    bndcoords=np.array(rainprop.subextent)
    
    xdim=rainprop.subdimensions[0]  
    ydim=rainprop.subdimensions[1] 
    
    if ngenfile:
        sys.exit("this isn't ready yet")
        project = pyproj.Transformer.from_proj(pyproj.Proj(init='epsg:4326'),pyproj.Proj(init='epsg:5070'))

    # this appears to work even if the shapefile has multiple polygons... it seems to just take the outline
    with fiona.open(shpname, "r") as shapefile:
        shapes = [shape(feature["geometry"]) for feature in shapefile]
        
        # trouble figuring out how to reproject a geojson file to WGS84
        #temp=shape(shapefile)
        #shapes=[]
        #for feature in shapefile:
        #    if ngenfile:
        #        temp=shape(shape(feature["geometry"]))
        #        t1=transform(project.transform, temp)
        #    else:
        #        shapes.append(shape(feature["geometry"]))
            
        
    
    if masktype=='simple':
        print('creating simple mask (0s and 1s)')
        trans = from_origin(bndcoords[0], bndcoords[3], rainprop.spatialres[0], rainprop.spatialres[1])
        rastertemplate=np.ones((ydim,xdim),dtype='float32')
        
        memfile = MemoryFile()
        rastermask = memfile.open(driver='GTiff',
                                 height = rastertemplate.shape[1], width = rastertemplate.shape[0],
                                 count=1, dtype=str(rastertemplate.dtype),
                                 crs='+proj=longlat +datum=WGS84 +no_defs',
                                 transform=trans)
        rastermask.write(rastertemplate,1)
        simplemask, out_transform = mask(rastermask, shapes, crop=False,all_touched=True)
        rastertemplate=simplemask[0,:]

    elif masktype=="fraction":
        print('creating fractional mask (range from 0.0-1.0)')
        n=10
        trans = from_origin(bndcoords[0], bndcoords[3], rainprop.spatialres[0]/np.float32(n), rainprop.spatialres[1]/np.float32(n))
        rastertemplate=np.ones((ydim,xdim),dtype='float32')

        memfile = MemoryFile()
        rastermask = memfile.open(driver='GTiff',
                                 height = n*rastertemplate.shape[1], width = n*rastertemplate.shape[0],
                                 count=1, dtype=str(rastertemplate.dtype),
                                 crs='+proj=longlat +datum=WGS84 +no_defs',
                                 transform=trans)
        rastermask.write(rastertemplate,1)
        simplemask, out_transform = mask(rastermask, shapes, crop=False,all_touched=True)
        rastertemplate=simplemask[0,:]
        
        from scipy.signal import convolve2d
        
        kernel = np.ones((n, n))
        convolved = convolve2d(rastertemplate, kernel, mode='valid')
        rastertemplate=convolved[::n, ::n] / n /n 
        
    else:
        sys.exit("You entered an incorrect mask type, options are 'simple' or 'fraction'")
    #delete('temp9999.tif')   
    return rastertemplate   



#==============================================================================
# WRITE SCENARIOS TO NETCDF ONE REALIZATION AT A TIME
#==============================================================================
def writerealization(scenarioname,rlz,nrealizations,writename,outrain,writemax,writestorm,writeperiod,writex,writey,writetimes,latrange,lonrange,whichorigstorm):
    # SAVE outrain AS NETCDF FILE
    dataset=Dataset(writename, 'w', format='NETCDF4')

    # create dimensions
    outlats=dataset.createDimension('latitude',len(latrange))
    outlons=dataset.createDimension('longitude',len(lonrange))
    time=dataset.createDimension('time',writetimes.shape[1])
    nyears=dataset.createDimension('nyears',len(writeperiod))

    # create variables
    times=dataset.createVariable('time',np.float64, ('nyears','time'))
    latitudes=dataset.createVariable('latitude',np.float32, ('latitude'))
    longitudes=dataset.createVariable('longitude',np.float32, ('longitude'))
    rainrate=dataset.createVariable('precrate',np.float32,('nyears','time','latitude','longitude'),zlib=True,complevel=4,least_significant_digit=1) 
    basinrainfall=dataset.createVariable('basinrainfall',np.float32,('nyears')) 
    xlocation=dataset.createVariable('xlocation',np.int16,('nyears')) 
    ylocation=dataset.createVariable('ylocation',np.int16,('nyears')) 
    returnperiod=dataset.createVariable('returnperiod',np.float32,('nyears')) 
    stormnumber=dataset.createVariable('stormnumber',np.int16,('nyears'))
    original_stormnumber=dataset.createVariable('original_stormnumber',np.int16,('nyears'))
    #stormtimes=dataset.createVariable('stormtimes',np.float64,('nyears'))          
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    rainrate.units = 'mm hr^-1'
    times.units = 'minutes since 1970-01-01 00:00.0'
    times.calendar = 'gregorian'
    basinrainfall.units='mm'
    xlocation.units='dimensionless'
    ylocation.units='dimensionless'
    returnperiod.units='years'
    stormnumber.units='dimensionless'
    original_stormnumber.units='dimensionless'
    
    times.long_name='time'
    latitudes.long_name='latitude'
    longitudes.long_name='longitude'
    rainrate.long_name='precipitation rate'
    basinrainfall.long_name='storm total basin averaged precipitation'
    xlocation.long_name='x index of storm'
    ylocation.long_name='y index of storm'
    returnperiod.long_name='return period of storm total rainfall'
    stormnumber.long_name='storm rank, from 1 to NYEARS'
    original_stormnumber.long_name='parent storm number from storm catalog'
    
    
    
    # Global Attributes
    dataset.description = 'SST Rainfall Scenarios Realization: '+str(rlz+1)+' of '+str(nrealizations)
    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'Realization '+str(rlz)+' from scenario '+scenarioname
    dataset.missing='-9999.'

    # fill the netcdf file
    latitudes[:]=latrange[::-1]
    longitudes[:]=lonrange
    outrain[np.isnan(outrain)]=-9999.
    rainrate[:]=outrain[:,:,::-1,:] 
    basinrainfall[:]=writemax
    times[:]=writetimes
    xlocation[:]=writex
    ylocation[:]=writey
    stormnumber[:]=writestorm
    returnperiod[:]=writeperiod
    original_stormnumber[:]=whichorigstorm
    #stormtimes[:]=writetimes
    
    dataset.close()
    
    
    
#==============================================================================
# WRITE SCENARIOS TO NETCDF ONE REALIZATION AT A TIME-USING THE NPERYEAR OPTION
#==============================================================================    
def writerealization_nperyear(scenarioname,writename,rlz,nperyear,nrealizations,outrain_large,outtime_large,subrangelat,subrangelon,rlz_order,nsimulations):
    # SAVE outrain AS NETCDF FILE
    #filename=writename+'_SSTrealization'+str(rlz+1)+'_Top'+str(nperyear)+'.nc'
    dataset=Dataset(scenarioname, 'w', format='NETCDF4')

    # create dimensions
    outlats=dataset.createDimension('latitude',len(subrangelat))
    outlons=dataset.createDimension('longitude',len(subrangelon))
    time=dataset.createDimension('time',outtime_large.shape[2])
    nyears=dataset.createDimension('nyears',nsimulations)
    topN=dataset.createDimension('nperyear',nperyear)

    # create variables
    times=dataset.createVariable('time',np.float64, ('nyears','nperyear','time'))
    latitudes=dataset.createVariable('latitude',np.float32, ('latitude'))
    longitudes=dataset.createVariable('longitude',np.float32, ('longitude'))
    rainrate=dataset.createVariable('precrate',np.float32,('nyears','nperyear','time','latitude','longitude'),zlib=True,complevel=4,least_significant_digit=1) 
    top_event=dataset.createVariable('top_event',np.int16, ('nyears'))
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    rainrate.units = 'mm hr^-1'
    times.units = 'minutes since 1970-01-01 00:00.0'
    times.calendar = 'gregorian'
    top_event.units='dimensionless'
    
    times.long_name='time'
    latitudes.long_name='latitude'
    longitudes.long_name='longitude'
    rainrate.long_name='precipitation rate'
    top_event.long_name='largest event (storm number) of synthetic year'
    
    
    # Global Attributes
    dataset.description = 'NPERYEAR-type SST Rainfall Scenarios Realization: '+str(rlz+1)+' of '+str(nrealizations)
    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'Realization '+str(rlz)+' from scenario '+scenarioname
    dataset.missing='-9999.'

    # fill the netcdf file
    latitudes[:]=subrangelat[::-1]   # need to check this!
    longitudes[:]=subrangelon
    outrain_large[np.isnan(outrain_large)]=-9999.
    rainrate[:]=outrain_large[:,:,:,::-1,:]
    times[:]=outtime_large
    n_evnet = np.nansum(rlz_order>=0,axis=0)
    n_evnet[n_evnet>=nperyear]=nperyear
    top_event[:]= n_evnet
    dataset.close()
    
#==============================================================================
# WRITE The maximized storm
#==============================================================================
def writemaximized(scenarioname,writename,outrain,writemax,write_ts,writex,writey,writetimes,latrange,lonrange):
    # SAVE outrain AS NETCDF FILE
    dataset=Dataset(writename, 'w', format='NETCDF4')

    # create dimensions
    outlats=dataset.createDimension('latitude',len(latrange))
    outlons=dataset.createDimension('longitude',len(lonrange))
    time=dataset.createDimension('time',len(writetimes))

    # create variables
    times=dataset.createVariable('time',np.float64, ('time'))
    latitudes=dataset.createVariable('latitude',np.float32, ('latitude'))
    longitudes=dataset.createVariable('longitude',np.float32, ('longitude'))
    rainrate=dataset.createVariable('precrate',np.float32,('time','latitude','longitude'),zlib=True,complevel=4,least_significant_digit=1) 
    basinrainfall=dataset.createVariable('basinrainfall',np.float32) 
    xlocation=dataset.createVariable('xlocation',np.int16) 
    ylocation=dataset.createVariable('ylocation',np.int16) 
    #stormtimes=dataset.createVariable('stormtimes',np.float64,('nyears'))          
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    rainrate.units = 'mm hr^-1'
    times.units = 'minutes since 1970-01-01 00:00.0'
    times.calendar = 'gregorian'
    xlocation.units='dimensionless'
    ylocation.units='dimensionless'
    basinrainfall.units='mm'
    
    times.long_name='time'
    latitudes.long_name='latitude'
    longitudes.long_name='longitude'
    rainrate.long_name='precipitation rate'
    basinrainfall.long_name='storm total basin averaged precipitation'
    xlocation.long_name='x index of storm'
    ylocation.long_name='y index of storm'
    
    # Global Attributes
    dataset.description = 'SST Rainfall Maximum Storm'
    dataset.missing='-9999.'
    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = "RainyDay Y'all!"
    

    
    #print dataset.description
    #print dataset.history
    
    # fill the netcdf file
    latitudes[:]=latrange[::-1]
    longitudes[:]=lonrange
    outrain[np.isnan(outrain)]=-9999.
    rainrate[:]=outrain[:,::-1,:]
    basinrainfall[:]=writemax
    times[:]=writetimes
    xlocation[:]=writex
    ylocation[:]=writey
    
    dataset.close()
        
        

#==============================================================================
# READ RAINFALL FILE FROM NETCDF (ONLY FOR RAINYDAY NETCDF-FORMATTED DAILY FILES!
#==============================================================================

# def readnetcdf(rfile,variables,index = None,dropvars=False):
#     """
#     Used to trim the dataset with defined inbounds or transposition domain

#     Parameters
#     ----------
#     rfile : Dataset file path ('.nc' file)
#         This is the path to the dataset
#     variables : TYPE
#         DESCRIPTION.
#     inbounds : TYPE, optional
#         DESCRIPTION. The default is False.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
#     rain_name,lat_name,lon_name = variables.values()
#     if index is not None:
#         infile = Dataset(rfile, mode='r')
#         outrain = np.array(infile.variables[rain_name][:,index[0]:index[1]+1, index[3]:index[3]+1])
#         outtime = np.array(infile.variables['time'][:], dtype='datetime64[m]')
#         infile.close()
#     else:    
#         infile = xr.open_dataset(rfile, drop_variables=dropvars) if dropvars else xr.open_dataset(rfile)  # added DBW 07282023 to avoid reading in unnecessary variables
        
#         if max(infile[lon_name].values) > 180: # convert from positive degrees west to negative degrees west
#             infile[lon_name] = infile[lon_name] - 360 
#         # if np.any(inbounds!=False):
#         #     latmin,latmax,longmin,longmax = inbounds[2],inbounds[3],inbounds[0],inbounds[1]
#         #     outrain=infile[rain_name].sel(**{lat_name:slice(latmin,latmax)},\
#         #                                               **{lon_name:slice(longmin,longmax)})
#         # else:
#         outrain=infile[rain_name]
#         outlatitude=outrain[lat_name]
#         outlongitude=outrain[lat_name] 
#         outtime=np.array(infile['time'], dtype='datetime64[m]')
#         infile.close()
#     if index is not None:
#         return outrain,outtime
        
#     else:
#         return np.array(outrain),outtime,np.array(outlatitude),np.array(outlongitude)
def find_indices(rfile,inarea,variables):
    ds = Dataset(rfile, 'r')
    rain_name, lat_name, lon_name = variables.values()

    # Extract the latitude, longitude, and variable data
    lat = ds.variables[lat_name][:]
    lon = ds.variables[lon_name][:]
    if max(ds.variables[lon_name]) > 180: # convert from positive degrees west to negative degrees west
        lon = lon - 360
    # Specify the latitude and longitude bounds for clipping
    lat_min, lat_max = inarea[2],inarea[3]  # Example latitude range
    lon_min, lon_max = inarea[0],inarea[1]  # Example longitude range

    # Find the indices corresponding to the specified bounds
    lat_inds = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_inds = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    return [lat_inds.min(),lat_inds.max(),lon_inds.min(), lon_inds.max()]

def readnetcdf(rfile,variables,idxes=False,dropvars=False,setup=False,calendar=False,time_units=False,):
    """
    Used to trim the dataset with defined inbounds or transposition domain

    Parameters
    ----------
    rfile : Dataset file path ('.nc' file)
        This is the path to the dataset
    variables : TYPE
        DESCRIPTION.
    inbounds : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # infile = xr.open_dataset(rfile, drop_variables=dropvars,chunks='auto').load() if dropvars else xr.open_dataset(rfile).load()  # added DBW 07282023 to avoid reading in unnecessary variables
    rain_name,lat_name,lon_name = variables.values()
    if np.any(idxes!=False):
        # latmin,latmax,longmin,longmax = inbounds[2],inbounds[3],inbounds[0],inbounds[1]
        # outrain=infile[rain_name].sel(**{lat_name:slice(latmin,latmax)},\
        #                                           **{lon_name:slice(longmin,longmax)})

        infile = Dataset(rfile, 'r') ;
        outrain = infile.variables[rain_name][:, idxes[0]:idxes[1]+1, idxes[2]:idxes[3]+1];
        time_var = infile.variables['time'];time_converted = num2date(time_var, units=time_var.units, calendar=calendar);
        outtime = np.array(time_converted, dtype='datetime64[m]')
        infile.close()

        # # --- HERE
        # # Open dataset with Dask-enabled lazy loading
        # # Set the scheduler to use threads (great for I/O-bound tasks like NetCDF reads)
        # dask.config.set(scheduler='threads')
        # ds = xr.open_dataset(rfile,  decode_times=False, chunks={'time': 8, 'latitude': 295, 'longitude': 590})
        #  # Subset the rainfall variable lazily
        # outrain_lazy = ds[rain_name].isel(
        #     latitude=slice(idxes[0], idxes[1]+1),
        #     longitude=slice(idxes[2], idxes[3]+1)
        # )
        
        # # Load data into memory
        # outrain = outrain_lazy.compute().values
        # # Convert time using provided calendar and units
        # time_var = ds['time']
        # time_converted = num2date(time_var.values, units=time_units, calendar=calendar)
        # outtime = np.array(time_converted, dtype='datetime64[m]')
        # ds.close()

    else:
        infile = xr.open_dataset(rfile, drop_variables=dropvars, chunks='auto').load() if dropvars else xr.open_dataset(rfile).load()
        ncfile = Dataset(rfile, 'r') ;
        nctime = ncfile.variables['time']
        if max(infile[lon_name].values) > 180: # convert from positive degrees west to negative degrees west
            infile[lon_name] = infile[lon_name] - 360
        outrain=infile[rain_name]
        outlatitude=outrain[lat_name]
        outlongitude=outrain[lon_name]
        outtime=np.array(infile['time'],dtype='datetime64[m]')
        infile.close()
    
    if setup:
        return np.array(outrain),outtime,np.array(outlatitude),np.array(outlongitude),nctime
    else:
        return  outrain,outtime
  
  
#==============================================================================
# READ RAINFALL FILE FROM NETCDF
#==============================================================================

def readcatalog(rfile) :
    """
    Returns the properties of the storm including spatial range, storm center,
    storm depth, storm time by reading the already created storm catalogs.

    Parameters
    ----------
    rfile : string
        This takes in the path of the source file.

    Returns
    -------
    arrays
        returns all the properties of a storm including storm rain array, storm time, storm depth, storm center and the extent of the transpositio domain.
        The all storms cattime, catmax, catx and caty are also returned.

    """
    # infile=xr.open_dataset(rfile, engine='h5netcdf')
    infile=xr.open_dataset(rfile)
    outrain=infile['rain']
    outlatitude=infile['latitude']
    outmask=infile['gridmask']
    domainmask=np.array(infile['domainmask'])
    stormtime=np.array(infile['time'],dtype='datetime64[m]')
    outlongitude=infile['longitude']
    outlocx=np.array(infile['xlocation'])
    outlocy=np.array(infile['ylocation'])
    outmax=np.array(infile['basinrainfall'])
    cattime = np.array(infile['cattime'],dtype='datetime64[m]')

    try:
        timeresolution=np.int16(infile.timeresolution)
        resexists=True
    except:
        resexists=False
    infile.close()
    
    if resexists:
        	return outrain,stormtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask,cattime,timeresolution
    else:
        return outrain,stormtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask,cattime

def check_time(datetime_obj):
    """
    

    Parameters
    ----------
    datetime_obj : numpy datetime object
        Datetime object to check the time component of the object

    Returns
    -------
    boolean
        returns True if the datetime object is either '12:00:00' or '00:00:00' 
        else false

    """
    time_str = str(datetime_obj).split('T')[1][:8]  # Extract the time part
    return time_str == '00:00' or time_str == '12:00:00' or time_str == '00:00:00' or time_str =='12:00'

#==============================================================================
# WRITE RAINFALL FILE TO NETCDF
#==============================================================================


#RainyDay.writecatalog(scenarioname,catrain,catmax,catx,caty,cattime,latrange,lonrange,catalogname,nstorms,catmask,parameterfile,domainmask,timeresolution=rainprop.timeres)   
def writecatalog(scenarioname, catrain, catmax, catx, caty, cattime, latrange, lonrange, catalogname, gridmask,
                 parameterfile, dmask, nstorms, duration,storm_num,timeresolution=False):
    print(parameterfile)
    with open(parameterfile,'r') as f:
        params = json.loads(f.read())
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes_units,longitudes_units = 'degrees_north', 'degrees_east'
    rainrate_units,basinrainfall_units = 'mm hr^-1', 'mm'
    times_units, times_calendar = 'minutes since 1970-01-01 00:00.0' , 'gregorian'
    # Variable Names
    times_name = 'time'                     ## change here
    latitudes_name,longitudes_name  = 'latitude', 'longitude'
    rainrate_name, basinrainfall_name = 'precipitation rate', 'storm total basin averaged precipitation'
    xlocation_name, ylocation_name = 'x index of storm', 'y index of storm'
    gmask_name, domainmask_name = 'mask for Aw (control volume)', 'mask for transposition domain'
    if timeresolution!=False:
        timeresolution=timeresolution
    else:
        timeresolution = "None"
    catrain[np.isnan(catrain)] = -9999.
    history, missing = 'Created ' + str(datetime.now()), '-9999.'
    source = 'RainyDay Storm Catalog for scenario ' + scenarioname + '. See description for JSON file contents.'
    data_vars = dict(
                     rain = (("time","latitude", "longitude"),catrain,{'units': rainrate_units, 'long_name': rainrate_name}),
                    # rain = (("time","latitude", "longitude"),catrain[:, ::-1, :],{'units': rainrate_units, 'long_name': rainrate_name}),
                     basinrainfall = (("storm_dim"),catmax.reshape(nstorms),{'units': basinrainfall_units, 'long_name': basinrainfall_name}),
                     xlocation = (("storm_dim"),catx.reshape(nstorms),{'units': 'dimensionless', 'long_name': xlocation_name}),
                     ylocation = (("storm_dim"),caty.reshape(nstorms),{'units': 'dimensionless', 'long_name': ylocation_name}),
                     cattime = (("storm_dim","time"), cattime),
                     gridmask= (("latitude", "longitude"), gridmask,{'units': 'dimensionless', 'long_name': gmask_name}),
                     domainmask = (("latitude", "longitude"),dmask,{'units': 'dimensionless', 'long_name': domainmask_name}),
                     # gridmask= (("latitude", "longitude"), gridmask[::-1, :],{'units': 'dimensionless', 'long_name': gmask_name}),
                     # domainmask = (("latitude", "longitude"),dmask[::-1, :],{'units': 'dimensionless', 'long_name': domainmask_name}),
                     timeresolution = ((), timeresolution) )
    coords = dict(time = ((times_name),cattime[storm_num,:]),
                  longitude = (("longitude"), lonrange.data , {'units': longitudes_units, 'long_name': longitudes_name}),
                  latitude =  (("latitude"), latrange.data, {'units': latitudes_units, 'long_name': latitudes_name}),
                  #latitude =  (("latitude"), latrange[::-1].data, {'units': latitudes_units, 'long_name': latitudes_name}),
                  )
    

    attrs  = dict(history =history, source =  source, missing = missing, description = str(params),  calendar = times_calendar)
    catalog = xr.Dataset(data_vars = data_vars, coords = coords, attrs = attrs)
    catalog.time.encoding['units'] = "minutes since 1970-01-01 00:00:00"

    catalog.to_netcdf(catalogname)
    catalog.close()


def writeintensityfile(scenarioname,intenserain,filename,latrange,lonrange,intensetime):
    # SAVE outrain AS NETCDF FILE
    dataset=Dataset(filename, 'w', format='NETCDF4')
    
    # create dimensions
    outlats=dataset.createDimension('latitude',intenserain.shape[1])
    outlons=dataset.createDimension('longitude',intenserain.shape[2])
    nstorms=dataset.createDimension('nstorms',intenserain.shape[0])

    # create variables
    latitudes=dataset.createVariable('latitude',np.float32, ('latitude',))
    longitudes=dataset.createVariable('longitude',np.float32, ('longitude',))
    stormtotals=dataset.createVariable('stormtotals',np.float32,('nstorms','latitude','longitude',),zlib=True,complevel=4,least_significant_digit=1)
    times=dataset.createVariable('time',np.float64, ('nstorms','latitude','longitude',))

    dataset.Conventions ='CF1.8'
    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'RainyDay Storm Intensity File for scenario '+scenarioname
    dataset.description = 'this description should be improved :)!'
    dataset.missing='-9999.'
    
    times.long_name='time'
    latitudes.long_name='latitude'
    longitudes.long_name='longitude'
    stormtotals.long_name='storm total rainfall'

    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    stormtotals.units = 'mm'
    times.units = 'minutes since 1970-01-01 00:00.0'

    # fill the netcdf file
    latitudes[:]=latrange[::-1]
    longitudes[:]=lonrange
    intenserain[np.isnan(intenserain)]=-9999.
    stormtotals[:]=intenserain[:,::-1,:]
    times[:]=intensetime[:,::-1,:]
    dataset.close()
    
    
def readintensityfile(rfile,inbounds=False):
    infile=Dataset(rfile,'r')
    sys.exit("need to make sure that all CF-related file formatting issues are solved. This main revolves around flipping the rainfall vertically, and perhaps the latitude array as well.")
    if np.any(inbounds!=False):
        outrain=np.array(infile.variables['stormtotals'][:,inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1])
        outtime=np.array(infile.variables['time'][:,inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1],dtype='datetime64[m]')
        outlat=np.array(infile.variables['latitude'][inbounds[3]:inbounds[2]+1])
        outlon=np.array(infile.variables['longitude'][inbounds[0]:inbounds[1]+1])
    else:
        outrain=np.array(infile.variables['stormtotals'][:])
        outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
        outlat=np.array(infile.variables['latitude'][:])
        outlon=np.array(infile.variables['longitude'][:])        
    infile.close()
    return outrain,outtime,outlat,outlon

def readmeanfile(rfile,inbounds=False):
    infile=Dataset(rfile,'r')
    sys.exit("need to make sure that all CF-related file formatting issues are solved. This main revolves around flipping the rainfall vertically, and perhaps the latitude array as well.")
 
    if np.any(inbounds!=False):
        outrain=np.array(infile.variables['stormtotals'][inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1])
        outlat=np.array(infile.variables['latitude'][inbounds[3]:inbounds[2]+1])
        outlon=np.array(infile.variables['longitude'][inbounds[0]:inbounds[1]+1])
    else:
        outrain=np.array(infile.variables['stormtotals'][:])
        outlat=np.array(infile.variables['latitude'][:])
        outlon=np.array(infile.variables['longitude'][:])        
    infile.close()
    return outrain,outlat,outlon


def writedomain(domain,mainpath,latrange,lonrange,parameterfile):
    # SAVE outrain AS NETCDF FILE
    sys.exit("need to make sure that all CF-related file formatting issues are solved. This main revolves around flipping the rainfall vertically, and perhaps the latitude array as well.")
 
    dataset=Dataset(mainpath, 'w', format='NETCDF4')

    # create dimensions
    outlats=dataset.createDimension('latitude',domain.shape[0])
    outlons=dataset.createDimension('longitude',domain.shape[1])

    # create variables
    latitudes=dataset.createVariable('latitude',np.float32, ('latitude',))
    longitudes=dataset.createVariable('longitude',np.float32, ('longitude',))
    domainmap=dataset.createVariable('domain',np.float32,('latitude','longitude',))
    
    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'RainyDay Storm Transposition Domain Map File'
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    domainmap.units = 'dimensionless'
    
    # fill the netcdf file
    latitudes[:]=latrange
    longitudes[:]=lonrange
    domainmap[:]=domain
    
    with open(parameterfile, "r") as myfile:
        params=myfile.read()
    myfile.close
    dataset.description=params
    
    dataset.close()

# =============================================================================
# added Ashar 08162023: To extract numbers from storm files.
# =============================================================================
def extract_storm_number(file_path, catalogname):
    """
    Extracts the storm number from the filename of a storm file, using the catalog name for precise matching.

    Parameters
    ----------
    file_path : string
        File path for the storms .nc files.
    catalogname : string
        The specific prefix (catalog name) to match in the filename.

    Returns
    -------
    integer
        Returns the storm number from the path given in "file_path", or 0 if not found.
    """
    base_name = os.path.basename(file_path)
    pattern = re.escape(catalogname)+'_storm' + r'_(\d+)_\d{8}\.nc$'
    match = re.search(pattern, base_name)
    if match:
        return int(match.group(1))
    return 0
# def extract_storm_number(file_path, catalogname):
#     """
    

#     Parameters
#     ----------
#     file_path : string
#         File path for the storms .nc files
#     catalogname : string
#         Name of the storm catalog given in JSON file

#     Returns
#     -------
#     integer
#         returns the storm number from the path given in "file_path".

#     """
#     base_name = os.path.basename(file_path)
#     match = re.search(catalogname +r'(\d+)', base_name)
#     if match:
#         return np.int32(match.group(1))
#     return 0  

def extract_date(file_path, catalogname):
    """
    Extracts the date from the filename of a storm file, using the catalog name for precise matching.

    Parameters
    ----------
    file_path : string
        File path for the storm .nc files.
    catalogname : string
        The specific prefix (catalog name) to match in the filename.

    Returns
    -------
    string
        Returns the date of the storm in the YYYYMMDD format (string), or None if not found.
    """
    base_name = os.path.basename(file_path)
    pattern = re.escape(catalogname) +'_storm'+ r'_\d+_(\d{8})\.nc$'
    match = re.search(pattern, base_name)
    if match:
        return match.group(1)
    return None


# def extract_date(file_path, pattern):
#     """
    

#     Parameters
#     ----------
#     file_path : string
#         File path for the storm catalog file
#     pattern : string
#         catalogname gievn in the JSON file

#     Returns
#     -------
#     string
#         returns the date of the storm catalog in the YYYYMMDD format(string)

#     """
#     base_name = os.path.basename(file_path)
#     match = re.search(pattern + r'\d+_(\d{8})\.nc', base_name)
#     if match:
#         return match.group(1)
#     return None
# =============================================================================
# added DBW 08152023: delete existing scenario files recursively before writing new ones
# this was provided by ChatGPT
# =============================================================================
def delete_files_in_directory(directory_path):
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
                #print(f"Deleted: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
        elif os.path.isdir(item_path):
            delete_files_in_directory(item_path)  # Recursively call the function for subdirectories


# =============================================================================
# added DBW 08142023: writing single storm scenario file using xarray
# =============================================================================
def writescenariofile(catrain,raintime,rainlocx,rainlocy,name_scenariofile,tstorm,tyear,trealization,maskheight,maskwidth,subrangelat,subrangelon,scenarioname,mask):
    # the following line extracts only the transposed rainfall within the area of interest
    #transposedrain=np.multiply(catrain[:,rainlocy[0] : (rainlocy[0]+maskheight), rainlocx[0] : (rainlocx[0]+maskwidth)],mask)
    transposedrain=catrain[:,rainlocy[0] : (rainlocy[0]+maskheight), rainlocx[0] : (rainlocx[0]+maskwidth)]
    
    description_string='RainyDay storm scenario file for original storm '+str(tstorm)+', year '+str(tyear)+', realization '+str(trealization)+', created from ' + scenarioname
    latitudes_units,longitudes_units = 'degrees_north', 'degrees_east'
    rainrate_units = 'mm hr^-1'
    times_units, times_calendar = 'minutes since 1970-01-01 00:00.0' , 'gregorian'
    
    # Variable Names
    times_name = 'time'                     ## change here
    latitudes_name,longitudes_name  = 'latitude', 'longitude'
    rainrate_name= 'precipitation rate'
    xlocation_name, ylocation_name = 'x index of transposition', 'y index of transposition'
    
    history, missing = 'Created ' + str(datetime.now()), '-9999.'
    source = 'RainyDay storm scenario file created from ' + scenarioname + '. See description for JSON file contents.'
    
    data=xr.Dataset(
         {
                "rain": (["time","latitude", "longitude"], transposedrain),
                "xlocation": (["scalar_dim"], rainlocx),
                "ylocation": (["scalar_dim"], rainlocy)
                #"scenariotime":(["time"],raintime)
            },
            coords={
                "time": raintime,   
                "latitude": subrangelat, 
                "longitude": subrangelon,
                "scalar_dim": [0]
            },
            attrs={
            "history":history, 
            "source" :  source, 
            "missing" : missing, 
            "description" : description_string,  
            "calendar" : times_calendar    
            }   
    )
    
    # # 
    # data_vars = dict(
    #                  rain = (("time","latitude", "longitude"),transposedrain,{'units': rainrate_units, 'long_name': rainrate_name}),
    #                 # rain = (("time","latitude", "longitude"),catrain[:, ::-1, :],{'units': rainrate_units, 'long_name': rainrate_name}),
    #                  xlocation = (("scalar_dim"),[rainlocx],{'units': 'dimensionless', 'long_name': xlocation_name}),
    #                  ylocation = (("scalar_dim"),[rainlocy],{'units': 'dimensionless', 'long_name': ylocation_name}),
    #                  time = (("time"), raintime)),
    # coords = dict(time = ((times_name),raintime),
    #                  longitude = (("longitude"), subrangelon.data , {'units': longitudes_units, 'long_name': longitudes_name}),
    #                  latitude =  (("latitude"), subrangelat.data, {'units': latitudes_units, 'long_name': latitudes_name}),
    #                  scalar_dim=(("scalar_dim"),[0])
    #                  #latitude =  (("latitude"), latrange[::-1].data, {'units': latitudes_units, 'long_name': latitudes_name}),
    #               )
    
    #attrs  = dict(history =history, source =  source, missing = missing, description = description_string,  calendar = times_calendar)
    
    #scenario = xr.Dataset(data_vars = data_vars, coords = coords, attrs = attrs)
    #scenario.time.encoding['units'] = "minutes since 1970-01-01 00:00:00"
    
    data.to_netcdf(name_scenariofile)
    data.close()    





#==============================================================================    
# http://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list 
#============================================================================== 
def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(i)
    return matches
    
    
#==============================================================================
# CREATE FILE LIST:- This will create file list and will remove the years(EXCLUDEYEARS) from the input given in .sst file. Also, it will return the number of years included in the dataset.
#==============================================================================

def try_parsing_date(text):
    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%Y%m%d'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')  ### This fucntion is not tested yet
    
    
def createfilelist(inpath, includeyears, excludemonths):
    """
    

    Parameters
    ----------
    inpath : string
        inpath takes in the file path for the rainfall data .nc files.
    includeyears : list
        includeyears are the years user want to include in the storm catalog analysis
        Default: False
    excludemonths : list
        exludemonths are the list of months user want to exclude from the analysis
        Default: none    
    Returns
    -------
    new_list : list
        returns the list of files including mentioned years and excluding described months
    nyears : int
        returns the lenght of years inlcuded in the analysis.

    """
    flist = sorted(glob.glob(inpath))
    new_list = [] ; years = set()
    for file in flist:
        base = os.path.basename(file)
        match = re.search(r'\d{4}(?:\d{2})?(?:\d{2}|\-\d{2}\-\d{2}|\/\d{2}/\d{2})', base)
        ### The block below is added for file named in formats YYYYMMDD, YYYY-MM-DD or YYYY/MM/DD and it will
        ### errors after year 1299 in format DDMMYYYY or MMDDYYYY
        try:
            file_date = datetime.strptime(match.group().replace("-","").replace("/",""),'%Y%m%d')
        except:
            sys.exit("You need to give file names in YYYYMMDD, YYYY-MM-DD or YYYY/MM/DD formats")
        file_year = file_date.year; file_month = file_date.month
        # ## Added by GP to manually exclude Harvey August 2017
        # file_month = file_date.month                # GP
        # if file_year == 2017 and file_month == 8:   # GP
        #     continue                                # GP

        if includeyears == False:
            if file_month not in excludemonths:
                new_list.append(file); years.add(file_year)
        else:
            if file_year in includeyears and file_month not in excludemonths:
                new_list.append(file); years.add(file_year)
    nyears = len(years) ## can be made more efficient
    return new_list, nyears
    
    

#==============================================================================
# Get things set up
#==============================================================================
def rainprop_setup(infile,rainprop,variables,catalog=False):
    if catalog:
        inrain,intime,inlatitude,inlongitude,catx,caty,catmax,_,domainmask=readcatalog(infile)
    else:
        # configure things so that in the storm catalog creation loop, we only read in the necessary variables
        invars=copy.deepcopy(variables)
        # we don't want to drop these:
        del invars['latname']   
        del invars['longname']
        keepvars=list(invars.values())

        # open the "entire" netcdf file once in order to get the list of all variables:        
        inds=xr.open_dataset(infile)
        

        # this will only keep the variables that we need to read in. 
        droplist=find_unique_elements(inds.keys(),keepvars) # droplist will be passed to the 'drop_variables=' in xr.open_dataset within the storm catalog creation loop in RainyDay
        inds.close()
        
        inrain,intime,inlatitude,inlongitude,nctime=readnetcdf(infile,variables,dropvars=droplist,setup = True)
        time_units = nctime.units; calendar = nctime.calendar if hasattr(nctime, 'calendar') else 'gregorian';
        # inrain,intime,inlatitude,inlongitude=readnetcdf(infile,variables,dropvars=droplist)
        # if max(inlongitude) > 180:
        #     inarea = inarea + 360
    
    if len(inlatitude.shape)>1 or len(inlongitude.shape)>1:
        sys.exit("RainyDay isn't set up for netcdf files that aren't on a regular lat/lon grid!")
        #inlatitude=inlatitude[:,0]          # perhaps would be safer to have an error here...
        #inlongitude=inlongitude[0,:]        # perhaps would be safer to have an error here...
    yres=np.abs((inlatitude[1:] - inlatitude[:-1])).mean()
    xres=np.abs((inlongitude[1:] - inlongitude[:-1])).mean()
    if np.isclose(xres,yres)==False:
        sys.exit("Rainfall grid isn't square. RainyDay cannot support that.")

    unqtimes=np.unique(intime)
    if len(unqtimes)>1:
        # tdiff=unqtimes[1:]-unqtimes[0:-1]
        tdiff = np.diff(unqtimes).astype('timedelta64[m]')
        # tempres=np.min(unqtimes[1:]-unqtimes[0:-1])   # temporal resolution
        tempres = np.min(tdiff).astype('timedelta64[m]')
        # if np.any(np.not_equal(tdiff,tempres)):
        #     sys.exit("Uneven time steps. RainyDay can't handle that.")
        if not np.all(tdiff == tempres):
            sys.exit("Uneven time steps. RainyDay can't handle that.")
        
    else:
        #this is to catch daily data where you can't calculate a time resolution
        tempres=np.float32(1440.)
        tempres=tempres.astype('timedelta64[m]')      # temporal resolution in minutes-haven't checked to make sure this works right
    # print(type(tempres) , type(tdiff))
    tempres_minutes = tempres.astype('timedelta64[m]').astype(int)
    if len(intime) * tempres_minutes != 1440. and catalog==False:
        sys.exit("RainyDay requires daily input files, but has detected something different.")
    tempres=np.int32(np.float32(tempres))

    nodata=np.unique(inrain[inrain<0.])
    if len(nodata)>1:
        sys.exit("More than one missing value flag.")
    elif len(nodata)==0 and catalog==False:
        print("Warning: Missing data flag is ambiguous. RainyDay will probably handle this ok, especially if there is not missing data.")
        nodata==-999.
    elif catalog:
        nodata=-999.
    else:
        nodata=nodata[0]

    if catalog:
        return [xres,yres], [len(inlatitude),len(inlongitude)],[np.min(inlongitude),np.max(inlongitude),np.min(inlatitude),np.max(inlatitude)],tempres,nodata,inrain,intime,inlatitude,inlongitude,catx,caty,catmax,domainmask
    else:
        return [xres,yres], [len(inlatitude),len(inlongitude)],[np.min(inlongitude),np.max(inlongitude)+xres,np.min(inlatitude)-yres,np.max(inlatitude)],tempres,nodata,droplist,calendar,time_units


#==============================================================================
# READ REALIZATION
#==============================================================================

def readrealization(rfile):
    infile=Dataset(rfile,'r')
    if 'rainrate' in infile.variables.keys():
        oldfile=True
    else:
        oldfile=False
        
    if oldfile:
        outrain=np.array(infile.variables['rainrate'][:])
    else:
        outrain=np.array(infile.variables['precrate'][:])[:,:,::-1,:]
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
    outlatitude=np.array(infile.variables['latitude'][:])
    outlongitude=np.array(infile.variables['longitude'][:])
    outlocx=np.array(infile.variables['xlocation'][:])
    outlocy=np.array(infile.variables['ylocation'][:])
    outmax=np.array(infile.variables['basinrainfall'][:])
    outreturnperiod=np.array(infile.variables['returnperiod'][:])
    outstormnumber=np.array(infile.variables['stormnumber'][:])
    origstormnumber=np.array(infile.variables['original_stormnumber'][:])
    #outstormtime=np.array(infile.variables['stormtimes'][:],dtype='datetime64[m]')
    timeunits=infile.variables['time'].units
    
    infile.close()
    return outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outreturnperiod,outstormnumber,origstormnumber,timeunits


#==============================================================================
# READ NPERYEAR REALIZATION
#==============================================================================
def readrealization_nperyear(rfile):
    infile=Dataset(rfile,'r')
    if 'rainrate' in infile.variables.keys():
        oldfile=True
    else:
        oldfile=False
        
    if oldfile:
        outrain=np.array(infile.variables['rainrate'][:])
    else:
        outrain=np.array(infile.variables['precrate'][:])[:,:,::-1,:]
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
    outlatitude=np.array(infile.variables['latitude'][:])
    outlongitude=np.array(infile.variables['longitude'][:])
    #outlocx=np.array(infile.variables['xlocation'][:])
    #outlocy=np.array(infile.variables['ylocation'][:])
    #outmax=np.array(infile.variables['basinrainfall'][:])
    #outreturnperiod=np.array(infile.variables['returnperiod'][:])
    #outstormnumber=np.array(infile.variables['stormnumber'][:])
    #origstormnumber=np.array(infile.variables['original_stormnumber'][:])
    #outstormtime=np.array(infile.variables['stormtimes'][:],dtype='datetime64[m]')
    timeunits=infile.variables['time'].units
    
    infile.close()
    return outrain,outtime,outlatitude,outlongitude,timeunits



#==============================================================================
# READ A PREGENERATED SST DOMAIN FILE
#==============================================================================

def readdomainfile(rfile,inbounds=False):
    infile=Dataset(rfile,'r')
    if np.any(inbounds!=False):
        outmask=np.array(infile.variables['domain'][inbounds[3]:inbounds[2]+1,inbounds[0]:inbounds[1]+1])
        outlatitude=np.array(infile.variables['latitude'][inbounds[3]:inbounds[2]+1])
        outlongitude=np.array(infile.variables['longitude'][inbounds[0]:inbounds[1]+1])         
    else:
        outmask=np.array(infile.variables['domain'][:])
        outlatitude=np.array(infile.variables['latitude'][:])
        outlongitude=np.array(infile.variables['longitude'][:])
    infile.close()
    return outmask,outlatitude,outlongitude


#==============================================================================
# "Rolling sum" function to correct for short-duration biases
#==============================================================================
    
def rolling_sum(a, n):
    ret = np.nancumsum(a, axis=0, dtype=float)
    ret[n:,:] = ret[n:,:] - ret[:-n,: ]
    return ret[n - 1:,: ]


#==============================================================================
# Distance between two points
#==============================================================================
    
def latlondistance(lat1,lon1,lat2,lon2):    
    #if len(lat1)>1 or len(lon1)>1:
    #    sys.exit('first 2 sets of points must be length 1');

    R=6371000;
    dlat=np.radians(lat2-lat1)
    dlon=np.radians(lon2-lon1)
    a=np.sin(dlat/2.)*np.sin(dlat/2.)+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.)*np.sin(dlon/2.);
    c=2.*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R*c
 
#==============================================================================
# rescaling functions
#==============================================================================
        
@jit(fastmath=True)
def intenseloop(intenserain,tempintense,xlen_wmask,ylen_wmask,maskheight,maskwidth,trimmask,mnorm,domainmask):
    for i in range(0,xlen_wmask*ylen_wmask):
        y=i//xlen_wmask
        x=i-y*xlen_wmask
        if np.equal(domainmask[y,x],1.) and  np.any(np.isnan(intenserain[:,y,x]))==False:
        # could probably get this working in nopython if I coded the multiplication explicitly, rather than using using the axis argument of nansum, which isn't numba-supported
            tempintense[:,y,x]=np.sum(np.multiply(intenserain[:,y:(y+maskheight),x:(x+maskwidth)],trimmask),axis=(1,2))/mnorm    
        else:
            tempintense[:,y,x]=np.nan
    return tempintense

@jit(nopython=True,fastmath=True)
def intense_corrloop(intenserain,intensecorr,homerain,xlen_wmask,ylen_wmask,mnorm,domainmask):   
    for i in range(0,xlen_wmask*ylen_wmask): 
        y=i//xlen_wmask
        x=i-y*xlen_wmask
        if np.equal(domainmask[y,x],1.) and  np.any(np.isnan(intenserain[:,y,x]))==False:
            intensecorr[y,x]=np.corrcoef(homerain,intenserain[:,y,x])[0,1]
        else:
            intensecorr[y,x]=np.nan
    return intensecorr


#==============================================================================
# read arcascii files
#==============================================================================

def read_arcascii(asciifile):
    # note: should add a detection ability for cell corners vs. centers: https://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/esri-ascii-raster-format.htm
    temp=linecache.getline(asciifile, 1)
    temp=linecache.getline(asciifile, 2)
    xllcorner=linecache.getline(asciifile, 3)
    yllcorner=linecache.getline(asciifile, 4)
    cellsize=linecache.getline(asciifile, 5)
    nodata=linecache.getline(asciifile, 6)
    
    #ncols=np.int(ncols.split('\n')[0].split(' ')[-1])
    #nrows=np.int(nrows.split('\n')[0].split(' ')[-1])
    
    xllcorner=np.float32(xllcorner.split('\n')[0].split(' ')[-1])
    yllcorner=np.float32(yllcorner.split('\n')[0].split(' ')[-1])
    
    cellsize=np.float32(cellsize.split('\n')[0].split(' ')[-1])
    nodata=np.float32(nodata.split('\n')[0].split(' ')[-1])
    
    #asciigrid = np.loadtxt(asciifile, skiprows=6)
    asciigrid = np.array(pd.read_csv(asciifile, skiprows=6,delimiter=' ', header=None),dtype='float32')
    nrows=asciigrid.shape[0]
    ncols=asciigrid.shape[1]
    
    asciigrid[np.equal(asciigrid,nodata)]=np.nan

    return asciigrid,ncols,nrows,xllcorner,yllcorner,cellsize



#==============================================================================
# used for prepping "drop_variables" so we don't read in unnecessary variables using xarray
#==============================================================================
def find_unique_elements(list1, list2):
    """
    Used to return only the elements of list1 that are not present in list2

    Parameters
    ----------
    list1 : target list of values to be reduced according to list2
    variables : list of values used to identify the values to keep in list1

    Returns
    -------
    list with only the values in list1 that were not present in list2

    """
    unique_elements_in_list1 = [x for x in list1 if x not in list2]
    #unique_elements_in_list2 = [x for x in list2 if x not in list1]
    return unique_elements_in_list1


#==============================================================================
# 
#==============================================================================
def is_monotonic(arr):
    return np.all(np.diff(arr) >= 0) or np.all(np.diff(arr) <= 0)


#==============================================================================
# 
#==============================================================================
def day_of_year_to_datetime(year, day_of_year):
    # Create a datetime for the first day of the given year
    start_of_year = np.datetime64(str(year), 'Y')

    # Add the number of days to get to the desired day of the year
    result = start_of_year + np.timedelta64(day_of_year - 1, 'D')

    return result

#==============================================================================
# 
#==============================================================================
def replace_year(dt, new_year):
    # Extract the time part from the datetime
    time_part = dt - np.datetime64(dt, 'Y')

    # Get the current year of the datetime
    current_year = np.datetime64(dt, 'Y')

    # Calculate the difference between the current year and the new year
    year_diff = new_year - (current_year.astype(int)+1970)

    # Add the difference to the datetime
    result = np.datetime64(dt, 'Y') + np.timedelta64(year_diff, 'Y') + time_part

    return result