#!/usr/bin/env python

# give the right permissions (for a Mac or Linux): chmod +x {scriptname}


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


#==============================================================================
# IMPORT STUFF!
#==============================================================================
#%%
import os
import re
import sys
import numpy as np
import scipy as sp
import shapefile
import math 
import json
import shutil
from datetime import datetime
import time  
from copy import deepcopy
from scipy import ndimage, stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import glob
import xarray as xr
from cartopy.feature import ShapelyFeature
import cartopy.mpl.ticker as cticker
import matplotlib.patches as patches
#from numba import njit, prange
numbacheck=True
import pandas as pd
from shapely.affinity import translate

# plotting stuff, really only needed for diagnostic plots
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import RainyDay_functions as RainyDay
import RainyDay_utilities_Py3.RainyDay_functions as RainyDay
import RainyDay_utilities_Py3.fast_prof as FP        # GP 2026-10-22: RAINYDAY_TIMERS wall-clock timers (no effect on results)
import RainyDay_utilities_Py3.fast_io as FIO          # GP 2026-10-22: CATALOGSTORAGE "light" (light storm catalog)
import RainyDay_utilities_Py3.fast_sst as FST        # GP 2026-10-22: MULTIDURATIONENGINE "table" (RainyDay_fast fix 3)
import RainyDay_utilities_Py3.fast_record as FR      # GP 2026-10-26: RECORDCI, record (sampling) uncertainty (adds outputs only)
import RainyDay_utilities_Py3.fast_parse as FPA      # GP 2026-10-26: parsing fixes, README_CHANGES B9 (one-line hooks below)

import warnings
warnings.filterwarnings("ignore")

# GP 2026-10-22: RainyDay_fast: tracemalloc traced every allocation (numpy arrays and numba NRT arrays, e.g. one per
# window and draw in SSTalt_multidur) and was never read; most of the time of a 1x1-basin frequency analysis went
# to tracemalloc's traceback_new. No effect on results. RAINYDAY_TRACEMALLOC=1 restores it.
# import tracemalloc
# tracemalloc.start()
if os.environ.get("RAINYDAY_TRACEMALLOC","").strip() not in ("","0"):
    import tracemalloc
    tracemalloc.start()

# Storm-catalog engine for irregular domains (rectangular domains always use catalogNumba):
# (GP 2026-10-26, rectangular catalog search: rectangular domains now use the per-step catalogFFT_irregular_domainaware with an all-ones
#  domain mask, i.e. the 'fft_perstep' search; the engines below remain irregular-only)
#   'batchfft'    : batched multithreaded FFTs over blocks of files (default; verified
#                   bit-identical to 'fft_perstep' on the Talequah example, see RainyDay/tools)
#   'batchfft_mp' : 'batchfft' distributed over worker processes (fastest for long records)
#   'fft_perstep' : original loop, one catalogFFT_irregular_domainaware call per timestep
#   'batchtf'     : experimental TensorFlow (Metal GPU) convolution; equivalent storms but
#                   float rounding can shift near-tied storm windows by ±1-2 timesteps
#   'batchcuda'   : CuPy/cuFFT GPU FFT (NVIDIA); like 'batchtf' but keeps the FFT algorithm.
#                   Both GPU engines go through the exact CPU confirm (bit-identical catalog).
# The engine and its performance knobs (nworkers/blockfiles/gpu_chunk/gpu_delta/prefetch)
# are resolved per-run from the optional JSON "COMPUTE" block, env vars, and machine
# auto-detection (precedence env > JSON > auto) in RainyDay.resolve_compute_config; the
# line below is only the pre-resolution fallback for env-based use.
catalog_engine = os.environ.get('RAINYDAY_CATALOG_ENGINE', 'batchfft')

#%%
#==============================================================================
# RAINFALL CLASS
# THIS CONTAINS INFORMATION ABOUT THE SPECIFIED INPUT RAINFALL DATASET
#==============================================================================
class GriddedRainProperties(object):
    def __init__(self,dataset,bndbox,subind,subextent,dimensions,subdimensions,spatialres,timeres,timeunits,spatialunits,rainunits,nodata,notes):
        self.dataset=dataset                #  INPUT RAINFALL DATA SOURCE (TMPA, STAGE IV, ETC.)
        self.bndbox=bndbox                  # COORDINATES OF FULL DATASET BOUNDING BOX (THIS SHOULD BE THE IN THE ORDER [WEST LON,EAST LON,SOUTH LAT,NOTH LAT])
        self.subind=subind                  # MATRIX INDICIES OF BOUNDING BOX (XMIN,XMAX,YMIN,YMAX)
        self.subextent=subextent            # COORDINATES OF USER-DEFINED BOUNDING BOX (NOTE: THIS WILL BE AUTOMATICALLY CALCULATED)
        self.dimensions=dimensions          # WHAT ARE THE SPATIAL DIMENSIONS OF THE INPUT DATASET (NOTE: EVEN IF THE INPUT DATA YOU ARE USING IS FOR A SUBDOMAIN, THIS SHOULD BE THE DIMENSIONS OF THE FULL DATASET)
        self.subdimensions=subdimensions    # WHAT ARE THE DIMENSIONS OF THE SUBDOMAIN (IN THIS SCRIPT, THIS WILL BE RESET TO THE ACTUAL DIMENSION OF THE INPUT DATA, IF NEEDED)
        self.spatialres=spatialres          # WHAT IS THE SPATIAL RESOLUTION OF THE INPUT DATA [dx,dy]?  CURRENTLY THIS SCRIPT WILL ONLY HANDLE RECTANGULAR GRIDS IN DEGREES
        self.timeunits=timeunits            # TEMPORAL UNITS.  CURRENTLY MUST BE MINUTES
        self.spatialunits=spatialunits      # SPATIAL UNITS (CURRENTLY MUST BE DEGREES) [Xres,Yres]
        self.rainunits=rainunits            # RAINFALL UNITS (CURRENTLY MUST BE MM/HR)
        self.nodata=nodata                  # MISSING DATA FLAG
        self.notes=notes                    # ANY SPECIAL NOTES?


#==============================================================================
# RAINFALL INFO
# NOTE: "BOUNDING BOXES"-bndbox, subbox, CONSIDER THE COORDINATES OF THE EDGE OF THE BOUNDING BOX
#==============================================================================
    
emptyprop=GriddedRainProperties('emptyprop',
                            [-999.,-999.-999.-999.],
                            [999, 999, 999, 999],
                            [-999.,-999.-999.-999.],
                            [999, 999],
                            [999, 999],
                            [99.,99.],
                            99.,
                            "minutes",
                            "degrees",
                            "mm/hr",
                            -9999.,
                            "none")
   
                   
################################################################################
# "MAIN"
################################################################################

print('''Welcome to RainyDay, a framework for coupling remote sensing precipitation
        fields with Stochastic Storm Transposition for assessment of rainfall-driven hazards.
        Copyright (C) 2017  Daniel Benjamin Wright (danielb.wright@gmail.com)
        Distributed under the MIT Open-Source License: https://opensource.org/licenses/MIT
    
    
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
''')


#==============================================================================
# USER DEFINED INFO
#==============================================================================
start = time.time()
parameterfile='ttt'
try:
    parameterfile=sys.argv[1]
    #parameterfile='/Users/daniel/Google_Drive/RainyDay2/Summer2023_RefactorTesting/CONUS_Madison.json'
    print("reading in the parameter file...")
    ### Cardinfo takes in the  'JSON' file parameters
    with open(parameterfile, 'r') as read_file:
        cardinfo = json.loads(read_file.read(), object_pairs_hook=RainyDay.dict_raise_on_duplicates)   # the "hook" catches instances of duplicate keys in the json file
except FileNotFoundError:
    print("You either didn't specify a parameter file, or it doesn't exist on the source path given.")
# GP 2026-10-26, missing required keys: a missing key (MAINPATH, SCENARIONAME, CATALOGNAME, CREATECATALOG, DURATION, VARIABLES,
# POINTAREA; RAINPATH/AREA_EXTENT/WATERSHEDSHP/POINTLAT/POINTLON/BOX_* when the settings need them) stops here with a
# message; the reads below catch ImportError/ValueError/IndexError, so a missing key ended in a KeyError traceback
FPA.check_required(cardinfo)
#%%


# #==============================================================================
# # USER DEFINED VARIABLES
# #==============================================================================
#
#
# setting up basic paths:
try:
    wd=cardinfo["MAINPATH"]
    try:
        scenarioname=cardinfo["SCENARIONAME"]
    except ValueError:
        sys.exit("You didn't specify SCENARIONAME!")
    # GP 2026-10-21: MULTIPLEDURATIONS (optional): one run builds/reads the parent storm catalog (DURATION) and does a
    # frequency analysis for every duration in the list. Outputs go to MAINPATH/<SCENARIONAME>_<d>/; the storm catalog
    # and the diagnostics stay in MAINPATH/<SCENARIONAME>_<DURATION>/ (this run's main folder).
    multidurations=RainyDay.parse_multidurations(cardinfo.get("MULTIPLEDURATIONS",None))
    basescenario=scenarioname
    if multidurations is not None:
        if "DURATION" not in cardinfo:
            sys.exit("MULTIPLEDURATIONS needs DURATION (the duration of the parent storm catalog)")
        scenarioname=basescenario+'_'+RainyDay.durtag(cardinfo["DURATION"])
        print("MULTIPLEDURATIONS: frequency analyses for "+", ".join(RainyDay.durtag(d) for d in multidurations)+" h from the "+RainyDay.durtag(cardinfo["DURATION"])+" h storm catalog in "+wd+'/'+scenarioname)
    fullpath=wd+'/'+scenarioname
    if os.path.isdir(fullpath)==False:
        #os.system('mkdir -p -v %s' %(fullpath))
        os.mkdir(fullpath)
    os.chdir(wd)
except ImportError:
    sys.exit("You did't specify MAINPATH, which is a required field!")
    
#
#
# # PROPERTIES RELATED TO SST PROCEDURE:
try:
    variables= cardinfo["VARIABLES"]   # this defines the names of the lat,lon, and precip variables in the netcdf input files
except ValueError:
    sys.exit("Please enter the valid names of the variables and the coordinates in your dataset")
try:
    catalogname=cardinfo["CATALOGNAME"]
except ValueError:
    sys.exit("You didn't specify CATALOGNAME!")

    
try:
    CreateCatalog=cardinfo["CREATECATALOG"]
except Exception:
    sys.exit("You didn't specify CREATECATALOG, which is required!")

if CreateCatalog.lower()=='true':
    CreateCatalog=True
elif CreateCatalog.lower()=='false':
    CreateCatalog=False
else:
    sys.exit("CreateCatalog must be either 'true' or 'false'!")

# CATALOGPATH (optional, read-only): folder + file prefix of an existing storm catalog to reuse,
# e.g. "/path/to/StormCatalog/Talequah_3days.nc" -> reads /path/to/StormCatalog/Talequah_3days.nc_storm_*.nc
# If not given, the catalog is read from MAINPATH/SCENARIONAME/StormCatalog/CATALOGNAME (original behavior).
catalogpath=cardinfo.get("CATALOGPATH",None)
if isinstance(catalogpath,str) and catalogpath.strip()!='' and catalogpath.strip().lower()!='none':
    catalogpath=os.path.abspath(os.path.expanduser(catalogpath.strip()))
    if CreateCatalog:
        print("WARNING: CATALOGPATH is only used to read an existing catalog. Since CREATECATALOG is 'true', it will be ignored and the new catalog will be written to "+fullpath+'/StormCatalog')
        catalogpath=fullpath+'/StormCatalog/'+catalogname
    else:
        if os.path.isdir(catalogpath):
            sys.exit("CATALOGPATH must include the catalog prefix, not only the folder, e.g. '"+catalogpath+"/"+catalogname+"'")
        if os.path.isdir(os.path.dirname(catalogpath))==False:
            sys.exit("The folder given in CATALOGPATH doesn't exist: "+os.path.dirname(catalogpath))
        if os.path.basename(catalogpath)!=catalogname:
            print("NOTE: using the catalog prefix from CATALOGPATH ('"+os.path.basename(catalogpath)+"') instead of CATALOGNAME ('"+catalogname+"')")
            catalogname=os.path.basename(catalogpath)
else:
    catalogpath=fullpath+'/StormCatalog/'+catalogname
# GP 2026-10-22: CATALOGSTORAGE "full" (default: one file per storm, unchanged) | "light" (one small file without the
# rain, which is rebuilt from the input files when a storm is read; RainyDay_fast Fix 2, RainyDay_utilities_Py3/fast_io.py).
# Reading (CATALOGPATH) uses the storm files when they exist, otherwise the light catalog.
catalogstorage=FIO.parse_catalogstorage(cardinfo.get("CATALOGSTORAGE","full"))
# GP 2026-10-22: light catalogs rebuild the storm rain from this run's RAINPATH files (matched by file name), not from
# the paths stored in the catalog (fast_io.py); set before the first catalog read below
FIO.set_rainpath(cardinfo.get("RAINPATH",None))
# if CreateCatalog:
#     try:
#         os.mkdir(fullpath + '/StormCatalog')
#     except OSError as exc:
#         if exc.errno != 17:   ### This checks the file exist error. '17' this is for file exist error.
#             raise
#         pass
try:
    nstorms=cardinfo["NSTORMS"]
    defaultstorms=False
    print(str(nstorms)+" storms will be identified for storm catalog!")
except Exception:
    #nstorms=100
    print("you didn't specify NSTORMS, defaulting to 20 per year, or whatever is in the catalog!")
    defaultstorms=True

# if you are reusing a storm catalog, identify all the associated files and create a list of them:
if CreateCatalog==False:
    # GP 2026-10-22: storm files of a full catalog or the (same-named) virtual storm paths of a light catalog
    # stormlist = glob.glob(catalogpath + '_storm_' + '*' + '.nc')
    stormlist = FIO.list_storms(catalogpath, catalogname, catalogstorage)
    stormlist = sorted(stormlist, key=lambda path: RainyDay.extract_storm_number(path, catalogname))
    if len(stormlist)==0:
        sys.exit("You need to create a storm catalog first. No storm files found matching "+catalogpath+"_storm_*.nc")
    else:
        print("Reading an existing storm catalog from "+os.path.dirname(catalogpath)+" ("+str(len(stormlist))+" storm files)")
        # GP 2026-10-22: FIO.readcatalog = RainyDay.readcatalog, or the same tuple from a light catalog
        # catrain,stormtime,latrange,lonrange,catx,caty,catmax,catmask,domainmask,cattime,timeres=RainyDay.readcatalog(stormlist[-1])
        catrain,stormtime,latrange,lonrange,catx,caty,catmax,catmask,domainmask,cattime,timeres=FIO.readcatalog(stormlist[-1])
        # warn if the catalog was built with different key settings than this run (the catalog stores its original JSON)
        try:
            import ast
            # with xr.open_dataset(stormlist[-1]) as tempcat:
            with xr.open_dataset(FIO.attrs_path(stormlist[-1])) as tempcat:   # GP 2026-10-22: light catalog file
                catparams=ast.literal_eval(tempcat.attrs['description'])
            for key in ["DURATION","DOMAINTYPE","DOMAINSHP","POINTAREA","WATERSHEDSHP","POINTLAT","POINTLON"]:
                if catparams.get(key)!=cardinfo.get(key):
                    print("WARNING: the storm catalog was built with "+key+"="+str(catparams.get(key))+", but this run has "+key+"="+str(cardinfo.get(key)))
            if catparams.get("POINTAREA")!=cardinfo.get("POINTAREA") or catparams.get("WATERSHEDSHP")!=cardinfo.get("WATERSHEDSHP"):
                print("WARNING: when reusing a catalog, the analysis area (mask) is taken from the catalog, not from this run's POINTAREA/WATERSHEDSHP!")
        except Exception:
            print("Couldn't read the original parameters stored in the storm catalog, skipping the consistency check.")
        yres=np.abs(latrange.diff(dim='latitude')).mean()
        xres=np.abs(lonrange.diff(dim='longitude')).mean()
        catarea=[lonrange[0],lonrange[-1]+xres,latrange[-1]-yres,latrange[0]]
        if np.isclose(xres,yres,atol=1e-5)==False:
            sys.exit('RainyDay currently only supports equal x and y resolutions!')
        else:
            res=np.min([yres,xres])


try:
    nsimulations=cardinfo["NYEARS"]
    print(str(nsimulations)+" years of synthetic storms will be generated, for a max recurrence interval of "+str(nsimulations)+" years!")
except Exception:
    nsimulations=100
    print("you didn't specify NYEARS, defaulting to 100, for a max recurrence interval of 100 years!")

try:
    nrealizations=cardinfo["NREALIZATIONS"]
    print(str(nrealizations)+" realizations will be performed!")
except Exception:
    nrealizations=1

try:
    duration=cardinfo["DURATION"]
    if duration<=0.:
        sys.exit("Duration is zero or negative!")
except ImportError:
    sys.exit("You didn't specify 'DURATION', which is a required field!")
#
#
##this following bit is only needed in the very specific (and generally not recommended) case when the desired analysis/catalog duration is equal to the temporal resolution of the dataset
try:
    temptimeres=cardinfo["TIMERESOLUTION"]
    print("A resolution of "+str(temptimeres)+" minutes has been provided. Be careful with this, because if it is improperly specified, this will cause errors. Note that TIMERESOLUTION is not needed unless the duration of each storm is to be exactly equal to the temporal resolution of the input data or catalog. In other words, make sure you know what you're doing!")
except Exception:
    temptimeres=False
#
#
# here you define which type of temporal resampling scheme you will use: either poisson or empirical
try:
    samplingtype=cardinfo["RESAMPLING"]
    sampling_options = ['poisson', 'empirical', 'negbinom']
    if samplingtype.lower() not in sampling_options:
        sys.exit("unrecognized storm count resampling type, options are: 'poisson', 'empirical', or 'negbinom'")
except Exception:
    samplingtype='poisson'
#
if samplingtype=='poisson':
    print("Poisson-based temporal resampling scheme will be used!")
elif samplingtype=='empirical':
    print("Empirically-based temporal resampling scheme will be used!")
elif samplingtype=='negbinom':
    sys.exit("Negative binomial-based temporal resampling is not currently supported because it is a mess!")
#
#
# Here is where you define the area of interest-which can be a single grid cell ('point' or 'grid') via ptlat and ptlon,
# a rectangular box ('box' or 'rectangle') via "BOX_YMIN","BOX_YMAX","BOX_XMIN","BOX_XMAX, or 
# a polygon, typically a watershed boundary ('watershed' or 'basin'), which must be a polygon shapefile in WGS84
try:
    areatype=cardinfo["POINTAREA"]
    if areatype.lower()=="basin" or areatype.lower()=="watershed":
        areatype='basin'
        try:
            wsmaskshp=cardinfo["WATERSHEDSHP"]
            print("You selected 'watershed' for 'POINTAREA', please note that if the watershed is not in a regular lat/lon projection such as EPSG4326/WGS 84, the results will likely be incorrect!")
        except IndexError:
            sys.exit("You specified 'watershed' for 'POINTAREA' but didn't specify 'WATERSHEDSHP'")
    elif areatype.lower()=="point" or areatype.lower()=="grid":
        try:
            ptlat=np.float32(cardinfo["POINTLAT"])
            ptlon=np.float32(cardinfo["POINTLON"])                   
            ctrlat=ptlat
            ctrlon=ptlon
            areatype="point"
        except ImportError:
            sys.exit("You specified 'point' for 'POINTAREA' but didn't properly specify 'POINTLAT' and 'POINTLON'")
    elif areatype.lower()=="box" or areatype.lower()=="rectangle":
        try:
            box1,box2,box3,box4=cardinfo["BOX_YMIN"],cardinfo["BOX_YMAX"],cardinfo["BOX_XMIN"],cardinfo["BOX_XMAX"]
            boxarea=np.array([box3,box4,box1,box2])
            ctrlat, ctrlon=(box1+box2)/2., (box3+box4)/2
            areatype="box"
        except ImportError:
            sys.exit("You specified 'box' for 'POINTAREA' but didn't properly specify 'BOX_YMIN', 'BOX_YMAX', 'BOX_XMIN', and 'BOX_XMAX'")
    elif areatype.lower()=="pointlist":
        sys.exit("You selected 'pointlist'. This is untested in the refactored RainyDay.")
        if CreateCatalog:
            sys.exit("POINTLIST is currently not available when creating a new storm catalog.")
        ptlistname=cardinfo["POINTLIST"]
        #ptlistname=np.str(sys.argv[3])
        ptlistdat=np.loadtxt(ptlistname,skiprows=1,delimiter=',')
        ptlatlist=ptlistdat[:,0]
        ptlonlist=ptlistdat[:,1]
        npoints_list=len(ptlatlist)
    else:
        sys.exit("unrecognized area type")
except ImportError:
    sys.exit("You didn't specify 'POINTAREA', which needs to either be set to 'watershed', 'point', 'grid', or 'rectangle'")
#
#
# Here you define the type of transposition scheme you want to use.
try:
    transpotype=cardinfo["TRANSPOSITION"]
    if transpotype.lower()=='nonuniform':
        transpotype='nonuniform'
        print("You selected the kernel density-based non-uniform storm transposition scheme!")
    elif transpotype.lower()=='uniform' and areatype.lower()=="pointlist":
        print("You selected the spatially uniform storm transposition scheme and to perform IDF for a list of points!")
    elif transpotype.lower()=='user':
        transpotype='user'
        sys.exit("RainyDay isn't set up for the user-supplied transposition scheme yet")
    elif transpotype.lower()=='uniform':
        transpotype='uniform'
        print("You selected the spatially uniform storm transposition scheme!")
    else:
        sys.exit("You specified an unknown resampling method")
except Exception:
    transpotype='uniform'
    print("You didn't specify 'TRANSPOSITION', defaulting to spatially uniform scheme!")

#
#
# it is not well documented yet, and is not recommended for normal users
rescaletype='none'
try:
    rescalingtype=cardinfo["ENHANCEDSST"]
    if rescalingtype.lower()=='stochastic' or rescalingtype.lower()=='deterministic':
        print("You selected the 'Ratio Rescaling'! This is a more advanced option that should be used with caution")
        if rescalingtype.lower()=='stochastic':
            rescaletype='stochastic'
            print("You selected stochastic ratio rescaling. This has not been thoroughly vetted. Be careful!")
            if areatype.lower()=='pointlist':
                pass
                #sys.exit("You selected 'pointlist' for POINTAREA. This is currently not compatible with stochastic rescaling.")
        if rescalingtype.lower()=='deterministic':
            rescaletype='deterministic'
            print("You selected deterministic ratio rescaling. This has not been thoroughly vetted. Be careful!")

        try:
            rescalingfile=cardinfo["RAINDISTRIBUTIONFILE"]
            if os.path.isfile(rescalingfile)==False:
                sys.exit("The precipitation file specified in 'RAINDISTRIBUTIONFILE' cannot be found!")
        except IndexError:
            sys.exit("Even though you 'ratio rescaling', you didn't specify the file of precipitation distributions!")
    elif rescalingtype.lower()=='dimensionless':
        print("You selected 'dimensionless SST', modeled after Nathan et al. (2016). This has not been thoroughly vetted. Be careful!")
        rescaletype='dimensionless'
        try:
            rescalingfile=cardinfo["RAINDISTRIBUTIONFILE"]
            if os.path.isfile(rescalingfile)==False:
                sys.exit("The precipitation file specified in 'RAINDISTRIBUTIONFILE' cannot be found!")
        except IndexError:
            sys.exit("Even though you 'dimensionless SST', you didn't specify the file of precipitation distributions!")
    else:
        print("No rescaling will be performed!")
except Exception:
    print("No rescaling will be performed!")


# what kind of transposition domain are you going to use? 'rectangular' or 'irregular'
# the former is defined via "AREA_EXTENT", e.g. "AREA_EXTENT" : {"LATITUDE_MIN":42.5,"LATITUDE_MAX":44.0,"LONGITUDE_MIN": -90.5, "LONGITUDE_MAX": -88.0},
# the latter is defined via a polygon shapefile in WGS84 using "DOMAINSHP"
try:
    domain_type=cardinfo["DOMAINTYPE"]
except Exception:
    domain_type='rectangular'

if domain_type.lower()=='rectangular':
    shpdom=False
    ncfdom=False

if domain_type.lower()=='irregular':
    print("Irregular domain type selected!")
    domain_type='irregular'
    ncfdom=False
    shpdom=False
    if CreateCatalog:
        try:
            domainshp=cardinfo["DOMAINSHP"]
            if os.path.isfile(domainshp)==False:
                sys.exit("can't find the transposition domain shapefile!")
            else:
                print("You selected 'irregular' for 'DOMAINTYPE', please note that if the domain shapefile is not in a regular lat/lon projection such as EPSG4326/WGS 84, the results will likely be incorrect!")
                shpdom=True
                ds = shapefile.Reader(domainshp,'rb')
                tempbox= ds.bbox
                inarea=np.array([tempbox[0],tempbox[2],tempbox[1],tempbox[3]],dtype='float32')

        except Exception:
            pass
        try:
            domainncf=cardinfo["DOMAINFILE"]
            if os.path.isfile(domainncf)==False:
                sys.exit("This capability isn't tested. Unclear whether changes are needed due to updating for CF compliance.")
                sys.exit("Can't find the transposition domain NetCDF file!")
            else:
                sys.exit("This capability isn't tested. Unclear whether changes are needed due to updating for CF compliance.")
                print("Domain NetCDF file found!")
                ncfdom=True
                domainmask,domainlat,domainlon=RainyDay.readdomainfile(domainncf)
                res=np.abs(np.mean(domainlat[1:]-domainlat[0:-1]))
                xmin=np.min(np.where(np.sum(domainmask,axis=0)!=0))
                xmax=np.max(np.where(np.sum(domainmask,axis=0)!=0))
                ymin=np.min(np.where(np.sum(domainmask,axis=1)!=0))
                ymax=np.max(np.where(np.sum(domainmask,axis=1)!=0))
                domainmask=domainmask[ymin:(ymax+1),xmin:(xmax+1)]
                inarea=np.array([domainlon[xmin],domainlon[xmax]+res,domainlat[ymax]-res,domainlat[ymin]])
                domainlat=domainlat[ymin:(ymax+1)]
                domainlon=domainlon[xmin:(xmax+1)]
        except Exception:
            pass
    else:
        # try:
        #     domainshp=cardinfo["DOMAINSHP"]
        #     if os.path.isfile(domainshp)==False:
        #         sys.exit("can't find the transposition domain shapefile!")
        #     else:
        #         print("You selected 'irregular' for 'DOMAINTYPE', please note that if the domain shapefile is not in a regular lat/lon projection such as EPSG4326/WGS 84, the results will likely be incorrect!")
        #         shpdom=True
        # except Exception:
        #     print("Trouble finding the domain shapefile. Technically we don't need it, so we'll skip this part.")

        yres=np.abs(latrange.diff(dim='latitude')).mean()
        xres=np.abs(lonrange.diff(dim='longitude')).mean()
        inarea=np.array([lonrange[0],lonrange[-1]+res,latrange[-1]-res,latrange[0]])
        domainshp=cardinfo.get("DOMAINSHP","")
    if ncfdom==False and shpdom==False and CreateCatalog:
        sys.exit("You selected 'irregular' for 'DOMAINTYPE', but didn't provide a shapefile or NetCDF file for the domain!")
else:  ###### Ashar:  Dan I beleive we can remove this condition and add it to the if condition above, this may avoid extra condition.
    print("rectangular domain type selected!")
    domain_type='rectangular'
    if CreateCatalog==False:
        inarea=deepcopy(catarea)
    else:
        try:
            lim1,lim2,lim3,lim4 = cardinfo["AREA_EXTENT"].values()
            inarea=np.array([lim3,lim4,lim1,lim2])
        except ImportError:
            sys.exit("need to specify 'LATITUDE_MIN', 'LATITUDE_MAX', 'LONGITUDE_MIN', 'LONGITUDE_MAX'")
##
# this catches the problem of a user changing the transposition domain from what was in the catalog. That is not a good plan!
if CreateCatalog==False and np.allclose(inarea,catarea)==False:
        sys.exit("You are changing the domain size for an existing catalog. RainyDay can't handle that!")
#
#
# do you want to plot diagnostic plots? Storm total maps, hyetographs, catalog mean and std. deviation maps, and CDF of storm total rainfall
try:
    DoDiagnostics=cardinfo["DIAGNOSTICPLOTS"]
    if DoDiagnostics.lower()=='only-statistics':
        DoDiagnostics_stats=True
    else:
        DoDiagnostics_stats=False
    
    if DoDiagnostics.lower()=='true' or DoDiagnostics.lower()=='only-statistics':
        DoDiagnostics=True
    else:
        DoDiagnostics=False
except Exception:
    DoDiagnostics=True

diagpath=fullpath+'/Diagnostics/'
if DoDiagnostics==True:
    if os.path.isdir(fullpath+'/Diagnostics')==False:
        #os.system('mkdir %s' %(diagpath))
        try:
            os.mkdir(diagpath)
        except:
            pass
if DoDiagnostics and shpdom:
                import geopandas as gpd
                dshp = gpd.read_file(domainshp)
                #print stateshp1.crs
                # Comment the two below lines for parallel processing to avoid conflict while writing the shapefile in multiprocessing
                # dshp.crs={}
                # dshp.to_file(domainshp, driver='ESRI Shapefile')
#
#
# perform the frequency analysis?
try:
    FreqAnalysis=cardinfo["FREQANALYSIS"]
    if FreqAnalysis.lower()=='true':
        FreqAnalysis=True
    else:
        FreqAnalysis=False
except Exception:
    FreqAnalysis=True
#
#
# pointlist is a mess and not recommended for normal users. It might not even work anymore
if areatype.lower()=="pointlist":
    FreqFile_mean=fullpath+'/'+scenarioname+'_mean.FreqAnalysis'
    FreqFile_min=fullpath+'/'+scenarioname+'_min.FreqAnalysis'
    FreqFile_max=fullpath+'/'+scenarioname+'_max.FreqAnalysis'
else:
    FreqFile=fullpath+'/'+scenarioname+'_FreqAnalysis.csv'
#
#
# do you want to write output scenarios in netcdf format, e.g. for flood frequency simulations?
try:
    Scenarios=cardinfo["SCENARIOS"]
    if Scenarios.lower()=='true' and areatype.lower()!="pointlist":
        Scenarios=True
        FreqAnalysis=True
        WriteName=fullpath+'/Realizations'
        #os.system('mkdir %s' %(WriteName))
        if os.path.isdir(WriteName)==False:
            os.mkdir(WriteName)
        #WriteName=WriteName+'/'+scenarioname
        
        # DBW 08142023: added this since we're going to be creating a huge number of scenario files and need ways of keeping them organized:
        for i in range(0,nrealizations):
            if os.path.isdir(WriteName+'/realization'+str(i+1))==False:
                os.mkdir(WriteName+'/realization'+str(i+1))  # now there will be a directory of scenarios for each realization
            
        print("RainyDay will write "+str(nrealizations)+" realizations times "+str(nsimulations)+" years worth of output precipitation scenarios. If this is a big number, this could be very slow!")


    elif Scenarios and areatype.lower()=="pointlist":
        print("You specified 'POINTAREA pointlist', but want precipitation scenario outputs. The 'pointlist option' does not support precipitation scenarios!")
        Scenarios=False
    else:
        Scenarios=False
except Exception:
    Scenarios=False
    print("You didn't specify 'SCENARIOS', defaulting to 'false', no scenarios will be written!")
# GP 2026-10-22: scenario writing reads the storm files directly; light catalogs are not supported there
if Scenarios and (catalogstorage=='light' or (CreateCatalog==False and FIO.is_virtual(stormlist[-1]))):
    sys.exit("SCENARIOS needs a full storm catalog (CATALOGSTORAGE 'full')")

try:
    padscenarios=cardinfo["PADSCENARIOS"]
    if isinstance(padscenarios, int):
        if padscenarios<0:
           sys.exit("PADSCENARIOS is negative. That's pretty silly!") 
    else:
        sys.exit("PADSCENARIOS is not an integer! That's silly")
except:
    padscenarios=0         
    
    
#
#
# # EXLCUDE CERTAIN MONTHS
# GP 2026-10-26, EXCLUDEMONTHS spellings: a range across the new year ("12-3") gave range(12,4) = [] (nothing
# excluded), one month as a string ("8") was ignored, a list of strings (["12","1"]) excluded nothing; parse_excludemonths returns
# the same objects as the code below for every input that code read correctly. Old code:
# try:
#     excludemonths=cardinfo["EXCLUDEMONTHS"]
#     if isinstance(excludemonths, str):
#         if ',' in excludemonths:  ## For different months to be exlcuded from the analysis
#             excludemonths = [np.int32(exstorm) for exstorm in excludemonths.split(",")]
#         elif '-' in excludemonths:   ## For a range of months of storms to exclude from the analysis
#             start_year, end_year = map(int, excludemonths.split('-'))
#             excludemonths = [exstorm for exstorm in range(start_year, end_year + 1)]
#         elif excludemonths.lower() == 'none':
#             excludemonths = []
#         else:
#             excludemonths =[]
#     elif isinstance(excludemonths, list):
#         excludemonths = excludemonths
#     else:
#         excludemonths = []
# except Exception:
#     excludemonths=[]
excludemonths=FPA.parse_excludemonths(cardinfo)       # GP 2026-10-26: EXCLUDEMONTHS spellings
#
# # INCLUDE ONLY CERTAIN YEARS
# GP 2026-10-26, INCLUDEYEARS spellings: "all" only worked because np.int32("all") raised inside the try below
# (includeyr.lower == "all" compares the method); a list of strings gave an empty file list and a crash; same objects as the code below for
# every input that code read correctly. Old code:
# try:
#     includeyr=cardinfo["INCLUDEYEARS"]
#     if isinstance(includeyr , str):
#         if includeyr.lower == "all":
#             includeyears = False
#         elif ',' in includeyr:
#             includeyears = [np.int32(year) for year in includeyr.split(",")]
#         elif '-' in includeyr:
#             start_year, end_year = map(np.int32, includeyr.split('-'))
#             includeyears = [year for year in range(start_year, end_year + 1)]
#         else:
#             includeyears = [np.int32(includeyr)]
#     elif isinstance(includeyr, list):
#         if len(includeyr) == 0:
#             includeyears = False
#         else:
#             includeyears = includeyr
#     else:
#         includeyears=False
# except Exception:
#     includeyears=False
includeyears=FPA.parse_includeyears(cardinfo)         # GP 2026-10-26: INCLUDEYEARS spellings
#
# # MINIMUM RETURN PERIOD THRESHOLD FOR WRITING RAINFALL SCENARIO (THIS WILL HAVE A HUGE IMPACT ON THE SIZE OF THE OUTPUT FILES!)  IN PRINCIPLE IT IS PERHAPS BETTER TO USE AN INTENSITY BUT THIS IS CLEANER!
try:
    RainfallThreshYear=cardinfo["RETURNTHRESHOLD"]
except Exception:
    RainfallThreshYear=1
#
# # DIRECTORY INFO-WHERE DOES THE INPUT DATASET RESIDE?
if CreateCatalog:
    try:
        inpath=cardinfo["RAINPATH"]
    except ImportError:
        sys.exit("You didn't specify 'RAINPATH', which is a required field for creating a new storm catalog!")


# just ignore this part
if areatype=='point' or areatype=='pointlist':
    arfval=np.array([1.])

# GP 2026-10-26, unknown CALCTYPE: it printed "Defaulting to Annual Maxima Series" but never set calctype (NameError
# after the storm catalog was built); it now stops here. Missing CALCTYPE: 'ams', as before. Old code:
# try:
#     CalcType=cardinfo["CALCTYPE"]
#     if CalcType.lower()=='ams' or CalcType.lower()=='annmax':
#         calctype='ams'
#         print("You've selected to use Annual Maxima Series.")
#     elif CalcType.lower()=='pds' or CalcType.lower()=='partialduration':
#         calctype='pds'
#         print("You've selected to use Partial Duration Series.")
#     else:
#         print("Unrecognized entry for CALCTYPE. Defaulting to Annual Maxima Series.")
# except Exception:
#     calctype='ams'
#     print("Nothing provided for CALCTYPE. Defaulting to Annual Maxima Series.")
calctype=FPA.parse_calctype(cardinfo)                 # GP 2026-10-26: unknown CALCTYPE

#
#
# # here is where we'll determine if you want to write more than one storm per year to realization files
# # ideally would write the if statement better to provide more guidance if the user provides some weird combination, such as 'CALCTYPE pds' and 'NPERYEAR 5'
if Scenarios and areatype.lower()!="pointlist":
    try:
        nperyear=cardinfo["NPERYEAR"]
        if nperyear!='false' or np.int16(nperyear)!=1:
            print("RainyDay will output "+str(np.int16(nperyear))+" storms per synthetic year! If this is a big number, this could be very slow!")
            nperyear=np.int16(nperyear)
            #if nperyear==1:
            #    nperyear='false'
    #        else:
    #            print("RainyDay will output "+str(nperyear)+" storms per synthetic year!")
    except Exception:
        print("You specified writing output scenarios, but didn't specify 'NPERYEAR'. Defaulting to 1 per year!")
        nperyear=1
    
#
#
#
try:
    userdistr=cardinfo["INTENSDISTR"]
    if userdistr.lower()=='false':
        userdistr=np.zeros((1),dtype='bool')
    elif len(userdistr.split(','))==3:
        print("reading user-defined precipitation intensity distribution...")
        userdistr=np.array(userdistr.split(','),dtype='float32')
    else:
        sys.exit("There is a problem with the INTENSDISTR entry!")
except Exception:
    userdistr=np.zeros((1),dtype='bool')
    pass

#
#
# this is trying to catch an error related to an older feature that is not currently supported
if Scenarios:
    try:
        if np.any(np.core.defchararray.find(list(cardinfo.keys()),"SPINPERIOD")>-1):
                sys.exit("'SPINPERIOD' is not currently supported!")
    except Exception:
        pass


try:
    if np.any(np.core.defchararray.find(list(cardinfo.keys()),"UNCERTAINTY")>-1):
        spread=cardinfo["UNCERTAINTY"]
        # GP 2026-10-26, numeric UNCERTAINTY: a JSON number (90) fell back to 'ensemble' silently (spread.lower() failed); a numeric
        # string ("90") ran to the end and crashed in write_frequency_outputs (spreadmean undefined, fixed there). Both
        # now give the quantile band; other values as before. quantilecalc is None unless 'quantile'. Old code:
        # if spread.lower()=='ensemble':
        #     spreadtype='ensemble'
        #     print('"ensemble spread" will be calculated for precipitation frequency analysis...')
        # elif spread.lower()=='all_realizations':
        #     spreadtype='all_realizations'
        #     print(' The estimation from all the realization will be printed for precipitation frequency analysis...')
        # else:
        #     try:
        #         int(spread)
        #     except:
        #         print('unrecognized value for UNCERTAINTY.')
        #     if int(spread)>=0 and int(spread)<=100:
        #         spreadtype='quantile'
        #         quantilecalc=int(spread)
        #         printcalc=(100-quantilecalc)//2
        #         print(str(printcalc)+'th-'+str(quantilecalc+printcalc)+'th interquantile range will be calculated for precipitation frequency analysis...')
        #     else:
        #         sys.exit('invalid quantile range for frequency analysis...')
        spreadtype,quantilecalc=FPA.parse_uncertainty(spread)   # GP 2026-10-26: numeric UNCERTAINTY
    else:
        spreadtype='ensemble'
except Exception:
    spreadtype='ensemble'

try:
    if np.any(np.core.defchararray.find(list(cardinfo.keys()),"RETURNLEVELS")>-1):
        speclevels=cardinfo["RETURNLEVELS"]
        if isinstance(speclevels, str) and speclevels.lower() == 'all':
            print('using all return levels...')
            alllevels=True
        else:
            alllevels=False
            # GP 2026-10-26, RETURNLEVELS comma string: "2,5,10" (the original User's Guide format) stopped below since the
            # list fix of 07021dd; it is now read as the list [2.,5.,10.] (np.float32 gives the same array as for the
            # JSON list [2,5,10]); a JSON list and "all" are unchanged
            speclevels=FPA.returnlevels_from_string(speclevels)
            if isinstance(speclevels, (list, np.ndarray)):
                try:
                    speclevels=np.float32(speclevels)
                except:
                    print("Non-numeric value provided to RETURNLEVELS.")
                speclevels=speclevels[speclevels<=nsimulations+0.00001]
                speclevels=speclevels[speclevels>=0.99999]
            else:
                sys.exit("The format of RETURNLEVELS isn't recognized.  It should be 'all' or a comma separated list.")
    else:
        alllevels=True

except Exception:
    alllevels=True

try:
    rotation=False
    if np.any(np.core.defchararray.find(list(cardinfo.keys()),"ROTATIONANGLE")>-1):
        try:
            rotangle=cardinfo["ROTATIONANGLE"]
        except ImportError:
            "You're trying to use storm rotation, but didn't specify 'ROTATIONANGLE'"
        if rotangle.lower()=='none' or rotangle.lower()=='false' or areatype.lower()=="point":
            rotation=False
        else:
            rotation=True
            if len(rotangle.lower().split(','))!=3:
                sys.exit('Unrecognized entry provided for ROTATIONANGLE.  Should be "none" or "-X,+Y,Nangles".')
            else:
                minangle,maxangle,nanglebins=rotangle.split(',')
                try:
                    minangle=np.float32(minangle)
                except:
                    print('Unrecognized entry provided for ROTATIONANGLE minimum angle.')
                try:
                    maxangle=np.float32(maxangle)
                except:
                    print('Unrecognized entry provided for ROTATIONANGLE maximum angle.')
                try:
                    nanglebins=np.int32(nanglebins,dtype="float32")
                except:
                    print('Unrecognized entry provided for ROTATIONANGLE number of bins.')

                if minangle>0. or maxangle<0.:
                    sys.exit('The minimum angle should be negative and the maximum angle should be positive.')
except Exception:
    rotation=False

if rotation:
    print("storm rotation will be used...")
    delarray=[]


try:
    if np.any(np.core.defchararray.find(list(cardinfo.keys()),"DURATIONCORRECTION")>-1):
        durcorr=cardinfo["DURATIONCORRECTION"]
        if durcorr.lower()=='false':
            durcorrection=False
            print('DURATIONCORRECTION set to "false". No correction will be used.')
        elif durcorr.lower()=='true':
            durcorrection=True
        else:
            print('Invalid option provided for DURATIONCORRECTION (should be "true" or "false"). Defaulting to "false"!')
            durcorrection=False
    else:
        print('No value was given for DURATIONCORRECTION. If applicable, it will default to "false"!')
        durcorrection=False
except Exception:
    durcorrection=False

# Added by DBW, 16 Feb 2021, specifically to support Dr. Emad Habib's team in performing ARF analyses.
# If ARFANALYSIS is used, DURATIONCORRECTION will be turned on automatically, and the scenarios will have a duration equal to DURATION.
#This is sensible if the objective is to create rainfall scenario files for subsequent analysis (ARFs, scaling properties, etc.).
# If ARFANALYSIS is turned off and DURATIONCORRECTION is turned on, then the rainfall scenarios will under most circumstances have a longer duration than DURATION.
# This latter approach (DURATIONCORRECTION on, ARFANALYSIS off) is a more justifiable way of generating rainfall scenarios for flood modeling.
# Also, this ARFANALYSIS option isn't set up to work with 'pointlists'.
# try:
#     if np.any(np.core.defchararray.find(list(cardinfo.keys()),"ARFANALYSIS")>-1):
#         arfcorr=cardinfo["ARFANALYSIS"]
#         if arfcorr.lower()=='false':
#             arfcorrection=False
#         elif arfcorr.lower()=='true' and areatype.lower()!="pointlist":
#             durcorrection=True
#             arfcorrection=True
#             print("ARFANALYSIS set to 'true'. DURATIONCORRECTION will be used. This is only really recommended in you want to create rainfall scenarios for ARF or other spatial analysis. Make sure you know what you're doing!")
#                 else:
#             print('Invalid option provided for ARFANALYSIS (should be "true" or "false"). Defaulting to "false"!')
#             arfcorrection=False
#     else:
#         arfcorrection=False
# except Exception:
#     arfcorrection=False

if CreateCatalog and durcorrection:
    catduration=max([72.,3.*duration])    # this will be slow!
    print('Since DURCORRECTION will be used and a storm catalog will be created, the duration of the catalog will be '+"{0:0.2f}".format(catduration)+' hours')
elif CreateCatalog:
    catduration=duration


#sys.exit('decide what to do with duration correction and timeseparation!')
try:
    timeseparation=cardinfo["TIMESEPARATION"]
except Exception:
    timeseparation=0.

# Added 2/6/2024 by DBW
# This reads in a daily-scale probability of probabilities that will be used when resampling storms
try:
    seasonfile=cardinfo["SEASONALSAMPLING"]
    seasonalsampling=True
    seasonaldata=np.genfromtxt(seasonfile, delimiter=',',skip_header=1)
    seasonaldoy=np.array(seasonaldata[:,0],dtype='int16')
    seasonaldate=np.zeros((len(seasonaldoy)),dtype='datetime64[D]')
    for i in range(0,len(seasonaldate)):
        seasonaldate[i]=RainyDay.day_of_year_to_datetime(1776,seasonaldoy[i])
    if RainyDay.is_monotonic(seasonaldata[:,1]):
        # if data is a cdf already
        seasonalcdf=seasonaldata[:,1]/np.max(seasonaldata[:,1])
    else:
        # if data is a PMF
        seasonalcdf=np.cumsum(seasonaldata[:,2])/np.max(seasonaldata[:,1])
except Exception:
    seasonalsampling=False    


# added 3/1/2024 by DNW
# this tries to accelerate the storm catalog identification:
try:
    catalogstride=cardinfo["CATALOGACCELERATOR"] 
except Exception:
    catalogstride=1
if isinstance(catalogstride, int):
    if catalogstride>1:
        # GP 2026-10-26, rectangular catalog search: the catalog search is the exhaustive FFT for both domain types (catalogNumba, the only user
        # of catalogstride, is no longer called), so CATALOGACCELERATOR has no effect. Old message:
        # print("You've decided to speed up the storm catalog identification by not examining every possibility. Good for you!")
        print("CATALOGACCELERATOR is ignored: the storm catalog search examines every location (exhaustive FFT) for both domain types.")
        
# GP 2026-10-26: RECORDCI (optional, default off; RainyDay_fast fast_record.py): record (sampling) uncertainty of the
# frequency analysis. Parsed and checked here, before the storm catalog is built, so an unsupported run stops early.
recordopts=FR.parse_options(cardinfo)
if recordopts is not None:
    FR.check_options(recordopts,freqanalysis=FreqAnalysis,transpotype=transpotype,samplingtype=samplingtype,calctype=globals().get('calctype'),areatype=areatype,rescaletype=rescaletype,rotation=rotation,intensdistr=userdistr.all()!=False,scenarios=Scenarios,seasonalsampling=seasonalsampling)
# GP 2026-10-26: FREQENGINE (optional, default "montecarlo"; RainyDay_fast fast_record.ExactFA, README_CHANGES B8):
# "exact" computes the frequency analysis from the complete storm x location pool (no realizations: the Monte Carlo draws,
# ranking and _FreqAnalysis.csv are skipped, so no np.random call happens after the seed); "both" adds the exact outputs
# to RainyDay's unchanged Monte Carlo outputs. Same scope as RECORDCI, checked here before the storm catalog is built.
freqengine=FR.parse_engine(cardinfo)
freqexact=(freqengine=='exact')
if freqengine!='montecarlo':
    FR.check_options(None,freqanalysis=FreqAnalysis,transpotype=transpotype,samplingtype=samplingtype,calctype=globals().get('calctype'),areatype=areatype,rescaletype=rescaletype,rotation=rotation,intensdistr=userdistr.all()!=False,scenarios=Scenarios,seasonalsampling=seasonalsampling,who="FREQENGINE '"+freqengine+"'")

#==============================================================================
# THIS BLOCK CONFIGURES SEVERAL THINGS
#============================================================================== 
    
initseed=0
np.random.seed(initseed)
global rainprop   
rainprop=deepcopy(emptyprop) 
    

#==============================================================================
# CREATE NEW STORM CATALOG, IF DESIRED
#==============================================================================  
if CreateCatalog:
    print("creating a new storm catalog...")
    
    flist,nyears=RainyDay.createfilelist(inpath,includeyears,excludemonths)
    # GP 2026-10-22: RAINPATH is a crop of tools/make_domain_crop.py: stop if this run's domain is not inside it
    FIO.check_crop_coverage(inpath,inarea,variables,flist,parameterfile)
    if defaultstorms:
        nstorms=nyears*20
  
    
    # GET SUBDIMENSIONS, ETC. FROM THE NETCDF FILE RATHER THAN FROM RAINPROPERTIES  
    rainprop.spatialres,rainprop.dimensions,rainprop.bndbox,rainprop.timeres,rainprop.nodata,droplist,calendar,time_units=RainyDay.rainprop_setup(flist[0],rainprop,variables)
    spatres=rainprop.spatialres[0]
    
    
    #==============================================================================
    # SET UP THE SUBGRID INFO
    #==============================================================================
    # 'subgrid' defines the transposition domain
    # note that the order of 'latrange' has been changed from earlier efforts, so it starts with lower latutide
    rainprop.subextent,rainprop.subdimensions,latrange,lonrange,indices=RainyDay.findsubbox(inarea,variables,flist[0])
    if ncfdom and (np.any(domainmask.shape!=rainprop.subdimensions)):  ## we should remove it
        sys.exit("Something went terribly wrong :(")        # this shouldn't happen


#==============================================================================
# IF A STORM CATALOG ALREADY EXISTS, USE IT
#==============================================================================  
    
else:
    print("Using the existing storm catalog...")

    rainprop.spatialres=[xres,yres]
    rainprop.timeres=timeres
    no_data = np.unique(catrain.where(catrain < 0, drop = True))
    rainprop.nodata= no_data[~np.isnan(no_data)]
    delt=np.timedelta64(stormtime[-1]-(stormtime[0]))
    catduration=(delt.astype('int')+rainprop.timeres)/60.
    stm_num = len(stormlist)
    # GP 2026-10-26, NSTORMS absent: nstorms was undefined here (NameError), and nstorms_default below too; now
    # every storm of the catalog is used, the same run as NSTORMS = the catalog's storm count
    nstorms,defaultstorms=FPA.default_nstorms(globals().get('nstorms'),defaultstorms,stm_num)
    if nstorms>stm_num:
        print("WARNING: The storm catalog has fewer storms than the specified nstorms")
        nstorms=stm_num 
        
        
    if ncfdom and (np.any(domainmask.shape!=catrain.shape[2:])):
        sys.exit("The domain mask and the storm catalog are different sizes. Do you know what you're doing?")

    rainprop.bndbox=catarea
    rainprop.subextent=rainprop.bndbox
    
    rainprop.dimensions=np.array([len(latrange),len(lonrange)],dtype='int32') 
    rainprop.subdimensions=rainprop.dimensions

if int(duration*60/rainprop.timeres)<=0:
    sys.exit("it appears that you specified a duration shorter than the temporal resolution of the input data!")


if timeseparation<=0. and durcorrection==False:
    timeseparation=duration
elif timeseparation>0. and durcorrection==False:
    timeseparation=timeseparation+duration
elif durcorrection:
    timeseparation=np.max([timeseparation+duration,catduration])

spatres=rainprop.spatialres[0]
# ingridx,ingridy=np.meshgrid(np.arange(rainprop.subextent[0],rainprop.subextent[1]-spatres/1000.,spatres),np.arange(rainprop.subextent[3],rainprop.subextent[2]+spatres/1000.,-spatres)) 
ingridx,ingridy=np.meshgrid(lonrange,latrange)        


#============================================================================
# Do the setup to run for specific times of day!
#=============================================================================

# if CreateCatalog==False:
#     tdates = pd.DatetimeIndex(cattime[:,0])
#     tdates.year
#     nyears=np.max(np.array(tdates.year))-np.min(np.array(tdates.year))+1
# if starthour==0 and endhour==24:
#     hourinclude=np.ones((int(24*60/rainprop.timeres)),dtype='int32')
# else:
#     sys.exit("Restrictions to certain hours isn't currently tested or supported")
#     try:
#         sys.exit("need to fix this")
#         _,temptime,_,_=RainyDay.readnetcdf(flist[0],variables,inarea)
#     except Exception:
#         sys.exit("Can't find the input files necessary for setup to calculate time-of-day-specific IDF curves")

#     starthour=starthour+np.float(rainprop.timeres/60)   # because the netcdf file timestamps correspond to the end of the accumulation period
#     hourinclude=np.zeros((24*60/rainprop.timeres),dtype='int32')
#     temphour=np.zeros((24*60/rainprop.timeres),dtype='float32')

#     for i in np.arange(0,len(temptime)):
#         temphour[i]=temptime[i].astype(object).hour
    
#     # the following line will need to be adapted, due to UTC vs. local issues
#     hourinclude[np.logical_and(np.greater_equal(temphour,starthour),np.less_equal(temphour,endhour))]=1
#     if len(hourinclude)!=len(temptime):
#         sys.exit("Something is wrong in the hour exclusion calculation!")

# hourinclude=hourinclude.astype('bool')

#temptime[hourinclude] # checked, seems to be working right
 
        
#==============================================================================
# SET UP GRID MASK
#==============================================================================
if CreateCatalog:
    print("setting up the grid information and masks...")
    if areatype=="basin":
        if os.path.isfile(wsmaskshp)==False:
            sys.exit("can't find the basin shapefile!")
        else:
            catmask=RainyDay.rastermask(wsmaskshp,rainprop,'fraction')
    
            # DBW 08072023-this is to ensure consistency in orientation with precipitation fields from xarray:
            catmask=np.flipud(catmask)
            #catmask=catmask.reshape(ingridx.shape,order='F')
    elif areatype=="point":
        catmask=np.zeros((rainprop.subdimensions))
        yind=np.where((np.array(latrange)-ptlat)>0)[0][0]  # DBW 08082023: fixed this to account for the flipped north-south grid
        xind=np.where((ptlon-np.array(lonrange))>0)[0][-1]  
        if xind==0 or yind==0 or yind==(len(latrange)-1) or xind==(len(lonrange)-1):
            sys.exit('the point you defined is too close to the edge of the box you defined!')
        else: 
            catmask[yind,xind]=1.0   
    elif areatype=="pointlist":
        catmask=np.zeros((rainprop.subdimensions))
        yind_list=np.zeros_like(ptlatlist,dtype='int32')
        xind_list=np.zeros_like(ptlatlist,dtype='int32')
        for pt in np.arange(0,npoints_list):
            yind_list[pt]=np.where((np.array(latrange)-ptlatlist[pt])>0)[0][0]  # DBW 08082023: fixed this to account for the flipped north-south grid
            xind_list[pt]=np.where((ptlonlist[pt]-np.array(lonrange))>0)[0][-1]  
        if np.any(yind_list==0) or np.any(xind_list==0) or np.any(yind_list==(len(latrange)-1)) or np.any(xind_list==(len(lonrange)-1)):
            sys.exit('the point you defined is too close to the edge of the box you defined!')
        else:
            catmask[yind_list[1],xind_list[1]]=1.0          # the idea of the catmask gets a little goofy in this situation
    elif areatype=="box":
        finelat=np.arange(latrange[0],latrange[-1]+rainprop.spatialres[1]-rainprop.spatialres[0]/1000,rainprop.spatialres[1]/25)
        finelon=np.arange(lonrange[0],lonrange[-1]+rainprop.spatialres[0]-rainprop.spatialres[0]/1000,rainprop.spatialres[0]/25)
    
        subindy=np.logical_and(finelat>boxarea[2]+rainprop.spatialres[1]/1000,finelat<boxarea[3]+rainprop.spatialres[1]/1000)
        subindx=np.logical_and(finelon>boxarea[0]-rainprop.spatialres[0]/1000,finelon<boxarea[1]-rainprop.spatialres[0]/1000)
        
        tx,ty=np.meshgrid(subindx,subindy)
        catmask=np.array(np.logical_and(tx==True,ty==True),dtype='float32')
        
        if len(finelat[subindy])<25 and len(finelon[subindx])<25:    
            print('WARNING: you set POINTAREA to "box", but the box is smaller than a single pixel.  This is not advised.  Either set POINTAREA to "point" or increase the size of the box.')
        
        if len(finelat[subindy])==1 and len(finelon[subindx])==1:
            catmask=np.zeros((rainprop.subdimensions))
            yind=np.where((np.array(latrange)-ptlat)>0)[0][-1]
            xind=np.where((ptlon-np.array(lonrange))>0)[0][-1]  
            if xind==0 or yind==0:
                sys.exit('the point you defined is too close to the edge of the box you defined!')
            else:
                catmask[yind,xind]=1.0
        else:
            def block_mean(ar, fact):
                assert isinstance(fact, int), type(fact)
                sx, sy = ar.shape
                X, Y = np.ogrid[0:sx, 0:sy]
                regions = sy//fact * (X//fact) + Y//fact
                res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
                res.shape = (sx//fact, sy//fact)
                return res
    
            catmask=block_mean(catmask,25)          # this scheme is a bit of a numerical approximation but I doubt it makes much practical difference  
    else:
        sys.exit("unrecognized area type!")

       
# TRIM THE GRID DOWN TO GET THE RECTANGLE THAT BOUNDS THE NONZERO VALUES IN CATMASK. THIS IS NEEDED FOR IDENTIFYING EXTREME STORMS WITH RESPECT TO THAT SCALE
csum=np.where(np.sum(catmask,axis=0)==0)
rsum=np.where(np.sum(catmask,axis=1)==0)

xmin=np.min(np.where(np.sum(catmask,axis=0)!=0))
xmax=np.max(np.where(np.sum(catmask,axis=0)!=0))
ymin=np.min(np.where(np.sum(catmask,axis=1)!=0))
ymax=np.max(np.where(np.sum(catmask,axis=1)!=0))

trimmask=np.delete(catmask,csum,axis=1)
trimmask=np.delete(trimmask,rsum,axis=0)
maskwidth=trimmask.shape[1]
maskheight=trimmask.shape[0]
trimmask=np.array(trimmask,dtype='float32')
catmask=np.array(catmask,dtype='float32')

timeseparation=np.timedelta64(np.int32(timeseparation*60.),'m')
timestep=np.timedelta64(np.int32(rainprop.timeres),'m')

mnorm=np.sum(trimmask)

### Old catalognumba configuration
# xlen=rainprop.subdimensions[1]-maskwidth+1
# ylen=rainprop.subdimensions[0]-maskheight+1

### New catalognumba configuration
# xlen =rainprop.subdimensions[1]-maskwidth
xlen =rainprop.subdimensions[1]-maskwidth + 1  # GP
if (rainprop.subdimensions[1] - maskwidth ) % 2 != 0:
    xloop = (rainprop.subdimensions[1] - maskwidth - 1) / 2
else:
    xloop = (rainprop.subdimensions[1] - maskwidth) / 2
# ylen =rainprop.subdimensions[0]-maskheight-2 
ylen =rainprop.subdimensions[0]-maskheight +1 # GP
if (rainprop.subdimensions[0]-maskheight)% 2 != 0:
    yloop = (rainprop.subdimensions[0]-maskheight -1) / 2
else:
    yloop = (rainprop.subdimensions[0]-maskheight) / 2
halfheight=np.int32(np.ceil(maskheight/2))
halfwidth=np.int32(np.ceil(maskwidth/2))
# GP 2026-10-26, rectangular catalog search: xlen, ylen (= W-mw+1, H-mh+1) are the numbers of anchors per row/column: the range of the
# exhaustive catalog search (catalogFFT_irregular_domainaware, both domain types), of the rectangular uniform draws
# (randint(0,W-mw+1)) and of the KDE grid (kx, ky), so the KDE checkind below drops storms on the first and last anchor
# (0 and xlen-1 = W-mw) as intended. xloop, yloop, rainsum and catalogstride fed only catalogNumba, no longer called.

if CreateCatalog:
    if ncfdom:
        sys.exit("This isn't set up :(")
        ingridx_dom,ingridy_dom=np.meshgrid(domainlon,domainlat)        
    
        ingrid_domain=np.column_stack((ingridx_dom.flatten(),ingridy_dom.flatten())) 
        #ingridx,ingridy=np.meshgrid(np.arange(rainprop.subextent[0],rainprop.subextent[1]-rainprop.spatialres[0]/1000,rainprop.spatialres[0]),np.arange(rainprop.subextent[3],rainprop.subextent[2]+rainprop.spatialres[1]/1000,-rainprop.spatialres[1]))        
        grid_rainfall=np.column_stack((ingridx.flatten(),ingridy.flatten())) 
        
        delaunay=sp.spatial.qhull.Delaunay(ingrid_domain)
        interp=sp.interpolate.NearestNDInterpolator(delaunay,domainmask.flatten())
    
        # in case you want it back in a rectangular grid:
        domainmask=np.reshape(interp(grid_rainfall),ingridx.shape)    
    
    elif domain_type.lower()=='irregular' and shpdom and CreateCatalog:
        domainmask=RainyDay.rastermask(domainshp,rainprop,'simple').astype('float32')
        # DBW 08072023-this is to ensure consistency in orientation with precipitation fields from xarray:
        domainmask=np.flipud(domainmask)

    elif domain_type.lower()=='rectangular':
        domainmask=np.ones((catmask.shape),dtype='float32')    
    else:
        pass



if catmask.shape!=domainmask.shape:
    sys.exit("Oh dear, 'catmask' and 'domainmask' aren't the same size!")


# DBW 08082023: this checks to see if any of catmask is outside of the domainmask. That would be bad. This didn't work before, but now it should   
if np.any(np.logical_and(np.equal(catmask,1.),np.equal(domainmask,0.))):
    sys.exit("it looks as if the location specified in 'POINTAREA' is outside of the transposition domain!")

# exclude points that are outside of the transposition domain:
if areatype=="pointlist" and domain_type=='irregular':
    keeppoints=np.ones_like(yind_list,dtype='bool')
    for pt in np.arange(0,npoints_list):
        if domainmask[yind_list[pt],xind_list[pt]]==0:
            keeppoints[pt]=False
elif areatype=="pointlist" and domain_type=='rectangular':
    sys.exit("this might need to be fixed-check for lat flipping!")
    keeppoints=np.ones_like(yind_list,dtype='bool')
    for pt in np.arange(0,npoints_list):
        if ptlatlist[pt]>rainprop.subextent[3] or ptlatlist[pt]<rainprop.subextent[2] or ptlonlist[pt]>rainprop.subextent[1] or ptlonlist[pt]<rainprop.subextent[0]:
            keeppoints[pt]=False
            

if areatype=="pointlist":
    if np.any(keeppoints)==False:
        sys.exit("it looks as if none of the points in specified in 'POINTLIST' are inside the transposition domain!")
    else:
        yind_list=yind_list[keeppoints]
        xind_list=xind_list[keeppoints] 
        ptlatlist=ptlatlist[keeppoints]
        ptlonlist=ptlonlist[keeppoints]
        npoints_list=len(yind_list)   
    
        
#################################################################################
# STEP 1: CREATE STORM CATALOG
#################################################################################

if CreateCatalog: 
    print("reading precipitation files...")
    
    #==============================================================================
    # SET UP OUTPUT VARIABLE
    #==============================================================================

    rainarray=np.zeros((int(catduration*60/rainprop.timeres),rainprop.subdimensions[0],rainprop.subdimensions[1]),dtype='float32')  
    
    rainsum=np.zeros((ylen,xlen),dtype='float32')
    rainarray[:]=np.nan
    raintime=np.empty((int(catduration*60/rainprop.timeres)),dtype='datetime64[m]')
    raintime[:]=np.datetime64(datetime(1700,1,1,0,0,0))
    
    catmax=np.zeros((nstorms),dtype='float32')
    cattime=np.empty((nstorms,int(catduration*60/rainprop.timeres)),dtype='datetime64[m]')
    cattime[:]=np.datetime64(datetime(1700,1,1,0,0,0))
    catloc=np.empty((nstorms),dtype='float32')
    
    catx=np.zeros((nstorms),dtype='int32')
    caty=np.zeros((nstorms),dtype='int32')
    

    #==============================================================================
    # READ IN RAINFALL
    #==============================================================================
    filerange=range(0,len(flist))
    idxes = RainyDay.find_indices(flist[0],inarea,variables)
    proc_start = time.time()

    # Batched catalog engines (see RainyDay/tools): the constant kernel FFT and
    # domain-validity map are computed once, and blocks of files are processed
    # with batched FFTs (catalog_scan_files). catmax/cattime/catx/caty get the
    # same bookkeeping as the per-timestep loop below, which is then skipped by
    # emptying filerange.
    # Resolve the execution config (engine + performance knobs) from the optional
    # JSON "COMPUTE" block, environment variables, and machine auto-detection —
    # precedence env > JSON > auto. This picks/validates the engine and injects the
    # resolved RAINYDAY_* knobs the code below reads. Irregular domains only.
    if domain_type=='irregular':
        catalog_engine=RainyDay.resolve_compute_config(cardinfo.get('COMPUTE',{}),
            domainmask.shape, max(1,int(round(24*60/rainprop.timeres))),
            int(catduration*60/rainprop.timeres))

    if catalog_engine!='fft_perstep' and domain_type=='irregular':
        print("storm catalog engine: "+catalog_engine)
        nwindow=int(catduration*60/rainprop.timeres)
        blockfiles=int(os.environ.get('RAINYDAY_BLOCKFILES','16'))

        # Benchmark hook (no effect unless RAINYDAY_DUMP_SCANINPUTS is set): pickle
        # the scan inputs and exit, so tools/bench_mp_scan.py can time batchfft_mp
        # via a spawn pool on Windows (where 'fork' is unavailable) without a
        # __main__ guard around this whole script. See RainyDay/tools.
        if os.environ.get('RAINYDAY_DUMP_SCANINPUTS'):
            import pickle
            _,firsttime=RainyDay.readnetcdf(flist[0],variables,idxes,dropvars=droplist,calendar=calendar,time_units=time_units)
            nctx=int(np.ceil(float(nwindow-1)/max(1,len(firsttime))))
            with open(os.environ['RAINYDAY_DUMP_SCANINPUTS'],'wb') as _f:
                pickle.dump(dict(flist=flist,variables=variables,idxes=idxes,droplist=droplist,
                                 calendar=calendar,time_units=time_units,timestep=timestep,
                                 nwindow=nwindow,trimmask=trimmask,domainmask=domainmask,
                                 blockfiles=blockfiles,nctx=nctx),_f)
            sys.exit('scan inputs dumped to '+os.environ['RAINYDAY_DUMP_SCANINPUTS'])

        if catalog_engine=='batchfft_mp':
            import concurrent.futures
            import multiprocessing as mp
            nworkers=int(os.environ.get('RAINYDAY_NWORKERS',str(max(1,(os.cpu_count() or 2)//2))))
            # enough preceding files to cover the rolling window of each chunk
            _,firsttime=RainyDay.readnetcdf(flist[0],variables,idxes,dropvars=droplist,calendar=calendar,time_units=time_units)
            nctx=int(np.ceil(float(nwindow-1)/max(1,len(firsttime))))
            bounds=np.linspace(0,len(flist),nworkers+1).astype(int)
            tasks=[]
            for c in range(nworkers):
                lo,hi=bounds[c],bounds[c+1]
                if lo<hi:
                    tasks.append((flist[lo:hi],flist[max(0,lo-nctx):lo],variables,idxes,droplist,calendar,time_units,timestep,nwindow,trimmask,domainmask,blockfiles,False,'batchfft'))
            print("scanning "+str(len(flist))+" files with "+str(len(tasks))+" workers...")
            # 'fork' is required here: this script has no __main__ guard, and the
            # 'spawn'/'forkserver' start methods re-import (i.e. re-run) it in
            # every worker. The workers only use numpy/scipy/netCDF4, which are
            # fork-safe on macOS.
            with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers,mp_context=mp.get_context('fork')) as executor:
                parts=list(executor.map(RainyDay.catalog_scan_worker,tasks))
            scantimes=np.concatenate([p[0] for p in parts])
            scanmax=np.concatenate([p[1] for p in parts])
            scany=np.concatenate([p[2] for p in parts])
            scanx=np.concatenate([p[3] for p in parts])
        elif catalog_engine in ('batchtf','batchcuda'):
            scantimes,scanmax,scany,scanx,scanfidx=RainyDay.catalog_scan_files(flist,[],variables,idxes,droplist,calendar,time_units,timestep,nwindow,trimmask,domainmask,blockfiles,True,catalog_engine,return_fileidx=True)
        else:   # 'batchfft'
            scantimes,scanmax,scany,scanx=RainyDay.catalog_scan_files(flist,[],variables,idxes,droplist,calendar,time_units,timestep,nwindow,trimmask,domainmask,blockfiles,True,catalog_engine)

        if catalog_engine in ('batchtf','batchcuda'):
            # GPU values are only a screen: every timestep whose accept/merge
            # decision Metal's float rounding could flip is recomputed with the
            # exact CPU FFT inside catalog_select_confirmed, so the catalog is
            # bit-identical to the 'batchfft'/'fft_perstep' engines
            gpudelta=float(os.environ.get('RAINYDAY_GPU_DELTA','1e-3'))
            _,firsttime=RainyDay.readnetcdf(flist[0],variables,idxes,dropvars=droplist,calendar=calendar,time_units=time_units)
            nctx=int(np.ceil(float(nwindow-1)/max(1,len(firsttime))))
            RainyDay.catalog_select_confirmed(scantimes,scanmax,scany,scanx,scanfidx,flist,nctx,catmax,cattime,catx,caty,catduration,timestep,timeseparation,variables,idxes,droplist,calendar,time_units,nwindow,trimmask,domainmask,blockfiles,gpudelta)
        else:
            # identical catalog bookkeeping as in the per-timestep loop below
            for k in range(len(scantimes)):
                rainmax=scanmax[k]; ycat=int(scany[k]); xcat=int(scanx[k])
                starttime=scantimes[k]-np.timedelta64(int(catduration*60.),'m')
                subtime=np.arange(scantimes[k],starttime,-timestep)[::-1]
                minind=np.argmin(catmax)
                tempmin=catmax[minind]
                if rainmax>tempmin:
                    checksep=scantimes[k]-cattime[:,-1]
                    if (checksep<timeseparation).any():
                        checkind=np.where(checksep<timeseparation)
                        if rainmax>=catmax[checkind]:
                            catmax[checkind]=rainmax
                            cattime[checkind,:]=subtime
                            catx[checkind]=xcat
                            caty[checkind]=ycat
                    else:
                        catmax[minind]   =rainmax
                        cattime[minind,:]=subtime
                        catx[minind]     =xcat
                        caty[minind]     =ycat
        filerange=range(0)   # batched engine done; skip the per-timestep loop below

    for i in filerange:
        #startpc = time.time()
        infile=flist[i]
        startrd = time.time()

        inrain,intime=RainyDay.readnetcdf(infile,variables,idxes,dropvars =droplist,calendar=calendar,time_units=time_units)

        # endrd = time.time(); print("readnetcdf time:", endrd-startrd)
        
        #inrain=inrain[hourinclude,:]
        #intime=intime[hourinclude]
        inrain[inrain<0.]=np.nan
        # inrain = inrain.where(inrain >= 0, np.nan)
      
        print('Processing file '+str(i+1)+' out of '+str(len(flist))+' ('+"{0:0.0f}".format(100*(i+1)/len(flist))+'%): '+infile.split('/')[-1])
        
        # THIS FIRST PART BUILDS THE STORM CATALOG
        start = time.time()
        for k in np.arange(0,len(intime)):     
            starttime=intime[k]-np.timedelta64(int(catduration*60.),'m')
            raintime[-1]=intime[k]
            # stt = time.time()
            rainarray[-1,:]=inrain[k,:]
            # ett = time.time();print(ett-stt)
            #rainarray[-1,:]=np.reshape(inrain[k,:],(rainprop.subdimensions[0],rainprop.subdimensions[1]))
            subtimeind=np.where(np.logical_and(raintime>starttime,raintime<=raintime[-1]))
            subtime=np.arange(raintime[-1],starttime,-timestep)[::-1]
            temparray=np.squeeze(np.nansum(rainarray[subtimeind,:],axis=1))
            
            if domain_type=='irregular':
                temparray = temparray * domainmask
                # startct = time.time()
                
                # Parallelized CPU version of catalogNumba for irregular domains
                # rainmax,ycat,xcat=RainyDay.catalogNumba_irregular(temparray,trimmask,xlen,ylen,xloop,yloop,maskheight,maskwidth,rainsum,stride=catalogstride)
                # CPU version using convolution for irregular domains
                # rainmax,ycat,xcat=RainyDay.catalogFFT_irregular(temparray, trimmask)
                rainmax,ycat,xcat=RainyDay.catalogFFT_irregular_domainaware(temparray, trimmask, domainmask)
                # Parallelized GPU version of catalogTF for irregular domains - For Macs
                # rainmax,ycat,xcat=RainyDay.catalogTF_irregular(temparray,trimmask)
                # Parallelized GPU version of catalogNumba for irregular domains - For CUDA NVIDIA GPUs
                # rainmax,ycat,xcat=RainyDay.catalog_gpu_cupy(temparray,trimmask)
                
                # rainmax,ycat,xcat=RainyDay.catalogNumba_irregular(temparray,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask,stride=catalogstride)

            else:
                # GP 2026-10-26, rectangular catalog search: since 0cc1f2c xlen=W-mw+1, so the mirrored catalogNumba sliced mw-1 columns and
                # raised a numba broadcast ValueError for masks >= 3 cells wide or tall (1-2 cell masks ran but kept
                # the off-by-one stored location of three mirrored quadrants). The rectangular domain now uses the same
                # exhaustive search as irregular domains (every anchor 0..W-mw, 0..H-mh) with domainmask = all ones
                # (set above for DOMAINTYPE rectangular). Old code:
                # rainmax,ycat,xcat=RainyDay.catalogNumba(temparray,trimmask,xlen,ylen,xloop,yloop,maskheight,maskwidth,rainsum,stride=catalogstride)
                rainmax,ycat,xcat=RainyDay.catalogFFT_irregular_domainaware(temparray,trimmask,domainmask)   # GP 2026-10-26: rectangular catalog search
                
            minind=np.argmin(catmax)
            tempmin=catmax[minind]
            if rainmax>tempmin:
                checksep=intime[k]-cattime[:,-1]
                #if intime[k]>np.datetime64('2500-01-01T01:00'):
                #    sys.exit("weird time")
                
                if (checksep<timeseparation).any():
                    checkind=np.where(checksep<timeseparation)
                    if rainmax>=catmax[checkind]:
                        catmax[checkind]=rainmax
                        cattime[checkind,:]=subtime
                        catx[checkind]=xcat
                        caty[checkind]=ycat

                else:
                    catmax[minind]   =rainmax
                    cattime[minind,:]=subtime
                    catx[minind]     =xcat
                    caty[minind]     =ycat
            
            rainarray[0:-1,:]=rainarray[1:int(catduration*60/rainprop.timeres),:]
            raintime[0:-1]=raintime[1:int(catduration*60/rainprop.timeres)] 
        # print(f"✅ Time elapsed: {time.time() - start:.4f} seconds")
        #endpc = time.time()
        #print('overall time:', endpc-startpc)
    # scan-only timer: excludes the storm re-read/write part below, which is the
    # same for every catalog engine
    print(f"catalog scan timer: {(time.time()-proc_start)/60.:0.2f} minutes")
#%%
    if np.count_nonzero(catmax) < nstorms:
        zero_ind = np.where(catmax ==0)[0][0]
        catmax = catmax[0:zero_ind]
        nstorms = zero_ind
        print(f"The number of storms found are lesser than the storms defined in JSON, trimming the storm\
              to {zero_ind} storms")
    
    sind=np.argsort(catmax)
    cattime=cattime[sind,:]
    catx=catx[sind]
    caty=caty[sind]    
    catmax=catmax[sind]/mnorm*rainprop.timeres/60.
    # we might need something here that catches instances when NSTORMS is big, but the actual amount of data fed in isn't enough to find that many storms. This produced problems for me.
    if os.path.exists(fullpath + '/StormCatalog'):
        shutil.rmtree(fullpath + '/StormCatalog')
    os.mkdir(fullpath + '/StormCatalog')
    
    # This part saves each storm as single file #
    _,readtime = RainyDay.readnetcdf(flist[0],variables,idxes,dropvars=droplist,calendar=calendar,time_units=time_units)
    print("Writing Storm Catalog!")
    if catalogstorage=='light':                       # GP 2026-10-22: CATALOGSTORAGE "light" (fast_io.py)
        lightcat=FIO.LightCatalogWriter(fullpath+'/StormCatalog/'+catalogname,scenarioname,parameterfile,catmax,catx,caty,cattime,
                                        latrange,lonrange,catmask,domainmask,nstorms,rainprop.timeres,flist,variables,idxes,
                                        droplist,calendar,time_units,readtime[0])
    for i in range(nstorms):
        start_time = cattime[i,0]
        end_time   = cattime[i,-1]
        current_datetime = start_time
        dataset_date = None
        catrain = np.zeros((int(catduration*60/rainprop.timeres),rainprop.subdimensions[0]\
                            ,rainprop.subdimensions[1]),dtype='float32')
        k  = 0 
        stm_file = None
        count = 0
        while current_datetime <= end_time:
            if RainyDay.check_time(readtime[0]):
                current_date = np.datetime_as_string(current_datetime, unit='D')
            else:
                current_date = np.datetime_as_string(current_datetime - rainprop.timeres, unit='D')
            if current_date != dataset_date:
                dataset_date = current_date
                # This loop searches for the right file to open from the filelist
                for file in flist:
                    match = re.search(r'\d{4}(?:\d{2})?(?:\d{2}|\-\d{2}\-\d{2}|\/\d{2}/\d{2})',\
                                      os.path.basename(file))
                    if current_date.replace("-","") == match.group().replace("-","").replace("/",""):
                        stm_file = file
                        break
                # stm_rain,stm_time = RainyDay.readnetcdf(stm_file,variables,indices)
                stm_rain,stm_time = RainyDay.readnetcdf(stm_file,variables,idxes,dropvars=droplist,calendar=calendar,time_units=time_units)

            # If there is no match (missing file), fill with zeros. Useful for datasets with missing days in leap years
            match = np.where(stm_time == current_datetime)[0] if len(stm_time) > 0 else []
            if len(match) == 0:
                print(f"[Info] {current_datetime} not found in data. Filling with zeros.")
                catrain[k, :] = 0.0
            else:
                cind = match[0]
                catrain[k, :] = stm_rain[cind, :]
                
            # cind = np.where(stm_time == current_datetime)[0][0]
            # catrain[k,:] = stm_rain[cind,:]
            current_datetime += rainprop.timeres
            k += 1

        storm_time = np.datetime_as_string(start_time, unit='D').replace("-","")
        storm_name = fullpath +'/StormCatalog/' + catalogname+'_storm_'+str(i+1) +"_"+ storm_time+".nc"
        print("Writing Storm "+ str(i+1) + " out of " + str(nstorms))
        # GP 2026-10-22: CATALOGSTORAGE "light": no storm file, only its name and the CRC32 of its rain
        # (NaN -> -9999 in catrain as writecatalog does)
        if catalogstorage=='light':
            lightcat.add(i,catrain,storm_name)
            continue
        try:
            RainyDay.writecatalog(scenarioname,catrain,\
                                  catmax,\
                                      catx,caty,\
                                          cattime,latrange,lonrange,\
                                              storm_name,catmask,parameterfile,domainmask,\
                                                 nstorms,catduration,storm_num=int(i),timeresolution=rainprop.timeres)
        except OSError as exc:
            if exc.errno == 13:   ### This checks the permission error. ## '13' is error no for permission error.
                os.remove(storm_name)
                print("The file is removed to make way for a new file")
                RainyDay.writecatalog(scenarioname,catrain,\
                                      catmax,\
                                          catx,caty,\
                                              cattime,latrange,lonrange,\
                                                  storm_name,catmask,parameterfile,domainmask,\
                                                     nstorms,catduration,storm_num=int(i),timeresolution=rainprop.timeres)
            else:
                # previously this branch silently swallowed the error, leaving
                # storms missing from the catalog with no indication why
                print("ERROR: failed to write "+storm_name+": "+repr(exc))
                raise
    if catalogstorage=='light':                       # GP 2026-10-22: CATALOGSTORAGE "light"
        print("Light storm catalog written: "+lightcat.close())
    end = time.time()
    print(f"catalog timer: {(end-proc_start)/60.:0.2f} minutes")

    # GP 2026-10-22: storm files or the virtual storm paths of the light catalog
    # stormlist = glob.glob(fullpath+'/StormCatalog/'+catalogname + '_storm_'+'*' + '.nc')
    stormlist = FIO.list_storms(fullpath+'/StormCatalog/'+catalogname, catalogname, catalogstorage)
    stormlist = sorted(stormlist, key=lambda path: RainyDay.extract_storm_number(path, catalogname))
#%%
#################################################################################
# STEP 2: RESAMPLING
#################################################################################
       
print("trimming storm catalog...")

if CreateCatalog==False:
    if duration>catrain.shape[0]*rainprop.timeres/60.:
        sys.exit("The specified duration is longer than the length of the storm catalog")

origstormsno=np.arange(0,len(stormlist),dtype='int32')                 
# EXLCUDE "BAD" STORMS OR FOR DOING SENSITIVITY ANALYSIS TO PARTICULAR STORMS.  THIS IS PARTICULARLY NEEDED FOR ANY RADAR RAINFALL PRODUCT WITH SERIOUS ARTIFACTS
# includestorms=np.ones((len(stormlist)),dtype="bool")

try:
    exclude=cardinfo["EXCLUDESTORMS"]
    # GP 2026-10-26, EXCLUDESTORMS single string: one storm number as a string ("17") was ignored (nothing excluded); same objects as the code
    # below for every input that code read correctly. Old code:
    # if isinstance(exclude, str):
    #     if ',' in exclude:
    #         exclude = [np.int32(exstorm) for exstorm in exclude.split(",")]
    #     elif '-' in exclude:
    #         start_year, end_year = map(int, exclude.split('-'))
    #         exclude = [exstorm for exstorm in range(start_year, end_year + 1)]
    #     elif exclude.lower() == 'none':
    #         exclude = []
    #     else:
    #         exclude =[]
    # elif isinstance(exclude, list):
    #     exclude = exclude
    # else:
    #     exclude = []
    exclude=FPA.parse_excludestorms(exclude)          # GP 2026-10-26: EXCLUDESTORMS single string
    ## Check if storm numbe given in exlude exist in storm catalog.   
    if len(exclude) > 0:
        for storms in exclude:
            if storms in [RainyDay.extract_storm_number(storm, catalogname) for storm in stormlist]:
                continue
            else:
                sys.exit("something seems wrong! You are excluding storms that aren't in the catalog. Give the exact storm number if you are excluding storm and also nstorm is less than "\
                 "the original storm")
except Exception:
    exclude=[]

  
if len(exclude) != 0 and CreateCatalog==False:
    includestorms = [True if RainyDay.extract_storm_number(storm, catalogname) not in exclude else False for storm in stormlist]
    stormlist = [storm for storm in stormlist if RainyDay.extract_storm_number(storm, catalogname) not in exclude]  
    catmax=catmax[includestorms]
    catx=catx[includestorms]
    caty=caty[includestorms]
    cattime=cattime[includestorms,:]  
elif len(exclude) != 0 and CreateCatalog:
		sys.exit("You are excluding storms from a new storm catalog, without inspecting them first. This seems like a bad idea.")
else:
    includestorms = np.ones((len(stormlist)),dtype="bool")
    
# includestorms[np.isclose(catmax,0.)]=False   ## Do we need to check this
       
  
modstormsno=origstormsno[includestorms]


# EXCLUDE STORMS BY MONTH AND YEAR
# THIS SECTION EXISTS IN CASE YOU WANT TO USE A PRE-EXISTING STORM CATALOG THAT HASN'T CONSIDERED ANY MONTH-BASED OR YEAR-BASED EXCLUSION
if CreateCatalog==False:  
    if includeyears == False:
        # GP 2026-10-26, EXCLUDEMONTHS on an existing catalog: the file-name month is a string ("08") and excludemonths holds integers, so catinclude kept
        # every storm while stormlist (np.int32 below) dropped the excluded months: catmax/catx/caty/cattime no longer
        # lined up with stormlist. Same np.int32 month as stormlist and the INCLUDEYEARS branch. Old code:
        # catinclude = [True if RainyDay.extract_date(storm, catalogname)[4:6]\
        #          not in excludemonths else False for storm in stormlist]
        catinclude = [True if np.int32(RainyDay.extract_date(storm, catalogname)[4:6])\
                 not in excludemonths else False for storm in stormlist]
        stormlist= [storm for storm in stormlist if np.int32(RainyDay.extract_date(storm, catalogname)\
                    [4:6]) not in excludemonths]
    else:
        catinclude = [True if np.int32(RainyDay.extract_date(storm, catalogname)[:4])\
                 in includeyears and np.int32(RainyDay.extract_date(storm, catalogname)[4:6])\
                 not in excludemonths else False for storm in stormlist]
        stormlist = [storm for storm in stormlist if np.int32(RainyDay.extract_date(storm, catalogname)\
                    [4:6]) not in excludemonths and np.int32(RainyDay.extract_date(storm, catalogname)[:4])\
                     in includeyears]  
    catmax=catmax[catinclude]
    catx=catx[catinclude]
    caty=caty[catinclude]
    cattime=cattime[catinclude,:] 
    nstorms_cat=len(stormlist)
    if defaultstorms:
        if nstorms_default<modstormsno.shape[0]:
            modstormsno=modstormsno[-nstorms_default:]   ### we have deleted this nstorms_default
    else:
        modstormsno=modstormsno[catinclude]      #### How to handle this?
        
else:
    nstorms_cat=len(stormlist)
if nstorms<nstorms_cat:
    stormlist = stormlist[-nstorms:]
    nstorms = len(stormlist)

    catmax=catmax[-nstorms:]
    catx=catx[-nstorms:]
    caty=caty[-nstorms:]
    cattime=cattime[-nstorms:,:]
    modstormsno=modstormsno[-nstorms:]   ### What are we doing here?
else:    
    nstorms= len(stormlist)
stormnumber = [RainyDay.extract_storm_number(storm, catalogname) for storm in stormlist]   ## We can use this variable somewhere.



#==============================================================================
# If the storm catalog has a different duration than the specified duration, fix it!
# find the max rainfall for the N-hour duration, not the M-day duration
#==============================================================================

# DBW, 08012023: currently this is inactive, 

durationcheck=60./rainprop.timeres*duration==np.float32(catrain.shape[0])
if durationcheck==False:
    print("Storm catalog duration is longer than the specified duration...")
    print("Sorry, but we're turning DURATIONCORRECTION on. While there might be specific use cases where what you're trying to do makes sense, it is more likely that it doesn't. And it is difficult to sort out how to handle this situation in the refactored code.")

#  # if (durationcheck==False and durcorrection==True) or (durationcheck==False and DoDiagnostics): 
# if (durationcheck==False and durcorrection==False): 
#     print("checking storm catalog duration, and adjusting if needed...")
    
#     # if you are using a catalog that is longer in duration than your desired analysis, this happens:
#     print("Storm catalog duration is longer than the specified duration, finding max precipitation periods for specified duration...")

#     # the first run through trims the storm catalog duration to the desired duration, for diagnostic purposes if needed...       
#     dur_maxind=np.array((nstorms),dtype='int32')
#     dur_x=0
#     dur_y=0
#     dur_j=0
#     temprain=np.zeros((nstorms,int(duration*60/rainprop.timeres),rainprop.subdimensions[0],rainprop.subdimensions[1]),dtype='float32')
#     rainsum=np.zeros((rainprop.subdimensions[0]-maskheight+1,rainprop.subdimensions[1]-maskwidth+1),dtype='float32')

#     # I think the following commented block was wrong, but haven't fully tested the change-DBW 1/24/2020
# #        if durcorrection:
# #            catmax_subdur=np.zeros_like(catmax)
# #            catx_subdur=np.zeros_like(catx)
# #            caty_subdur=np.zeros_like(caty)
# #            cattime_subdur=cattime
    
#     catmax_subdur=  np.zeros_like(catmax)
#     catx_subdur  =  np.zeros_like(catx)
#     caty_subdur  =  np.zeros_like(caty)
#     cattime_subdur=cattime

#     temptime=np.empty((nstorms,int(duration*60/rainprop.timeres)),dtype='datetime64[m]')
#     for i in range(0,nstorms):
#         #if (100*((i+1)%(nstorms//10)))==0:
#         print('adjusting duration of storms, '+"{0:0.0f}".format(100*(i+1)/nstorms)+'% complete...')
#         dur_max=0.
#         for j in range(0,catrain.shape[1]-int(duration*60/rainprop.timeres)):
#             maxpass=np.nansum(catrain[i,j:j+int(duration*60./rainprop.timeres),:],axis=0)
            
#             if domain_type.lower()=='irregular':
#                 maxtemp,tempy,tempx=RainyDay.catalogNumba_irregular(maxpass,trimmask,xlen,ylen,maskheight,maskwidth,rainsum,domainmask)   
#             else:
#                 maxtemp,tempy,tempx=RainyDay.catalogNumba(maxpass,trimmask,xlen,ylen,maskheight,maskwidth,rainsum)                       
 
#             if maxtemp>dur_max:
#                 dur_max=maxtemp
#                 dur_x=tempx
#                 dur_y=tempy
#                 dur_j=j

#         catmax_subdur[i]=dur_max
#         catx_subdur[i]=dur_x
#         caty_subdur[i]=dur_y  

#         temprain[i,:]=catrain[i,dur_j:dur_j+int(duration*60./rainprop.timeres),:]
#         temptime[i,:]=cattime[i,dur_j:dur_j+int(duration*60./rainprop.timeres)]
    
#     catrain_subdur=temprain
#     cattime_subdur=temptime

#     sind=np.argsort(catmax_subdur)
    
#     cattime=cattime_subdur[sind,:]
#     catx=catx_subdur[sind]
#     caty=caty_subdur[sind]
#     catrain=catrain_subdur[sind,:]
#     catmax=catmax_subdur[sind]/mnorm*rainprop.timeres/60.        


#==============================================================================
# IF THE USER IS SUPPLYING A DISTRIBUTION FOR THE INTENSITY, NORMALIZE THE FIELDS
# SO THAT THE INTENSITY CAN BE APPLIED PROPERLY
#==============================================================================




#==============================================================================
# Create kernel density smoother of transposition probability, even if you don't use it for resampling
#==============================================================================
print("calculating transposition probabilities...")

kx,ky=np.meshgrid(np.arange(0,rainprop.subdimensions[1]-maskwidth+1),np.arange(0,rainprop.subdimensions[0]-maskheight+1))
kpositions=np.vstack([ky.ravel(),kx.ravel()])

if domain_type=='rectangular':
    # this "checkind" line seems to cause some unrealistic issues with irregular domains, which makes sense
    checkind=np.where(np.logical_and(np.logical_and(caty!=0,catx!=0),np.logical_and(caty!=ylen-1,catx!=xlen-1)))
    invalues=np.vstack([caty[checkind], catx[checkind]])
else:
    invalues=np.vstack([caty, catx])
    
# Comment the three below lines to conduct realizations with a low number of events. 
# This because gaussian_kde needs a relatevely large number of events.
stmkernel=stats.gaussian_kde(invalues,bw_method=RainyDay.my_kde_bandwidth)
pltkernel=np.multiply(np.reshape(stmkernel(kpositions), kx.shape),domainmask[0:rainprop.subdimensions[0]-maskheight+1,0:rainprop.subdimensions[1]-maskwidth+1])
pltkernel=pltkernel/np.nansum(pltkernel)
tempmask=deepcopy(domainmask[0:rainprop.subdimensions[0]-maskheight+1,0:rainprop.subdimensions[1]-maskwidth+1])

if transpotype=='uniform':
    tempsum=np.nansum(tempmask)
    temparr=np.arange(1.,tempsum+1.)/tempsum
    tempmask[np.equal(tempmask,0.)]=np.nan
    tempmask[~np.isnan(tempmask)]=temparr
    tempmask[np.isnan(tempmask)]=100.
    cumkernel=tempmask
    cumkernel=np.expand_dims(cumkernel,2)        # do we need this???
elif transpotype=='nonuniform' or rescaletype!='none':
    # stmkernel=stats.gaussian_kde(invalues,bw_method=RainyDay.my_kde_bandwidth)
    # pltkernel=np.multiply(np.reshape(stmkernel(kpositions), kx.shape),domainmask[0:rainprop.subdimensions[0]-maskheight+1,0:rainprop.subdimensions[1]-maskwidth+1])
    # pltkernel=pltkernel/np.nansum(pltkernel)
    smoothsig=5
    # the following is an important fix. All prior kernel-based results are conceptually incorrect-DBW 1/24/2018. But this hasn't been thoroughly vetted!
    if areatype=="pointlist":
        transpokernel=np.empty(pltkernel.shape,dtype='float64')
        cumkernel=np.empty((tempmask.shape[0],tempmask.shape[1],npoints_list),dtype='float32')
        for pt in range(0,npoints_list):
            phome=pltkernel[yind_list[pt],xind_list[pt]]
            if np.isclose(phome,0.):
                sys.exit("There is a zero probability of transposition to your location of interest. RainyDay's kernel-based non-uniform transposition scheme doesn't know how to cope with this...")
            transpokernel[np.less(pltkernel,phome)]=pltkernel[np.less(pltkernel,phome)]/phome
            transpokernel[np.greater(pltkernel,phome)]=phome/pltkernel[np.greater(pltkernel,phome)]
            transpokernel[np.equal(pltkernel,phome)]=1.0
            transpokernel=RainyDay.mysmoother(transpokernel,sigma=[smoothsig,smoothsig])
            #transpokernel=sp.ndimage.filters.gaussian_filter(transpokernel, [3,3], mode='nearest')
            transpokernel[np.equal(pltkernel,0.)]=0.
            rescaler=np.nansum(transpokernel)
            transpokernel=transpokernel/rescaler  
            
            cumkernel[:,:,pt]=np.array(np.reshape(np.cumsum(transpokernel),(kx.shape)),dtype='float32')
            tempmask[np.equal(tempmask,0.)]=100.
            cumkernel[np.equal(tempmask,100.),pt]=100.         
    else:
        transpokernel=np.empty(pltkernel.shape,dtype='float64')
        phome=pltkernel[ymin,xmin]          # I hope this is right... I think so.
        if np.isclose(phome,0.):
            sys.exit("There is a zero probability of transposition to your location of interest. RainyDay's kernel-based non-uniform transposition scheme doesn't know how to cope with this...")
        transpokernel[np.less(pltkernel,phome)]=pltkernel[np.less(pltkernel,phome)]/phome
        transpokernel[np.greater(pltkernel,phome)]=phome/pltkernel[np.greater(pltkernel,phome)]
        transpokernel[np.equal(pltkernel,phome)]=1.0
        transpokernel=RainyDay.mysmoother(transpokernel,sigma=[smoothsig,smoothsig])
        #transpokernel=sp.ndimage.filters.gaussian_filter(transpokernel, [3,3], mode='nearest')
        transpokernel[np.equal(pltkernel,0.)]=0.
        rescaler=np.nansum(transpokernel)
        transpokernel=transpokernel/rescaler  
        
        cumkernel=np.array(np.reshape(np.cumsum(transpokernel),(kx.shape)),dtype=np.float64)
        tempmask[np.equal(tempmask,0.)]=100.
        cumkernel[np.equal(tempmask,100.)]=100.
        cumkernel=np.expand_dims(cumkernel,2)    # do we need this??? 
        
        
#==============================================================================
# DO YOU WANT TO CREATE DIAGNOSTIC PLOTS?
#==============================================================================
#%%
FP.tic('diagnostics')                                 # GP 2026-10-22: RAINYDAY_TIMERS
if DoDiagnostics: 
    for f in glob.glob(diagpath+'*.png'):
        os.remove(f)      
    if areatype.lower()=="box":
        from shapely.geometry.polygon import LinearRing
        lons = [boxarea[0], boxarea[0], boxarea[1], boxarea[1]]
        lats = [boxarea[2], boxarea[3], boxarea[3], boxarea[2]]
        ring = LinearRing(list(zip(lons, lats)))
    elif areatype.lower()=="point":
        from shapely.geometry.polygon import LinearRing
        # lons = [ptlon-rainprop.spatialres[0]/2., ptlon-rainprop.spatialres[0]/2., ptlon+rainprop.spatialres[0]/2., ptlon+rainprop.spatialres[0]/2.]
        # lats = [ptlat-rainprop.spatialres[1]/2., ptlat+rainprop.spatialres[1]/2.,ptlat+rainprop.spatialres[1]/2., ptlat-rainprop.spatialres[1]/2.]
        closest_lon = lonrange[np.abs(lonrange - ptlon).argmin()] + rainprop.spatialres[0]/2
        closest_lat = latrange[np.abs(latrange - ptlat).argmin()] - rainprop.spatialres[1]/2
        lons = [closest_lon - rainprop.spatialres[0]/2, closest_lon - rainprop.spatialres[0]/2, closest_lon + rainprop.spatialres[0]/2, closest_lon + rainprop.spatialres[0]/2]
        lats = [closest_lat - rainprop.spatialres[1]/2, closest_lat + rainprop.spatialres[1]/2, closest_lat + rainprop.spatialres[1]/2, closest_lat - rainprop.spatialres[1]/2]
        
        ring = LinearRing(list(zip(lons, lats)))
        
    print("preparing diagnostic plots (this could take a while)...")
    
    if rainprop.subdimensions[0]>rainprop.subdimensions[1]:
        figsizex=5
        figsizey=5+0.25*5*np.float32(rainprop.subdimensions[0])/rainprop.subdimensions[1]
    elif rainprop.subdimensions[0]<rainprop.subdimensions[1]: 
        figsizey=5
        figsizex=5+0.25*5*np.float32(rainprop.subdimensions[0])/rainprop.subdimensions[1]
    else:
        figsizey=5   
        figsizex=5
        
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        try:
            #wmap = shpreader.Reader(wsmaskshp)
            wmap_feature=ShapelyFeature(Reader(wsmaskshp).geometries(),  crs=ccrs.PlateCarree())
        except ValueError:
                print("problem plotting the watershed map; skipping...")
                
    if domain_type.lower()=="irregular" and os.path.isfile(domainshp):
        domain_feature=ShapelyFeature(Reader(domainshp).geometries(),  crs=ccrs.PlateCarree())
#    if BaseMap.lower()!='none':
#        try: 
#            sys.exit("fix this")
#        except ValueError:
#                print("problem plotting the basemap; skipping...")   
                
                
    outerextent=np.array(rainprop.subextent,dtype='float32')
    coast_10m = cfeature.NaturalEarthFeature("physical", "land", "10m", edgecolor="k", facecolor="0.8")
    
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')  
    
    if rainprop.subdimensions[1]>rainprop.subdimensions[0]:
        orientation='horizontal'
    else:
        orientation='vertical'
    proj = ccrs.PlateCarree()
    
    plotlat=latrange.values
    plotlon=lonrange.values

    
    # =============================================================================
    #     redoing plotting to be consistent with 1 storm per file configuration
    # =============================================================================
    
    for i in np.arange(0,nstorms):
        # plotrain,plottime,_,_,_,_,_,_,_,_,_ = RainyDay.readcatalog(stormlist[i])
        plotrain,plottime,_,_,_,_,_,_,_,_,_ = FIO.readcatalog(stormlist[i])     # GP 2026-10-22: full or light catalog
        plotrain = plotrain.where(plotrain >= 0) ##Replace the missing flags
        temprain = plotrain.sum(dim = 'time', skipna =True) * rainprop.timeres / 60.0
        # Shift coordinates to the center for plotting. RainyDay coordinates are upper left
        temprain['longitude'] = temprain['longitude'] + rainprop.spatialres[0]/2
        temprain['latitude'] = temprain['latitude'] - rainprop.spatialres[1]/2
        
        if i == 0:
            mu_t = temprain    
            M2 = temprain * 0.  # Initialize M2 as a DataArray with the same shape as temprain but all values set to 0.
        else:
            oldmu = mu_t
            mu_t = oldmu + (temprain - oldmu) / (i + 1)
            M2 = M2 + (temprain - oldmu) * (temprain - mu_t)
            if i > 1:
                std_t = np.sqrt(M2 / i)
        
        if DoDiagnostics_stats == False:
            print("plotting diagnostics for storm "+str(i+1)+" out of "+str(nstorms))
            fig = plt.figure(figsize=(figsizex,figsizey))
            ax=plt.axes(projection=proj)
            #ax.set_extent(outerextent)
            if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
                ax.add_feature(wmap_feature,edgecolor="red", facecolor='none',zorder = 3)
                # Add the basin but transposed to the new location
                xnew_trans = lonrange[catx[i]]+(maskwidth/2)*rainprop.spatialres[0]
                ynew_trans = latrange[caty[i]]+(maskheight/2)*rainprop.spatialres[1]
                # Read the original geometries
                original_geometries = list(Reader(wsmaskshp).geometries())
                # Use the trimmed-mask window anchor rather than the polygon centroid.
                # The catalog placement is based on the trimmed raster mask window, so the
                # diagnostic overlay should use the same reference point.
                xorig_trans = lonrange[xmin]+(maskwidth/2)*rainprop.spatialres[0]
                yorig_trans = latrange[ymin]+(maskheight/2)*rainprop.spatialres[1]
                Delta_x = xnew_trans - xorig_trans
                Delta_y = ynew_trans - yorig_trans
                # % Compute the centroid of the first geometry
                # % original_centroid = original_geometries[0].centroid
                # % Compute translation distances
                # % Delta_x = xnew_trans - original_centroid.x
                # % Delta_y = ynew_trans - original_centroid.y
                # Apply translation to all geometries
                transposed_geometries = [translate(geom, xoff=Delta_x, yoff=Delta_y) for geom in original_geometries]
                # Create a new ShapelyFeature for the transposed geometries
                wmap_feature_transposed = ShapelyFeature(transposed_geometries, crs=ccrs.PlateCarree())
                # Add the transposed feature to the plot
                ax.add_feature(wmap_feature_transposed, edgecolor="green", facecolor='none', zorder=4)
                circle = patches.Circle((lonrange[catx[i]]+(maskwidth/2)*rainprop.spatialres[0], latrange[caty[i]]+(maskheight/2)*rainprop.spatialres[1]), radius = 0.2, edgecolor='green', facecolor='none',zorder = 3)
            elif areatype.lower()=="box" or areatype.lower()=="point":
                ax.add_geometries([ring], facecolor='none',edgecolor='red',crs=ccrs.PlateCarree(),zorder = 3)
                circle = patches.Circle((lonrange[catx[i]] + rainprop.spatialres[0]/2, latrange[caty[i]] - rainprop.spatialres[1]/2), radius = 0.2, edgecolor='green', facecolor='none',zorder = 3)

            if domain_type.lower()=="irregular" and os.path.isfile(domainshp):
                ax.add_feature(domain_feature,edgecolor="black",facecolor="None",zorder = 3)
                
            temprain.plot(x='longitude', y ='latitude',cmap='Blues',cbar_kwargs={'orientation':orientation,'label':"Storm Total precipitation [mm]"}, ax=ax, zorder=1)
           
            ax.add_patch(circle)
            ax.add_feature(states_provinces, zorder = 3)
            #ax.add_feature(coast_10m)
            ax.set_xticks(np.linspace(outerextent[0],outerextent[1],2))
            lon_formatter = cticker.LongitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
        
            # Define the yticks for latitude
            ax.set_yticks(np.linspace(outerextent[3],outerextent[2],2))
            lat_formatter = cticker.LatitudeFormatter()
            ax.set(xlabel=None,ylabel=None)
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.set_title('Storm '+str(i+1)+': '+str(plottime[-1])
                        +' Storm Total Rainfall\nMax Precipitation:'+str(np.round(catmax[i]))
                        +' mm @ Lat/Lon:'+"{:6.1f}".format(np.array(plotlat[caty[i]]-(maskheight/2+maskheight%2)*rainprop.spatialres[0]))
                        +u'\N{DEGREE SIGN}'+','+"{:6.1f}".format(np.array(plotlon[catx[i]]
                        +(maskwidth/2+maskwidth%2)*rainprop.spatialres[0]))
                        +u'\N{DEGREE SIGN}')
            plt.tight_layout()
            plt.savefig(diagpath+'Storm'+str(i+1)+'_'+str(plottime[-1]).split('T')[0]+'.png',dpi=250)
            plt.close()     
        
            
            # create hyetograph diagnostic plots:
            selected_region = plotrain.isel(latitude=slice(caty[i], caty[i] + maskheight), 
                                    longitude=slice(catx[i], catx[i] + maskwidth))
            masked_data = selected_region * trimmask
            raints = masked_data.sum(dim=('latitude', 'longitude'), skipna=True) / mnorm
            # raints=np.nansum(np.multiply(plotrain[:,caty[i]:caty[i]+maskheight,catx[i]:catx[i]+maskwidth],trimmask),axis=(1,2))/mnorm
            fig = plt.figure()
            ax  = fig.add_subplot(111)
            fig.set_size_inches(6,4)
            ax.bar(np.arange(0,raints.shape[0]*rainprop.timeres/60.,rainprop.timeres/60.), raints,width=rainprop.timeres/60., align='edge')
            ax.set_title('Storm '+str(i+1)+': '+str(plottime[-1])
                        +' Hyetograph\nMax Precipitation:'+str(np.round(catmax[i]))
                        +' mm @ Lat/Lon:'+"{:6.1f}".format(np.array(plotlat[caty[i]]-(maskheight/2+maskheight%2)*rainprop.spatialres[0]))
                        +u'\N{DEGREE SIGN}'+','+"{:6.1f}".format(np.array(plotlon[catx[i]]
                        +(maskwidth/2+maskwidth%2)*rainprop.spatialres[0]))
                        +u'\N{DEGREE SIGN}')
            ax.set_xlabel('Time [hours]')
            ax.set_ylabel('Precipitation Rate [mm/hr]')
            plt.tight_layout()
            plt.savefig(diagpath+'Hyetograph_Storm'+str(i+1)+'_'+str(plottime[-1]).split('T')[0]+'.png',dpi=250)
            plt.close()
    
    
    # PLOT STORM OCCURRENCE PROBABILITIES-there is a problem with the "alignment of the raster and the storm locations
    print("     Creating storm probability map...")
    
    #plot_kernel=np.column_stack([pltkernel,np.zeros((pltkernel.shape[0],catmask.shape[1]-pltkernel.shape[1]))])
    #plot_kernel=np.row_stack([plot_kernel,np.zeros((catmask.shape[0]-pltkernel.shape[0],plot_kernel.shape[1]))])

    #padleft=math.floor(maskwidth/2)-1
    #padright=math.ceil(maskwidth/2)
    padleft=0
    padright=maskwidth-1
    
    plot_kernel=np.column_stack([np.zeros((pltkernel.shape[0],padleft)),pltkernel,np.zeros((pltkernel.shape[0],padright))])
    
    #padtop=math.floor(maskheight/2)
    #padbottom=math.ceil(maskheight/2)-1
    padtop=0
    padbottom=maskheight-1
    
    plot_kernel=np.row_stack([np.zeros((padtop,plot_kernel.shape[1])),plot_kernel,np.zeros((padbottom,plot_kernel.shape[1]))])

    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        xplot_kernel=xr.Dataset(
            data_vars=dict(plot_kernel=(["y","x"],plot_kernel)),
            coords=dict(
                lat=(["y"],latrange.data + (maskheight / 2) * rainprop.spatialres[1].item()),
                lon=(["x"],lonrange.data + (maskwidth / 2) * rainprop.spatialres[0].item())),
            attrs=dict(description="diagnostic plotting of the storm probability density"),
        )
    elif areatype.lower()=="box" or areatype.lower()=="point":
        xplot_kernel=xr.Dataset(
            data_vars=dict(plot_kernel=(["y","x"],plot_kernel)),
            coords=dict(
                lat=(["y"],latrange.data - rainprop.spatialres[1].item()/2),
                lon=(["x"],lonrange.data + rainprop.spatialres[0].item()/2)),
            attrs=dict(description="diagnostic plotting of the storm probability density"),
        )
    
    fig = plt.figure(figsize=(figsizex,figsizey))
    ax=plt.axes(projection=proj)
    #ax.set_extent(outerextent)
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        ax.add_feature(wmap_feature,edgecolor="red",facecolor="None",zorder = 3)
    elif areatype.lower()=="box" or areatype.lower()=="point":
        ax.add_geometries([ring], facecolor='none',edgecolor='black',crs=ccrs.PlateCarree(),zorder = 3)
        
    if domain_type.lower()=="irregular" and os.path.isfile(domainshp):
        ax.add_feature(domain_feature,edgecolor="black",facecolor="None",zorder = 3)
    xplot_kernel.plot_kernel.plot(x="lon",y="lat",cmap='Reds',cbar_kwargs={'orientation':'horizontal','label':"Probability of storm occurrence [-]"})
    #xplot_kernel.pltkernel.plot(x="lon",y="lat",cmap='Reds',cbar_kwargs={'orientation':orientation,'label':"Probability of storm occurrence [-]"})
        
    # plt.scatter(lonrange[catx]+(maskwidth/2)*rainprop.spatialres[0],latrange[caty]-(maskheight/2)*rainprop.spatialres[1],s=catmax/2,facecolors='k',edgecolors='none',alpha=0.75)
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        for k in range(0,nstorms):
            # plt.scatter(plotlon[catx[k]],plotlat[caty[k]],s=catmax[k]*2,facecolors='k',edgecolors='none',alpha=0.75)
            # plt.scatter(plotlon[catx[k]]+(maskwidth/2)*rainprop.spatialres[0],plotlat[caty[k]]+(maskheight/2)*rainprop.spatialres[1],s=catmax[k]*1,facecolors='gray',edgecolors='k',alpha=0.5)
            plt.scatter(lonrange[catx[k]]+(maskwidth/2)*rainprop.spatialres[0],latrange[caty[k]]+(maskheight/2)*rainprop.spatialres[1],s=catmax[k]*1,facecolors='gray',edgecolors='k',alpha=0.5)
    elif areatype.lower()=="box" or areatype.lower()=="point":
        for k in range(0,nstorms):
            plt.scatter(lonrange[catx[k]]+rainprop.spatialres[0]/2,latrange[caty[k]]- rainprop.spatialres[1]/2,s=catmax[k]*1,facecolors='gray',edgecolors='k',alpha=0.5)
  

    # ax.add_feature(states_provinces)
    ax.set_xticks(np.linspace(outerextent[0],outerextent[1],2))
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

# Define the yticks for latitude
    ax.set_yticks(np.linspace(outerextent[3],outerextent[2],2))
    lat_formatter = cticker.LatitudeFormatter()
    ax.set(xlabel=None,ylabel=None)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    # ax.set_title('Prob. of storm occurrence from\n'+catalogname.split('/')[-1]+'\nNOTE: The prob. map may not extend to lower/right edges. That is not a mistake!')
    #ax.axes.set_title(xlabel=None)

    plt.tight_layout()
    plt.savefig(diagpath+'ProbabilityOfStorms.png',dpi=250)
    # plt.show()
    plt.close()   
    
    
    # PLOT AVERAGE STORM RAINFALL
    print ("     Creating mean precipitation map...")
    outerextent=np.array(rainprop.subextent,dtype='float32')
    
    fig = plt.figure(figsize=(figsizex,figsizey))
    ax=plt.axes(projection=proj)
    #ax.set_extent(outerextent)
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        ax.add_feature(wmap_feature,edgecolor="red",facecolor="None",zorder = 3)
    elif areatype.lower()=="box" or areatype.lower()=="point":
        ax.add_geometries([ring], facecolor='none',edgecolor='red',crs=ccrs.PlateCarree(),zorder = 3)
        
    if domain_type.lower()=="irregular" and os.path.isfile(domainshp):
        ax.add_feature(domain_feature,edgecolor="black",facecolor="None",zorder = 3)
    mu_t.plot(x='longitude', y ='latitude',cmap='Blues',cbar_kwargs={'orientation':orientation,'label':"Mean Storm Total precipitation [mm]"})
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        for k in range(0,nstorms):
            plt.scatter(lonrange[catx[k]]+(maskwidth/2)*rainprop.spatialres[0],latrange[caty[k]]+(maskheight/2)*rainprop.spatialres[1],s=catmax[k]*1,facecolors='gray',edgecolors='k',alpha=0.5)
    elif areatype.lower()=="box" or areatype.lower()=="point":
        for k in range(0,nstorms):
            plt.scatter(lonrange[catx[k]]+rainprop.spatialres[0]/2,latrange[caty[k]]- rainprop.spatialres[1]/2,s=catmax[k]*1,facecolors='gray',edgecolors='k',alpha=0.5)

    ax.add_feature(states_provinces)
    #ax.add_feature(coast_10m)
    ax.set_xticks(np.linspace(outerextent[0],outerextent[1],2))
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

# Define the yticks for latitude
    ax.set_yticks(np.linspace(outerextent[3],outerextent[2],2))
    lat_formatter = cticker.LatitudeFormatter()
    ax.set(xlabel=None,ylabel=None)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.set_title('Mean of Storm Catalog Rainfall from\n'+catalogname.split('/')[-1])
    #ax.axes.set_title(xlabel=None)
    plt.tight_layout()
    plt.savefig(diagpath+'MeanStormRain.png',dpi=250)
    # plt.show()
    plt.close()   
    
    
    
    
    # PLOT STORM RAINFALL Standard deviation
    print("     Creating precipitation standard deviation map...")
    
    fig = plt.figure(figsize=(figsizex,figsizey))
    ax=plt.axes(projection=proj)
    #ax.set_extent(outerextent)
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        ax.add_feature(wmap_feature,edgecolor="black", facecolor='none',zorder = 3)
    elif areatype.lower()=="box" or areatype.lower()=="point":
        ax.add_geometries([ring], facecolor='none',edgecolor='red',crs=ccrs.PlateCarree(),zorder = 3)
        
    if domain_type.lower()=="irregular" and os.path.isfile(domainshp):
        ax.add_feature(domain_feature,edgecolor="black",facecolor="None",zorder = 3)
    std_t.plot(x='longitude', y ='latitude',cmap='Greys',cbar_kwargs={'orientation':orientation,'label':"STD Storm Total precipitation [mm]"})
    if areatype.lower()=="basin" and os.path.isfile(wsmaskshp):
        for k in range(0,nstorms):
            plt.scatter(lonrange[catx[k]]+(maskwidth/2)*rainprop.spatialres[0],latrange[caty[k]]+(maskheight/2)*rainprop.spatialres[1],s=catmax[k]*1,facecolors='gray',edgecolors='k',alpha=0.5)
    elif areatype.lower()=="box" or areatype.lower()=="point":
        for k in range(0,nstorms):
            plt.scatter(lonrange[catx[k]]+rainprop.spatialres[0]/2,latrange[caty[k]]- rainprop.spatialres[1]/2,s=catmax[k]*1,facecolors='gray',edgecolors='k',alpha=0.5)


    ax.add_feature(states_provinces)
    #ax.add_feature(coast_10m)
    ax.set_xticks(np.linspace(outerextent[0],outerextent[1],2))
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

# Define the yticks for latitude
    ax.set_yticks(np.linspace(outerextent[3],outerextent[2],2))
    lat_formatter = cticker.LatitudeFormatter()
    ax.set(xlabel=None,ylabel=None)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    ax.set_title('Std. Dev. of Storm Catalog Rainfall from\n'+catalogname.split('/')[-1])
    #ax.axes.set_title(xlabel=None)
    plt.tight_layout()
    plt.savefig(diagpath+'StdDevStormRain.png',dpi=250)
    plt.close()   
    
    
    
    
    # PLOT CATALOG CDF/PDF
    print ("     Creating storm catalog PDF and CDF...")
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    fig.set_size_inches(6,4)
    n, bins, patches = ax.hist(catmax, nstorms, density=True, histtype='stepfilled',cumulative=True, label='Empirical',color='lightgray')
    ax.annotate('Largest Storm: '+"{0:0.1f}".format(np.nanmax(catmax))+" mm\n"+str(plottime[-1]), xy=(np.nanmax(catmax), 1),  xycoords='data',
            xytext=(0.75, 0.75), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
    ax.set_title('CDF of '+str(nstorms)+' Storm Catalog Entries')
    ax.set_xlabel('Storm Total Precipitation [mm]')
    ax.set_ylabel('Nonexceedance Probability [-]')
    plt.tight_layout()
    plt.savefig(diagpath+'CDF_StormCatalog.png',dpi=250)
    plt.close()
FP.toc('diagnostics')                                 # GP 2026-10-22: RAINYDAY_TIMERS
    
        

#%%
#==============================================================================
# STEP 2 (OPTIONAL): STORM TRANSPOSITION
#==============================================================================

# GP 2026-10-21: MULTIPLEDURATIONS checks. Each duration d < DURATION must give exactly what a separate run gives
# (CATALOGPATH = this catalog, DURATION d, DURATIONCORRECTION true): the random draws (seed 0, Poisson counts, storms,
# locations) do not depend on the duration, so all durations share them; only the per-storm rainfall field differs
# (rolling sums of d hours and the max over windows, as SSTalt(durcheck=True)). Validated configuration only.
subdurs=[]
# GP 2026-10-21: engine for every duration-corrected resampling (the shorter durations of MULTIPLEDURATIONS and single
# runs with DURATIONCORRECTION true, watershed/box, no rescaling/rotation). "fast" (default): compiled SSTalt_multidur,
# verified bit-identical to "exact" and to the old code; "exact": the original SSTalt (numba object mode).
mdengine=str(cardinfo.get("MULTIDURATIONENGINE","fast")).lower()
# GP 2026-10-22: "table" (RainyDay_fast, Source/RainyDay_utilities_Py3/fast_sst.py): FFT basin-sum table per storm,
# exact sums only for the candidate windows; bit-identical to "fast" and "exact"
# if mdengine not in ('exact','fast'):
#     sys.exit("MULTIDURATIONENGINE must be 'exact' or 'fast'")
if mdengine not in ('exact','fast','table'):
    sys.exit("MULTIDURATIONENGINE must be 'exact', 'fast' or 'table'")
if multidurations is not None:
    if FreqAnalysis==False:
        sys.exit("MULTIPLEDURATIONS needs FREQANALYSIS 'true'")
    if durcorrection:
        sys.exit("MULTIPLEDURATIONS: DURATIONCORRECTION must be 'false' (it applies to the parent DURATION; the shorter durations always use the max over windows of the parent catalog)")
    if durationcheck==False:
        sys.exit("MULTIPLEDURATIONS: the storm catalog duration must equal DURATION")
    if Scenarios:
        sys.exit("MULTIPLEDURATIONS doesn't support SCENARIOS yet")
    if rescaletype!='none' or rotation or userdistr.all()!=False:
        sys.exit("MULTIPLEDURATIONS doesn't support ENHANCEDSST, ROTATIONANGLE or INTENSDISTR")
    if areatype.lower() in ('point','pointlist'):
        sys.exit("MULTIPLEDURATIONS supports watershed or box areas only (not POINTAREA point/pointlist)")
    if calctype.lower() not in ('ams','pds'):
        sys.exit("MULTIPLEDURATIONS supports CALCTYPE 'ams' or 'pds' only")
    if multidurations[-1]>duration+1e-9:
        sys.exit("MULTIPLEDURATIONS: every duration must be <= DURATION ("+RainyDay.durtag(duration)+" h)")
    subdurs=[d for d in multidurations if not np.isclose(d,duration)]
    substeps=[]
    for d in subdurs:
        if not np.isclose(d*60./rainprop.timeres,np.round(d*60./rainprop.timeres)):
            sys.exit("MULTIPLEDURATIONS: "+RainyDay.durtag(d)+" h is not a multiple of the time resolution ("+str(rainprop.timeres)+" min)")
        substeps.append(int(np.round(d*60./rainprop.timeres)))
    # (MULTIDURATIONENGINE is now read above, for single duration-corrected runs too)
    # mdengine=str(cardinfo.get("MULTIDURATIONENGINE","fast")).lower()
    # if mdengine not in ('exact','fast'):
    #     sys.exit("MULTIDURATIONENGINE must be 'exact' or 'fast'")
    if CreateCatalog:
        # a separate CATALOGPATH run reads the masks from the written catalog and filters storms by file-name year/month:
        # make sure this run uses exactly the same masks and storms
        # _,_,_,_,_,_,_,mdmask,mddomain,_,_=RainyDay.readcatalog(stormlist[-1])
        _,_,_,_,_,_,_,mdmask,mddomain,_,_=FIO.readcatalog(stormlist[-1])     # GP 2026-10-22: full or light catalog
        mdmask=np.array(mdmask)
        mdtrim=np.delete(mdmask,np.where(np.sum(mdmask,axis=0)==0),axis=1)
        mdtrim=np.array(np.delete(mdtrim,np.where(np.sum(mdmask,axis=1)==0),axis=0),dtype='float32')
        if mdtrim.dtype!=trimmask.dtype or not np.array_equal(mdtrim,trimmask) or not np.array_equal(np.array(mddomain)>0,domainmask>0):
            sys.exit("MULTIPLEDURATIONS: the masks stored in the new catalog differ from the ones in memory; can't guarantee results identical to separate runs")
        if includeyears==False:
            mdkeep=[np.int32(RainyDay.extract_date(storm, catalogname)[4:6]) not in excludemonths for storm in stormlist]
        else:
            mdkeep=[np.int32(RainyDay.extract_date(storm, catalogname)[4:6]) not in excludemonths and np.int32(RainyDay.extract_date(storm, catalogname)[:4]) in includeyears for storm in stormlist]
        if not all(mdkeep):
            sys.exit("MULTIPLEDURATIONS: some catalog storms fall outside INCLUDEYEARS/EXCLUDEMONTHS; a separate CATALOGPATH run would drop them")
    print("MULTIPLEDURATIONS: parent "+RainyDay.durtag(duration)+" h + "+", ".join(RainyDay.durtag(d) for d in subdurs)+" h, engine '"+mdengine+"'")
# GP 2026-10-26: RECORDCI: not on the duration-rewrite path (catalog longer than DURATION, DURATIONCORRECTION false)
if recordopts is not None:
    FR.check_duration(durationcheck,durcorrection)
if freqengine!='montecarlo':                         # GP 2026-10-26: FREQENGINE: the same duration check
    FR.check_duration(durationcheck,durcorrection,who="FREQENGINE '"+freqengine+"'")

if FreqAnalysis:
    print("resampling and transposing...")
    
    if includeyears==False:
        nyears=len(np.unique(cattime[:,-1].astype('datetime64[Y]')))
    else:
        nyears=len(includeyears)
        
        
    # resampling counts options:
    # GP 2026-10-26: FREQENGINE "exact": no synthetic years are drawn (no ncounts, whichstorms, locations): the exact
    # engine computes the frequency analysis from the pool of every storm at every location, so the np.random calls of
    # the draws are skipped on purpose (README_CHANGES B8); "montecarlo"/"both": unchanged (the elif chain below)
    if freqexact:
        print("FREQENGINE 'exact': no Monte Carlo draws (exact frequency analysis from the storm x location pool)")
    # if samplingtype.lower()=='poisson':            # GP 2026-10-26: old line, now an elif of the branch above
    elif samplingtype.lower()=='poisson':
        #lrate=len(catmax)/nyears*FrequencySens 
        lrate=len(catmax)/nyears      
        ncounts=np.random.poisson(lrate,(nsimulations,nrealizations))
        #ncounts[ncounts==0]=1
        if calctype.lower()=='npyear' and lrate<nperyear:   
            sys.exit("You specified to write multiple storms per year, but you specified a number that is too large relative to the resampling rate!")
    elif samplingtype.lower()=='negbinom':
        sys.exit("Sorry, the negative binomial resampling isn't set up yet :(")
        _,yrscount=np.unique(cattime[:,-1].astype('datetime64[Y]').astype(int)+1970,return_counts=True)
        if len(yrscount)<nyears:
            yrscount=np.append(yrscount,np.ones(nyears-len(yrscount),dtype='int32'))
        rparam=np.mean(yrscount)*np.mean(yrscount)/(np.var(yrscount)-np.mean(yrscount))   
        pparam=1.-np.mean(yrscount)/np.var(yrscount)
        ncounts=np.random.negative_binomial(rparam,pparam,size=(nsimulations,nrealizations)) 
        if calctype.lower()=='npyear' and np.mean(yrscount)<nperyear :   
            sys.exit("You specified to write multiple storms per year, but you specified a number that is too large relative to the resampling rate!")
    else:
        _,yrscount=np.unique(cattime[:,-1].astype('datetime64[Y]').astype(int)+1970,return_counts=True)
        if len(yrscount)<nyears:
            yrscount=np.append(yrscount,np.ones(nyears-len(yrscount),dtype='int32'))
        ncounts=np.random.choice(yrscount,(nsimulations,nrealizations),replace=True)   
        if calctype.lower()=='npyear' and np.mean(yrscount)<nperyear :   
            sys.exit("You specified to write multiple storms per year, but you specified a number that is too large relative to the resampling rate!")
        #ncounts[ncounts==0]=1
            
    # GP 2026-10-26: FREQENGINE "exact": no draws (old lines kept commented; the same lines run otherwise)
    # whichstorms=np.empty((np.nanmax(ncounts),ncounts.shape[0],ncounts.shape[1]),dtype='int32')
    # whichstorms[:]=-9999
    if not freqexact: whichstorms=np.empty((np.nanmax(ncounts),ncounts.shape[0],ncounts.shape[1]),dtype='int32')
    if not freqexact: whichstorms[:]=-9999
    
    if rotation==True:
        randangle=(maxangle-minangle)*np.random.random_sample(((np.nanmax(ncounts),ncounts.shape[0],ncounts.shape[1])))+minangle
        
        angbins=np.linspace(minangle,maxangle,nanglebins)
        angs=math.pi/180*angbins
        anglebins=np.digitize(randangle.ravel(),angbins).reshape(np.nanmax(ncounts),ncounts.shape[0],ncounts.shape[1])
    
    
    # DOES THIS PROPERLY HANDLE STORM EXCLUSIONS???  I think so...
    # for i in range(0, np.nanmax(ncounts)):        # GP 2026-10-26: old line; FREQENGINE "exact": no storm draws
    for i in (range(0, np.nanmax(ncounts)) if not freqexact else ()):
        whichstorms[i, ncounts >= i + 1] = np.random.randint(0, nstorms, (len(ncounts[ncounts >= i + 1])))  # why was this previously "nstorms-1"??? Bug
    # added 2/6/2024 DBW to support seasonally-dependent sampling
    if seasonalsampling:
        cat_doy=np.zeros((2*cattime.shape[0]),dtype='datetime64[m]')
        
        # convert the starting time of each storm into the same date of an arbitrary year that matches the seasonal sampling probabilities
        for i in range(0,cattime.shape[0]):
            cat_doy[i]=RainyDay.replace_year(cattime[i,0],1776)
        # double up the length of the cattime series to account for the fact that end of December storms might be closer to January    
        for i in range(cattime.shape[0],2*cattime.shape[0]):
            cat_doy[i]=RainyDay.replace_year(cattime[i-cattime.shape[0],0],1777)
            # convert date of storm (assumed to be the starttime of storm) into day of year, to be compared against the annual sampling CDF
            #cat_doy[i]=(cattime[i,0] - np.datetime64(cattime[i,0].astype('datetime64[Y]'))).astype('timedelta64[D]').astype(int) + 1
        for i in range(0,np.nanmax(ncounts)):
            cdf_2d=seasonalcdf[:, np.newaxis]
            cdf_2d = np.tile(cdf_2d, (1,len(ncounts[ncounts>=i+1]))) 
            sampprob=np.random.rand(len(ncounts[ncounts>=i+1])) # why was this previously "nstorms-1"??? Bug?
            pdiff=cdf_2d-sampprob
            pdiff[pdiff>0.]=-999.
            closest_day=np.argmax(pdiff,axis=0)
            closest_date = pd.to_datetime('1776-01-01') + pd.to_timedelta(closest_day.tolist(), unit='D')
            closest_date=closest_date.values.astype('datetime64[D]')
            
            # find the storm that is closest to the randomly-sampled date:
            absolute_diff = np.abs(closest_date[:, np.newaxis] - cat_doy)
            closest_storm = np.argmin(absolute_diff, axis=1)
    
    # the next three lines were commented out when adding the "pointlist" option
    #whichrain=np.zeros((whichstorms.shape),dtype='float32')
    #whichx=np.zeros((whichstorms.shape),dtype='int32')
    #whichy=np.zeros((whichstorms.shape),dtype='int32')
    
    if freqexact:                                     # GP 2026-10-26: FREQENGINE "exact": no draws to store
        pass
    # if areatype.lower()=="pointlist":               # GP 2026-10-26: old line, now an elif of the branch above
    elif areatype.lower()=="pointlist":
        whichx=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],npoints_list),dtype='int32')
        whichy=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],npoints_list),dtype='int32')
        whichrain=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],npoints_list),dtype='float32')
    else:
        whichx=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],1),dtype='int32')
        whichy=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],1),dtype='int32')  
        whichrain=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],1),dtype='float32')
        whichstep=np.zeros((whichstorms.shape[0],whichstorms.shape[1],whichstorms.shape[2],1),dtype='int32')
    # GP 2026-10-21: MULTIPLEDURATIONS: depths of the shorter durations for the same draws (storm, location)
    # if len(subdurs)>0:                              # GP 2026-10-26: old line; FREQENGINE "exact": no draws
    if len(subdurs)>0 and not freqexact:
        whichrain_md=np.zeros((len(subdurs),)+whichrain.shape,dtype='float32')

    
    # if durcorrection:                               # GP 2026-10-26: old line; FREQENGINE "exact": no draws
    if durcorrection and not freqexact:
        whichtimeind=np.zeros((whichstorms.shape),dtype='float32')
    
    # ------------------------------------------------------------------------------------------------
    # GP 2026-10-16: OLD transposition locations (kept commented for reference).
    # Only the anchor cell (corner of the basin bounding box) had to be inside the domain; the last
    # maskheight rows and maskwidth columns of the ARRAY were zeroed so the box stays inside the array,
    # but NOT inside the domain polygon. On irregular domains (e.g. SE Texas coast, Galveston Bay cut out)
    # transposed basins could hang up to ~55% outside the domain and still collect rain there, which is
    # inconsistent with the storm-catalog search (whole box inside the domain, catalogFFT_irregular_domainaware).
    # It also modified domainmask in place, which changed the later sub-duration window search.
    # For SE Texas this made the 100-yr depth increase with basin area (Harvey's core lies on the domain edge).
    #
    # # For irregular domains under uniform transposition, we mask out border cells to avoid placing storm windows
    # # where the mask would exceed domain bounds. However, for single-pixel masks (maskheight = maskwidth = 1),
    # # this masking is unnecessary and would incorrectly remove valid storm locations.
    # # Therefore, we apply border masking only if maskheight or maskwidth is greater than 1.
    #
    # # Trim all edges of the domain mask to ensure storm centroids stay within bounds.
    # # This avoids placing storm centers too close to the edges where the mask footprint (e.g., 5x5)
    # # would exceed the domain and cause indexing issues or partial storms.
    #
    # if transpotype=='uniform' and domain_type=='irregular':
    #     if maskheight > 1:
    #         # domainmask[:maskheight, :] = 0.    # Trim northern edge
    #         domainmask[-maskheight:,:]= 0.      # Trim southern edge
    #     if maskwidth > 1:
    #         # domainmask[:, :maskwidth] = 0.     # Trim western edge
    #         domainmask[:, -maskwidth:] = 0.    # Trim eastern edge
    #
    #     xmask,ymask=np.meshgrid(np.arange(0,domainmask.shape[1],1),np.arange(0,domainmask.shape[0],1))
    #     xmask=xmask[np.equal(domainmask,True)]
    #     ymask=ymask[np.equal(domainmask,True)]
    # ------------------------------------------------------------------------------------------------
    # GP 2026-10-16: NEW transposition locations = same rule as the storm-catalog search.
    # A location (anchor = corner of the basin bounding box) is valid only if the WHOLE maskheight x maskwidth
    # box lies inside the domain. domainmask is no longer modified. For single-pixel masks (1x1) this is
    # simply every domain cell.
    if transpotype=='uniform' and domain_type=='irregular':
        from scipy.signal import fftconvolve
        insidecount=fftconvolve((domainmask>0).astype('float32'),np.ones((maskheight,maskwidth),dtype='float32'),mode='valid')
        validanchor=np.zeros(domainmask.shape,dtype=bool)
        validanchor[:insidecount.shape[0],:insidecount.shape[1]]=insidecount>(maskheight*maskwidth-0.5)
        if np.sum(validanchor)==0:
            sys.exit("No transposition location fits the whole watershed inside the domain. Check DOMAINSHP and WATERSHEDSHP.")

        xmask,ymask=np.meshgrid(np.arange(0,domainmask.shape[1],1),np.arange(0,domainmask.shape[0],1))
        xmask=xmask[validanchor]
        ymask=ymask[validanchor]
        print("transposition locations: "+str(len(xmask))+" (whole watershed inside the domain) out of "+str(int(np.sum(domainmask>0)))+" domain cells")

    # if np.sum(domainmask)==0:
    #     sys.exit("It seems the transposition domain is relatively small with respect to the basin mask. Stcohastic Storms cannot be created")

    if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
        whichmultiplier=np.empty_like(whichrain)
        whichmultiplier[:]=np.nan

        
    #==============================================================================
    # If you're using intensity-dependent resampling, get ready for it!
    #==============================================================================


    if rescaletype=='stochastic' or rescaletype=='deterministic':
        smoothsig=5
        
        print("reading in precipitation intensity data...")
        intenserain,_,intenselat,intenselon=RainyDay.readintensityfile(rescalingfile)
        intensemask=np.equal(np.sum(intenserain,axis=0),0.)
        intenserain[:,intensemask]=np.nan
        int_xmin=np.abs(intenselon-rainprop.bndbox[0]).argmin()
        int_ymin=np.abs(intenselat-rainprop.bndbox[3]).argmin()
        int_xmax=np.abs(intenselon-rainprop.bndbox[1]).argmin()
        int_ymax=np.abs(intenselat-rainprop.bndbox[2]).argmin()
        #int_xmax=np.abs(intenselon-rainprop.bndbox[1]).argmin()+1
        #int_ymax=np.abs(intenselat-rainprop.bndbox[2]).argmin()+1
        intenserain=intenserain[:,int_ymin:int_ymax,int_xmin:int_xmax]
        intenselat=intenselat[int_ymin:int_ymax]
        intenselon=intenselon[int_xmin:int_xmax]
        
        #intenserain[np.equal(intenserain,0.)]=np.nan
        nintstorms=np.min((intenserain.shape[0],2*nyears))
        intenserain=intenserain[-nintstorms:,:]

        if np.array_equal(intenselat,latrange)==False or np.array_equal(intenselon,lonrange)==False:  
            intensegridx,intensengridy=np.meshgrid(intenselon,intenselat)        
            ingrid_intense=np.column_stack((intensegridx.flatten(),intensengridy.flatten())) 
            grid_out=np.column_stack((ingridx.flatten(),ingridy.flatten())) 
            delaunay=sp.spatial.qhull.Delaunay(ingrid_intense)
            tempintense=np.empty((intenserain.shape[0],ingridx.shape[0],ingridx.shape[1]),dtype='float32')
            for i in range(0,intenserain.shape[0]):
                interp=sp.interpolate.LinearNDInterpolator(delaunay,intenserain[i,:].flatten(),fill_value=np.nan)
                tempintense[i,:]=np.reshape(interp(grid_out),ingridx.shape)
            intenserain=tempintense 
        
        #intenserain[:,np.equal(domainmask,0.)]=np.nan
      

        # the stochastic multiplier approach uses the log of the rainfall:
        intenserain=np.log(intenserain)
        hometemp=np.nansum(np.multiply(intenserain,catmask),axis=(1,2))/mnorm
        xlen_wmask=rainprop.subdimensions[1]-maskwidth+1
        ylen_wmask=rainprop.subdimensions[0]-maskheight+1
                
        
        # there is some goofy numba stuff below...
        if maskheight>1 or maskwidth>1:
            tempintense=np.empty((intenserain.shape[0],ylen_wmask,xlen_wmask),dtype='float32')
            intenserain=RainyDay.intenseloop(intenserain,tempintense,xlen_wmask,ylen_wmask,maskheight,maskwidth,trimmask,mnorm,domainmask)
        intensecorr=np.empty((ylen_wmask,xlen_wmask),dtype='float32')    
        intensecorr=RainyDay.intense_corrloop(intenserain,intensecorr,hometemp,xlen_wmask,ylen_wmask,mnorm,domainmask)
            
        #homemean=np.mean(homerain)
        #homestd=np.std(homerain)
        intensemean=np.mean(intenserain,axis=0)
        intensestd=np.std(intenserain,axis=0) 
        
        intensemean=RainyDay.mysmoother(intensemean,sigma=[smoothsig,smoothsig])
        intensestd=RainyDay.mysmoother(intensestd,sigma=[smoothsig,smoothsig])
        intensecorr=RainyDay.mysmoother(intensecorr,sigma=[smoothsig,smoothsig])
        
        if maskheight>1 or maskwidth>1:
            homemean=np.nansum(np.multiply(intensemean,catmask[:-maskheight+1,:-maskwidth+1]),axis=(0,1))/mnorm
            homestd=np.nansum(np.multiply(intensestd,catmask[:-maskheight+1,:-maskwidth+1]),axis=(0,1))/mnorm        
        else:    
            homemean=np.nansum(np.multiply(intensemean,catmask),axis=(0,1))/mnorm
            homestd=np.nansum(np.multiply(intensestd,catmask),axis=(0,1))/mnorm
        
        # just in case you don't have any data to inform the rescaling:
        intensemean[np.isneginf(intensemean)]=homemean
        intensestd[np.isneginf(intensestd)]=0.
        intensecorr[np.isneginf(intensecorr)]=1.0
        
        intensemean[np.isnan(intensemean)]=homemean
        intensestd[np.isnan(intensestd)]=0.
        intensecorr[np.isnan(intensecorr)]=1.0
        
    elif rescaletype=='dimensionless':
        print("reading in precipitation map for dimensionless SST...")
        if '.nc' in rescalingfile:
            sys.exit('need to set this up')
            #intenserain,_,intenselat,intenselon=RainyDay.readintensityfile(rescalingfile)
        elif '.asc' in rescalingfile:
            asciigrid,ncols,nrows,xllcorner,yllcorner,cellsize=RainyDay.read_arcascii(rescalingfile)
            dlsstarea=[xllcorner,xllcorner+ncols*cellsize,yllcorner,yllcorner+nrows*cellsize]
            atlasgridx,atlasgridy=np.meshgrid(np.arange(dlsstarea[0],dlsstarea[1]-cellsize/10.,cellsize),np.arange(dlsstarea[3],dlsstarea[2]+cellsize/10.,-cellsize))
            atlas14_domain=np.column_stack((atlasgridx.flatten(),atlasgridy.flatten())) 
        
            delaunay=sp.spatial.qhull.Delaunay(atlas14_domain)
            interp=sp.interpolate.LinearNDInterpolator(delaunay,asciigrid.flatten(),fill_value=np.nan)
            
            grid_out=np.column_stack((ingridx.flatten(),ingridy.flatten())) 
            atlas_regridded=np.reshape(interp(grid_out),ingridx.shape) 
            atlas_regridded=np.log(atlas_regridded)
            if areatype.lower()!='pointlist':
                hometemp=np.nansum(np.multiply(atlas_regridded,catmask))/mnorm
            else:
                hometemp=np.nanmean(atlas_regridded[domainmask==True])
            atlas_regridded[np.isnan(atlas_regridded)]=hometemp
   
        else:
            sys.exit('Unrecognized file format provided for dimensionless SST')   
    else:
        pass
        
   
    # GP 2026-10-26: RECORDCI (fast_record.py): the pool of basin-average depths is filled per storm in the loop below
    # (from a copy of catrain) and the record outputs are written after the FreqAnalysis CSVs; own random generator
    # (RECORDSEED), no np.random call; anchors = the valid transposition locations (xmask/ymask, or the whole grid).
    recordci=None
    if recordopts is not None:
        recordci=FR.RecordCI(recordopts,durations=[duration]+list(subdurs),steps=[int(duration*60/rainprop.timeres)]+(list(substeps) if len(subdurs)>0 else []),
                             parent_fullsum=(durcorrection==False),trimmask=trimmask,mnorm=mnorm,timeres=rainprop.timeres,shape=rainprop.subdimensions,
                             anchors=(ymask,xmask) if domain_type=='irregular' else None,cattime=cattime,includeyears=includeyears,nyears_rd=nyears,
                             calctype=calctype,nsim=nsimulations,alllevels=alllevels,speclevels=globals().get('speclevels'),folder=fullpath,basename=basescenario)
    # GP 2026-10-26: FREQENGINE "exact"/"both" (fast_record.ExactFA): the exact frequency analysis from the same pool
    # (shared with RECORDCI when it is on: one copy); nyears = RainyDay's own record years (divisor of the storm rate)
    exactfa=None
    if freqengine!='montecarlo':
        exactfa=FR.ExactFA(freqengine,durations=[duration]+list(subdurs),steps=[int(duration*60/rainprop.timeres)]+(list(substeps) if len(subdurs)>0 else []),
                           parent_fullsum=(durcorrection==False),trimmask=trimmask,mnorm=mnorm,timeres=rainprop.timeres,shape=rainprop.subdimensions,
                           anchors=(ymask,xmask) if domain_type=='irregular' else None,nstorms=nstorms,nyears_rd=nyears,calctype=calctype,
                           nsim=nsimulations,nreal=nrealizations,alllevels=alllevels,speclevels=globals().get('speclevels'),
                           spreadtype=spreadtype,quantilecalc=globals().get('quantilecalc'),record=recordci)

    # here is the main resampling and transposition loop
    FP.tic('resampling: total')                       # GP 2026-10-22: RAINYDAY_TIMERS
    # GP 2026-10-22: RainyDay_fast: the draws of storm i as one index tuple, wsel = np.nonzero(whichstorms==i) (same
    # elements, same order), built for all storms from one stable argsort, instead of a new boolean mask of whichstorms
    # for every use (~10 full scans per storm). Used on the uniform/irregular/watershed/no-rescaling path below.
    # stormsel=FST.StormSelector(whichstorms,nstorms)   # GP 2026-10-26: old line; FREQENGINE "exact": no draws
    stormsel=FST.StormSelector(whichstorms,nstorms) if not freqexact else None
    for i in np.arange(0,nstorms):
        # wsel=stormsel(i)                            # GP 2026-10-22 (GP 2026-10-26: old line)
        wsel=stormsel(i) if not freqexact else None   # GP 2026-10-26: FREQENGINE "exact": no draws
        FP.tic('resampling: storm read')              # GP 2026-10-22: RAINYDAY_TIMERS
        # catrain,_,_,_,_,_,_,_,_,_,_ = RainyDay.readcatalog(stormlist[i])
        catrain,_,_,_,_,_,_,_,_,_,_ = FIO.readcatalog(stormlist[i])   # GP 2026-10-22: full or light catalog
        catrain = np.array(catrain)
        catrain[np.less(catrain,0.)]=np.nan
        FP.toc('resampling: storm read')              # GP 2026-10-22: RAINYDAY_TIMERS
        if freqexact:                                 # GP 2026-10-26: FREQENGINE "exact": this storm's pool, no draws
            print('Exact engine: pool of storm '+str(i+1)+' out of '+str(nstorms)+' ('"{0:0.0f}".format(100*(i+1)/nstorms)+'%)')
            exactfa.add_storm(i,catrain)
            continue
        
        if (durationcheck==False and durcorrection==False): 
            print("checking storm catalog duration, and adjusting if needed...")
            
            # if you are using a catalog that is longer in duration than your desired analysis, this happens:
            print("Storm catalog duration is longer than the specified duration, finding max precipitation periods for specified duration...")

            # the first run through trims the storm catalog duration to the desired duration, for diagnostic purposes if needed...       
            dur_x=0
            dur_y=0
            dur_j=0
            rainsum=np.zeros((rainprop.subdimensions[0]-maskheight+1,rainprop.subdimensions[1]-maskwidth+1),dtype='float32')

            # I think the following commented block was wrong, but haven't fully tested the change-DBW 1/24/2020
        #        if durcorrection:
        #            catmax_subdur=np.zeros_like(catmax)
        #            catx_subdur=np.zeros_like(catx)
        #            caty_subdur=np.zeros_like(caty)
        #            cattime_subdur=cattime

            temptime=np.empty((nstorms,int(duration*60/rainprop.timeres)),dtype='datetime64[m]')
                #if (100*((i+1)%(nstorms//10)))==0:
            print('adjusting duration of storms, '+"{0:0.0f}".format(100*(i+1)/nstorms)+'% complete...')
            dur_max=0.
            for j in range(0,catrain.shape[0]-int(duration*60/rainprop.timeres)):
                maxpass=np.nansum(catrain[j:j+int(duration*60./rainprop.timeres),:],axis=0)
                
                if domain_type.lower()=='irregular':
                    maxpass = maxpass * domainmask
                    # Parallelized CPU version of catalogNumba for irregular domains
                    # maxtemp,tempy,tempx=RainyDay.catalogNumba_irregular(maxpass,trimmask,xlen,ylen,xloop,yloop,maskheight,maskwidth,rainsum,stride=catalogstride)
                    # CPU version using convolution for irregular domains
                    # maxtemp,tempy,tempx=RainyDay.catalogFFT_irregular(maxpass,trimmask)
                    maxtemp,tempy,tempx=RainyDay.catalogFFT_irregular_domainaware(maxpass,trimmask,domainmask)

                    # Parallelized GPU version of catalogTF for irregular domains - For Macs
                    # maxtemp,tempy,tempx=RainyDay.catalogTF_irregular(maxpass,trimmask)
                    # Parallelized GPU version of catalogNumba for irregular domains - For CUDA NVIDIA GPUs
                    # maxtemp,tempy,tempx=RainyDay.catalog_gpu_cupy(maxpass,trimmask)

                else:
                    # GP 2026-10-26, rectangular catalog search: same crash as the scan above; now the same exhaustive search,
                    # over every anchor of the grid (rectangular transposition draws from the whole grid). Old code:
                    # maxtemp,tempy,tempx=RainyDay.catalogNumba(maxpass,trimmask,xlen,ylen,xloop,yloop,maskheight,maskwidth,rainsum,stride=catalogstride)
                    maxtemp,tempy,tempx=RainyDay.catalogFFT_irregular_domainaware(maxpass,trimmask,np.ones(maxpass.shape,dtype='float32'))   # GP 2026-10-26: rectangular catalog search
     
                if maxtemp>dur_max:
                    dur_max=maxtemp
                    dur_x=tempx
                    dur_y=tempy
                    dur_j=j

            catmax[i]=dur_max
            catx[i]=dur_x
            caty[i]=dur_y  
            catrain=catrain[dur_j:dur_j+int(duration*60./rainprop.timeres),:]  # I think this line is a problem-DBW 9/26/2023
            temptime[i,:]=cattime[i,dur_j:dur_j+int(duration*60./rainprop.timeres)]
        print('Resampling and transposing storm '+str(i+1)+' out of '+str(nstorms)+' ('"{0:0.0f}".format(100*(i+1)/nstorms)+'%)')
        # UNIFORM RESAMPLING
        if transpotype=='uniform' and domain_type=='rectangular':
            whichx[whichstorms==i,0]=np.random.randint(0,np.int16(rainprop.subdimensions[1])-maskwidth+1,len(whichx[whichstorms==i]))
            whichy[whichstorms==i,0]=np.random.randint(0,np.int16(rainprop.subdimensions[0])-maskheight+1,len(whichy[whichstorms==i]))
     
        # KERNEL-BASED AND INTENSITY-BASED RESAMPLING (ALSO NEEDED FOR IRREGULAR TRANSPOSITION DOMAINS)
        elif transpotype=='nonuniform':
            rndloc=np.array(np.random.random_sample(len(whichx[whichstorms==i])),dtype='float32')
            tempx=np.empty((len(rndloc)),dtype='int32')
            tempy=np.empty((len(rndloc)),dtype='int32')
            for pt in np.arange(0,whichx.shape[3]):
                whichx[whichstorms==i,pt],whichy[whichstorms==i,pt]=RainyDay.numbakernel_fast(rndloc,cumkernel[:,:,pt],tempx,tempy,rainprop.subdimensions[1])
                

        if transpotype=='uniform' and domain_type=='irregular':
            # GP 2026-10-16: OLD (kept for reference) drew from the number of domain cells, which only matched len(xmask)
            # because domainmask had been trimmed in place:
            # rndloc=np.random.randint(0,np.sum(np.equal(domainmask,True)),np.sum(whichstorms==i))
            # GP 2026-10-16: NEW: draw uniformly among the valid transposition locations
            FP.tic('resampling: location draws')      # GP 2026-10-22: RAINYDAY_TIMERS
            # GP 2026-10-22: wsel (same count, same elements and order as whichstorms==i)
            # rndloc=np.random.randint(0,len(xmask),np.sum(whichstorms==i))
            # for pt in np.arange(0,whichx.shape[3]):
            #     whichx[whichstorms==i,pt]=xmask[rndloc].reshape(len(xmask[rndloc]))
            #     whichy[whichstorms==i,pt]=ymask[rndloc].reshape(len(ymask[rndloc]))
            rndloc=np.random.randint(0,len(xmask),len(wsel[0]))
            for pt in np.arange(0,whichx.shape[3]):
                whichx[wsel+(pt,)]=xmask[rndloc].reshape(len(xmask[rndloc]))
                whichy[wsel+(pt,)]=ymask[rndloc].reshape(len(ymask[rndloc]))
            FP.toc('resampling: location draws')      # GP 2026-10-22: RAINYDAY_TIMERS
        
        # SET UP MANUAL PDF RESAMPLING
        elif transpotype=='manual':  
            sys.exit("not configured for manually supplied pdf yet!")
    
        FP.tic('resampling: parent (passrain + SST)') # GP 2026-10-22: RAINYDAY_TIMERS
        if durcorrection:
            passrain=np.array(RainyDay.rolling_sum(catrain, int(duration*60/rainprop.timeres)),dtype='float32')
            
        else:
            passrain=np.nansum(catrain,axis=0)         # time-average the rainfall
    
        if rotation: 
            print('rotating storms for transposition, '+str(100*(i+1)/nstorms)+'% complete...')
            delarray.append([])
             
            xctr=catx[i]+maskwidth/2.
            yctr=caty[i]+maskheight/2.
            xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
            ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
            ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
            ingridx=ingridx.flatten()
            ingridy=ingridy.flatten()
            outgrid=np.column_stack((ingridx,ingridy))       
            
    
            binctr=0
            for cbin in np.unique(anglebins):
                #print "really should fix the center of rotation! to be the storm center"
                rotx=ingridx*np.cos(angs[binctr])+ingridy*np.sin(angs[binctr])
                roty=-ingridx*np.sin(angs[binctr])+ingridy*np.cos(angs[binctr])
                rotgrid=np.column_stack((rotx,roty))
                delaunay=sp.spatial.qhull.Delaunay(rotgrid)
                delarray[i].append(delaunay)
                interp=sp.interpolate.LinearNDInterpolator(delaunay,passrain.flatten(),fill_value=0.)
                tpass=np.reshape(interp(outgrid),rainprop.subdimensions)
                if rescaletype=='stochastic':
                    sys.exit("WARNING: rotation + intensity-based resampling = not tested!")
                    #whichrain[np.logical_and(whichstorms==i,anglebins==cbin)]=RainyDay.SSTalt(tpass,whichx[np.logical_and(whichstorms==i,anglebins==cbin)],whichy[np.logical_and(whichstorms==i,anglebins==cbin)],trimmask,maskheight,maskwidth,intense_data=intense_dat,durcheck=durcorrection)*rainprop.timeres/60./mnorm
                #else:
                    #whichrain[np.logical_and(whichstorms==i,anglebins==cbin)]=RainyDay.SSTalt(tpass,whichx[np.logical_and(whichstorms==i,anglebins==cbin)],whichy[np.logical_and(whichstorms==i,anglebins==cbin)],trimmask,maskheight,maskwidth,durcheck=durcorrection)*rainprop.timeres/60./mnorm     
                binctr=binctr+1
        else:
            for pt in np.arange(0,whichx.shape[3]):
                if rescaletype=='stochastic' and areatype.lower()!='pointlist' and areatype.lower!='point':                    
                    temprain,whichmultiplier[whichstorms==i,pt],whichstep=RainyDay.SSTalt(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth,intensemean=intensemean,intensestd=intensestd,intensecorr=intensecorr,homemean=homemean,homestd=homestd,durcheck=durcorrection)
                    whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60./mnorm    
                elif rescaletype=='deterministic' and areatype.lower()!='pointlist' and areatype.lower()!='point':
                    temprain,whichmultiplier[whichstorms==i,pt],whichstep=RainyDay.SSTalt(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth,intensemean=intensemean,homemean=homemean,durcheck=durcorrection)
                    whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60./mnorm 
                elif areatype.lower()!='pointlist' and areatype.lower()!='point' and rescaletype=='none':
                    if durcorrection and mdengine=='fast':
                        # GP 2026-10-21: single duration-corrected run with the compiled kernel (one duration),
                        # bit-identical to SSTalt(durcheck=True); whichstep (not used by any output) stays 0
                        # temprain=RainyDay.SSTalt_multidur(np.ascontiguousarray(passrain,dtype='float32'),np.array([0,passrain.shape[0]],dtype=np.int64),whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth)[0]   # GP 2026-10-22: old, whichstorms==i -> wsel
                        temprain=RainyDay.SSTalt_multidur(np.ascontiguousarray(passrain,dtype='float32'),np.array([0,passrain.shape[0]],dtype=np.int64),whichx[wsel+(pt,)],whichy[wsel+(pt,)],trimmask,maskheight,maskwidth)[0]
                    # GP 2026-10-22: "table" engine (fast_sst.py): duration-corrected single run with the basin-sum
                    # table, bit-identical to the line above; whichstep stays 0 as with "fast"
                    elif durcorrection and mdengine=='table':
                        # temprain=FST.table_durcorrection(catrain,passrain,int(duration*60/rainprop.timeres),whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth)   # GP 2026-10-22: old, whichstorms==i -> wsel
                        temprain=FST.table_durcorrection(catrain,passrain,int(duration*60/rainprop.timeres),whichx[wsel+(pt,)],whichy[wsel+(pt,)],trimmask,maskheight,maskwidth)
                    else:
                        # GP 2026-10-22: "table" engine: the parent duration with the compiled kernel (one window)
                        # instead of SSTalt in numba object mode, bit-identical; whichstep = 0 as SSTalt(durcheck=False)
                        # temprain,whichstep[whichstorms==i,pt]=RainyDay.SSTalt(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth,durcheck=durcorrection)
                        temprain=None
                        if mdengine=='table' and durcorrection==False:
                            # temprain=FST.parent_sst(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth)   # GP 2026-10-22: old, whichstorms==i -> wsel
                            temprain=FST.parent_sst(passrain,whichx[wsel+(pt,)],whichy[wsel+(pt,)],trimmask,maskheight,maskwidth)
                        if temprain is None:
                            # temprain,whichstep[whichstorms==i,pt]=RainyDay.SSTalt(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth,durcheck=durcorrection)   # GP 2026-10-22: old, whichstorms==i -> wsel
                            temprain,whichstep[wsel+(pt,)]=RainyDay.SSTalt(passrain,whichx[wsel+(pt,)],whichy[wsel+(pt,)],trimmask,maskheight,maskwidth,durcheck=durcorrection)
                    # whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60./mnorm   # GP 2026-10-22: old, whichstorms==i -> wsel
                    whichrain[wsel+(pt,)]=temprain*rainprop.timeres/60./mnorm

                elif areatype.lower()=='pointlist':
                    if rescaletype=='deterministic':
                        homemeanpt=intensemean[yind_list[pt],xind_list[pt]]
                        temprain,whichmultiplier[whichstorms==i,pt],_=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,durcheck=durcorrection,intensemean=intensemean,homemean=homemeanpt)
                    elif rescaletype=='dimensionless':
                        homemeanpt=atlas_regridded[yind_list[pt],xind_list[pt]]
                        temprain,whichmultiplier[whichstorms==i,pt],_=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,durcheck=durcorrection,intensemean=atlas_regridded,homemean=homemeanpt)
                    elif rescaletype=='stochastic':
                        homemeanpt=intensemean[yind_list[pt],xind_list[pt]]
                        homestdpt=intensestd[yind_list[pt],xind_list[pt]]
                         
                        intensemeanpt=np.mean(intenserain,axis=0)
                        intensestdpt=np.std(intenserain,axis=0) 
                        
                        intensemeanpt=RainyDay.mysmoother(intensemeanpt,sigma=[smoothsig,smoothsig])
                        intensestdpt=RainyDay.mysmoother(intensestdpt,sigma=[smoothsig,smoothsig])
                         
                        intensecorrpt=np.empty((ylen_wmask,xlen_wmask),dtype='float32')    
                        intensecorrpt=RainyDay.intense_corrloop(intenserain,intensecorrpt,intenserain[:,yind_list[pt],xind_list[pt]],xlen_wmask,ylen_wmask,mnorm,domainmask)    
                        intensecorrpt=RainyDay.mysmoother(intensecorrpt,sigma=[smoothsig,smoothsig])
                         
                        intensecorrpt[np.isneginf(intensecorrpt)]=1.0
                    
                        intensemeanpt[np.isnan(intensemeanpt)]=homemeanpt
                        intensestdpt[np.isnan(intensestdpt)]=0.
                        intensecorrpt[np.isnan(intensecorrpt)]=1.0
                        temprain,whichmultiplier[whichstorms==i,pt],_=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth,intensemean=intensemeanpt,intensestd=intensestdpt,intensecorr=intensecorrpt,homemean=homemeanpt,homestd=homestdpt,durcheck=durcorrection)
                    else:
                        temprain,_=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,maskheight,maskwidth,durcheck=durcorrection)

                    whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60.   

                   
                elif areatype.lower()=='point':
                    
                    if rescaletype=='deterministic':
                        homemeanpt=intensemean[ymin,xmin]
                        temprain,whichmultiplier[whichstorms==i,pt],whichstep[whichstorms==i,pt]=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,durcheck=durcorrection,intensemean=intensemean,homemean=homemeanpt)
                        whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60.
                    elif rescaletype=='dimensionless':
                        homemeanpt=atlas_regridded[ymin,xmin]
                        temprain,whichmultiplier[whichstorms==i,pt],whichstep[whichstorms==i,pt]=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,durcheck=durcorrection,intensemean=atlas_regridded,homemean=homemeanpt)
                        whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60.
                    elif rescaletype=='stochastic':
                        homemeanpt=intensemean[ymin,xmin]
                        homestdpt=intensemean[ymin,xmin]
                        temprain,whichmultiplier[whichstorms==i,pt],whichstep[whichstorms==i,pt]=RainyDay.SSTalt(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,intensemean=intensemean,intensestd=intensestd,intensecorr=intensecorr,homemean=homemeanpt,homestd=homestdpt,durcheck=durcorrection)
                        whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60. 
                    else:
                        temprain,whichstep[whichstorms==i,pt]=RainyDay.SSTalt_singlecell(passrain,whichx[whichstorms==i,pt],whichy[whichstorms==i,pt],trimmask,1,1,durcheck=durcorrection)
                        whichrain[whichstorms==i,pt]=temprain*rainprop.timeres/60.

        FP.toc('resampling: parent (passrain + SST)') # GP 2026-10-22: RAINYDAY_TIMERS
        FP.tic('resampling: sub-durations')           # GP 2026-10-22: RAINYDAY_TIMERS
        # GP 2026-10-21: MULTIPLEDURATIONS: the shorter durations of this storm, same draws (whichx/whichy), exactly as
        # a separate run with DURATIONCORRECTION true: passrain = rolling sums of d hours, SSTalt(durcheck=True)
        if len(subdurs)>0:
            # GP 2026-10-22: mdsel = wsel (index tuple, same elements and order as whichstorms==i): [mdsel,0] -> [mdsel+(0,)]
            # mdsel=whichstorms==i
            # mdx=whichx[mdsel,0]
            # mdy=whichy[mdsel,0]
            mdsel=wsel
            mdx=whichx[mdsel+(0,)]
            mdy=whichy[mdsel+(0,)]
            if mdengine=='exact':
                for jd in range(0,len(subdurs)):
                    mdpass=np.array(RainyDay.rolling_sum(catrain, substeps[jd]),dtype='float32')
                    mdrain,_=RainyDay.SSTalt(mdpass,mdx,mdy,trimmask,maskheight,maskwidth,durcheck=True)
                    whichrain_md[jd][mdsel+(0,)]=mdrain*rainprop.timeres/60./mnorm
            # GP 2026-10-22: "table" engine (fast_sst.py), bit-identical to "fast" below
            elif mdengine=='table':
                mdrain=FST.table_multidur(catrain,substeps,mdx,mdy,trimmask,maskheight,maskwidth)
                for jd in range(0,len(subdurs)):
                    whichrain_md[jd][mdsel+(0,)]=mdrain[jd]*rainprop.timeres/60./mnorm
            else:
                mdwins=[np.array(RainyDay.rolling_sum(catrain, s),dtype='float32') for s in substeps]
                mdoff=np.cumsum([0]+[w.shape[0] for w in mdwins]).astype(np.int64)
                mdrain=RainyDay.SSTalt_multidur(np.ascontiguousarray(np.concatenate(mdwins,axis=0)),mdoff,mdx,mdy,trimmask,maskheight,maskwidth)
                for jd in range(0,len(subdurs)):
                    whichrain_md[jd][mdsel+(0,)]=mdrain[jd]*rainprop.timeres/60./mnorm
        FP.toc('resampling: sub-durations')           # GP 2026-10-22: RAINYDAY_TIMERS
        if recordci is not None:                      # GP 2026-10-26: RECORDCI pool of this storm (copy of catrain)
            recordci.add_storm(i,catrain)
        if exactfa is not None and recordci is None:  # GP 2026-10-26: FREQENGINE "both" (with RECORDCI: shared pool)
            exactfa.add_storm(i,catrain)
    FP.toc('resampling: total')                       # GP 2026-10-22: RAINYDAY_TIMERS
    FST.report()                                      # GP 2026-10-22: "table" engine statistics (prints nothing otherwise)

    if (durationcheck==False and durcorrection==False):
        cattime=temptime
    
    if areatype.lower()=='pointlist' or areatype.lower()=='point':
        if len(arfval)==1 and np.isclose(arfval[0],1.):
            sortrain=whichrain*arfval
        else:
            if len(arfval)==1:
                arfrand=np.random.exponential(arfval[0]-1.,size=whichrain.shape)+1.
                arflimit=sp.stats.expon.ppf(0.90,1./(arfval[0]-1.))+1.
                arfmed=sp.stats.expon.ppf(0.5,1./(arfval[0]-1.))+1.
            elif len(arfval)==2:
                arfrand=np.random.gamma(shape=arfval[0],scale=arfval[1],size=whichrain.shape)
                arflimit=sp.stats.gamma.ppf(0.90,a=arfval[0],scale=arfval[1])
                arfmed=sp.stats.gamma.ppf(0.5,a=arfval[0],scale=arfval[1])
            #arfrand[arfrand>1.5]=1.0
            arfrand[arfrand>arflimit]=arfmed
            whichrain=np.multiply(whichrain,arfrand)
            
            
    # modified by DBW to account for k=0 situations, 9/22/2022
    # GP 2026-10-26: FREQENGINE "exact" has no draws (no whichstorms/whichrain): the k=0 marks, the dump hook and the
    # ranking below are skipped (old lines kept commented; the same lines run otherwise). The try/except NameError
    # lines below then do nothing (nostorm_index is not defined).
    # nostorm_index=np.equal(whichstorms,-9999)
    # whichrain[nostorm_index]=-9999.
    # if len(subdurs)>0:      # GP 2026-10-21: MULTIPLEDURATIONS
    if not freqexact: nostorm_index=np.equal(whichstorms,-9999)
    if not freqexact: whichrain[nostorm_index]=-9999.
    if len(subdurs)>0 and not freqexact:      # GP 2026-10-21: MULTIPLEDURATIONS
        whichrain_md[:,nostorm_index]=-9999.
    # GP 2026-10-21: debugging/verification hook: RAINYDAY_DUMP_WHICHRAIN=<folder> saves the raw transposed depths
    # (before ranking) of every duration as <name>_whichrain.npy; does nothing when the variable is not set
    mddump=os.environ.get("RAINYDAY_DUMP_WHICHRAIN","").strip()
    # if mddump!='':                                  # GP 2026-10-26: old line; FREQENGINE "exact": no raw depths
    if mddump!='' and not freqexact:
        os.makedirs(mddump,exist_ok=True)
        np.save(os.path.join(mddump,scenarioname+'_whichrain.npy'),whichrain)
        for jd in range(0,len(subdurs)):
            np.save(os.path.join(mddump,basescenario+'_'+RainyDay.durtag(subdurs[jd])+'_whichrain.npy'),whichrain_md[jd])
    # whichx[nostorm_index]=-9999                     # GP 2026-10-26: old lines; FREQENGINE "exact": no draws
    # whichy[nostorm_index]=-9999
    if not freqexact: whichx[nostorm_index]=-9999
    if not freqexact: whichy[nostorm_index]=-9999
    try:
        whichstep[nostorm_index]=-9999
    except NameError:
        pass
    try:
        whichmultiplier[nostorm_index]=-9999.
    except NameError:
        pass  
    try:
        whichtimeind[nostorm_index]=-9999
    except NameError:
        pass  
            
    
    # HERE ARE THE ANNUAL MAXIMA!!!
    if areatype.lower()=="pointlist":
        sortrain=np.empty((whichrain.shape[1],whichrain.shape[2],npoints_list),dtype='float32')
        #if rescaletype=='stochastic' or rescaletype=='deterministic':
        #    sortmultiplier=np.empty((whichrain.shape[1],whichrain.shape[2],npoints_list),dtype='float32')
        for pt in np.arange(0,whichx.shape[3]):
            
            if calctype.lower()=='ams':
                maxrain=np.nanmax(whichrain[:,:,:,pt],axis=0)
                
            elif calctype.lower()=='pds':
                temprain=np.squeeze(whichrain[:,:,:,pt])
                temppds=temprain.reshape(-1, temprain.shape[-1])
                maxrain=np.sort(temppds,axis=0)[-nsimulations:,:]
            
            
            # PULL OUT THE CORRESPONDING TRANSPOSITION INFORMATION
            maxind=np.nanargmax(whichrain[:,:,:,pt],axis=0)
            
            #added for k=0 catching, DBW 9/22/22
            maxind[np.equal(maxrain,-9999.)]=-9999
        
            
            # THIS ISN'T VERY ELEGANT
            maxx=np.empty((maxind.shape),dtype="int32")
            maxy=np.empty((maxind.shape),dtype="int32")
            maxstorm=np.empty((maxind.shape),dtype="int32")
            maxx[:]=-9999
            maxy[:]=-9999
            maxstorm[:]=-9999
            #if rescaletype=='stochastic' or rescaletype=='deterministic':
            #    maxmultiplier=np.empty((maxind.shape),dtype="float32") 
                
            for i in range(0,np.max(ncounts)):
                maxx[maxind==i]=whichx[i,maxind==i,pt]
                maxy[maxind==i]=whichy[i,maxind==i,pt]
                maxstorm[maxind==i]=whichstorms[i,maxind==i]
             #   if rescaletype=='stochastic' or rescaletype=='deterministic':
             #       maxmultiplier[maxind==i]=whichmultiplier[i,maxind==i,pt]
            
            
            # RANK THE STORMS BY INTENSITY AND ASSIGN RETURN PERIODS
            exceedp=np.linspace(1,1./nsimulations,nsimulations)
            
            returnperiod=1/exceedp
            sortind=np.argsort(maxrain,axis=0)
            sortrain[:,:,pt]=np.sort(maxrain,axis=0)
            #if rescaletype=='stochastic' or rescaletype=='deterministic':
            #    sortmultiplier[:,:,pt]=maxmultiplier[sortind]
                
                
        if alllevels==False:
            reducedlevind=[]
            for i in range(0,len(speclevels)):
                reducedlevind.append(RainyDay.find_nearest(returnperiod,speclevels[i]))  
            
            returnperiod=returnperiod[reducedlevind]
            sortrain=sortrain[reducedlevind,:]
            exceedp=exceedp[reducedlevind]
                
    else:
        # if calctype=='ams' or calctype=='pds':          # this isn't very elegant!   (GP 2026-10-26: old line)
        if (calctype=='ams' or calctype=='pds') and not freqexact:   # GP 2026-10-26: FREQENGINE "exact": no realizations to rank
            if calctype.lower()=='ams':
                maxrain=np.nanmax(whichrain[:,:,:,pt],axis=0)
                maxind=np.nanargmax(whichrain,axis=0)
            elif calctype.lower()=='pds':
                temprain=np.squeeze(whichrain)
                temppds=temprain.reshape(-1, temprain.shape[-1])
                maxrain=np.sort(temppds,axis=0)[-nsimulations:,:]
                maxind=np.nanargmax(whichrain,axis=0)
            
            #added for k=0 catching, DBW 9/22/22
            maxind[np.equal(maxrain,-9999.)]=-9999
            
            #elif calctype.lower()=='npyear':
            #    temprain=np.squeeze(whichrain)
            #    maxrain=np.sort(temprain,axis=0)
            #    maxind=np.argsort(temprain,axis=0)
                
                #maxrain=np.sort(temprain,axis=0)[-5:,:]
                #maxind=np.argsort(temprain,axis=0)[-5:,:]
      
            
            # HERE THE OPTIONAL USER SPECIFIED INTENSITY DISTRIBUTION IS APPLIED    
            if userdistr.all()!=False:
                rvs=sp.stats.genextreme.rvs(userdistr[2],loc=userdistr[0],scale=userdistr[1],size=maxrain.shape).astype('float32')
                maxrain=maxrain*rvs
                maxrain[np.equal(maxind,-9999)]=-9999.
                
            # PULL OUT THE CORRESPONDING TRANSPOSITION INFORMATION
            
            
            # THIS ISN'T VERY ELEGANT
            maxx=np.empty((maxind.shape),dtype="int32")
            maxy=np.empty((maxind.shape),dtype="int32")
            maxx[:]=-9999
            maxy[:]=-9999
            maxstorm=np.empty((maxind.shape),dtype="int32")
            maxstorm[:]=-9999
            # if arfcorrection:
            #     maxstep=np.empty((maxind.shape),dtype="int32")
            #     maxstep[:]=-9999
            if rotation:
                maxangles=np.empty((maxind.shape),dtype="float32")
                sortangle=np.empty((maxind.shape),dtype="float32")
                maxangles[:]=-9999.
                sortangle[:]=-9999.
            if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
                maxmultiplier=np.empty((maxind.shape),dtype="float32") 
                sortmultiplier=np.empty((maxind.shape),dtype="float32")
                maxmultiplier[:]=-9999.
                sortmultiplier[:]=-9999.
                
            for i in range(0,np.max(ncounts)):
                maxx[maxind==i]=np.squeeze(whichx[i,np.squeeze(maxind==i)])
                maxy[maxind==i]=np.squeeze(whichy[i,np.squeeze(maxind==i)])
                maxstorm[maxind==i]=np.squeeze(whichstorms[i,np.squeeze(maxind==i)])
                # if arfcorrection:
                #     maxstep[maxind==i]=np.squeeze(whichstep[i,np.squeeze(maxind==i)])
                
                if rotation:
                    maxangles[maxind==i]=randangle[i,maxind==i]
                if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
                    maxmultiplier[maxind==i]=np.squeeze(whichmultiplier[i,maxind==i])
    #            elif calctype.lower()=='npyear':
    #                sys.exit("having problems here")
    #                for stm in range(0,nperyear):
    #                    maxx[maxind==i]=np.squeeze(whichx[i,np.squeeze(maxind==i)])
    #                    maxy[stm,maxind==i]=np.squeeze(whichy[i,np.squeeze(maxind==i)])
    #                    maxstorm[stm,maxind==i]=np.squeeze(whichstorms[i,np.squeeze(maxind==i)])
    #                    if rotation:
    #                        print("Warning: We haven't tested ROTATION with NPERYEAR!")
    #                        maxangles[stm,maxind==i]=randangle[stm,i,maxind==i]
    #                    if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
    #                        maxmultiplier[stm,maxind==i]=np.squeeze(whichmultiplier[stm,i,maxind==i])
                    
                
            
            # RANK THE STORMS BY INTENSITY AND ASSIGN RETURN PERIODS
            exceedp=np.linspace(1,1./nsimulations,nsimulations)
            returnperiod=1/exceedp
            #rp_pds=1./(1.-np.exp(-1./(returnperiod)))
            #rp_pds=1./np.log(returnperiod/(returnperiod-1))
            sortind=np.argsort(maxrain,axis=0)
            sortrain=np.sort(maxrain,axis=0)
            sortx=np.empty((maxind.shape),dtype="int32")
            sorty=np.empty((maxind.shape),dtype="int32")
            sortstorms=np.empty((maxind.shape),dtype="int32")
            # if arfcorrection:
            #     sortstep=np.empty((maxind.shape),dtype="int32")
            
            
            for i in range(0,nrealizations):
                sortx[:,i]=maxx[sortind[:,i],i]
                sorty[:,i]=maxy[sortind[:,i],i]
                sortstorms[:,i]=maxstorm[sortind[:,i],i]
                # if arfcorrection:
                #     sortstep[:,i]=maxstep[sortind[:,i],i]
                if rotation:
                    sortangle[:,i]=maxangles[sortind[:,i],i]
                if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
                    sortmultiplier[:,i]=maxmultiplier[sortind[:,i],i]
            
                
            # FIND THE TIMES:
            # if arfcorrection==False:
            #     sorttimes=np.zeros((maxind.shape[0],maxind.shape[1],cattime.shape[1]),dtype="datetime64[m]")
            # else:       # using ARFANALYSIS
            #     sorttimes=np.zeros((maxind.shape[0],maxind.shape[1],int(duration*60/rainprop.timeres)),dtype="datetime64[m]")
            
            sorttimes=np.zeros((maxind.shape[0],maxind.shape[1],cattime.shape[1]),dtype="datetime64[m]")
            whichorigstorm=np.zeros((maxind.shape[0],maxind.shape[1]),dtype='int32')
            for i in range(0,nstorms):
                if np.sum(np.squeeze(sortstorms==i))>0:
                    #if arfcorrection==False:
                    sorttimes[np.squeeze(sortstorms==i),:]=cattime[i,:]
                        
                    # else:  # using ARFANALYSIS
                    #     tstep=sortstep[np.squeeze(sortstorms==i),:]
                    #     if len(tstep)>1:
                    #         tempstep=np.squeeze(sortstep[np.squeeze(sortstorms==i),:])
                    #         temptime=np.empty((tempstep.shape[0],int(duration*60/rainprop.timeres)),dtype="datetime64[m]")
                    #         for j in range(0,tempstep.shape[0]):
                    #             temptime[j,:]=cattime[i,tempstep[j]:tempstep[j]+int(duration*60/rainprop.timeres)]
                    #     else:
                    #         tempstep=sortstep[np.squeeze(sortstorms==i),:][0]
                    #         temptime=np.empty((tempstep.shape[0],int(duration*60/rainprop.timeres)),dtype="datetime64[m]")
                    #         for j in range(0,tempstep.shape[0]):
                    #             temptime[j,:]=cattime[i,tempstep[j]:tempstep[j]+int(duration*60/rainprop.timeres)]
    
                    #     sorttimes[np.squeeze(sortstorms==i),:]=temptime
                    whichorigstorm[np.squeeze(sortstorms==i)]=modstormsno[i]+1
                else:
                    continue
                
                
            if alllevels==False:
                reducedlevind=[]
                for i in range(0,len(speclevels)):
                    reducedlevind.append(RainyDay.find_nearest(returnperiod,speclevels[i]))  
                
                returnperiod=returnperiod[reducedlevind]
                sortrain=sortrain[reducedlevind,:]
                sortstorms=sortstorms[reducedlevind,:]
                sorttimes=sorttimes[reducedlevind,:]
                exceedp=exceedp[reducedlevind]
                sortx=sortx[reducedlevind,:]
                sorty=sorty[reducedlevind,:]
                # if arfcorrection:
                #     sortstep=sortstep[reducedlevind,:]

                whichorigstorm=whichorigstorm[reducedlevind,:]
                if rotation:    
                    sortangle=sortangle[reducedlevind,:]
                if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':        
                    sortmultiplier=sortmultiplier[reducedlevind,:]
            
        nanmask=deepcopy(trimmask)
        nanmask[np.isclose(nanmask,0.)]=np.nan
        nanmask[np.isclose(nanmask,0.)==False]=1.0
         
        #################################################################################
        # STEP 2a (OPTIONAL): Find the single storm maximized storm rainfall-added DBW 7/19/2017
        #################################################################################    
        
        
        # if deterministic:
        #     print("finding maximizing precipitation...")
            
        #     max_trnsx=catx[-1]
        #     max_trnsy=caty[-1]
        #     if rotation==False:
        #         # there is some small bug that I don't understand either here or in the storm catalog creation, in which maxstm_avgrain will not exactly match catmax[-1] unless areatype is a point
        #         maxstm_rain=np.multiply(catrain[-1,:,max_trnsy:(max_trnsy+maskheight),max_trnsx:(max_trnsx+maskwidth)],nanmask)
        #         maxstm_avgrain=np.nansum(np.multiply(catrain[-1,:,max_trnsy:(max_trnsy+maskheight),max_trnsx:(max_trnsx+maskwidth)],trimmask))/mnorm
        #         maxstm_ts=np.nansum(np.multiply(maxstm_rain,trimmask)/mnorm,axis=(1,2))
        #         maxstm_time=cattime[-1,:]
        #     else:  
        #         prevmxstm=0.
        #         maxstm_rain=np.empty((catrain.shape[1],nanmask.shape[0],nanmask.shape[1]),dtype='float32')
        #         for i in range(0,nstorms):
        #             passrain=np.nansum(catrain[i,:],axis=0)
        #             xctr=catx[i]+maskwidth/2.
        #             yctr=caty[i]+maskheight/2.
        #             xlinsp=np.linspace(-xctr,rainprop.subdimensions[1]-xctr,rainprop.subdimensions[1])
        #             ylinsp=np.linspace(-yctr,rainprop.subdimensions[0]-yctr,rainprop.subdimensions[0])
        #             ingridx,ingridy=np.meshgrid(xlinsp,ylinsp)
        #             ingridx=ingridx.flatten()
        #             ingridy=ingridy.flatten()
        #             outgrid=np.column_stack((ingridx,ingridy))       
                
        #             for tempang in angbins:
        #                 #print "really should fix the center of rotation! to be the storm center"
        #                 rotx=ingridx*np.cos(tempang)+ingridy*np.sin(tempang)
        #                 roty=-ingridx*np.sin(tempang)+ingridy*np.cos(tempang)
        #                 rotgrid=np.column_stack((rotx,roty))
        #                 delaunay=sp.spatial.qhull.Delaunay(rotgrid)
        #                 interp=sp.interpolate.LinearNDInterpolator(delaunay,passrain.flatten(),fill_value=0.)
        #                 train=np.reshape(interp(outgrid),rainprop.subdimensions)
        #                 temp_maxstm_avgrain=np.nansum(np.multiply(train[max_trnsy:(max_trnsy+maskheight),max_trnsx:(max_trnsx+maskwidth)],trimmask))/mnorm
        #                 if temp_maxstm_avgrain>prevmxstm:
        #                     maxstm_avgrain=temp_maxstm_avgrain
        #                     prevmxstm=maxstm_avgrain
        #                     maxstm_time=cattime[-i,:]
                            
        #                     for k in range(0,len(maxstm_time)):
        #                         interp=sp.interpolate.LinearNDInterpolator(delaunay,catrain[i,k,:].flatten(),fill_value=0.)
        #                         maxstm_rain[k,:]=np.reshape(interp(outgrid),rainprop.subdimensions)[max_trnsy:(max_trnsy+maskheight),max_trnsx:(max_trnsx+maskwidth)]
        #                     maxstm_rain=np.multiply(maxstm_rain,nanmask)
        #                     maxstm_ts=np.nansum(np.multiply(maxstm_rain,trimmask)/mnorm,axis=(1,2))
        	
        
        
    #################################################################################
    # STEP 3 (OPTIONAL): RAINFALL FREQUENCY ANALYSIS
    #################################################################################

    print("preparing frequency analysis...")

    if areatype.lower()=='pointlist':
        spreadmean=np.nanmean(sortrain,1)
        
        if spreadtype=='ensemble':
            spreadmin=np.nanmin(sortrain,axis=1)
            spreadmax=np.nanmax(sortrain,axis=1)   
        else:
            spreadmin=np.percentile(sortrain,(100-quantilecalc)/2,axis=1)
            spreadmax=np.percentile(sortrain,quantilecalc+(100-quantilecalc)/2,axis=1)
        
        if spreadmean.shape[0]!=returnperiod.shape[0] or spreadmax.shape[0]!=returnperiod.shape[0] or spreadmean.shape[1]!=ptlatlist.shape[0]:
            sys.exit("There is some dimension inconsistency in the pointlist scheme!")
        
        #fmean=open(FreqFile_mean,'w')
        #fmin=open(FreqFile_min,'w')
        #fmax=open(FreqFile_max,'w')
        
        #fmean.write('#prob.exceed,returnperiod,meanrain\n'+ptlistname+'\n')
        #fmax.write('#prob.exceed,returnperiod,maxrain\n'+ptlistname+'\n')
        #fmin.write('#prob.exceed,returnperiod,minrain\n'+ptlistname+'\n')

        freqanalysis_mean=np.column_stack((exceedp,returnperiod,spreadmean))
        freqanalysis_min=np.column_stack((exceedp,returnperiod,spreadmin))
        freqanalysis_max=np.column_stack((exceedp,returnperiod,spreadmax))
        
        outlat_line=np.append([-999.,-999.],ptlatlist)
        outlon_line=np.append([-999.,-999.],ptlonlist)
        coordline=np.row_stack((outlat_line,outlon_line))
        freqanalysis_mean=np.row_stack((coordline,freqanalysis_mean))
        freqanalysis_min=np.row_stack((coordline,freqanalysis_min))
        freqanalysis_max=np.row_stack((coordline,freqanalysis_max))
        
        np.savetxt(FreqFile_mean,freqanalysis_mean,delimiter=',',header='prob.exceed,returnperiod,meanrain',fmt='%6.2f',comments='#',footer=ptlistname)
        np.savetxt(FreqFile_min,freqanalysis_min,delimiter=',',header='prob.exceed,returnperiod,minrain',fmt='%6.2f',comments='#',footer=ptlistname)
        np.savetxt(FreqFile_max,freqanalysis_max,delimiter=',',header='prob.exceed,returnperiod,maxrain',fmt='%6.2f',comments='#',footer=ptlistname)
        
    else:
        # GP 2026-10-21: the CSV/PNG writing below moved unchanged to RainyDay.write_frequency_outputs, so that every
        # MULTIPLEDURATIONS duration writes its outputs with the same code (old code kept commented):
        # if spreadtype=='ensemble' or spreadtype=='all_realizations':
        #     spreadmin=np.nanmin(sortrain,axis=1)
        #     spreadmax=np.nanmax(sortrain,axis=1)
        #     spreadmean=np.nanmean(sortrain,1)
        # else:
        #     spreadmin=np.percentile(sortrain,(100-quantilecalc)/2,axis=1)
        #     spreadmax=np.percentile(sortrain,quantilecalc+(100-quantilecalc)/2,axis=1)
        #
        # if spreadtype=='ensemble' or spreadtype=='quantile':
        #     freqanalysis=np.column_stack((exceedp,returnperiod,spreadmin,spreadmean,spreadmax))
        #     np.savetxt(FreqFile,freqanalysis,delimiter=',',header='prob.exceed,returnperiod,minrain,meanrain,maxrain',fmt='%6.2f',comments='')
        # elif spreadtype=='all_realizations':
        #     freqanalysis=np.column_stack((exceedp,returnperiod,sortrain))
        #     rlz_headers = [f'rlz{i+1}' for i in range(nrealizations)]
        #     header = 'prob.exceed,returnperiod,' + ','.join(rlz_headers)
        #     np.savetxt(FreqFile,freqanalysis,delimiter=',',header=header,fmt='%6.2f',comments='')
        # 
        # import matplotlib.patches as mpatches
        # from matplotlib.font_manager import FontProperties
        # from matplotlib import pyplot as plt
        #       # warnings.filterwarnings('ignore')
        # fontP = FontProperties()
        # fontP.set_size('xx-small')
        # fig, ax = plt.subplots(1)
        # line1, = plt.plot(exceedp[exceedp<=0.5], RainyDay.np.nanmean(sortrain,1)[exceedp<=0.5], lw=1, label='Average', color='blue')
        #
        # ax.fill_between(exceedp[exceedp<=0.5], spreadmin[exceedp<=0.5], spreadmax[exceedp<=0.5], facecolor='dodgerblue', alpha=0.5,label='Ensemble Variability')
        # blue_patch = mpatches.Patch(color='dodgerblue', label='Spread')
        # plt.legend(handles=[line1,blue_patch],loc='lower right',prop = fontP)
        #
        # if np.nanmax(spreadmax[exceedp<=0.5])<10.:
        #     upperlimit=10.
        # elif np.nanmax(spreadmax[exceedp<=0.5])<100.:
        #     upperlimit=100.
        # elif np.nanmax(spreadmax[exceedp<=0.5])<1000.:
        #     upperlimit=1000.
        # else:
        #     upperlimit=10000.
        #     
        # if np.nanmin(spreadmin[exceedp<=0.5])<1.:
        #     lowerlimit=0.1
        # elif np.nanmin(spreadmin[exceedp<=0.5])<10.:
        #     lowerlimit=1 
        # elif np.nanmin(spreadmin[exceedp<=0.5])<100.:
        #     lowerlimit=10. 
        # else:
        #     lowerlimit=100.
        #         
        #         
        # plt.ylim(lowerlimit,upperlimit)
        # ax.set_xlabel('Annual Exceed. Prob. [-]\n1/(Return Period) [year]')
        # ax.set_ylabel('Precip. Depth [mm]')
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # plt.gca().invert_xaxis()
        # ax.grid()
        # plt.tight_layout()
        # plt.savefig(fullpath+'/'+scenarioname+'_FrequencyAnalysis.png',dpi=250)
        # plt.close('all')
        # GP 2026-10-26: FREQENGINE "exact"/"both": <name>_FreqExact.csv/.png of every duration (fast_record.ExactFA;
        # "exact" with RECORDCI also writes the RECORDCI outputs from the exact distribution). Before the Monte Carlo
        # writes, so that RECORDCI frees the shared pool only afterwards. "exact" skips the Monte Carlo writes below.
        if exactfa is not None:
            exactfa.write_all(wd,basescenario,fullpath,scenarioname)
        FP.tic('frequency: write parent + rank/write sub-durations')   # GP 2026-10-22: RAINYDAY_TIMERS
        # RainyDay.write_frequency_outputs(sortrain,exceedp,returnperiod,FreqFile,fullpath+'/'+scenarioname+'_FrequencyAnalysis.png',spreadtype,globals().get('quantilecalc'),nrealizations)   # GP 2026-10-26: old line
        if not freqexact: RainyDay.write_frequency_outputs(sortrain,exceedp,returnperiod,FreqFile,fullpath+'/'+scenarioname+'_FrequencyAnalysis.png',spreadtype,globals().get('quantilecalc'),nrealizations)
        # if recordci is not None:                    # GP 2026-10-26: old line; FREQENGINE "exact": written by ExactFA
        if recordci is not None and not freqexact:    # GP 2026-10-26: RECORDCI outputs of the parent duration
            recordci.write(0,sortrain,exceedp,returnperiod,fullpath,scenarioname,spreadtype,globals().get('quantilecalc'))

        # GP 2026-10-21: MULTIPLEDURATIONS: rank and write every shorter duration to MAINPATH/<SCENARIONAME>_<d>/
        # for jd in range(0,len(subdurs)):           # GP 2026-10-26: old line; FREQENGINE "exact": no realizations
        for jd in (range(0,len(subdurs)) if not freqexact else ()):
            mdname=basescenario+'_'+RainyDay.durtag(subdurs[jd])
            mdpath=wd+'/'+mdname
            os.makedirs(mdpath,exist_ok=True)
            mdsort,mdexceed,mdrp=RainyDay.rank_frequency(whichrain_md[jd],calctype,nsimulations,alllevels,globals().get('speclevels'))
            RainyDay.write_frequency_outputs(mdsort,mdexceed,mdrp,mdpath+'/'+mdname+'_FreqAnalysis.csv',mdpath+'/'+mdname+'_FrequencyAnalysis.png',spreadtype,globals().get('quantilecalc'),nrealizations)
            print('MULTIPLEDURATIONS: wrote '+mdpath+'/'+mdname+'_FreqAnalysis.csv')
            if recordci is not None:                  # GP 2026-10-26: RECORDCI outputs of this duration
                recordci.write(1+jd,mdsort,mdexceed,mdrp,mdpath,mdname,spreadtype,globals().get('quantilecalc'))
        FP.toc('frequency: write parent + rank/write sub-durations')   # GP 2026-10-22: RAINYDAY_TIMERS
        if recordci is not None:                      # GP 2026-10-26: RECORDCI timings
            recordci.report()
            
        
    #################################################################################
    # STEP 4 (OPTIONAL): WRITE RAINFALL SCENARIOS
    # this was heavily rewritten in August 2023 by DBW
    # Not many fancy features survived that rewriting
    #################################################################################

    if Scenarios:
        print("writing spacetime precipitation scenarios...")
        # Track how many storms have been written per year
        written_per_year = {}
        # check if scenario files already exist there and if so, delete them:
        if os.path.exists(fullpath+'/Realizations') and os.path.isdir(fullpath+'/Realizations'):
            RainyDay.delete_files_in_directory(fullpath+'/Realizations')
        
        subrangelat=np.array(latrange[ymin:ymax+1])
        subrangelon=np.array(lonrange[xmin:xmax+1])
        minind=RainyDay.find_nearest(returnperiod,RainfallThreshYear)
        
        # sortind=np.argsort(whichrain[:,:,:,0],axis=0)
        # whichrain=np.take_along_axis(np.squeeze(whichrain),sortind,axis=0)
        # whichstorms=np.take_along_axis(np.squeeze(whichstorms),sortind,axis=0)
        #
        # whichx=np.take_along_axis(np.squeeze(whichx),sortind,axis=0)
        # whichy=np.take_along_axis(np.squeeze(whichy),sortind,axis=0)

        # Get sorting indices based on the first channel of whichrain along the first axis
        sortind_first_axis = np.argsort(whichrain[:, :, :, 0], axis=0)

        # # Sort all arrays along the first axis
        # whichrain_sorted_first = np.take_along_axis(np.squeeze(whichrain), sortind_first_axis, axis=0)
        # whichstorms_sorted_first = np.take_along_axis(np.squeeze(whichstorms), sortind_first_axis, axis=0)
        # whichx_sorted_first = np.take_along_axis(np.squeeze(whichx), sortind_first_axis, axis=0)
        # whichy_sorted_first = np.take_along_axis(np.squeeze(whichy), sortind_first_axis, axis=0)

        # Sort all arrays along the first axis: Changed by Gabriel 2025-03-12
        whichrain_sorted_first = np.take_along_axis(np.squeeze(whichrain, axis=3), sortind_first_axis, axis=0)
        whichstorms_sorted_first = np.take_along_axis(whichstorms, sortind_first_axis, axis=0)
        whichx_sorted_first = np.take_along_axis(np.squeeze(whichx, axis=3), sortind_first_axis, axis=0)
        whichy_sorted_first = np.take_along_axis(np.squeeze(whichy, axis=3), sortind_first_axis, axis=0)

        # Get sorting indices based on the first channel of whichrain_sorted_first along the second axis
        sortind_second_axis = np.argsort(whichrain_sorted_first[-1, :, :],axis=0)

        # Sort all arrays along the second axis
        for i in range(whichrain_sorted_first.shape[0]):
            for j in range(whichrain_sorted_first.shape[2]):
                whichrain_sorted_first[i, :, j] = whichrain_sorted_first[i, sortind_second_axis[:,j], j]
                whichstorms_sorted_first[i, :, j] = whichstorms_sorted_first[i, sortind_second_axis[:,j], j]
                whichx_sorted_first[i, :, j] = whichx_sorted_first[i, sortind_second_axis[:,j], j]
                whichy_sorted_first[i, :, j] = whichy_sorted_first[i, sortind_second_axis[:,j], j]

        # Optionally, replace the original arrays with the sorted ones
        whichrain = whichrain_sorted_first
        whichstorms = whichstorms_sorted_first
        whichx = whichx_sorted_first
        whichy = whichy_sorted_first



        whichstorms=whichstorms[-nperyear:,minind:,:]
        writex=whichx[-nperyear:,minind:,:]
        writey=whichy[-nperyear:,minind:,:]
        
        writemask=trimmask
        writemask[np.greater(trimmask,0.)]=1.   # we don't want fractional masks here
        
        
        # if rotation:
        #     sys.exit("We haven't set this up yet after the major refactoring")
        #     writeangle=sortangle[minind:,:]
        #     binwriteang=np.digitize(writeangle.ravel(),angbins).reshape(writeangle.shape)
        # if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
        #     sys.exit("We haven't set this up yet after the major refactoring")
        #     writemultiplier=sortmultiplier[minind:,:]       
        
        for i in np.arange(0,nstorms):
            print("writing scenarios for storm "+str(i+1))
            catrain,raintime,_,_,_,_,_,_,_,_,_ = RainyDay.readcatalog(stormlist[i])
            catrain = np.array(catrain)
            catrain[np.less(catrain,0.)]=np.nan
            if padscenarios>0:
                padrain=np.zeros((np.int16(padscenarios/(rainprop.timeres/60.)),catrain.shape[1],catrain.shape[2]),dtype='float32')
                catrain=np.concatenate([catrain,padrain],axis=0)
                padtime=np.arange(raintime[-1]+np.timedelta64(rainprop.timeres,'m'),raintime[-1]+np.timedelta64(rainprop.timeres,'m')*np.int16(padscenarios/(rainprop.timeres/60.))+np.timedelta64(rainprop.timeres-1,'m'),np.timedelta64(rainprop.timeres,'m'))
                raintime=np.concatenate([raintime,padtime])
            
            howmanystorms=np.sum(whichstorms==i)        # how many transposed storms are due to parent storm i?
            # print(howmanystorms)

            if howmanystorms>0:
                stormindex=np.zeros_like(whichstorms) 
                stormindex[:]=-999
                stormindex[whichstorms==i]=np.arange(0,howmanystorms)    # this will assign a unique identifier to each transposed storm based on parent storm i
                for k in np.arange(0,howmanystorms):
                    tstorm,tyear,trealization=np.where(stormindex==k)    ####tstorm is not the original storm number here but "i" is.
                    outx=writex[stormindex==k]
                    outy=writey[stormindex==k]
                    
                    name_scenariofile=fullpath+'/Realizations/realization'+str(trealization[0]+1)+'/scenario_'+scenarioname+'_rlz'+str(trealization[0]+1)+'year'+str(tyear[0]+1)+'storm'+str(tstorm[0]+1)+'.nc'
                    #outrain=RainyDay.SSTspin_write_v2(catrain,np.squeeze(writex[:,rlz]),np.squeeze(writey[:,rlz]),np.squeeze(writestorm[:,rlz]),nanmask,maskheight,maskwidth,precat,cattime[:,-1],rainprop,spin=prependrain,flexspin=False,samptype=transpotype,cumkernel=cumkernel,rotation=rotation,domaintype=domain_type)
                    RainyDay.writescenariofile(catrain,raintime,outx,outy,name_scenariofile,i,tyear[0],trealization[0],maskheight,maskwidth,subrangelat,subrangelon,scenarioname,writemask)
    
    
    
    #testrain=np.nansum(np.multiply(catrain[:,21 : 21+maskheight, 29 : 29+maskwidth],trimmask),axis=(1,2))/mnorm 
    
    #np.nansum(np.multiply(plotrain[:,caty[i]:caty[i]+maskheight,catx[i]:catx[i]+maskwidth],trimmask),axis=(1,2))/mnorm
    
    # if Scenarios and calctype!='npyear':
    #     print("writing spacetime precipitation scenarios...")
        
    # if alllevels:
    #     minind=RainyDay.find_nearest(returnperiod,RainfallThreshYear)
    #     writemax=sortrain[minind:,:]
    #     writex=sortx[minind:,:]
    #     writey=sorty[minind:,:]
    #     writestorm=sortstorms[minind:,:]
    #     writeperiod=returnperiod[minind:]
    #     writeexceed=exceedp[minind:]
    #     writetimes=sorttimes[minind:,:]
    #     whichorigstorm=whichorigstorm[minind:,:]
    #     if rotation:
    #         writeangle=sortangle[minind:,:]
    #         binwriteang=np.digitize(writeangle.ravel(),angbins).reshape(writeangle.shape)
    #     if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
    #         writemultiplier=sortmultiplier[minind:,:]
    #     #if arfcorrection:
    #     #    writesteps=sortstep[minind:,:]
    # else:
    #     writemax=sortrain
    #     writex=sortx
    #     writey=sorty
    #     writestorm=sortstorms
    #     writeperiod=returnperiod
    #     writeexceed=exceedp
    #     writetimes=sorttimes
    #     #if arfcorrection:
    #     #    writesteps=sortstep
    #     if rotation:
    #         writeangle=sortangle
    #         binwriteang=np.digitize(writeangle.ravel(),angbins).reshape(writeangle.shape)
    #     if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
    #         writemultiplier=sortmultiplier

    # for rlz in range(0,nrealizations):
    #     print("writing scenarios for realization "+str(rlz+1)+"/"+str(nrealizations))
        
    #     # this statement is only really needed if you are prepending rainfall, but it is fast so who cares?
    #     # if arfcorrection!=True:
    #     #     outtime=np.empty((writetimes.shape[0],cattime.shape[1]),dtype='datetime64[m]')
    #     #     unqstm=np.unique(writestorm[:,rlz])
    #     #     for i in range(0,len(unqstm)):
    #     #         outtime[np.squeeze(writestorm[:,rlz])==unqstm[i],:]=cattime[unqstm[i],:]
    #     # else:
    #         # I'm not sure why the above statements for outtime are needed, rather than the following line... I should have commented my code better back in 2015 or whenever I wrote that!
    #     outtime=writetimes[:,rlz,:]
        
    #     if rotation:
    #         if rescaletype=='stochastic':
    #             sys.exit("not tested!")
    #         # elif arfcorrection==True:
    #         #     sys.exit("not tested!")
    #         else:
    #             outrain=RainyDay.SSTspin_write_v2(catrain,writex[:,rlz],writey[:,rlz],writestorm[:,rlz],nanmask,maskheight,maskwidth,precat,cattime[:,-1],rainprop,rlzanglebin=binwriteang[:,rlz],delarray=delarray,spin=prependrain,flexspin=False,samptype=transpotype,cumkernel=cumkernel,rotation=rotation,domaintype=domain_type)
    #     else:
    #         if rescaletype=='stochastic' or rescaletype=='deterministic' or rescaletype=='dimensionless':
    #             outrain=RainyDay.SSTspin_write_v2(catrain,np.squeeze(writex[:,rlz]),np.squeeze(writey[:,rlz]),np.squeeze(writestorm[:,rlz]),nanmask,maskheight,maskwidth,precat,cattime[:,-1],rainprop,spin=prependrain,flexspin=False,samptype=transpotype,cumkernel=cumkernel,rotation=rotation,domaintype=domain_type)
    #             for rl in range(0,outrain.shape[0]):
    #                 outrain[rl,tlen:,:]=outrain[rl,tlen:,:]*writemultiplier[rl,rlz,0]
    #         else:
    #             outrain=RainyDay.SSTspin_write_v2(catrain,np.squeeze(writex[:,rlz]),np.squeeze(writey[:,rlz]),np.squeeze(writestorm[:,rlz]),nanmask,maskheight,maskwidth,precat,cattime[:,-1],rainprop,spin=prependrain,flexspin=False,samptype=transpotype,cumkernel=cumkernel,rotation=rotation,domaintype=domain_type)
        
    #     outrain[:,:,np.isclose(trimmask,0.)]=-9999.               # this line produced problems in CUENCAS CONVERSIONS :(
    #     writename=WriteName+'_SSTrealizationAMS_rlz'+str(rlz+1)+'.nc'
        
            
        
    #     #print "need to write angles to the realization files"
    #     RainyDay.writerealization(scenarioname,rlz,nrealizations,writename,outrain,writemax[:,rlz],np.squeeze(writestorm[:,rlz]),writeperiod,np.squeeze(writex[:,rlz]),np.squeeze(writey[:,rlz]),outtime,subrangelat,subrangelon,whichorigstorm[:,rlz])
    
    
    
    
    
    
    
    
    
    # if calctype=='npyear':
    #     # this code is adapted from Guo Yu's version of RainyDay
    #     # this isn't pretty, but it works. It doesn't have all the functionality of the "1 per year" scenarios above.
    #     print("Writing top N precipitation scenarios")
            
    #     if rescaletype=='none':
    #         whichmultiplier=np.ones_like(whichrain)

    #     for i in range(0,nrealizations):
    #         print("writing scenarios for realization "+str(i+1)+"/"+str(nrealizations))
    #         outrain_large = np.zeros((nsimulations,nperyear,int(catduration),maskheight,maskwidth),dtype='float32')
    #         outrain_large[:] = -9999.
    #         outtime_large =  np.empty((nsimulations,nperyear,int(catduration)),dtype='datetime64[m]')
    #         outtime_large[:] =  np.datetime64(datetime(1900,1,1,0,0,0))
            
    #         rlz_rain = whichrain[:,:,i]*whichmultiplier[:,:,i]
    #         rlz_whichstorms = whichstorms[:,:,i]
    #         rlz_order = np.argsort(-rlz_rain,axis=0)
    #         rlz_order[rlz_whichstorms<0]=-9999
        
    #         rlz_whichx = whichx[:,:,i]
    #         rlz_whichy = whichy[:,:,i]
        
    #         for j in range(0,nsimulations):
    #             # loop through synthetic year
    #             if np.sum(rlz_order[:,j]>=0) >= nperyear:
    #                 n_largest = nperyear
    #             else:
    #                 n_largest = np.sum(rlz_order[:,j]>=0)
                
    #             for k in range(0,n_largest):
                    
    #                 y1 = rlz_whichy[rlz_order[k,j],j][0][0]
    #                 y2 = rlz_whichy[rlz_order[k,j],j][0][0]+ maskheight 
    #                 x1 = rlz_whichx[rlz_order[k,j],j][0][0]
    #                 x2 = rlz_whichx[rlz_order[k,j],j][0][0]+ maskwidth
                    
    #                 sst_storm=rlz_whichstorms[rlz_order[k,j],j][0]
    #                 if sst_storm<=0:
    #                     continue
                    
    #                 SST_rain = np.array(catrain[sst_storm,:,y1:y2,x1:x2])*whichmultiplier[rlz_order[k,j],j,i,0]
    #                 SST_rain[:,np.isclose(trimmask,0.)]=-9999.
                    
    #                 outrain_large[j,k,:,:,:] = SST_rain[:]
    #                 outtime_large[j,k,:] = cattime[sst_storm,:]
            
    #         writename=WriteName+'_SSTrealization'+ str(n_largest) +'PYEAR_rlz'+str(i+1)+'.nc'
    #         RainyDay.writerealization_nperyear(scenarioname,writename,i,nperyear,nrealizations,outrain_large,outtime_large,subrangelat,subrangelon,rlz_order,nsimulations)
    
    #         #### THIS COMMENTED SECTION CAN BE DELETED ONCE 'writerealization_nperyear()' is tested
    #         # dataset=Dataset(filename, 'w', format='NETCDF4')
        
    #         # # create dimensions
        
    #         # outlats=dataset.createDimension('outlat',len(subrangelat))
    #         # outlons=dataset.createDimension('outlon',len(subrangelon))
    #         # time=dataset.createDimension('time',outtime_large.shape[2])
    #         # nyears=dataset.createDimension('nyears',nsimulations)
    #         # topN=dataset.createDimension('topN',nperyear)
        
        
    #         # # create variables
    #         # times=dataset.createVariable('time',np.float64, ('nyears','topN','time'))
    #         # latitudes=dataset.createVariable('latitude',np.float32, ('outlat'))
    #         # longitudes=dataset.createVariable('longitude',np.float32, ('outlon'))
    #         # rainrate=dataset.createVariable('rainrate',np.float32,('nyears','topN','time','outlat','outlon'),zlib=True,complevel=4,least_significant_digit=2)
    #         # top_event=dataset.createVariable('top_event',np.int16, ('nyears'))
    #         # # Global Attributes
        
    #         # dataset.history = 'Created ' + str(datetime.now())
           
    #         # # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
        
    #         # rainrate.units = 'mm/h'
    #         # times.units = 'minutes since 1970-01-01 00:00.0'
    #         # times.calendar = 'gregorian'
        
    #         # # fill the netcdf file
    #         # latitudes[:]=subrangelat
    #         # longitudes[:]=subrangelon
    #         # rainrate[:]=outrain_large 
    #         # times[:]=outtime_large
    #         # n_evnet = np.nansum(rlz_order>=0,axis=0)
    #         # n_evnet[n_evnet>=nperyear]=nperyear
    #         # top_event[:]= n_evnet
           
    #         # dataset.close()
else:
    print("skipping the frequency analysis!")
    
import time     # reimporting due to goofy issue with nperyear writing above      
end = time.time()   
print("RainyDay has successfully finished!")
print("Elapsed time: "+"{0:0.2f}".format((end - start)/60.)+" minutes")
FP.report()                                           # GP 2026-10-22: RAINYDAY_TIMERS

   
#################################################################################
# THE END
#################################################################################

