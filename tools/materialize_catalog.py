#!/usr/bin/env python
"""
GP 2026-10-22: write the full per-storm catalog of a light catalog (CATALOGSTORAGE "light"), exactly as RainyDay's
writecatalog writes it during a run with CATALOGSTORAGE "full": same file names (<CATALOGNAME>_storm_<n>_<date>.nc),
variables, encodings, attributes and values (the "history" attribute gets the new creation time).

    python tools/materialize_catalog.py <light catalog .nc> <out StormCatalog dir> [--rainpath GLOB] [--overwrite]

The rain of each storm is rebuilt from the input files (fast_io.get_storm_rain: this RAINPATH glob, matched by file
name, or the paths stored in the catalog; CRC32 checked) and written with RainyDay_functions.writecatalog. Use it to
share or archive a catalog, or for SCENARIOS (which needs full catalogs). Check:
    python tools/compare_catalogs.py <reference StormCatalog> <out dir> --rtol 0
"""
import argparse
import ast
import glob
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Source"))
import RainyDay_utilities_Py3.RainyDay_functions as RD   # noqa: E402
import RainyDay_utilities_Py3.fast_io as FIO              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("light"); ap.add_argument("outdir")
    ap.add_argument("--rainpath", help="input files glob (default: the files stored in the catalog)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    light = os.path.abspath(a.light)
    FIO.set_rainpath(a.rainpath)
    c = FIO._load(light)
    ds = c['ds']
    os.makedirs(a.outdir, exist_ok=True)
    if glob.glob(os.path.join(a.outdir, "*_storm_*.nc")) and not a.overwrite:
        sys.exit(a.outdir + " already holds storm files (use --overwrite)")

    # writecatalog takes the JSON file and stores str(params) as "description": rebuild a file that gives the same string
    desc = ds.attrs['description']
    params = ast.literal_eval(desc)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(params, tmp); tmp.close()
    with open(tmp.name) as f:
        same_desc = str(json.loads(f.read())) == desc
    src = ds.attrs['source']
    scenarioname = src[len('RainyDay Storm Catalog for scenario '):-len('. See description for JSON file contents.')]

    catmax = np.array(ds['basinrainfall'])
    catx, caty = np.array(ds['xlocation']), np.array(ds['ylocation'])
    cattime = c['cattime'].copy()                                  # datetime64[m], as in the run
    gridmask, dmask = np.array(ds['gridmask']), np.array(ds['domainmask'])
    timeres = ds['timeresolution'].values[()]                      # np.int32, as rainprop.timeres
    nstorms = len(c['names'])
    catduration = cattime.shape[1] * float(timeres) / 60.
    t0 = time.perf_counter()
    for i, name in enumerate(c['names']):
        rain = FIO.get_storm_rain(light, i)                        # CRC32 checked
        out = os.path.join(a.outdir, name)
        if os.path.exists(out):
            os.remove(out)
        RD.writecatalog(scenarioname, rain, catmax, catx, caty, cattime, ds['latitude'], ds['longitude'], out,
                        gridmask, tmp.name, dmask, nstorms, catduration, storm_num=int(i), timeresolution=timeres)
        if not same_desc:
            import netCDF4
            with netCDF4.Dataset(out, 'a') as f:
                f.setncattr('description', desc)
    os.remove(tmp.name)
    print("materialized %d storms into %s in %.1f s (%.2f GB)" % (nstorms, a.outdir, time.perf_counter() - t0,
          sum(os.path.getsize(p) for p in glob.glob(os.path.join(a.outdir, "*_storm_*.nc"))) / 1e9))


if __name__ == "__main__":
    main()
