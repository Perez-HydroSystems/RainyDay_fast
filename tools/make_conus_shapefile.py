#!/usr/bin/env python
"""Build a CONUS-mainland polygon shapefile (EPSG:4326) for use as a RainyDay
transposition domain, from Natural Earth admin_0 countries (110m)."""
import os
import sys

import cartopy.io.shapereader as shpreader
from shapely.geometry import mapping, MultiPolygon
import fiona

outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
os.makedirs(outdir, exist_ok=True)
outshp = os.path.join(outdir, 'CONUS_domain.shp')

shpfile = shpreader.natural_earth(resolution='110m', category='cultural',
                                  name='admin_0_countries')
usa = None
for rec in shpreader.Reader(shpfile).records():
    if rec.attributes.get('ADM0_A3') == 'USA':
        usa = rec.geometry
        break
if usa is None:
    sys.exit('USA not found in natural_earth admin_0_countries')

# largest polygon = CONUS mainland (drops Alaska, Hawaii, islands)
polys = list(usa.geoms) if isinstance(usa, MultiPolygon) else [usa]
conus = max(polys, key=lambda p: p.area)
print('CONUS polygon bounds:', ['%.2f' % b for b in conus.bounds])

schema = {'geometry': 'Polygon', 'properties': {'name': 'str'}}
crs_wkt = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",'
           '6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
           'UNIT["Degree",0.0174532925199433]]')
with fiona.open(outshp, 'w', driver='ESRI Shapefile', schema=schema,
                crs_wkt=crs_wkt) as dst:
    dst.write({'geometry': mapping(conus), 'properties': {'name': 'CONUS'}})
print('wrote', outshp)
