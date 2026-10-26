"""
GP 2026-10-26: RainyDay_fast: fixes of the JSON parsing of RainyDay_Py3.py (README_CHANGES B9). Each function is called
from a one-line hook in RainyDay_Py3.py (marked "GP 2026-10-26", the replaced code kept as comments next to it).

Rule: every input that already worked parses to the SAME objects as before (same values, same container types, same
element types: np.int32 where the old code made np.int32, Python int where it made Python int, the JSON list itself
where it kept the JSON list). Only inputs that crashed, or that were silently ignored, behave differently.
tools/test_parsing.py checks this against verbatim copies of the old parsing code.

    check_required        missing required keys: a clear message instead of a KeyError traceback
    parse_excludemonths   wrap-around ranges ("12-3" -> [12, 1, 2, 3]); a single month as a string ("8");
                          lists of strings (["12", "1"]) -> integers
    parse_includeyears    "all" (and "none", as the original User's Guide shows) explicitly -> all years;
                          lists of strings -> integers
    parse_excludestorms   a single storm number as a string ("17")
    parse_calctype        an unknown CALCTYPE stops before the storm catalog is built (missing: AMS, as before)
    parse_uncertainty     UNCERTAINTY as a number (JSON 90) or a numeric string ("90"): quantile band
    returnlevels_from_string  RETURNLEVELS as a comma string ("2,5,10"), as the original User's Guide documents
    default_nstorms       NSTORMS absent with an existing catalog: the number of storms in the catalog

Strings that cannot be parsed (e.g. EXCLUDEMONTHS "summer", INCLUDEYEARS "2002to2010") used to be ignored silently
(nothing excluded, all years); they now stop the run with a message that lists the accepted formats.
No np.random call here.
"""
import sys

import numpy as np

CALCTYPES = {"ams": "ams", "annmax": "ams", "pds": "pds", "partialduration": "pds"}


# =====================================================================================================================
# missing required keys
# =====================================================================================================================
def required_keys(cardinfo):
    """Keys that RainyDay_Py3.py reads unconditionally (or under the given condition) with no usable default; a missing
    one used to end in a KeyError traceback (the old code caught ImportError/ValueError/IndexError, never KeyError)."""
    keys = [("MAINPATH", "the folder where the outputs are written"),
            ("SCENARIONAME", "the name of the run (output folder and file prefix)"),
            ("CATALOGNAME", "the name (file prefix) of the storm catalog"),
            ("CREATECATALOG", "'true' (build a new storm catalog) or 'false' (use an existing one)"),
            ("DURATION", "the duration (hours) of the storm catalog / analysis"),
            ("VARIABLES", 'the variable names of the input files, e.g. {"precip": "RAINRATE", "latname": "latitude", '
                          '"longname": "longitude"}'),
            ("POINTAREA", "'watershed', 'point', 'grid', 'box' or 'rectangle'")]
    create = isinstance(cardinfo.get("CREATECATALOG"), str) and cardinfo["CREATECATALOG"].lower() == "true"
    area = cardinfo.get("POINTAREA")
    area = area.lower() if isinstance(area, str) else ""
    domain = cardinfo.get("DOMAINTYPE", "rectangular")
    domain = domain.lower() if isinstance(domain, str) else ""
    if create:
        keys.append(("RAINPATH", "the input files, e.g. /data/AORC.*.preciptemp.nc (CREATECATALOG 'true')"))
    if area in ("basin", "watershed"):
        keys.append(("WATERSHEDSHP", "the watershed polygon shapefile (POINTAREA '" + area + "')"))
    elif area in ("point", "grid"):
        keys += [("POINTLAT", "the latitude of the point (POINTAREA '" + area + "')"),
                 ("POINTLON", "the longitude of the point (POINTAREA '" + area + "')")]
    elif area in ("box", "rectangle"):
        keys += [(k, "the box edges BOX_YMIN, BOX_YMAX, BOX_XMIN, BOX_XMAX (POINTAREA '" + area + "')")
                 for k in ("BOX_YMIN", "BOX_YMAX", "BOX_XMIN", "BOX_XMAX")]
    if create and domain != "irregular":      # the driver takes the rectangular branch for any DOMAINTYPE but irregular
        keys.append(("AREA_EXTENT", 'the rectangular transposition domain (DOMAINTYPE rectangular, the default), e.g. '
                                    '{"LATITUDE_MIN": 42.5, "LATITUDE_MAX": 44.0, "LONGITUDE_MIN": -90.5, '
                                    '"LONGITUDE_MAX": -88.0}'))
    return keys


def check_required(cardinfo):
    """Stop with one message that names every missing required key."""
    missing = [(k, why) for k, why in required_keys(cardinfo) if k not in cardinfo]
    if missing:
        sys.exit("The parameter file is missing required keys:\n"
                 + "\n".join("  " + k + ": " + why for k, why in missing))


# =====================================================================================================================
# helpers
# =====================================================================================================================
def _stop(key, value, formats):
    sys.exit("Can't read " + key + " = " + repr(value) + ". Accepted: " + formats)


def _int32(token, key, value, formats):
    try:
        return np.int32(token)
    except (TypeError, ValueError, OverflowError):
        _stop(key, value, formats)


def _int_list(values, key, formats):
    """Lists of strings: a JSON list as RainyDay used it, with the string elements converted to np.int32. A list without strings is
    returned unchanged (the same object, as the old code kept it)."""
    if not any(isinstance(v, str) for v in values):
        return values
    return [_int32(v, key, values, formats) if isinstance(v, str) else v for v in values]


MONTHS_FORMATS = ('"none"; one month "8"; a comma list "6,7,8"; a range "6-8" or a range across the new year "12-3" '
                  '(= 12,1,2,3); or a JSON list [6,7,8] (numbers or numeric strings)')
YEARS_FORMATS = ('"all" or "none" (all years); one year "2002"; a comma list "2002,2005"; a range "2002-2023"; or a JSON '
                 'list [2002,2005] (numbers or numeric strings)')
STORMS_FORMATS = '"none"; one storm "17"; a comma list "3,17"; a range "1-20"; or a JSON list [3,17]'


# =====================================================================================================================
# EXCLUDEMONTHS
# =====================================================================================================================
def parse_excludemonths(cardinfo):
    """EXCLUDEMONTHS -> list of months to exclude ([] = none). Same objects as the old parser for every input it read
    correctly (comma string -> np.int32 list, range -> Python int list, JSON list -> the list itself)."""
    if "EXCLUDEMONTHS" not in cardinfo:
        return []
    value = cardinfo["EXCLUDEMONTHS"]
    if isinstance(value, str):
        if ',' in value:
            return [_int32(tok, "EXCLUDEMONTHS", value, MONTHS_FORMATS) for tok in value.split(",")]
        if '-' in value:
            try:
                first, last = map(int, value.split('-'))
            except ValueError:
                _stop("EXCLUDEMONTHS", value, MONTHS_FORMATS)
            if first <= last:
                return [m for m in range(first, last + 1)]                       # as before
            # a range across the new year; the old code gave range(12, 4) = [] (nothing excluded, silently)
            if not (1 <= last < first <= 12):
                _stop("EXCLUDEMONTHS", value, MONTHS_FORMATS)
            return [m for m in range(first, 13)] + [m for m in range(1, last + 1)]
        if value.strip().lower() in ('none', ''):
            return []
        # one month as a string; the old code ignored it (nothing excluded, silently)
        return [_int32(value, "EXCLUDEMONTHS", value, MONTHS_FORMATS)]
    if isinstance(value, list):
        return _int_list(value, "EXCLUDEMONTHS", MONTHS_FORMATS)                 # lists of strings
    return []                                                                    # as before (other JSON types)


# =====================================================================================================================
# INCLUDEYEARS
# =====================================================================================================================
def parse_includeyears(cardinfo):
    """INCLUDEYEARS -> list of years, or False (all years). Same objects as the old parser for every input it read
    correctly (comma string and single year -> np.int32 list, range -> Python int list, JSON list -> the list itself)."""
    if "INCLUDEYEARS" not in cardinfo:
        return False
    value = cardinfo["INCLUDEYEARS"]
    if isinstance(value, str):
        # "all": the old test was includeyr.lower == "all" (the method, never equal); "all" only worked because
        # np.int32("all") raised a ValueError that the surrounding try/except turned into False
        if value.strip().lower() in ('all', 'none', ''):
            return False
        if ',' in value:
            return [_int32(tok, "INCLUDEYEARS", value, YEARS_FORMATS) for tok in value.split(",")]
        if '-' in value:
            try:
                first, last = map(np.int32, value.split('-'))
            except (TypeError, ValueError, OverflowError):
                _stop("INCLUDEYEARS", value, YEARS_FORMATS)
            return [year for year in range(first, last + 1)]                     # as before
        return [_int32(value, "INCLUDEYEARS", value, YEARS_FORMATS)]
    if isinstance(value, list):
        if len(value) == 0:
            return False
        return _int_list(value, "INCLUDEYEARS", YEARS_FORMATS)                   # lists of strings
    return False                                                                 # as before (other JSON types)


# =====================================================================================================================
# EXCLUDESTORMS
# =====================================================================================================================
def parse_excludestorms(value):
    """EXCLUDESTORMS value -> list of storm numbers ([] = none). Same objects as the old parser for every input it read
    correctly."""
    if isinstance(value, str):
        if ',' in value:
            return [_int32(tok, "EXCLUDESTORMS", value, STORMS_FORMATS) for tok in value.split(",")]
        if '-' in value:
            try:
                first, last = map(int, value.split('-'))
            except ValueError:
                _stop("EXCLUDESTORMS", value, STORMS_FORMATS)
            return [s for s in range(first, last + 1)]                           # as before
        if value.strip().lower() in ('none', ''):
            return []
        # one storm number as a string; the old code ignored it (nothing excluded, silently)
        return [_int32(value, "EXCLUDESTORMS", value, STORMS_FORMATS)]
    if isinstance(value, list):
        return value                                                             # as before
    return []                                                                    # as before


# =====================================================================================================================
# CALCTYPE
# =====================================================================================================================
def parse_calctype(cardinfo):
    """CALCTYPE -> 'ams' or 'pds' (same messages as before). Missing (or null): 'ams', as before. Anything else stops
    here, before the storm catalog is built (the old code printed "Defaulting to Annual Maxima Series" but never set
    calctype: NameError after the catalog)."""
    value = cardinfo.get("CALCTYPE", None)
    if value is None:
        print("Nothing provided for CALCTYPE. Defaulting to Annual Maxima Series.")
        return 'ams'
    calctype = CALCTYPES.get(value.strip().lower()) if isinstance(value, str) else None
    if calctype is None:
        sys.exit("Unrecognized CALCTYPE " + repr(value) + ". Valid values: 'ams' or 'annmax' (annual maxima), "
                 "'pds' or 'partialduration' (partial duration series); omit it for 'ams'.")
    if calctype == 'ams':
        print("You've selected to use Annual Maxima Series.")
    else:
        print("You've selected to use Partial Duration Series.")
    return calctype


# =====================================================================================================================
# UNCERTAINTY
# =====================================================================================================================
def parse_uncertainty(spread):
    """UNCERTAINTY value -> (spreadtype, quantilecalc). quantilecalc is None unless spreadtype is 'quantile' (the old
    code left it undefined; the driver only reads it through globals().get('quantilecalc')).
    'ensemble' / 'all_realizations' as before; an integer X in 0..100 (JSON number or numeric string) -> 'quantile':
    the (100-X)/2 and X+(100-X)/2 percentiles of the realizations (X = 90: 5th-95th); any other value -> 'ensemble'
    with a message, as before (e.g. "false" in the Talequah example)."""
    if isinstance(spread, str):
        s = spread.strip().lower()
        if s == 'ensemble':
            print('"ensemble spread" will be calculated for precipitation frequency analysis...')
            return 'ensemble', None
        if s == 'all_realizations':
            print(' The estimation from all the realization will be printed for precipitation frequency analysis...')
            return 'all_realizations', None
        try:
            quantile = int(s)
        except ValueError:
            print('unrecognized value for UNCERTAINTY (' + repr(spread) + '); using "ensemble".')
            return 'ensemble', None
    elif isinstance(spread, (int, float, np.integer, np.floating)) and not isinstance(spread, (bool, np.bool_)):
        # a JSON number; the old code called spread.lower() and fell back to 'ensemble' silently
        if not float(spread).is_integer():
            sys.exit('UNCERTAINTY must be "ensemble", "all_realizations" or an integer from 0 to 100 (the central '
                     'percent range of the realizations), not ' + repr(spread))
        quantile = int(spread)
    else:
        print('unrecognized value for UNCERTAINTY (' + repr(spread) + '); using "ensemble".')
        return 'ensemble', None
    if quantile < 0 or quantile > 100:
        sys.exit('invalid quantile range for frequency analysis...')
    printcalc = (100 - quantile) // 2
    print(str(printcalc) + 'th-' + str(quantile + printcalc) + 'th interquantile range will be calculated for '
          'precipitation frequency analysis...')
    return 'quantile', quantile


# =====================================================================================================================
# RETURNLEVELS
# =====================================================================================================================
def returnlevels_from_string(speclevels):
    """RETURNLEVELS "2,5,10" (the original User's Guide format; the fork's list fix 07021dd made it stop) -> the
    list [2.0, 5.0, 10.0], which the driver then handles exactly as the JSON list [2, 5, 10] (np.float32 of either gives
    the same array). Anything else, including a string that is not a list of numbers, is returned unchanged (a JSON
    list goes on as before; another string still stops with the old message)."""
    if not isinstance(speclevels, str):
        return speclevels
    try:
        return [float(tok) for tok in speclevels.split(',')]
    except ValueError:
        return speclevels


# =====================================================================================================================
# NSTORMS absent with an existing storm catalog
# =====================================================================================================================
def default_nstorms(nstorms, defaultstorms, ncatalog):
    """(nstorms, defaultstorms) for a run on an existing catalog. Without NSTORMS the old code compared an undefined
    nstorms (NameError), and later read an undefined nstorms_default. Now the run uses every storm of the catalog, i.e.
    it is the same run as with NSTORMS = the catalog's storm count (defaultstorms becomes False so the month/year filter
    is applied to every storm array, as with NSTORMS given)."""
    if not defaultstorms:
        return nstorms, defaultstorms
    print("NSTORMS not given: using the " + str(ncatalog) + " storms of the existing catalog")
    return ncatalog, False
