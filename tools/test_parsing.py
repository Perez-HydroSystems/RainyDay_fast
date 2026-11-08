#!/usr/bin/env python
"""
GP 2026-10-26: unit tests of the parsing fixes of README_CHANGES B9 (Source/RainyDay_utilities_Py3/fast_parse.py and their
hooks in Source/RainyDay_Py3.py; numeric UNCERTAINTY also in RainyDay_functions.write_frequency_outputs). No input data needed.

    python tools/test_parsing.py          (RainyDay_Env; a few seconds; PASS/FAIL per check, ALL PASS at the end)

How: OLD_BLOCKS below are verbatim copies of the parsing blocks of RainyDay_Py3.py at 34d6248 (fast/record, before
the fixes), run with exec(). The NEW behaviour is the CURRENT driver code: the blocks are cut out of
Source/RainyDay_Py3.py by markers and run with exec() in the same way (so the hooks themselves are tested, including the
try/except around them). Each test documents the old behaviour (asserted on OLD_BLOCKS) and asserts the new one.

  wrap range: EXCLUDEMONTHS "12-3"           -> old [] (nothing excluded, silently)        new [12, 1, 2, 3]
  single month: EXCLUDEMONTHS "8"            -> old []                                    new [8]
  string lists: EXCLUDEMONTHS / INCLUDEYEARS lists of strings -> old kept the strings (no month excluded; no year matched:
      empty file list, crash)            new np.int32
  INCLUDEYEARS "all":                    -> old False only through a caught ValueError  new False explicitly
  existing catalog: EXCLUDEMONTHS on an existing catalog without INCLUDEYEARS -> old catinclude kept every storm while stormlist
      dropped some (arrays misaligned)    new aligned
  EXCLUDESTORMS "17":                    -> old []                                    new [17]
  unknown CALCTYPE "foo":                -> old calctype undefined (NameError after the catalog)  new SystemExit at
      parse time, before the catalog block; missing CALCTYPE -> 'ams' (unchanged)
  numeric UNCERTAINTY 90 (JSON)        -> old 'ensemble' silently; "90" -> old 'quantile' then UnboundLocalError
      (a NameError) in write_frequency_outputs          new ('quantile', 90) and the 5th/mean/95th percentile CSV is written
  missing required keys:                 -> old KeyError traceback                    new SystemExit naming the keys
  NSTORMS absent, existing catalog:    -> old NameError (nstorms, then nstorms_default)  new nstorms = catalog count
  RETURNLEVELS "2,5,10":                 -> old SystemExit                            new the same float32 array as
      the JSON list [2, 5, 10]
  Production configuration (and the Talequah / matrix JSONs, and a battery of valid inputs): every parsed object identical
  to the old code's, including container and element types.
"""
import io
import json
import os
import sys
import tempfile
import textwrap
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Source"))
import RainyDay_utilities_Py3.RainyDay_functions as RDF   # noqa: E402
import RainyDay_utilities_Py3.fast_parse as FPA           # noqa: E402

DRIVER = (REPO / "Source" / "RainyDay_Py3.py").read_text().split("\n")
RES = []

# Verbatim copies of the old code, RainyDay_Py3.py and RainyDay_functions.py at 34d6248 (line numbers in brackets).
OLD_BLOCKS = {}
OLD_BLOCKS['excludemonths'] = r'''  # [F:632-649]
try:
    excludemonths=cardinfo["EXCLUDEMONTHS"]
    if isinstance(excludemonths, str):
        if ',' in excludemonths:  ## For different months to be exlcuded from the analysis
            excludemonths = [np.int32(exstorm) for exstorm in excludemonths.split(",")]
        elif '-' in excludemonths:   ## For a range of months of storms to exclude from the analysis
            start_year, end_year = map(int, excludemonths.split('-'))
            excludemonths = [exstorm for exstorm in range(start_year, end_year + 1)]
        elif excludemonths.lower() == 'none': 
            excludemonths = []
        else:
            excludemonths =[]
    elif isinstance(excludemonths, list):
        excludemonths = excludemonths
    else:
        excludemonths = []
except Exception:
    excludemonths=[]
'''
OLD_BLOCKS['includeyears'] = r'''  # [F:652-672]
try:
    includeyr=cardinfo["INCLUDEYEARS"]
    if isinstance(includeyr , str):
        if includeyr.lower == "all":
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
        includeyears=False
except Exception:
    includeyears=False
'''
OLD_BLOCKS['calctype'] = r'''  # [F:692-704]
try:
    CalcType=cardinfo["CALCTYPE"]
    if CalcType.lower()=='ams' or CalcType.lower()=='annmax':
        calctype='ams'
        print("You've selected to use Annual Maxima Series.")
    elif CalcType.lower()=='pds' or CalcType.lower()=='partialduration':
        calctype='pds'
        print("You've selected to use Partial Duration Series.")
    else:
        print("Unrecognized entry for CALCTYPE. Defaulting to Annual Maxima Series.")
except Exception:
    calctype='ams'
    print("Nothing provided for CALCTYPE. Defaulting to Annual Maxima Series.")
'''
OLD_BLOCKS['uncertainty'] = r'''  # [F:751-775]
try:
    if np.any(np.core.defchararray.find(list(cardinfo.keys()),"UNCERTAINTY")>-1):
        spread=cardinfo["UNCERTAINTY"]
        if spread.lower()=='ensemble':
            spreadtype='ensemble'
            print('"ensemble spread" will be calculated for precipitation frequency analysis...')
        elif spread.lower()=='all_realizations':
            spreadtype='all_realizations'
            print(' The estimation from all the realization will be printed for precipitation frequency analysis...')
        else:
            try:
                int(spread)
            except:
                print('unrecognized value for UNCERTAINTY.')
            if int(spread)>=0 and int(spread)<=100:
                spreadtype='quantile'
                quantilecalc=int(spread)
                printcalc=(100-quantilecalc)//2
                print(str(printcalc)+'th-'+str(quantilecalc+printcalc)+'th interquantile range will be calculated for precipitation frequency analysis...')
            else:
                sys.exit('invalid quantile range for frequency analysis...')
    else:
        spreadtype='ensemble'
except Exception:
    spreadtype='ensemble'
'''
OLD_BLOCKS['returnlevels'] = r'''  # [F:777-798]
try:
    if np.any(np.core.defchararray.find(list(cardinfo.keys()),"RETURNLEVELS")>-1):
        speclevels=cardinfo["RETURNLEVELS"]
        if isinstance(speclevels, str) and speclevels.lower() == 'all':
            print('using all return levels...')
            alllevels=True
        else:
            alllevels=False
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
'''
OLD_BLOCKS['nstorms'] = r'''  # [F:978-981]
stm_num = len(stormlist)
if nstorms>stm_num:
    print("WARNING: The storm catalog has fewer storms than the specified nstorms")
    nstorms=stm_num 
'''
OLD_BLOCKS['excludestorms'] = r'''  # [F:1537-1584]
origstormsno=np.arange(0,len(stormlist),dtype='int32')                 
# EXLCUDE "BAD" STORMS OR FOR DOING SENSITIVITY ANALYSIS TO PARTICULAR STORMS.  THIS IS PARTICULARLY NEEDED FOR ANY RADAR RAINFALL PRODUCT WITH SERIOUS ARTIFACTS
# includestorms=np.ones((len(stormlist)),dtype="bool")

try:
    exclude=cardinfo["EXCLUDESTORMS"]
    if isinstance(exclude, str):
        if ',' in exclude:
            exclude = [np.int32(exstorm) for exstorm in exclude.split(",")]
        elif '-' in exclude:
            start_year, end_year = map(int, exclude.split('-'))
            exclude = [exstorm for exstorm in range(start_year, end_year + 1)]
        elif exclude.lower() == 'none':
            exclude = []
        else:
            exclude =[]
    elif isinstance(exclude, list):
        exclude = exclude
    else:
        exclude = []
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
'''
OLD_BLOCKS['excludestorms_parse'] = r'''  # [F:1542-1556]
exclude=cardinfo["EXCLUDESTORMS"]
if isinstance(exclude, str):
    if ',' in exclude:
        exclude = [np.int32(exstorm) for exstorm in exclude.split(",")]
    elif '-' in exclude:
        start_year, end_year = map(int, exclude.split('-'))
        exclude = [exstorm for exstorm in range(start_year, end_year + 1)]
    elif exclude.lower() == 'none':
        exclude = []
    else:
        exclude =[]
elif isinstance(exclude, list):
    exclude = exclude
else:
    exclude = []
'''
OLD_BLOCKS['filtering'] = r'''  # [F:1587-1626]
# EXCLUDE STORMS BY MONTH AND YEAR
# THIS SECTION EXISTS IN CASE YOU WANT TO USE A PRE-EXISTING STORM CATALOG THAT HASN'T CONSIDERED ANY MONTH-BASED OR YEAR-BASED EXCLUSION
if CreateCatalog==False:  
    if includeyears == False:
        catinclude = [True if RainyDay.extract_date(storm, catalogname)[4:6]\
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
'''
OLD_BLOCKS['write'] = r'''  # [FF:3131-3140 (write_frequency_outputs)]
if spreadtype=='ensemble' or spreadtype=='all_realizations':
    spreadmin=np.nanmin(sortrain,axis=1)
    spreadmax=np.nanmax(sortrain,axis=1)
    spreadmean=np.nanmean(sortrain,1)
else:
    spreadmin=np.percentile(sortrain,(100-quantilecalc)/2,axis=1)
    spreadmax=np.percentile(sortrain,quantilecalc+(100-quantilecalc)/2,axis=1)

if spreadtype=='ensemble' or spreadtype=='quantile':
    freqanalysis=np.column_stack((exceedp,returnperiod,spreadmin,spreadmean,spreadmax))
'''


def report(ok, msg):
    RES.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def snippet(first, last, first_offset=0, last_offset=0, col0_last=False):
    """Lines of the CURRENT driver from the first line containing `first` (+offset) to the next line containing `last`
    (+offset); col0_last: `last` must start the line (column 0). Dedented."""
    i = next(k for k, s in enumerate(DRIVER) if first in s)
    j = next(k for k in range(i, len(DRIVER))
             if (DRIVER[k].startswith(last) if col0_last else last in DRIVER[k]) and k > i - 1)
    return textwrap.dedent("\n".join(DRIVER[i + first_offset:j + 1 + last_offset])) + "\n"


NEW_BLOCKS = dict(
    excludemonths=lambda: snippet("excludemonths=FPA.parse_excludemonths(cardinfo)", "excludemonths=FPA.parse_excludemonths(cardinfo)"),
    includeyears=lambda: snippet("includeyears=FPA.parse_includeyears(cardinfo)", "includeyears=FPA.parse_includeyears(cardinfo)"),
    calctype=lambda: snippet("calctype=FPA.parse_calctype(cardinfo)", "calctype=FPA.parse_calctype(cardinfo)"),
    uncertainty=lambda: snippet('find(list(cardinfo.keys()),"UNCERTAINTY")', "except Exception:", -1, 1, True),
    returnlevels=lambda: snippet('find(list(cardinfo.keys()),"RETURNLEVELS")', "except Exception:", -1, 1, True),
    nstorms=lambda: snippet("nstorms,defaultstorms=FPA.default_nstorms(", "nstorms=stm_num"),
    excludestorms=lambda: snippet("origstormsno=np.arange(0,len(stormlist),dtype='int32')", "modstormsno=origstormsno[includestorms]"),
    filtering=lambda: snippet("# EXCLUDE STORMS BY MONTH AND YEAR", "stormnumber = ["),
)
MISSING = object()


def run(code, **ns):
    """exec a driver block; returns (namespace, None) or (namespace, the exception). Prints are swallowed."""
    g = dict(np=np, sys=sys, RainyDay=RDF, FPA=FPA)
    g.update(ns)
    try:
        with redirect_stdout(io.StringIO()):
            exec(compile(code, "<driver block>", "exec"), g)
    except BaseException as e:                     # SystemExit, NameError, ...
        return g, e
    return g, None


def parse(which, card, var, **extra):
    """(old value, new value) of variable `var` after the old and new block `which` (MISSING if undefined; the
    exception if the block raised)."""
    out = []
    for code in (OLD_BLOCKS[which], NEW_BLOCKS[which]()):
        g, e = run(code, cardinfo=card, **extra)
        out.append(e if e is not None else g.get(var, MISSING))
    return out


def same(a, b):
    """Equal values AND the same container / element types (list vs ndarray, int vs np.int32, dtype)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, BaseException):
        return type(a) is type(b) and str(a) == str(b)
    return a == b


def is_exit(x, text=None):
    return isinstance(x, SystemExit) and (text is None or text in str(x.code))


# ---------------------------------------------------------------------------------------------------------------------
# EXCLUDEMONTHS / INCLUDEYEARS spellings
# ---------------------------------------------------------------------------------------------------------------------
def test_wrap_range():
    old, new = parse("excludemonths", {"EXCLUDEMONTHS": "12-3"}, "excludemonths")
    report(old == [] and same(new, [12, 1, 2, 3]),
           "[wrap range] EXCLUDEMONTHS '12-3': old %r (nothing excluded) -> new %r" % (old, new))
    old, new = parse("excludemonths", {"EXCLUDEMONTHS": "9-7"}, "excludemonths")
    report(old == [] and same(new, [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]), "[wrap range] EXCLUDEMONTHS '9-7': old %r -> new %r" % (old, new))
    old, new = parse("excludemonths", {"EXCLUDEMONTHS": "13-2"}, "excludemonths")
    report(old == [] and is_exit(new, "EXCLUDEMONTHS"), "[wrap range] EXCLUDEMONTHS '13-2' (invalid wrap): old [] -> new stops with a message")


def test_single_month():
    old, new = parse("excludemonths", {"EXCLUDEMONTHS": "8"}, "excludemonths")
    report(old == [] and same(new, [np.int32(8)]), "[single month] EXCLUDEMONTHS '8': old %r (ignored) -> new %r" % (old, new))
    old, new = parse("excludemonths", {"EXCLUDEMONTHS": "summer"}, "excludemonths")
    report(old == [] and is_exit(new, "Accepted"), "[single month] EXCLUDEMONTHS 'summer': old [] (ignored) -> new stops with the accepted formats")


def test_string_lists():
    old, new = parse("excludemonths", {"EXCLUDEMONTHS": ["12", "1", "2"]}, "excludemonths")
    old_drops = [m for m in range(1, 13) if m in old]
    new_drops = [m for m in range(1, 13) if m in new]
    report(old_drops == [] and new_drops == [1, 2, 12] and same(new, [np.int32(12), np.int32(1), np.int32(2)]),
           "[string list] EXCLUDEMONTHS ['12','1','2']: old excludes %r -> new excludes %r (np.int32)" % (old_drops, new_drops))
    old, new = parse("includeyears", {"INCLUDEYEARS": ["2002", "2003"]}, "includeyears")
    files = ["AORC.%d0701.preciptemp.nc" % y for y in (2001, 2002, 2003, 2004)]
    with tempfile.TemporaryDirectory() as d:
        for f in files:
            open(os.path.join(d, f), "w").close()
        fo, _ = RDF.createfilelist(os.path.join(d, "AORC.*.preciptemp.nc"), old, [])
        fn, ny = RDF.createfilelist(os.path.join(d, "AORC.*.preciptemp.nc"), new, [])
    report(len(fo) == 0 and [os.path.basename(f) for f in fn] == files[1:3] and ny == 2
           and same(new, [np.int32(2002), np.int32(2003)]),
           "[string list] INCLUDEYEARS ['2002','2003']: old createfilelist -> %d files (crash at flist[0]) -> new %d files, %d years"
           % (len(fo), len(fn), ny))
    old, new = parse("excludemonths", {"EXCLUDEMONTHS": [12, "1"]}, "excludemonths")
    report(same(new, [12, np.int32(1)]), "[string list] mixed list [12,'1'] -> %r" % (new,))


def test_includeyears_all():
    g, _ = run("includeyr='all'\nflag = includeyr.lower == 'all'\n")
    for v in ("all", "ALL", "none", ""):
        old, new = parse("includeyears", {"INCLUDEYEARS": v}, "includeyears")
        report(g["flag"] is False and old is False and new is False,
               "[INCLUDEYEARS all] INCLUDEYEARS %r: old False only via the caught ValueError ('all'.lower == 'all' is %r) -> new False"
               % (v, g["flag"]))
    old, new = parse("includeyears", {"INCLUDEYEARS": "2002to2010"}, "includeyears")
    report(old is False and is_exit(new, "INCLUDEYEARS"),
           "[INCLUDEYEARS all] INCLUDEYEARS '2002to2010': old False (all years, silently) -> new stops with a message")


# ---------------------------------------------------------------------------------------------------------------------
# existing catalog: months, EXCLUDESTORMS, NSTORMS (synthetic storm list; the driver's own filtering code)
# ---------------------------------------------------------------------------------------------------------------------
def synthetic_catalog():
    cat = "Test.nc"
    dates = ["2002%02d15" % m for m in range(1, 13)] + ["2003%02d03" % m for m in (1, 6, 7, 8, 12)]
    n = len(dates)
    stormlist = ["/c/StormCatalog/%s_storm_%d_%s.nc" % (cat, i + 1, d) for i, d in enumerate(dates)]
    cattime = np.array([[np.datetime64(d[:4] + "-" + d[4:6] + "-" + d[6:8] + "T00:00", "m")
                         + np.timedelta64(60 * h, "m") for h in range(3)] for d in dates])
    return dict(catalogname=cat, stormlist=stormlist, catmax=np.arange(1, n + 1, dtype=np.float32) * 10,
                catx=np.arange(n, dtype=np.int32) + 100, caty=np.arange(n, dtype=np.int32) + 200, cattime=cattime,
                n=n)


def aligned(g, base):
    """every array row belongs to the storm file at the same position, and the lengths agree"""
    sl = g["stormlist"]
    if not (len(sl) == len(g["catmax"]) == len(g["catx"]) == len(g["caty"]) == g["cattime"].shape[0]):
        return False
    for k, s in enumerate(sl):
        i = RDF.extract_storm_number(s, base["catalogname"]) - 1
        if not (g["catmax"][k] == base["catmax"][i] and g["catx"][k] == base["catx"][i]
                and g["caty"][k] == base["caty"][i] and np.array_equal(g["cattime"][k], base["cattime"][i])):
            return False
    return True


def run_filter(which, base, excludemonths, includeyears, defaultstorms=False, nstorms=None):
    ns = dict(CreateCatalog=False, includeyears=includeyears, excludemonths=excludemonths, defaultstorms=defaultstorms,
              catalogname=base["catalogname"], stormlist=list(base["stormlist"]), catmax=base["catmax"].copy(),
              catx=base["catx"].copy(), caty=base["caty"].copy(), cattime=base["cattime"].copy(),
              modstormsno=np.arange(base["n"], dtype=np.int32), includestorms=np.ones(base["n"], dtype=bool),
              origstormsno=np.arange(base["n"], dtype=np.int32))
    if nstorms is not None:
        ns["nstorms"] = nstorms
    code = OLD_BLOCKS["filtering"] if which == "old" else NEW_BLOCKS["filtering"]()
    return run(code, **ns)


def test_existing_catalog_months():
    base = synthetic_catalog()
    old_card = {"EXCLUDEMONTHS": [12, 1, 2, 3]}                       # what "12-3" now parses to (wrap range)
    em = FPA.parse_excludemonths(old_card)
    go, eo = run_filter("old", base, em, False, nstorms=base["n"])
    gn, en = run_filter("new", base, em, False, nstorms=base["n"])
    nkeep = sum(int(RDF.extract_date(s, base["catalogname"])[4:6]) not in em for s in base["stormlist"])
    kept = [RDF.extract_date(s, base["catalogname"])[4:6] for s in gn["stormlist"]]
    report(eo is None and len(go["stormlist"]) == nkeep and len(go["catmax"]) == base["n"] and not aligned(go, base),
           "[existing-catalog months] old: EXCLUDEMONTHS [12,1,2,3], no INCLUDEYEARS: stormlist %d storms but catmax %d (misaligned)"
           % (len(go["stormlist"]), len(go["catmax"])))
    report(en is None and aligned(gn, base) and len(gn["stormlist"]) == nkeep
           and not any(int(m) in em for m in kept) and len(gn["modstormsno"]) == len(gn["stormlist"]),
           "[existing-catalog months] new: %d of %d storms kept, catmax/catx/caty/cattime/modstormsno aligned with stormlist, kept months %s"
           % (len(gn["stormlist"]), base["n"], sorted(set(kept))))
    # with INCLUDEYEARS the old code was already right: old == new
    go, _ = run_filter("old", base, em, [2002], nstorms=base["n"])
    gn, _ = run_filter("new", base, em, [2002], nstorms=base["n"])
    report(all(same(go[k], gn[k]) for k in ("stormlist", "catmax", "catx", "caty", "cattime", "modstormsno", "nstorms"))
           and aligned(gn, base), "[existing-catalog months] with INCLUDEYEARS [2002]: new == old (already aligned)")
    # no exclusion at all (production configuration on an existing catalog): new == old
    go, _ = run_filter("old", base, [], False, nstorms=base["n"] - 2)
    gn, _ = run_filter("new", base, [], False, nstorms=base["n"] - 2)
    report(all(same(go[k], gn[k]) for k in ("stormlist", "catmax", "catx", "caty", "cattime", "modstormsno", "nstorms")),
           "[existing-catalog months] EXCLUDEMONTHS none, NSTORMS < catalog: new == old")


def test_nstorms_absent():
    base = synthetic_catalog()
    # old: the existing-catalog branch compared an undefined nstorms
    g, e = run(OLD_BLOCKS["nstorms"], stormlist=base["stormlist"], defaultstorms=True)
    report(isinstance(e, NameError), "[NSTORMS absent] old: NSTORMS absent, existing catalog -> %r" % (e,))
    g, e = run(NEW_BLOCKS["nstorms"](), stormlist=base["stormlist"], defaultstorms=True, stm_num=base["n"])
    report(e is None and g["nstorms"] == base["n"] and g["defaultstorms"] is False,
           "[NSTORMS absent] new: nstorms = %r (the catalog's %d storms), defaultstorms -> False" % (g.get("nstorms"), base["n"]))
    # then the month filter: old NameError on nstorms_default; new == the run with NSTORMS = catalog count
    em = [12, 1, 2, 3]
    go, eo = run_filter("old", base, em, False, defaultstorms=True, nstorms=base["n"])
    gn, en = run_filter("new", base, em, False, defaultstorms=g["defaultstorms"], nstorms=g["nstorms"])
    gx, ex = run_filter("new", base, em, False, defaultstorms=False, nstorms=base["n"])
    report(isinstance(eo, NameError) and en is None and ex is None and aligned(gn, base)
           and all(same(gn[k], gx[k]) for k in ("stormlist", "catmax", "catx", "caty", "cattime", "modstormsno", "nstorms")),
           "[NSTORMS absent] old %r in the month filter -> new identical to NSTORMS = %d" % (eo, base["n"]))
    # NSTORMS given: unchanged (same object)
    v = np.int64(400)
    r = FPA.default_nstorms(v, False, 10)
    report(r[0] is v and r[1] is False, "[NSTORMS absent] NSTORMS given -> unchanged")


def test_excludestorms_string():
    base = synthetic_catalog()
    for card, want in (({"EXCLUDESTORMS": "5"}, [5]), ({"EXCLUDESTORMS": "3,5"}, [3, 5]), ({"EXCLUDESTORMS": "none"}, [])):
        res = []
        for code in (OLD_BLOCKS["excludestorms"], NEW_BLOCKS["excludestorms"]()):
            g, e = run(code, cardinfo=card, CreateCatalog=False, catalogname=base["catalogname"],
                       stormlist=list(base["stormlist"]), catmax=base["catmax"].copy(), catx=base["catx"].copy(),
                       caty=base["caty"].copy(), cattime=base["cattime"].copy())
            res.append((g, e))
        (go, eo), (gn, en) = res
        dropped_new = sorted(set(range(1, base["n"] + 1)) - set(RDF.extract_storm_number(s, base["catalogname"]) for s in gn["stormlist"]))
        dropped_old = sorted(set(range(1, base["n"] + 1)) - set(RDF.extract_storm_number(s, base["catalogname"]) for s in go["stormlist"]))
        okalign = len(gn["stormlist"]) == len(gn["catmax"]) == len(gn["modstormsno"])
        if card["EXCLUDESTORMS"] == "5":
            report(eo is None and en is None and dropped_old == [] and dropped_new == want and okalign,
                   "[EXCLUDESTORMS string] EXCLUDESTORMS '5': old drops %r (ignored) -> new drops %r" % (dropped_old, dropped_new))
        else:
            report(eo is None and en is None and dropped_old == dropped_new == want and same(go["exclude"], gn["exclude"]),
                   "[EXCLUDESTORMS string] EXCLUDESTORMS %r: new == old (drops %r)" % (card["EXCLUDESTORMS"], dropped_new))


# ---------------------------------------------------------------------------------------------------------------------
# CALCTYPE, UNCERTAINTY, required keys, RETURNLEVELS
# ---------------------------------------------------------------------------------------------------------------------
def test_calctype():
    old, new = parse("calctype", {"CALCTYPE": "foo"}, "calctype")
    report(old is MISSING and is_exit(new, "'pds'"),
           "[CALCTYPE] CALCTYPE 'foo': old calctype undefined (NameError later) -> new stops listing the valid values")
    for card in ({}, {"CALCTYPE": None}, {"CALCTYPE": "ams"}, {"CALCTYPE": "PDS"}, {"CALCTYPE": "annmax"},
                 {"CALCTYPE": "partialduration"}):
        old, new = parse("calctype", card, "calctype")
        report(same(old, new), "[CALCTYPE] CALCTYPE %r: new %r == old" % (card.get("CALCTYPE", "(missing)"), new))
    # the hook runs before the storm catalog is built
    hook = next(k for k, s in enumerate(DRIVER) if "calctype=FPA.parse_calctype(cardinfo)" in s)
    cat = next(k for k, s in enumerate(DRIVER) if 'print("creating a new storm catalog...")' in s)
    report(hook < cat, "[CALCTYPE] CALCTYPE parsed at driver line %d, before the catalog block (line %d)" % (hook + 1, cat + 1))


def test_numeric_uncertainty():
    for spread in (90, "90"):
        old_t, new_t = parse("uncertainty", {"UNCERTAINTY": spread}, "spreadtype")
        old_q, new_q = parse("uncertainty", {"UNCERTAINTY": spread}, "quantilecalc")
        if spread == 90:
            report(old_t == "ensemble" and old_q is MISSING and new_t == "quantile" and same(new_q, 90),
                   "[numeric UNCERTAINTY] UNCERTAINTY 90 (JSON): old 'ensemble' (silently) -> new ('quantile', 90)")
        else:
            report(old_t == "quantile" and same(old_q, 90) and new_t == "quantile" and same(new_q, 90),
                   "[numeric UNCERTAINTY] UNCERTAINTY '90': parse unchanged ('quantile', 90)")
    # the write: old NameError (spreadmean), new 5th / mean / 95th percentiles
    rng = np.random.default_rng(1)
    sortrain = np.sort(rng.gamma(2., 20., (50, 100)).astype(np.float32), axis=0)
    exceedp = np.linspace(1, 1. / 50, 50)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "x_FreqAnalysis.csv")
        g, e = run(OLD_BLOCKS["write"], sortrain=sortrain, quantilecalc=90, spreadtype="quantile", exceedp=exceedp,
                   returnperiod=1 / exceedp)
        report(isinstance(e, NameError) and "spreadmean" in str(e), "[numeric UNCERTAINTY] old write_frequency_outputs, quantile: %r" % (e,))
        with redirect_stdout(io.StringIO()):
            RDF.write_frequency_outputs(sortrain, exceedp, 1 / exceedp, f, os.path.join(d, "x.png"), "quantile", 90, 100)
        out = np.loadtxt(f, delimiter=",", skiprows=1)
        want = np.column_stack((exceedp, 1 / exceedp, np.percentile(sortrain, 5., axis=1), np.nanmean(sortrain, 1),
                                np.percentile(sortrain, 95., axis=1)))
        hdr = open(f).readline().strip()
        report(np.allclose(out, np.round(want, 2), atol=0.0051) and os.path.exists(os.path.join(d, "x.png"))
               and hdr == "prob.exceed,returnperiod,minrain,meanrain,maxrain",
               "[numeric UNCERTAINTY] new: UNCERTAINTY 90 writes the 5th percentile, mean and 95th percentile of the realizations (+ PNG)")
    for v in (150, "150", -1):
        _, new = parse("uncertainty", {"UNCERTAINTY": v}, "spreadtype")
        report(is_exit(new, "invalid quantile range"), "[numeric UNCERTAINTY] UNCERTAINTY %r: stops ('invalid quantile range')" % (v,))
    _, new = parse("uncertainty", {"UNCERTAINTY": 90.5}, "spreadtype")
    report(is_exit(new, "integer"), "[numeric UNCERTAINTY] UNCERTAINTY 90.5: stops (integer 0-100)")


def test_required_keys():
    prod = production_config()
    g, e = run("try:\n    wd=cardinfo['MAINPATH']\nexcept ImportError:\n    sys.exit('x')\n", cardinfo={})
    report(isinstance(e, KeyError), "[required keys] old: missing MAINPATH -> %r (traceback)" % (e,))
    for drop in ("MAINPATH", "SCENARIONAME", "CATALOGNAME", "CREATECATALOG", "DURATION", "VARIABLES", "POINTAREA",
                 "RAINPATH", "WATERSHEDSHP"):
        c = dict(prod)
        del c[drop]
        try:
            FPA.check_required(c)
            r = None
        except SystemExit as ex:
            r = ex
        report(is_exit(r, drop + ":"), "[required keys] new: missing %s -> stops naming it" % drop)
    c = dict(prod, DOMAINTYPE="rectangular")
    try:
        FPA.check_required(c)
        r = None
    except SystemExit as ex:
        r = ex
    report(is_exit(r, "AREA_EXTENT:"), "[required keys] new: DOMAINTYPE rectangular + CREATECATALOG true without AREA_EXTENT -> stops")
    c = dict(prod, CREATECATALOG="false")
    del c["RAINPATH"]
    FPA.check_required(c)
    report(True, "[required keys] new: RAINPATH not required with CREATECATALOG false")
    c = {k: v for k, v in prod.items() if k not in ("MAINPATH", "DURATION")}
    try:
        FPA.check_required(c)
        r = None
    except SystemExit as ex:
        r = ex
    report(is_exit(r, "MAINPATH:") and "DURATION:" in str(r.code), "[required keys] new: every missing key is named in one message")


def test_returnlevels_string():
    ny = 1000
    old, new = parse("returnlevels", {"RETURNLEVELS": "2,5,10"}, "speclevels", nsimulations=ny)
    ref_old, _ = parse("returnlevels", {"RETURNLEVELS": [2, 5, 10]}, "speclevels", nsimulations=ny)
    al_old, al_new = parse("returnlevels", {"RETURNLEVELS": "2,5,10"}, "alllevels", nsimulations=ny)
    report(is_exit(old, "isn't recognized") and same(new, ref_old) and al_new is False,
           "[RETURNLEVELS string] RETURNLEVELS '2,5,10': old stops -> new %r %s == JSON [2,5,10]" % (new, getattr(new, "dtype", "")))
    old, new = parse("returnlevels", {"RETURNLEVELS": "2, 5, 166.67, 5000"}, "speclevels", nsimulations=ny)
    ref_old, _ = parse("returnlevels", {"RETURNLEVELS": [2, 5, 166.67, 5000]}, "speclevels", nsimulations=ny)
    report(same(new, ref_old), "[RETURNLEVELS string] RETURNLEVELS '2, 5, 166.67, 5000' == JSON list (filtered to <= NYEARS): %r" % (new,))
    old, new = parse("returnlevels", {"RETURNLEVELS": "2,five"}, "speclevels", nsimulations=ny)
    report(is_exit(old) and is_exit(new, "isn't recognized"), "[RETURNLEVELS string] RETURNLEVELS '2,five': still stops with the old message")


# ---------------------------------------------------------------------------------------------------------------------
# identical objects for the configurations that already worked
# ---------------------------------------------------------------------------------------------------------------------
def production_config():
    """the parsing-relevant keys of the production / validation-matrix JSONs (jsons/ref/*.json)"""
    return {"MAINPATH": "/x", "SCENARIONAME": "Basin", "RAINPATH": "/d/AORC.*.preciptemp.nc", "CATALOGNAME": "Basin.nc",
            "CREATECATALOG": "true", "DURATION": 72, "DURATIONCORRECTION": "false", "NSTORMS": 400, "NYEARS": 1000,
            "CALCTYPE": "PDS", "NREALIZATIONS": 100, "UNCERTAINTY": "all_realizations", "TIMESEPARATION": 24,
            "DOMAINTYPE": "irregular", "DOMAINSHP": "/d/domain.shp", "DIAGNOSTICPLOTS": "only-statistics",
            "FREQANALYSIS": "true", "EXCLUDESTORMS": "none", "EXCLUDEMONTHS": "none", "INCLUDEYEARS": "2002-2023",
            "TRANSPOSITION": "uniform", "RESAMPLING": "poisson",
            "RETURNLEVELS": [2, 5, 10, 25, 50, 100, 200, 250, 500, 1000], "RETURNTHRESHOLD": 2,
            "POINTAREA": "watershed", "WATERSHEDSHP": "/d/basin.shp",
            "VARIABLES": {"precip": "RAINRATE", "latname": "latitude", "longname": "longitude"},
            "MULTIPLEDURATIONS": [1, 3, 6, 12, 24, 48, 72], "MULTIDURATIONENGINE": "table", "CATALOGSTORAGE": "light"}


def talequah_config():
    return {"EXCLUDEMONTHS": [1, 2, 3, 4, 5, 6, 10, 11, 12], "INCLUDEYEARS": "2002", "UNCERTAINTY": "false",
            "CALCTYPE": "pds", "EXCLUDESTORMS": "none", "NYEARS": 500,
            "RETURNLEVELS": [2, 5, 10, 25, 50, 100, 166.67, 500]}


PARSED = [("excludemonths", "excludemonths"), ("includeyears", "includeyears"), ("calctype", "calctype"),
          ("uncertainty", "spreadtype"), ("uncertainty", "quantilecalc"), ("returnlevels", "alllevels"),
          ("returnlevels", "speclevels")]


def compare_config(card, label):
    bad = []
    for which, var in PARSED:
        old, new = parse(which, card, var, nsimulations=card.get("NYEARS", 100))
        if var == "quantilecalc":                            # read through globals().get('quantilecalc') only
            old, new = [None if x is MISSING else x for x in (old, new)]
        if not same(old, new):
            bad.append("%s: old %r new %r" % (var, old, new))
    report(not bad, "%s parses to identical objects (types included)%s" % (label, ": " + "; ".join(bad) if bad else ""))


def test_identical_configs():
    prod = production_config()
    compare_config(prod, "production configuration")
    em, iy = FPA.parse_excludemonths(prod), FPA.parse_includeyears(prod)
    report(em == [] and type(iy) is list and iy == list(range(2002, 2024)) and all(type(y) is int for y in iy),
           "production: EXCLUDEMONTHS [] and INCLUDEYEARS = Python ints 2002..2023 (as before)")
    try:
        FPA.check_required(prod)
        ok = True
    except SystemExit:
        ok = False
    report(ok, "production configuration passes the required-key check")
    compare_config(talequah_config(), "Talequah example")
    root = Path(os.environ.get("FAST_TEST_ROOT", str(Path.home() / "Databases" / "RainyDay_fast_tests"))) / "jsons" / "ref"
    for f in sorted(root.glob("*.json")) if root.is_dir() else []:     # optional: the matrix JSONs, when present
        card = json.load(open(f))
        compare_config(card, "matrix JSON " + f.name)
        try:
            FPA.check_required(card)
            ok = True
        except SystemExit:
            ok = False
        report(ok, "matrix JSON %s passes the required-key check" % f.name)
    # a battery of inputs the old code read correctly (or ignored in a way that is unchanged)
    battery = {
        "EXCLUDEMONTHS": [MISSING, "none", "None", "", "1,2,3", "1, 2, 12", "6-8", "1-12", "3-3", [1, 2, 3], [],
                          [12, 1], 5, None, {"a": 1}],
        "INCLUDEYEARS": [MISSING, "2002-2023", "2002,2005", "2002, 2005", "2002", " 2002", [2002, 2003], [], "all",
                         "ALL", "none", "", 2002, None],
        "CALCTYPE": [MISSING, "ams", "AMS", "annmax", "pds", "PDS", "partialduration", None],
        "UNCERTAINTY": [MISSING, "ensemble", "Ensemble", "all_realizations", "ALL_REALIZATIONS", "false", "true",
                        "90", "0", "100", None, True, [90]],
        "RETURNLEVELS": [MISSING, "all", "ALL", [2, 5, 10], [2, 5, 166.67, 500], [2., 5.], [0.5, 2, 2000], []],
    }
    bad = []
    for key, values in battery.items():
        for v in values:
            card = {} if v is MISSING else {key: v}
            for which, var in PARSED:
                if which != {"EXCLUDEMONTHS": "excludemonths", "INCLUDEYEARS": "includeyears", "CALCTYPE": "calctype",
                             "UNCERTAINTY": "uncertainty", "RETURNLEVELS": "returnlevels"}[key]:
                    continue
                old, new = parse(which, card, var, nsimulations=1000)
                if var == "quantilecalc":                    # read through globals().get('quantilecalc') only
                    old, new = [None if x is MISSING else x for x in (old, new)]
                if not same(old, new):
                    bad.append("%s=%r %s: old %r new %r" % (key, v, var, old, new))
    report(not bad, "battery of %d previously valid inputs: identical objects%s"
           % (sum(len(v) for v in battery.values()), (": " + "; ".join(bad)) if bad else ""))
    # EXCLUDESTORMS battery (value-level parser)
    bad = []
    for v in ("none", "None", "", "1,2", "1-20", [3, 17], [], None, 5):
        g, e = run(OLD_BLOCKS["excludestorms_parse"], cardinfo={"EXCLUDESTORMS": v})
        new = FPA.parse_excludestorms(v)
        if e is not None or not same(g["exclude"], new):
            bad.append("%r: old %r new %r" % (v, g.get("exclude"), new))
    report(not bad, "EXCLUDESTORMS battery: identical objects%s" % ((": " + "; ".join(bad)) if bad else ""))


def main():
    test_wrap_range()
    test_single_month()
    test_string_lists()
    test_includeyears_all()
    test_existing_catalog_months()
    test_excludestorms_string()
    test_calctype()
    test_numeric_uncertainty()
    test_required_keys()
    test_nstorms_absent()
    test_returnlevels_string()
    test_identical_configs()
    print()
    print("ALL PASS (%d checks)" % len(RES) if RES and all(RES) else "SOME TESTS FAILED (%d of %d)" % (RES.count(False), len(RES)))
    sys.exit(0 if RES and all(RES) else 1)


if __name__ == "__main__":
    main()
