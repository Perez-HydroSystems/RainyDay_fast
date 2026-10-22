#!/bin/bash
# GP 2026-10-21: regression test for MULTIPLEDURATIONS on the Talequah example (small: 2002 Jul-Sep, 10 storms).
#
# Usage: test_multidurations.sh [conda_env|python_path] [aorc_dir]
#   conda_env  : env name under /opt/anaconda3/envs (default RainyDay_Env) or the full path of a python executable
#   aorc_dir   : folder with AORC.YYYYMMDD.preciptemp.nc (default: the RAINPATH folder of Examples/Talequah/Talequah.json)
#   (GP 2026-10-21: portable — the Talequah shapefiles are taken from this repo, so it also runs on Linux)
#
# 1. MULTIPLEDURATIONS [1,3,6,24,72], CREATECATALOG true, engine "exact"   -> builds the catalog
# 2. the same with CATALOGPATH -> that catalog, engine "fast"
# 3. separate runs (as before MULTIPLEDURATIONS existed): 1 h, 6 h (CATALOGPATH, DURATIONCORRECTION true) and 72 h
# 4. tools/compare_multidurations.py: raw transposed depths of every duration bitwise identical (exact and fast)
# Output in tools/logs/multidur_<timestamp>/ (Talequah data from Examples/Talequah/Talequah.json).
set -e
ENV="${1:-RainyDay_Env}"
AORC="${2:-}"
TOOLSDIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$TOOLSDIR")"
if [ -x "$ENV" ]; then PY="$ENV"; else PY="/opt/anaconda3/envs/$ENV/bin/python"; fi
OUT="$TOOLSDIR/logs/multidur_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
export RAINYDAY_CATALOG_ENGINE="${RAINYDAY_CATALOG_ENGINE:-batchfft}"

"$PY" - "$REPO/Examples/Talequah/Talequah.json" "$OUT" "$AORC" <<'PYEOF'
import json, os, sys
src, out, aorc = sys.argv[1:4]
b = json.load(open(src))
shp = os.path.join(os.path.dirname(src), "Shapefiles")
b.update(DOMAINSHP=os.path.join(shp, "Talequah_transposition_domain.shp"), WATERSHEDSHP=os.path.join(shp, "Talequah_basin.shp"))
if aorc:
    b["RAINPATH"] = os.path.join(aorc, "AORC.*.preciptemp.nc")
b.update(FREQANALYSIS="true", DIAGNOSTICPLOTS="false", NYEARS=100, NREALIZATIONS=5, UNCERTAINTY="all_realizations",
         RETURNLEVELS=[2, 5, 10, 25, 50, 100], SCENARIOS="false", DURATIONCORRECTION="false", DURATION=72)
md = [1, 3, 6, 24, 72]
cat = os.path.join(out, "exact", "Talequah_72", "StormCatalog", b["CATALOGNAME"])
runs = {"exact": dict(b, MAINPATH=os.path.join(out, "exact"), SCENARIONAME="Talequah", CREATECATALOG="true",
                      MULTIPLEDURATIONS=md, MULTIDURATIONENGINE="exact"),
        "fast": dict(b, MAINPATH=os.path.join(out, "fast"), SCENARIONAME="Talequah", CREATECATALOG="false",
                     CATALOGPATH=cat, MULTIPLEDURATIONS=md, MULTIDURATIONENGINE="fast")}
for d in (1, 6, 72):
    runs[f"sep_{d}h"] = dict(b, MAINPATH=os.path.join(out, f"sep_{d}h"), SCENARIONAME=f"Talequah_{d}h", CREATECATALOG="false",
                             CATALOGPATH=cat, DURATION=d, DURATIONCORRECTION="true" if d < 72 else "false")
for k, j in runs.items():
    os.makedirs(j["MAINPATH"], exist_ok=True)
    os.makedirs(os.path.join(out, "dump_" + k), exist_ok=True)
    json.dump(j, open(os.path.join(out, k + ".json"), "w"), indent=2)
    if k.startswith("sep_"):
        json.dump(j, open(os.path.join(out, "dump_" + k, j["SCENARIONAME"] + ".json"), "w"), indent=2)
PYEOF

run() { RAINYDAY_DUMP_WHICHRAIN="$OUT/dump_$1" "$PY" "$REPO/Source/RainyDay_Py3.py" "$OUT/$1.json" > "$OUT/$1.log" 2>&1 \
        || { echo "run $1 FAILED (see $OUT/$1.log)"; exit 1; }; echo "run $1 ok"; }
run exact                                       # builds the catalog used by the others
run fast & run sep_1h & run sep_6h & run sep_72h & wait
echo "--- exact engine vs separate runs ---"
"$PY" "$TOOLSDIR/compare_multidurations.py" "$OUT/dump_exact" Talequah "$OUT/dump_sep_1h" "$OUT/dump_sep_6h" "$OUT/dump_sep_72h"
echo "--- fast engine vs separate runs ---"
"$PY" "$TOOLSDIR/compare_multidurations.py" "$OUT/dump_fast" Talequah "$OUT/dump_sep_1h" "$OUT/dump_sep_6h" "$OUT/dump_sep_72h"
