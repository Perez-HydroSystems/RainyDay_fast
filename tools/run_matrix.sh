#!/bin/bash
# GP 2026-10-22: run the validation matrix JSONs of one variant in parallel (one job per shape), with whichrain dumps,
# wall time and peak memory.
#   bash tools/run_matrix.sh <repo_dir> <variant> [name ...]
#   <repo_dir>  RainyDay repo whose Source/RainyDay_Py3.py runs (RainyDay_mod for references, RainyDay_fast otherwise)
#   <variant>   JSON folder $FAST_TEST_ROOT/jsons/<variant>/ (tools/make_test_jsons.py)
# env: PY (default RainyDay_Env python), FAST_TEST_ROOT, any RAINYDAY_* variable (passed through)
# Out: dumps/<variant>/<name>/*_whichrain.npy, logs/<variant>/<name>.log, logs/<variant>/summary.tsv
REPO="$1"; VAR="$2"; shift 2
ROOT="${FAST_TEST_ROOT:-/Users/gjperezm/Databases/RainyDay_fast_tests}"
PY="${PY:-/opt/anaconda3/envs/RainyDay_Env/bin/python}"
NAMES="$*"; [ -n "$NAMES" ] || NAMES=$(cd "$ROOT/jsons/$VAR" && ls *.json | sed 's/\.json$//')
mkdir -p "$ROOT/logs/$VAR"
if [ "$(uname)" = "Darwin" ]; then TIMEFLAG=-l; else TIMEFLAG=-v; fi
for N in $NAMES; do
    (
        t0=$(date +%s)
        RAINYDAY_CATALOG_ENGINE="${RAINYDAY_CATALOG_ENGINE:-batchfft}" RAINYDAY_BLOCKFILES="${RAINYDAY_BLOCKFILES:-16}" \
        RAINYDAY_DUMP_WHICHRAIN="$ROOT/dumps/$VAR/$N" \
            /usr/bin/time $TIMEFLAG "$PY" "$REPO/Source/RainyDay_Py3.py" "$ROOT/jsons/$VAR/$N.json" \
            > "$ROOT/logs/$VAR/$N.log" 2>&1
        rc=$?
        rss=$(grep -E "maximum resident set size|Maximum resident set size" "$ROOT/logs/$VAR/$N.log" | grep -oE "[0-9]+" | head -1)
        [ "$(uname)" = "Darwin" ] && rss=$((rss / 1048576)) || rss=$((rss / 1024))
        printf "%s\t%s\t%s\t%s\t%s\n" "$N" "$rc" "$(( $(date +%s) - t0 ))" "$rss" "$(git -C "$REPO" rev-parse --short HEAD)" \
            >> "$ROOT/logs/$VAR/summary.tsv"
        echo "$(date '+%T') $VAR $N rc=$rc $(( $(date +%s) - t0 ))s peak ${rss} MB"
    ) &
done
wait
echo "summary (name, rc, seconds, peak MB, commit): $ROOT/logs/$VAR/summary.tsv"
