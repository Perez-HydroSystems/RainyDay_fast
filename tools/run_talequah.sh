#!/bin/bash
# Run the Talequah reference example and capture the log + catalog timer.
#
# Usage: run_talequah.sh [conda_env] [engine] [logname]
#   conda_env : conda environment name (default: RainyDay_Env)
#   engine    : catalog engine, exported as RAINYDAY_CATALOG_ENGINE
#               (fft_perstep | batchfft | batchfft_mp | batchtf).
#               Empty = whatever default is set in RainyDay_Py3.py
#   logname   : optional explicit log file path

set -e
ENV="${1:-RainyDay_Env}"
ENGINE="${2:-}"
TOOLSDIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$TOOLSDIR")"
mkdir -p "$TOOLSDIR/logs"
LOG="${3:-$TOOLSDIR/logs/run_$(date +%Y%m%d_%H%M%S)_${ENV}${ENGINE:+_$ENGINE}.log}"
PY="/opt/anaconda3/envs/$ENV/bin/python"

if [ ! -x "$PY" ]; then
    echo "ERROR: python not found at $PY" >&2
    exit 1
fi
if [ -n "$ENGINE" ]; then
    export RAINYDAY_CATALOG_ENGINE="$ENGINE"
fi

echo "env=$ENV engine=${ENGINE:-default}"
echo "log=$LOG"
SECONDS=0
"$PY" "$REPO/Source/RainyDay_Py3.py" "$REPO/Examples/Talequah/Talequah.json" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
echo "total wall time: ${SECONDS}s (exit=$STATUS)" | tee -a "$LOG"
echo "--- summary ---"
grep -i "catalog timer" "$LOG" || echo "WARNING: no 'catalog timer' line found"
exit $STATUS
