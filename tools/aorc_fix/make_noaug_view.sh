#!/bin/bash
# GP 2026-10-21 (arguments added 2026-11-04): the corrected AORC view without August 2017 (runs without Hurricane
# Harvey). Every file of the corrected view except AORC.201708* gets a link to its final target (patched copy or
# original), so nothing is copied and the corrected view itself is not changed.
#   bash make_noaug_view.sh <corrected_view_dir> <new_no_aug2017_dir>
set -euo pipefail
SRC="${1:?usage: make_noaug_view.sh <corrected_view_dir> <new_no_aug2017_dir>}"
DST="${2:?usage: make_noaug_view.sh <corrected_view_dir> <new_no_aug2017_dir>}"

realpath_of() {   # readlink -f exists on Linux and macOS >= 12.3; python3 otherwise
    readlink -f "$1" 2>/dev/null || python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

mkdir -p "$DST"
n=0
for f in "$SRC"/AORC.*.preciptemp.nc; do
    b=$(basename "$f")
    case "$b" in AORC.201708*) continue;; esac
    [ -e "$DST/$b" ] || { ln -s "$(realpath_of "$f")" "$DST/$b"; n=$((n+1)); }
done
nall=$(ls "$DST" | grep -c '^AORC\..*\.preciptemp\.nc$' || true)
naug=$(ls "$DST" | grep -c '^AORC\.201708' || true)
echo "$DST: $nall files ($n new links), $naug files of August 2017 (must be 0)"
