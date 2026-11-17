"""
Build one folder that RainyDay can read as a single AORC record (a "view"): for every day of the period, a symbolic
link to the file of that day in the first source folder that has it. Nothing is copied and the source folders are
never modified. Instructions: AORC_DOWNLOAD_INSTRUCTIONS.md in this folder.

    python make_view.py --start 20020101 --end 20251231 --dst NEW_DIR --src DIR_A --src DIR_B [...]

Example: --src data_aorcfix2 (corrected 2002-2023, see tools/aorc_fix) --src data_2024_2025 (built from NOAA AORC v1.1).
Links are absolute paths to the final file (links in a source folder are followed), so do not move or rename the
source folders afterwards. The script stops without writing anything if a day is missing from every source folder, or
if NEW_DIR already holds files. Only the Python standard library is needed.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Folder of links to RainyDay-format AORC daily files from several folders")
    ap.add_argument("--start", required=True, help="first day, YYYYMMDD")
    ap.add_argument("--end", required=True, help="last day, YYYYMMDD")
    ap.add_argument("--dst", required=True, help="the new folder (must not exist, or be empty)")
    ap.add_argument("--src", required=True, action="append", help="source folder; repeat, first match wins")
    a = ap.parse_args()
    d0, d1 = datetime.strptime(a.start, "%Y%m%d"), datetime.strptime(a.end, "%Y%m%d")
    srcs = [Path(s).resolve() for s in a.src]
    for s in srcs:
        if not s.is_dir():
            sys.exit(f"Not a folder: {s}")
    dst = Path(a.dst).resolve()
    if dst in srcs:
        sys.exit("The new folder must not be one of the source folders.")
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"{dst} already holds files. Choose a new folder.")
    links, missing = [], []
    count = {s: 0 for s in srcs}
    for i in range((d1 - d0).days + 1):
        name = f"AORC.{d0 + timedelta(days=i):%Y%m%d}.preciptemp.nc"
        for s in srcs:
            if (s / name).exists():                          # False for a broken link
                links.append((name, os.path.realpath(s / name)))
                count[s] += 1
                break
        else:
            missing.append(name)
    if missing:
        sys.exit(f"{len(missing)} days are in no source folder, nothing written. First: {missing[0]}, "
                 f"last: {missing[-1]}")
    dst.mkdir(parents=True, exist_ok=True)
    for name, target in links:
        os.symlink(target, dst / name)
    print(f"{dst}: {len(links)} links, {a.start} to {a.end}, no day missing")
    for s in srcs:
        print(f"  {count[s]:6d} from {s}")


if __name__ == "__main__":
    main()
