"""
Download the ready-made RainyDay-format AORC daily files (AORC.YYYYMMDD.preciptemp.nc, CONUS, hourly rain rate and
temperature, about 19 MB per day) from the public bucket of the Hydroclimate Extremes Group, University of
Wisconsin-Madison:

    https://web.s3.wisc.edu/aorc/<year>/AORC.YYYYMMDD.preciptemp.nc        (1979-02-01 to 2023-12-31)

    python download_aorc_wisc.py --start 20020101 --end 20231231 --dst DIR [--workers 4]

Instructions: AORC_DOWNLOAD_INSTRUCTIONS.md in this folder. Only the Python standard library is needed.
A file is written to <name>.part and renamed when its size equals the size announced by the server, so an existing
file is complete; existing files are skipped (resumable: rerun the same command). Days that the bucket does not have
(HTTP 404) are listed at the end: build them with make_aorc_from_v11.py.
"""
import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

BASE = "https://web.s3.wisc.edu/aorc"
AGENT = {"User-Agent": "RainyDay_fast-aorc-download"}


def fetch(day, dst):
    """Returns (day, status), status in 'exists', 'ok', 'missing' or 'error: ...'."""
    name = f"AORC.{day:%Y%m%d}.preciptemp.nc"
    out = dst / name
    if out.exists():
        return day, "exists"
    part = dst / (name + ".part")
    url = f"{BASE}/{day:%Y}/{name}"
    err = ""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120) as r:
                size = int(r.headers.get("Content-Length", -1))
                with open(part, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            got = part.stat().st_size
            if size >= 0 and got != size:
                err = f"{got} of {size} bytes"
            else:
                part.replace(out)
                return day, "ok"
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return day, "missing"
            err = f"HTTP {e.code}"
        except Exception as e:                               # noqa: network errors are retried
            err = f"{type(e).__name__}: {e}"
        time.sleep(10 * (attempt + 1))
    if part.exists():
        part.unlink()
    return day, f"error: {err}"


def main():
    ap = argparse.ArgumentParser(description="Download RainyDay-format AORC daily files from web.s3.wisc.edu/aorc")
    ap.add_argument("--start", required=True, help="first day, YYYYMMDD")
    ap.add_argument("--end", required=True, help="last day, YYYYMMDD")
    ap.add_argument("--dst", required=True, help="folder for the files (created if needed)")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    ap.add_argument("--min-free-gb", type=float, default=5., help="stop if less free space than this")
    a = ap.parse_args()
    d0, d1 = datetime.strptime(a.start, "%Y%m%d"), datetime.strptime(a.end, "%Y%m%d")
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dst).free / 1e9
    need = 0.02 * sum(1 for d in days if not (dst / f"AORC.{d:%Y%m%d}.preciptemp.nc").exists())
    print(f"{len(days)} days -> {dst}; about {need:.1f} GB to download, {free:.0f} GB free", flush=True)
    if free < need + a.min_free_gb:
        sys.exit(f"Not enough free space on {dst}")
    count = {"exists": 0, "ok": 0}
    missing, errors = [], []
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        for i, (day, status) in enumerate(pool.map(lambda d: fetch(d, dst), days), 1):
            if status == "missing":
                missing.append(day)
            elif status.startswith("error"):
                errors.append((day, status))
            else:
                count[status] += 1
            if i % 100 == 0 or i == len(days):
                print(f"  {i}/{len(days)} days ({time.time() - t0:.0f} s)", flush=True)
    print(f"downloaded {count['ok']}, already there {count['exists']}, not in the bucket {len(missing)}, "
          f"failed {len(errors)}")
    if missing:
        print(f"not in the bucket: {missing[0]:%Y%m%d} .. {missing[-1]:%Y%m%d} ({len(missing)} days). "
              "Build them with make_aorc_from_v11.py.")
    for day, status in errors:
        print(f"  {day:%Y%m%d} {status}")
    if errors:
        sys.exit("Some downloads failed: rerun the same command.")


if __name__ == "__main__":
    main()
