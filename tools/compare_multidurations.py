"""
GP 2026-10-21: verify MULTIPLEDURATIONS against separate single-duration runs.

    python tools/compare_multidurations.py <multi_dump_dir> <multi_base_name> <ref_dump_dir> [<ref_dump_dir> ...]

<multi_dump_dir>  folder written by a MULTIPLEDURATIONS run with RAINYDAY_DUMP_WHICHRAIN=<folder>:
                  <base>_<d>_whichrain.npy for every duration d
<multi_base_name> the run's SCENARIONAME (the base, without _<d>)
<ref_dump_dir>    folders written by separate runs (one duration each) with RAINYDAY_DUMP_WHICHRAIN; their single
                  <scenario>_whichrain.npy is matched to a duration through the run's JSON copy <scenario>.json in the
                  same folder (written by tools/test_multidurations.sh) or through a trailing _<d>h / _<d> in the name.
Checks, per duration: raw transposed depths (whichrain, before ranking) bitwise identical. Exit 0 = ALL PASS.
"""
import glob
import json
import os
import re
import sys

import numpy as np


def ref_duration(path):
    base = os.path.basename(path)[:-len("_whichrain.npy")]
    js = os.path.join(os.path.dirname(path), base + ".json")
    if os.path.exists(js):
        return float(json.load(open(js))["DURATION"])
    m = re.search(r"_(\d+(?:\.\d+)?)h?$", base)
    return float(m.group(1)) if m else None


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    mdir, mbase, refs = sys.argv[1], sys.argv[2], sys.argv[3:]
    ok_all, n = True, 0
    for rdir in refs:
        for f in sorted(glob.glob(os.path.join(rdir, "*_whichrain.npy"))):
            d = ref_duration(f)
            tag = str(int(d)) if d is not None and float(d).is_integer() else ('%g' % d)
            mf = os.path.join(mdir, f"{mbase}_{tag}_whichrain.npy")
            if not os.path.exists(mf):
                print(f"FAIL  {tag:>4} h: no multi-duration dump {mf}"); ok_all = False; continue
            a, b = np.load(f), np.load(mf)
            same = a.shape == b.shape and np.array_equal(a, b)
            diff = float(np.max(np.abs(a.astype(float) - b))) if a.shape == b.shape else float("nan")
            print(f"{'PASS' if same else 'FAIL'}  {tag:>4} h: separate {os.path.basename(f)} vs multi, shape {a.shape}, "
                  f"max |diff| {diff:.3g}")
            ok_all &= same; n += 1
    print(("ALL PASS" if ok_all and n else "FAILED") + f" ({n} durations compared)")
    sys.exit(0 if ok_all and n else 1)


if __name__ == "__main__":
    main()
