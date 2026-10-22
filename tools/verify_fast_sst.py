"""
GP 2026-10-22: offline brute-force check of MULTIDURATIONENGINE "table" (Source/RainyDay_utilities_Py3/fast_sst.py)
on real storm-catalog storms.

    python tools/verify_fast_sst.py <StormCatalog dir> [--storms N] [--anchors N] [--durs 1 3 6 12 24 48] [--seed S]
                                    [--all-anchors] [--asm] [--synthetic N]

For N storms (random, default_rng) and N valid transposition anchors per storm (whole bounding box inside the domain,
the rule of the driver), all sub-durations:
  1. window stack (shared nancumsum) == rolling_sum per duration, bitwise
  2. table_sst == SSTalt_multidur (the "fast" engine, itself verified == SSTalt object mode), bitwise
  3. parent: parent_sst (compiled, one window) == SSTalt(passrain, durcheck=False) in object mode, bitwise
  4. FFT basin sums: max |A_fft - A_direct| over the sampled anchors and windows, relative to eta (must be << 1)
  5. candidate-count histogram
--asm: numba_multimask_calc is a real call (not inlined) in SSTalt_multidur and table_kernel, and its machine code is
the same in both modules and in the object-mode (int64) specialization.
Exit 0 = ALL PASS.
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Source"))
import RainyDay_utilities_Py3.RainyDay_functions as RDF  # noqa: E402
import RainyDay_utilities_Py3.fast_sst as FST  # noqa: E402


def load_storm(f):
    rain, _, _, _, _, _, _, mask, domain, _, _ = RDF.readcatalog(f)
    rain = np.array(rain)
    rain[np.less(rain, 0.)] = np.nan
    return rain, np.array(mask), np.array(domain)


def trim(mask):
    t = np.delete(mask, np.where(np.sum(mask, axis=0) == 0), axis=1)
    t = np.delete(t, np.where(np.sum(mask, axis=1) == 0), axis=0)
    return np.array(t, dtype="float32")


def valid_anchors(domain, mh, mw):
    from scipy.signal import fftconvolve
    inside = fftconvolve((domain > 0).astype("float32"), np.ones((mh, mw), dtype="float32"), mode="valid")
    va = np.zeros(domain.shape, dtype=bool)
    va[:inside.shape[0], :inside.shape[1]] = inside > (mh * mw - 0.5)
    yy, xx = np.nonzero(va)
    return xx.astype(np.int32), yy.astype(np.int32)


def func_asm(asm, name_re):
    """machine code of one function of a numba asm listing, up to the next global symbol; local labels, symbol names
    and comments normalized"""
    out, on = [], False
    for ln in asm.splitlines():
        if re.match(r"^_?_?\S*" + name_re + r"\S*:\s*$", ln) and "cpython" not in ln and "cfunc" not in ln:
            on = True
            continue
        if on:
            if re.match(r"^_\S*:\s*$", ln) or re.match(r"^__\S*:\s*$", ln):
                break
            s = ln.strip()
            if not s or s.startswith((";", ".loh", ".p2align", ".cfi")):
                continue
            s = re.sub(r"\bLloh\d+", "L", s)
            s = re.sub(r"\bL\w+", "L", s)
            s = re.sub(r"_?_ZN\w+", "SYM", s)
            out.append(s)
    return out


def sum_loop(body):
    """float data flow of the np.sum reduction (from the zeroed vector accumulator to the end of the scalar tail):
    float instructions only, integer registers normalized"""
    k = max(i for i, s in enumerate(body) if s.startswith("movi.16b"))
    seg = []
    for s in body[k:]:
        if s.startswith("blr"):
            break
        if re.match(r"^(f\w+|ext|dup|movi|ldr\s+[qs]\d)", s):
            seg.append(re.sub(r"\bx\d+\b", "xN", s))
    return seg


def asm_checks():
    """compiled-code identity of numba_multimask_calc: called (not inlined) in SSTalt_multidur and table_kernel, same
    machine code in both modules; the object-mode (int64 anchor) specialization used by SSTalt for the reference parent
    has the same float data flow (products are elementwise IEEE multiplies, no FMA; same summation order)"""
    ok = True
    bodies = {}
    for name, disp in (("SSTalt_multidur", RDF.SSTalt_multidur), ("table_kernel", FST.table_kernel)):
        llvm = list(disp.inspect_llvm().values())[0]
        called = any(("call" in ln and "numba_multimask_calc" in ln) for ln in llvm.splitlines()
                     if not ln.lstrip().startswith(("declare", "define")))
        bodies[name] = func_asm(list(disp.inspect_asm().values())[0], r"numba_multimask_calc\w*Eiixx")
        print(f"  {name}: numba_multimask_calc {'CALLED (not inlined)' if called else 'INLINED'}; "
              f"{len(bodies[name])} instructions")
        ok &= called and len(bodies[name]) > 0
    same = bodies["SSTalt_multidur"] == bodies["table_kernel"]
    print(f"  callee machine code identical in the SSTalt_multidur and table_kernel modules: {same}")
    ok &= same
    fma = [s for b in bodies.values() for s in b if re.match(r"^(fmadd|fmsub|fnmadd|fnmsub|fmla|fmls)", s)]
    sig64 = [s for s in RDF.numba_multimask_calc.signatures if str(s[2]) == "int64"]
    if sig64:
        b64 = func_asm(RDF.numba_multimask_calc.inspect_asm(sig64[0]), r"numba_multimask_calc\w*Exxxx")
        fma += [s for s in b64 if re.match(r"^(fmadd|fmsub|fnmadd|fnmsub|fmla|fmls)", s)]
        sl64, sl32 = sum_loop(b64), sum_loop(bodies["SSTalt_multidur"])
        print(f"  object-mode int64 specialization (reference parent): summation float data flow identical to the "
              f"copy in SSTalt_multidur: {sl64 == sl32} ({len(sl64)} float instructions)")
        ok &= sl64 == sl32
    print(f"  FMA instructions in numba_multimask_calc: {len(fma)}")
    ok &= len(fma) == 0
    return ok


def synthetic(nstorm, rng):
    """random storms with zeros, NaNs (missing data, as catrain<0 -> NaN in the driver), -0.0 (dedup disabled) and
    exact repeats, fractional masks; table vs SSTalt_multidur at every anchor, numba helpers vs numpy references"""
    ok = []
    for k in range(nstorm):
        T, H, W = 30, int(rng.integers(20, 60)), int(rng.integers(20, 60))
        mh, mw = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        rain = (rng.gamma(0.3, 2., (T, H, W)) * (rng.random((T, H, W)) < 0.4)).astype(np.float32)
        rain[rng.random((T, H, W)) < 0.02] = np.nan
        if k % 3 == 1:
            rain[rng.random((T, H, W)) < 0.05] = -0.0
        if k % 3 == 2:
            rain[5:9] = 0.                                   # whole zero hours: identical windows
        tm = (rng.random((mh, mw)) * (rng.random((mh, mw)) < 0.8)).astype(np.float32)
        tm[0, 0] = 1.; tm[-1, -1] = max(tm[-1, -1], 0.1)
        durs = [1, 2, 3, 5, 8, 13]
        allwin, off, _ = FST.window_stack(rain, durs, check=True)
        yy, xx = np.meshgrid(np.arange(H - mh + 1, dtype=np.int32), np.arange(W - mw + 1, dtype=np.int32), indexing="ij")
        xx, yy = np.ascontiguousarray(xx.ravel()), np.ascontiguousarray(yy.ravel())
        tab = FST.table_sst(rain, allwin, off, durs, xx, yy, tm, mh, mw)
        ref = RDF.SSTalt_multidur(allwin, off, xx, yy, tm, mh, mw)
        r = [np.array_equal(tab.view(np.uint32), ref.view(np.uint32)),
             np.array_equal(FST.zero_hours(rain, mh, mw)[0], FST.zero_hours_ref(rain, mh, mw)),
             np.array_equal(FST.zero_maps(allwin, off, mh, mw), FST.zero_maps_ref(allwin, off, mh, mw))]
        ok += r
        if not all(r):
            print(f"synthetic {k}: table {r[0]}, zero_hours {r[1]}, zero_maps {r[2]} ({T}x{H}x{W}, box {mh}x{mw})")
    print(f"synthetic storms: {nstorm}, all anchors, all checks {'PASS' if all(ok) else 'FAIL'}")
    return all(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog")
    ap.add_argument("--storms", type=int, default=10)
    ap.add_argument("--anchors", type=int, default=2000)
    ap.add_argument("--durs", type=int, nargs="*", default=[1, 3, 6, 12, 24, 48])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--all-anchors", action="store_true")
    ap.add_argument("--asm", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0, help="also N random storms with NaN/-0.0/zero hours")
    a = ap.parse_args()
    FST.ALLOW_OBJECTMODE_SIG[0] = True                                     # this tool calls SSTalt (object mode)
    rng = np.random.default_rng(a.seed)
    files = sorted(glob.glob(os.path.join(a.catalog, "*_storm_*.nc")))
    pick = sorted(rng.choice(len(files), size=min(a.storms, len(files)), replace=False))
    res, maxerr, hist, nev, nbr = [], 0., np.zeros(len(FST.HBINS), np.int64), 0, 0
    t_tab = t_bf = 0.
    for si in pick:
        rain, mask, domain = load_storm(files[si])
        tm = trim(mask)
        mh, mw = tm.shape
        xx, yy = valid_anchors(domain, mh, mw)
        if not a.all_anchors and len(xx) > a.anchors:
            sel = np.sort(rng.choice(len(xx), size=a.anchors, replace=False))
            xx, yy = xx[sel], yy[sel]
        allwin, off, c = FST.window_stack(rain, a.durs, check=True)        # 1. exits on a mismatch
        zh_ok = np.array_equal(FST.zero_hours(rain, mh, mw)[0], FST.zero_hours_ref(rain, mh, mw))
        zm_ok = np.array_equal(FST.zero_maps(allwin, off, mh, mw), FST.zero_maps_ref(allwin, off, mh, mw))
        res += [zh_ok, zm_ok]
        if not (zh_ok and zm_ok):
            print(f"storm {si + 1}: zero_hours ok {zh_ok}, zero_maps ok {zm_ok} (numba vs summed-area reference)")
        t0 = time.perf_counter()
        hist0 = FST.STATS["hist"].copy(); ev0 = FST.STATS["exact_evals"]; br0 = FST.STATS["brute_evals"]
        tab = FST.table_sst(rain, allwin, off, a.durs, xx, yy, tm, mh, mw)
        t1 = time.perf_counter()
        ref = RDF.SSTalt_multidur(allwin, off, xx, yy, tm, mh, mw)
        t2 = time.perf_counter()
        t_tab += t1 - t0; t_bf += t2 - t1
        hist += FST.STATS["hist"] - hist0; nev += FST.STATS["exact_evals"] - ev0; nbr += FST.STATS["brute_evals"] - br0
        same = np.array_equal(tab.view(np.uint32), ref.view(np.uint32))
        # 3. parent
        passrain = np.nansum(rain, axis=0)
        par = FST.parent_sst(passrain, xx, yy, tm, mh, mw)
        pref, _ = RDF.SSTalt(passrain, xx, yy, tm, mh, mw, durcheck=False)
        psame = np.array_equal(par.view(np.uint32), np.asarray(pref, np.float32).view(np.uint32))
        # 4. FFT error vs direct float64 sums (60 anchors)
        P = FST.anchor_cumsums(rain, tm)
        _, eta4 = FST.tolerances(mh, mw, P)
        h = np.nan_to_num(np.asarray(rain, np.float64))
        for k in rng.choice(len(xx), size=min(60, len(xx)), replace=False):
            y, x = int(yy[k]), int(xx[k])
            direct = np.concatenate([[0.], np.cumsum(np.einsum("tij,ij->t", h[:, y:y + mh, x:x + mw], tm.astype(np.float64)))])
            maxerr = max(maxerr, float(np.max(np.abs(P[y, x] - direct))) / max(eta4 / 4., 1e-300))
        res += [same, psame]
        print(f"storm {si + 1:4d} box {mh}x{mw} anchors {len(xx)}: table==SSTalt_multidur {same}, parent==SSTalt "
              f"{psame}, zero {int(np.sum(ref == 0))}/{ref.size}, table {t1 - t0:.2f} s, brute force {t2 - t1:.2f} s",
              flush=True)
    nz = max(int(hist.sum()), 1)
    print("candidates per nonzero draw x duration: " + ", ".join(f"{b}: {100. * c / nz:.2f}%" for b, c in zip(FST.HBINS, hist)))
    print(f"exact sums {nev} vs brute force {nbr} ({nbr / max(nev, 1):.1f}x fewer); time table {t_tab:.1f} s, "
          f"brute force {t_bf:.1f} s")
    print(f"FFT basin sums: max |A_fft - A_direct| / eta = {maxerr:.3g} (must be << 1)")
    res.append(maxerr < 1e-2)
    if a.synthetic:
        res.append(synthetic(a.synthetic, rng))
    if a.asm:
        res.append(asm_checks())
    print("ALL PASS" if all(res) else f"FAILED ({res.count(False)} of {len(res)})")
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
