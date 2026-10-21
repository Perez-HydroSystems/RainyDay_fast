#!/usr/bin/env python
"""
GP 2026-10-20: regression test for build_windowed_block (batched catalog engines).

Chains random blocks through build_windowed_block exactly as catalog_scan_files does and compares every rolling
nwindow-step total with a brute-force sum over the concatenated record (zeros before the first step), for window
lengths 1, 2, 3, 24 and 72 and blocks shorter and longer than the window. nwindow = 1 (a catalog whose DURATION
equals the data time step, e.g. a native 1 h catalog of hourly data) used to fail: full[-(nwindow-1):] = full[-0:]
returned the whole array as the next context, so the context grew without bound and later windows were wrong.

    python tools/test_windowed_block.py          # prints ALL PASS / FAIL, exit 0/1
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Source'))
from RainyDay_utilities_Py3 import RainyDay_functions as R  # noqa: E402

rng = np.random.RandomState(7)
H, W = 9, 11
domainmask = (rng.rand(H, W) > 0.2).astype('float32')
fails = 0
for nwindow in (1, 2, 3, 24, 72):
    for blocks in ([5, 5, 5], [1, 1, 1, 1], [100, 3, 48], [24] * 6):
        record = [rng.gamma(0.3, 2.0, size=(b, H, W)).astype('float32') for b in blocks]
        allrain = np.concatenate(record, axis=0)
        csum = np.cumsum(np.concatenate([np.zeros((nwindow, H, W)), allrain]), axis=0)
        brute = (csum[nwindow:] - csum[:-nwindow]) * domainmask          # window ending at each step
        context = np.zeros((nwindow - 1, H, W), dtype='float32')
        got, start = [], 0
        for blk in record:
            windowed, context = R.build_windowed_block(blk, context, nwindow, domainmask)
            if context.shape[0] != nwindow - 1:
                print(f"FAIL nwindow={nwindow} blocks={blocks}: context has {context.shape[0]} steps, expected {nwindow - 1}")
                fails += 1
                break
            got.append(windowed)
        else:
            d = np.max(np.abs(np.concatenate(got) - brute))
            if not d < 1e-4:
                print(f"FAIL nwindow={nwindow} blocks={blocks}: max |diff| {d:.3e}")
                fails += 1
print("ALL PASS" if fails == 0 else f"{fails} FAIL")
sys.exit(1 if fails else 0)
