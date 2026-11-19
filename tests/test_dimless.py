"""ENHANCEDSST "dimensionless" (fast_dimless.py): the kernel against a direct numpy transcription of RainyDay's
SSTalt_normalized, a flat reference field (no rescaling), and the scenario multiplier.

    python tests/test_dimless.py            (env rainyday_env)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Source'))
import RainyDay_utilities_Py3.fast_dimless as FDL                      # noqa: E402


def reference(passrain, sstx, ssty, trimmask, mh, mw, intensegrid, homegrid, durcheck):
    """RainyDay ff2b809, RainyDay_functions.SSTalt_normalized, without the scenario bookkeeping."""
    exprain = passrain if durcheck else np.expand_dims(passrain, 0)
    out = np.zeros(len(sstx), 'float32')
    for k in range(len(sstx)):
        y, x = int(ssty[k]), int(sstx[k])
        if np.all(np.less(exprain[:, y:y + mh, x:x + mw], 0.5)):
            continue
        m = np.exp(homegrid - intensegrid[y:y + mh, x:x + mw]); m = np.where(np.isfinite(m), m, 0.0)
        s = [np.sum(exprain[kk, y:y + mh, x:x + mw] * m * trimmask) for kk in range(exprain.shape[0])]
        out[k] = max(max(s), 0.) if durcheck else s[0]
    return out


rng = np.random.default_rng(1)
ny, nx, mh, mw, n = 60, 70, 9, 13, 4000
trimmask = (rng.random((mh, mw)) > 0.3).astype('float32') * rng.random((mh, mw)).astype('float32')
ref = np.exp(rng.normal(5, 0.2, (ny, nx)))
intense = np.log(ref); home = intense[20:20 + mh, 30:30 + mw].copy()
sstx, ssty = rng.integers(0, nx - mw + 1, n), rng.integers(0, ny - mh + 1, n)
for durcheck, rain in ((False, rng.gamma(0.3, 20, (ny, nx)).astype('float32')), (True, rng.gamma(0.3, 20, (5, ny, nx)).astype('float32'))):
    rain[..., :10, :] = 0.                                             # a dry band: the "no storm" branch
    got, mult, step = FDL.sst_dimensionless(rain, sstx, ssty, trimmask, mh, mw, intense, home, durcheck=durcheck)
    want = reference(rain, sstx, ssty, trimmask, mh, mw, intense, home, durcheck)
    assert np.allclose(got, want, rtol=2e-5, atol=1e-3), np.abs(got - want).max()
    flat, m1, _ = FDL.sst_dimensionless(rain, sstx, ssty, trimmask, mh, mw, np.zeros_like(intense), np.zeros_like(home), durcheck=durcheck)
    plain = reference(rain, sstx, ssty, trimmask, mh, mw, np.zeros_like(intense), np.zeros_like(home), durcheck)
    assert np.allclose(flat, plain, rtol=2e-5, atol=1e-3) and np.allclose(m1[m1 > -9000], 1.0, atol=1e-5)
    print(f'durcheck={durcheck}: kernel = RainyDay reference (max diff {np.abs(got - want).max():.2e}); flat field -> multiplier 1; '
          f'effective multiplier {np.nanmin(mult[mult > -9000]):.2f}-{np.nanmax(mult):.2f}; dry anchors {(mult < -9000).sum()}')
# at the home position the multiplier is exactly 1 and the scenario multiplier reproduces the kernel's sum
m = FDL.box_multiplier(intense, home, 20, 30); assert np.allclose(m, 1.0)
y, x = int(ssty[5]), int(sstx[5]); rain2 = rng.gamma(0.3, 20, (ny, nx)).astype('float32')
g, _, _ = FDL.sst_dimensionless(rain2, [x], [y], trimmask, mh, mw, intense, home)
assert np.isclose(g[0], np.sum(rain2[y:y + mh, x:x + mw] * FDL.box_multiplier(intense, home, y, x) * trimmask), rtol=2e-5)
print('scenario multiplier consistent with the frequency-analysis sum; all checks passed')
