#!/usr/bin/env python
"""
Adversarial equivalence test: catalogFFT_irregular_domainaware (per-timestep
reference) vs catalogFFT_precompute + catalogFFT_batch (batched engines used
by batchfft / batchfft_mp), on a deliberately nasty irregular domain:
concave boundary, an interior hole, speckled edge cells, fractional mask
weights, plus targeted rain spikes placed so the unconstrained argmax falls
where the storm footprint sticks OUT of the domain. If the batched engines
did not reproduce the "invalidate placements where the full storm footprint
is not inside domain" logic exactly, these cases would diverge.
"""
import os
import sys

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Source'))
from RainyDay_utilities_Py3 import RainyDay_functions as R

rng = np.random.RandomState(42)
H, W = 120, 150
mh, mw = 17, 23

# --- very irregular domain: one big concave blob + interior hole + jagged
# boundary + a detached speckled island (footprints there can never fit) ---
yy, xx = np.mgrid[0:H, 0:W]
domain = np.zeros((H, W), dtype='float32')
domain[((yy - 60.) / 55.) ** 2 + ((xx - 70.) / 68.) ** 2 <= 1.] = 1.   # big ellipse
domain[30:75, 95:150] = 0.                          # concave bite out of the east side
domain[45:58, 40:55] = 0.                           # interior hole
scratch = ((yy + 2 * xx) % 11 == 0) & (xx < 28)     # jagged scratches, west band only
domain[scratch] = 0.
domain[100:115, 120:145] = ((xx[100:115, 120:145] + yy[100:115, 120:145]) % 2 == 0)  # island
print('domain fill fraction: %.2f' % domain.mean())

# fractional storm mask (like a fractional watershed raster)
trimmask = rng.rand(mh, mw).astype('float32')
trimmask[rng.rand(mh, mw) < 0.3] = 0.

pre = R.catalogFFT_precompute(trimmask, domain)

# --- 1) validity map must equal the per-step inside_count_map test exactly ---
foot = np.ones_like(np.nan_to_num(trimmask).astype('float32'))
ic = fftconvolve((domain > 0).astype('float32'), foot[::-1, ::-1], mode='valid')
valid_ref = ~(ic < foot.sum() - 0.5)
assert np.array_equal(valid_ref, pre['valid']), "validity maps differ!"
print('validity maps identical: %d valid / %d total placements'
      % (pre['valid'].sum(), pre['valid'].size))

# --- 2) build test fields ---
T = 80
fields = []
for t in range(T):
    rain = (rng.gamma(0.5, 2.0, size=(H, W)) * (rng.rand(H, W) < 0.4)).astype('float32')
    rain *= domain                      # main script masks before the call
    fields.append(rain)
fields[10][:] = 0.                      # all-dry timestep
# targeted spikes: rain maxima placed at the hole edge, the concave bite, the
# speckled island, the scratched band, and one in the clean interior (control)
spikes = [(20, (44, 41)), (21, (33, 93)), (22, (106, 128)), (23, (52, 22)), (24, (62, 68))]
for t, (sy, sx) in spikes:
    fields[t][:] = 0.
    fields[t][max(0, sy - 2):sy + 3, max(0, sx - 2):sx + 3] = 500.
    fields[t][:] += (rng.rand(H, W) * domain).astype('float32')  # weak background
    fields[t] *= domain
block = np.stack(fields)

# --- 3) per-step reference vs batched ---
rmax_b, y_b, x_b = R.catalogFFT_batch(block, pre)
nloc = nval = 0
maxdiff = 0.
for t in range(T):
    r0, y0, x0 = R.catalogFFT_irregular_domainaware(block[t], trimmask, domain)
    if (y0, x0) != (int(y_b[t]), int(x_b[t])):
        nloc += 1
        print('  LOCATION MISMATCH t=%d ref=(%d,%d) batch=(%d,%d)' % (t, y0, x0, y_b[t], x_b[t]))
    d = abs(r0 - float(rmax_b[t]))
    maxdiff = max(maxdiff, d)
    if d != 0.:
        nval += 1

# spike timesteps: confirm the invalidation actually binds (the unconstrained
# argmax must be an INVALID placement that both engines rejected identically)
nbind = 0
for t, _ in spikes:
    conv = fftconvolve(block[t], np.nan_to_num(trimmask)[::-1, ::-1], mode='valid')
    uy, ux = np.unravel_index(np.argmax(conv), conv.shape)
    chosen_valid = bool(pre['valid'][int(y_b[t]), int(x_b[t])])
    binds = not pre['valid'][uy, ux]
    nbind += binds
    print('t=%d: unconstrained argmax (%d,%d) valid=%s -> chosen (%d,%d) valid=%s%s'
          % (t, uy, ux, bool(pre['valid'][uy, ux]), y_b[t], x_b[t], chosen_valid,
             '  [invalidation binding]' if binds else ''))
    assert chosen_valid, "an engine chose an invalid placement!"
assert nbind >= 3, "spikes did not exercise the footprint-inside-domain constraint enough"

print('locations: %d/%d identical; values: %d timesteps with nonzero diff, max |diff| = %g'
      % (T - nloc, T, nval, maxdiff))
assert nloc == 0, "batched engine placements differ from per-step reference!"
print('PASS: batched domain-aware engine is placement-identical to catalogFFT_irregular_domainaware')
