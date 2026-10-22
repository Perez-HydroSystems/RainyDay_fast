"""
GP 2026-10-22: RainyDay_fast wall-clock timers (no effect on results). Active only with RAINYDAY_TIMERS=1.

    from RainyDay_utilities_Py3 import fast_prof as FP
    with FP.timer("resampling: storm read"):
        ...
    FP.report()          # table of total seconds and calls per label (printed at the end of the run)
"""
import os
import time
from contextlib import contextmanager

ON = os.environ.get("RAINYDAY_TIMERS", "").strip() not in ("", "0")
_TOT, _N = {}, {}
_T0 = time.perf_counter()


@contextmanager
def timer(label):
    if not ON:
        yield
        return
    t = time.perf_counter()
    try:
        yield
    finally:
        _TOT[label] = _TOT.get(label, 0.) + time.perf_counter() - t
        _N[label] = _N.get(label, 0) + 1


_OPEN = {}


def tic(label):
    """Start a timer (flat statements in the driver, no re-indentation of the timed block)."""
    if ON:
        _OPEN[label] = time.perf_counter()


def toc(label):
    if ON and label in _OPEN:
        _TOT[label] = _TOT.get(label, 0.) + time.perf_counter() - _OPEN.pop(label)
        _N[label] = _N.get(label, 0) + 1


def report():
    if not ON or not _TOT:
        return
    wall = time.perf_counter() - _T0
    print(f"\nRAINYDAY_TIMERS (wall since import {wall:.1f} s)")
    for k in sorted(_TOT, key=_TOT.get, reverse=True):
        print(f"  {_TOT[k]:10.2f} s  {100 * _TOT[k] / wall:5.1f} %  {_N[k]:8d} calls  {k}")
