"""Shared helpers: JSON dumping of numpy-laden records, formatting, subprocess pools."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections.abc import Sequence

import numpy as np

log = logging.getLogger("mfa")


def timestamp() -> str:
    """Filesystem-safe local timestamp used for backups."""
    return time.strftime("%Y%m%d_%H%M%S")


def _to_builtin(o: object) -> object:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _to_builtin(x) for k, x in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_to_builtin(x) for x in o]
    return o


def json_dump(obj: object, fn: str) -> None:
    """Write ``obj`` as indented JSON, converting numpy scalars/arrays and
    dropping dictionary keys that start with an underscore (private arrays)."""
    os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
    with open(fn, "w") as fp:
        json.dump(_to_builtin(obj), fp, indent=1)


def json_load(fn: str) -> dict:
    """Read a JSON file."""
    with open(fn) as fp:
        return json.load(fp)


def fmt_g(x: object, nd: int = 2) -> str:
    """Format a table cell: floats with ``nd`` decimals, non-finite as 'nan'."""
    if x is None:
        return "-"
    if isinstance(x, (float, np.floating)):
        return "nan" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def run_jobs(jobs: Sequence[tuple[str, Sequence[str], str]], max_parallel: int, threads: int,
             poll_secs: float = 2.0) -> dict[str, int]:
    """Run ``(label, argv, logfile)`` jobs as subprocesses, at most ``max_parallel`` at once.

    Each child gets ``OMP/OPENBLAS/MKL_NUM_THREADS = threads`` so that the BLAS
    reduction order (and hence the last bits of the SLIM result) is fixed.
    Returns ``{label: returncode}``.
    """
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    pending = list(jobs)
    running: dict[str, tuple[subprocess.Popen, object, float]] = {}
    rc: dict[str, int] = {}
    t0 = time.time()
    while pending or running:
        while pending and len(running) < max_parallel:
            label, argv, logfile = pending.pop(0)
            fh = open(logfile, "w")
            running[label] = (subprocess.Popen(list(argv), stdout=fh, stderr=subprocess.STDOUT, env=env),
                              fh, time.time())
            log.info("  [jobs] start %s (%d running, %d queued)", label, len(running), len(pending))
        for label, (p, fh, ts) in list(running.items()):
            r = p.poll()
            if r is not None:
                fh.close()
                rc[label] = r
                del running[label]
                log.info("  [jobs] done  %s rc=%d (%.1f min; total %.1f min)", label, r,
                         (time.time() - ts) / 60, (time.time() - t0) / 60)
        if running:
            time.sleep(poll_secs)
    return rc
