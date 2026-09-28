"""FreeSurfer ``.patch.3d`` reading and writing.

File format (big-endian binary, as written by FreeSurfer's ``MRISwritePatch``)::

    int32   -1                       (version marker)
    int32   npts                     (number of vertices in the patch)
    npts x { int32 vno; float32 x, y, z }

``vno`` is the 1-based vertex index into the parent surface; a *negative* value
flags the vertex as a border vertex (``-(index + 1)``).  Faces are not stored:
FreeSurfer (and this package) rebuild them as the faces of the parent surface
whose three vertices are all in the patch.  A flat patch (``*.flat.patch.3d``)
uses the same format with ``z = 0`` and the flat coordinates in ``(x, y)``.
"""

from __future__ import annotations

import struct

import numpy as np

_REC = np.dtype(">i4, >3f4")


def read_patch(fn: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a ``.patch.3d`` file.

    Parameters
    ----------
    fn : str
        Path of the patch file.

    Returns
    -------
    idx : (n,) int ndarray
        0-based vertex indices into the parent surface.
    border : (n,) bool ndarray
        Border flags (negative ``vno`` in the file).
    xyz : (n, 3) float ndarray
        Coordinates stored in the file (3D for a cut patch, ``z = 0`` for a flat one).
    """
    with open(fn, "rb") as fp:
        _ver, npts = struct.unpack(">ii", fp.read(8))
        d = np.frombuffer(fp.read(npts * 16), dtype=_REC)
    vno = d["f0"]
    return np.abs(vno) - 1, vno < 0, d["f1"].astype(float)


def write_patch(fn: str, idx: np.ndarray, xyz: np.ndarray, border: np.ndarray) -> None:
    """Write a ``.patch.3d`` file.

    Parameters
    ----------
    fn : str
        Output path.
    idx : (n,) int array
        0-based vertex indices into the parent surface.
    xyz : (N, 3) float array
        Coordinates indexed by the *parent* vertex id (``xyz[idx[i]]`` is written).
    border : (n,) bool array
        Border flags, written as negative vertex numbers.
    """
    with open(fn, "wb") as fp:
        fp.write(struct.pack(">ii", -1, len(idx)))
        for i, b in zip(idx, border):
            vno = int(i) + 1
            fp.write(struct.pack(">i3f", -vno if b else vno, *xyz[i]))
