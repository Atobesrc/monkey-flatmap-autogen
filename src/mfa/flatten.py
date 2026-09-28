"""Flattening (uniform Tutte + SLIM), canonical orientation, quality gates and silhouette metric.

The canonical orientation is a rigid placement of the finished map defined by parcel
centroids only (:func:`anatomical_frame`), never by a scanner axis, so two subjects
whose heads lay differently still end up in the same display frame.

Everything here is deterministic: the Tutte initialisation is a sparse linear
solve and SLIM is a fixed sequence of local/global steps with no randomness.
The last bits of the result depend on the BLAS reduction order, i.e. on the
thread count; set ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` /
``MKL_NUM_THREADS`` to a fixed value when byte-identical results matter.
"""

from __future__ import annotations

import logging
import os
import shutil
import time

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .mesh import (
    boundary_loops,
    face_area3,
    flip_count,
    loop_perimeter,
    patch_faces,
    poly_centroid,
    resample_closed_loop,
    unique_edges,
)
from .patch_io import read_patch, write_patch
from .recipe import TEMPORAL_CORRIDOR_PARCELS
from .surface import DEFAULT_PARCELLATION, Parcellation

log = logging.getLogger("mfa")

DEFAULT_SLIM_ITERS = 800
DEFAULT_SLIM_TOL = 1e-7
#: Radius (flat mm) of the disk used by the local area-packing (swirl) detector.
HOT_PACK_MM = 2.0
#: An edge is 'extreme' when its stretch is > HOT_EXTREME or < 1 / HOT_EXTREME.
HOT_EXTREME = 2.5
#: Gate on the maximum local area-packing factor (1 = locally isometric).
HOT_MAX = 3.0
#: Minimum correlation between the display frame and the anatomical anterior/dorsal plane.
#: Smallest separation, as a fraction of sqrt(flat area), that the two anatomical frame
#: axes must have for the canonical orientation to be well defined (``gate_orient``).
FRAME_SEP_MIN = 0.05
#: Reporting only: below this, the map sits far from the scanner's anterior/dorsal plane.
#: NOT a gate -- the head's pose in the scanner must not decide whether a map is acceptable.
ORIENT_CORR_MIN = 0.5


# ---------------------------------------------------------------- SLIM
def jacobian_svals(V: np.ndarray, F: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-face singular values (s1 >= s2) and signed determinant of the 3D -> 2D map."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    nrm = np.cross(e1, e2)
    x = e1 / np.maximum(np.linalg.norm(e1, axis=1), 1e-12)[:, None]
    y = np.cross(nrm, x)
    y /= np.maximum(np.linalg.norm(y, axis=1), 1e-12)[:, None]
    M = np.zeros((len(F), 2, 2))
    M[:, 0, 0] = (e1 * x).sum(1)
    M[:, 1, 0] = (e1 * y).sum(1)
    M[:, 0, 1] = (e2 * x).sum(1)
    M[:, 1, 1] = (e2 * y).sum(1)
    q1, q2 = uv[F[:, 1]] - uv[F[:, 0]], uv[F[:, 2]] - uv[F[:, 0]]
    Q = np.zeros((len(F), 2, 2))
    Q[:, :, 0] = q1
    Q[:, :, 1] = q2
    J = Q @ np.linalg.inv(M)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    fro2 = (J**2).sum((1, 2))
    a = np.sqrt(np.maximum(fro2 + 2 * np.abs(det), 0))
    b = np.sqrt(np.maximum(fro2 - 2 * np.abs(det), 0))
    return (a + b) / 2, (a - b) / 2, det


def symdir_energy(V: np.ndarray, F: np.ndarray, uv: np.ndarray) -> tuple[float, int]:
    """Area-weighted mean symmetric-Dirichlet energy and number of flipped faces."""
    s1, s2, det = jacobian_svals(V, F, uv)
    A = face_area3(V, F)
    s1 = np.maximum(s1, 1e-9)
    s2 = np.maximum(np.abs(s2), 1e-9)
    e = s1**2 + s2**2 + s1**-2 + s2**-2
    return float((A * e).sum() / A.sum()), int((det < 0).sum())


def _slim_boundary(F: np.ndarray) -> np.ndarray:
    import igl

    loops = igl.boundary_loop_all(F)
    loops = sorted(loops, key=len, reverse=True)
    log.info("  boundary loops: %d (sizes %s)", len(loops), [len(loop) for loop in loops[:5]])
    return np.asarray(loops[0], dtype=np.int64)


def init_tutte_uniform(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Tutte embedding with uniform weights onto a disk whose area equals the patch area.

    Bijective by Tutte's theorem (unlike cotangent-weight harmonic maps, which can fold).

    Parameters
    ----------
    V : (n, 3) float64
        Patch vertices.
    F : (m, 3) int64
        Patch-local faces.
    """
    import igl

    bnd = _slim_boundary(F)
    if len(np.unique(bnd)) != len(bnd):
        log.warning("  boundary loop is not simple (%d repeats)", len(bnd) - len(np.unique(bnd)))
    n = len(V)
    e = unique_edges(F)
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    L = (diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()
    isb = np.zeros(n, bool)
    isb[bnd] = True
    interior = np.where(~isb)[0]
    bc = np.asarray(igl.map_vertices_to_circle(np.asfortranarray(V), bnd.astype(np.int32)))
    R = np.sqrt(face_area3(V, F).sum() / np.pi)
    uv = np.zeros((n, 2))
    uv[bnd] = bc * R
    rhs = -(L[interior][:, bnd] @ uv[bnd])
    uv[interior] = splu(L[interior][:, interior].tocsc()).solve(rhs)
    return uv


def run_slim(V: np.ndarray, F: np.ndarray, uv0: np.ndarray, iters: int = DEFAULT_SLIM_ITERS,
             tol: float = DEFAULT_SLIM_TOL) -> tuple[np.ndarray, list[tuple[int, float, int, float]]]:
    """libigl SLIM (symmetric Dirichlet, no soft constraints) started from ``uv0``.

    Stops when ``|E_prev - E| < tol * E`` or after ``iters`` iterations.

    Returns
    -------
    uv : (n, 2) ndarray
    trace : list of (iteration, energy, flips, seconds)
    """
    import igl

    et = igl.MappingEnergyType.SYMMETRIC_DIRICHLET
    b = np.zeros(0, dtype=np.int32)
    bc = np.asfortranarray(np.zeros((0, 2)))
    data = igl.slim_precompute(np.asfortranarray(V), np.asfortranarray(F.astype(np.int32)),
                               np.asfortranarray(uv0.astype(np.float64)), et, b, bc, 0.0)
    E0, fl0 = symdir_energy(V, F, uv0)
    trace = [(0, E0, fl0, 0.0)]
    log.info("  slim it 0: E=%.4f flips=%d", E0, fl0)
    t0 = time.time()
    uv = uv0
    Eprev = E0
    E, fl = E0, fl0
    for k in range(1, iters + 1):
        uv = np.asarray(igl.slim_solve(data, 1))
        E, fl = symdir_energy(V, F, uv)
        trace.append((k, E, fl, time.time() - t0))
        if k % 10 == 0 or k <= 5:
            log.info("  slim it %d: E=%.4f flips=%d (%.1fs)", k, E, fl, time.time() - t0)
        if abs(Eprev - E) < tol * E:
            log.info("  slim converged at it %d: E=%.4f flips=%d (%.1fs)", k, E, fl, time.time() - t0)
            break
        Eprev = E
    else:
        log.info("  slim stopped at the iteration cap (%d): E=%.4f flips=%d", iters, E, fl)
    return uv, trace


def slim_flatten(sd: str, hemi: str, name: str, iters: int = DEFAULT_SLIM_ITERS,
                 tol: float = DEFAULT_SLIM_TOL) -> list[tuple[int, float, int, float]]:
    """Flatten ``surf/<hemi>.<name>.patch.3d`` and write ``surf/<hemi>.<name>.flat.patch.3d``.

    Uniform-Tutte initialisation followed by SLIM; ``z = 0``, orientation is left
    to :func:`canonical_orient`.  Returns the SLIM trace.
    """
    pf = f"{sd}/surf/{hemi}.{name}.patch.3d"
    idx, border, _ = read_patch(pf)
    v, f = nib.freesurfer.read_geometry(f"{sd}/surf/{hemi}.white")
    F = patch_faces(idx, f, len(v)).astype(np.int64)
    used = np.zeros(len(idx), bool)
    used[F.ravel()] = True
    if not used.all():
        log.warning("  [%s] %d patch vertices in no face", hemi, int((~used).sum()))
    V = v[idx].astype(np.float64)
    t0 = time.time()
    uv0 = init_tutte_uniform(V, F)
    log.info("  [%s] tutte-uniform init: %.1fs", hemi, time.time() - t0)
    uv, trace = run_slim(V, F, uv0, iters=iters, tol=tol)
    full = np.zeros((len(v), 3))
    full[idx, :2] = uv
    fn = f"{sd}/surf/{hemi}.{name}.flat.patch.3d"
    write_patch(fn, idx, full, border)
    log.info("  [%s] SLIM done: %d its, E=%.4f, flips=%d, %.1fs -> %s", hemi, len(trace) - 1,
             trace[-1][1], trace[-1][2], time.time() - t0, fn)
    return trace


# ---------------------------------------------------------------- orientation
#: Parcels defining the canonical display frame.  All are in ``REQUIRED_PARCELS``, so every
#: subject has them: occipital and frontal fix the anterior axis, ventral fixes which way is up.
ORIENT_OCCIPITAL = ("V1", "V2-V4")
ORIENT_FRONTAL = ("OFC", "lat_PFC", "ACgG")
ORIENT_VENTRAL = ("Amy", "MTL", "TG", "ITC")


def anatomical_frame(P2: np.ndarray, lab: np.ndarray, names: list[str], hemi: str) -> np.ndarray:
    """Orthogonal 2x2 matrix taking flat coordinates into the canonical display frame.

    The frame is fixed by three parcel groups and nothing else:

    * the anterior axis is the direction from the occipital centroid
      (:data:`ORIENT_OCCIPITAL`) to the frontal centroid (:data:`ORIENT_FRONTAL`);
    * the perpendicular axis points away from the ventral centroid
      (:data:`ORIENT_VENTRAL`), so dorsal is up;
    * the anterior axis points right on the right hemisphere and left on the left one,
      which is the butterfly convention of a merged flatmap.

    Those three constraints have exactly one orthogonal solution, so the chirality SLIM
    happened to produce is fixed as well.  The result is a rotation or a reflection: no
    distance in the map is changed, and no coordinate axis of the scanner is used, so two
    subjects whose heads lay differently still come out in the same frame.
    """
    def cent(group):
        m = np.isin(lab, [names.index(p) for p in group if p in names])
        if not m.any():
            raise ValueError(f"[{hemi}] canonical orientation needs {group}; none of them is in the patch")
        return P2[m].mean(0)

    u = cent(ORIENT_FRONTAL) - cent(ORIENT_OCCIPITAL)   # anterior
    nu = float(np.linalg.norm(u))
    if nu < 1e-9:
        raise ValueError(f"[{hemi}] canonical orientation: occipital and frontal centroids coincide")
    u = u / nu
    w = np.array([-u[1], u[0]])                          # perpendicular, sign still open
    if float((cent(ORIENT_VENTRAL) - P2.mean(0)) @ w) > 0:
        w = -w                                           # ... now dorsal
    sign = -1.0 if hemi == "lh" else 1.0
    return np.c_[sign * u, w].T                          # rows: display X (anterior), display Y (dorsal)


#: A cut may pass beside an area but must not leave part of it stranded from the rest.  Above
#: this fraction of an area detached, :func:`hemi_quality` warns (large, heavily folded areas
#: that have to be opened for the map to lie flat are the accepted exception).
FRAG_WARN = 0.05


def parcel_fragmentation(idx: np.ndarray, faces: np.ndarray, lab: np.ndarray,
                         names: list[str], min_vertices: int = 20) -> dict[str, float]:
    """For each area in the patch, the fraction of it detached from its own largest piece.

    0 means the area came out in one connected piece, which is what a cut should leave.  A
    cut that runs *across* an area instead of beside it strands a piece of it, and the area
    then appears twice in two places on the map, so any measurement of it is split in two.
    """
    inpatch = np.zeros(len(lab), bool)
    inpatch[idx] = True
    out: dict[str, float] = {}
    for pi, nm in enumerate(names):
        if nm in ("unknown", "Unknown", "???", "medial_wall"):
            continue
        m = (lab == pi) & inpatch
        if m.sum() < min_vertices:
            continue
        loc = np.where(m[idx])[0]
        ff = faces[np.isin(faces, loc).all(1)]
        if not len(ff):
            continue
        ren = np.full(len(idx), -1)
        ren[loc] = np.arange(len(loc))
        g = ren[ff]
        e = np.r_[g[:, [0, 1]], g[:, [1, 2]], g[:, [2, 0]]]
        A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(loc),) * 2)
        _, comp = connected_components(A + A.T, directed=False)
        sizes = np.sort(np.bincount(comp))[::-1] / len(loc)
        out[nm] = float(sizes[1:].sum())
    return out


def frame_separations(P2: np.ndarray, lab: np.ndarray, names: list[str]) -> tuple[float, float]:
    """How well defined the anatomical frame is, in units of ``sqrt(flat area)``.

    Returns ``(axis_sep, updown_sep)``: the distance between the occipital and frontal
    centroids, which fixes the anterior axis, and the size of the ventral centroid's offset
    along the perpendicular axis, which fixes which way is up.  Both are intrinsic to the
    map, so neither depends on how the head lay in the scanner.  If either collapses the
    frame is ill defined and :func:`hemi_quality` fails ``gate_orient``.
    """
    def cent(group):
        m = np.isin(lab, [names.index(p) for p in group if p in names])
        return P2[m].mean(0) if m.any() else np.full(2, np.nan)

    occ, fro, ven = cent(ORIENT_OCCIPITAL), cent(ORIENT_FRONTAL), cent(ORIENT_VENTRAL)
    scale = np.sqrt(_polygon_area_proxy(P2)) or 1.0
    d = fro - occ
    nd = float(np.linalg.norm(d))
    if not np.isfinite(nd) or nd < 1e-9:
        return 0.0, 0.0
    u = d / nd
    w = np.array([-u[1], u[0]])
    return nd / scale, abs(float((ven - P2.mean(0)) @ w)) / scale


def _polygon_area_proxy(P2: np.ndarray) -> float:
    """Spread of the map (mean squared radius x pi), a scale that needs no face list."""
    c = P2 - P2.mean(0)
    return float(np.pi * np.mean(np.sum(c * c, axis=1)))


def canonical_orient(sd: str, hemi: str, name: str, backup: bool = True,
                     parc: Parcellation = DEFAULT_PARCELLATION) -> tuple[np.ndarray, float]:
    """Rotate a flat patch into the canonical display frame (in place, with optional backup).

    The frame comes from :func:`anatomical_frame`: anterior along the occipital-to-frontal
    direction (pointing outwards, so the two hemispheres face away from each other), dorsal
    up.  pycortex's ``import_flat`` maps patch ``(x, y)`` to display ``(y, -x)``; that
    transform is inverted here (``patch_x = -display_Y``, ``patch_y = display_X``).

    Rigid: no distance is changed.  Returns ``(R, corr)``, ``R`` the 2x2 orthogonal matrix
    and ``corr`` a diagnostic correlation between the oriented map and the white surface's
    ``(anterior, dorsal)`` plane.  ``corr`` is reported, never used to build the frame: it
    is low only when the head lay far from the scanner axes, which no longer matters.
    """
    fn = f"{sd}/surf/{hemi}.{name}.flat.patch.3d"
    idx, borderflag, xyz = read_patch(fn)
    v, _ = nib.freesurfer.read_geometry(f"{sd}/surf/{hemi}.white")
    lab, names = parc.read(sd, hemi)
    R = anatomical_frame(xyz[:, :2] - xyz[:, :2].mean(0), lab[idx], names, hemi)
    D = (xyz[:, :2] - xyz[:, :2].mean(0)) @ R.T
    sign = -1.0 if hemi == "lh" else 1.0
    Q = np.c_[sign * v[idx][:, 1], v[idx][:, 2]]
    Q = Q - Q.mean(0)
    fit = float(np.corrcoef(D.ravel(), Q.ravel())[0, 1])
    newxyz = np.zeros_like(xyz)
    newxyz[:, 0] = -D[:, 1]
    newxyz[:, 1] = D[:, 0]
    if backup:
        bk = fn + time.strftime(".precanon.%Y%m%d_%H%M.bak")
        if not os.path.exists(bk):
            shutil.copy2(fn, bk)
    full = np.zeros((len(v), 3))
    full[idx] = newxyz
    write_patch(fn, idx, full, borderflag)
    log.info("  [%s] canonical orient: anatomical frame det=%+.0f, corr(flat,(A,D))=%.3f, mirror sign=%+.0f -> %s",
             hemi, np.linalg.det(R), fit, sign, fn)
    return R, fit

def flat_metrics(sd: str, hemi: str, name: str) -> tuple[dict, tuple]:
    """Basic distortion metrics of ``surf/<hemi>.<name>.flat.patch.3d``.

    Returns ``(metrics, (idx, fk, p2, flipped, v))``; metrics = vertex/face counts,
    flipped triangles (count and %), edge-stretch median / p5 / p95.
    """
    fn = f"{sd}/surf/{hemi}.{name}.flat.patch.3d"
    idx, borderflag, xyz = read_patch(fn)
    v, f = nib.freesurfer.read_geometry(f"{sd}/surf/{hemi}.white")
    fk = patch_faces(idx, f, len(v))
    p2 = xyz[:, :2]
    a, b, c = p2[fk[:, 0]], p2[fk[:, 1]], p2[fk[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    sgn = np.sign(np.median(area2))
    flipped = (area2 * sgn) < 0
    e = unique_edges(fk)
    Lf = np.linalg.norm(p2[e[:, 0]] - p2[e[:, 1]], axis=1)
    L3 = np.linalg.norm(v[idx][e[:, 0]] - v[idx][e[:, 1]], axis=1)
    r = Lf / np.maximum(L3, 1e-6)
    m = dict(nvert=len(idx), nface=len(fk), nborder=int(borderflag.sum()),
             flipped=int(flipped.sum()), flipped_pct=100 * float(flipped.mean()),
             stretch_med=float(np.median(r)), stretch_p5=float(np.percentile(r, 5)),
             stretch_p95=float(np.percentile(r, 95)))
    return m, (idx, fk, p2, flipped, v)


def display_loop(hemi: str, p2: np.ndarray, fk: np.ndarray) -> np.ndarray:
    """Largest boundary loop in the pycortex display frame ``(X, Y) = (patch_y, -patch_x)``,
    with the right hemisphere mirrored into the left frame."""
    L = boundary_loops(fk)[0]
    P = p2[L]
    X, Y = P[:, 1], -P[:, 0]
    if hemi == "rh":
        X = -X
    return np.c_[X, Y]


def silhouette_from_loops(Ll: np.ndarray, Lr: np.ndarray, N: int = 4096) -> dict[str, float]:
    """Silhouette agreement of two display-frame boundary loops.

    Symmetric mean nearest-neighbour distance between the arc-length-resampled,
    centroid-aligned loops, in mm and as a percentage of the mean perimeter (the
    percentage is the scale-free number to compare across flatteners and subjects).
    """
    Ql, Qr = resample_closed_loop(Ll, N), resample_closed_loop(Lr, N)
    pl, pr = loop_perimeter(Ll), loop_perimeter(Lr)
    Ql = Ql - poly_centroid(Ql)
    Qr = Qr - poly_centroid(Qr)
    d = 0.5 * (cKDTree(Qr).query(Ql)[0].mean() + cKDTree(Ql).query(Qr)[0].mean())
    return dict(sil_mm=float(d), sil_pct=float(100 * d / (0.5 * (pl + pr))), perim_lh=pl, perim_rh=pr)


def silhouette_norm(sd: str, lh_name: str, rh_name: str) -> dict[str, float]:
    """Normalised lh-vs-mirrored-rh silhouette of two canonical-frame flat patches."""
    loops = {}
    for hemi, name in (("lh", lh_name), ("rh", rh_name)):
        idx, _, xyz = read_patch(f"{sd}/surf/{hemi}.{name}.flat.patch.3d")
        v, f = nib.freesurfer.read_geometry(f"{sd}/surf/{hemi}.white")
        loops[hemi] = display_loop(hemi, xyz[:, :2], patch_faces(idx, f, len(v)))
    return silhouette_from_loops(loops["lh"], loops["rh"])


def hemi_quality(sd: str, hemi: str, name: str, parc: Parcellation = DEFAULT_PARCELLATION,
                 nbhd_mm: float = HOT_PACK_MM, extreme: float = HOT_EXTREME, arrays: bool = False) -> dict:
    """Gates and quality metrics of one flattened hemisphere ``surf/<hemi>.<name>.flat.patch.3d``.

    Gates (all must hold, ``gates_ok``): one boundary loop, no isolated vertices,
    one connected component (``gate_topology``); zero flipped triangles
    (``gate_flips``); canonical orientation present, i.e. the display frame
    sits in a well-defined anatomical frame (``gate_orient``: both separations of
    :func:`frame_separations` exceed :data:`FRAME_SEP_MIN`).  ``orient_corr``, the
    correlation with the white surface's anterior/dorsal plane, is reported but does NOT
    gate: it drops when the head lay away from the scanner axes, which says nothing about
    the map.  ``orient_near_scanner_ad`` flags that case for information only.  The
    area-packing gate (``hot_pack <= hot_max``) is applied by the caller.

    Quality: edge stretch ``r`` = flat / 3D edge length (median, p95, p99);
    ``hot_pack`` = maximum over vertices of sum(3D area) / sum(flat area) of the
    faces whose flat centroid lies within ``nbhd_mm`` of the vertex (1 = locally
    isometric; a swirl packs > ~2x the surface into the disk) with its location
    (``hot_vertex``, ``hot_parcel``, ``hot_dbnd`` = flat distance to the boundary);
    ``edge_hot``, ``hot_p999`` and ``hot_cluster`` are edge-based hotspot measures
    dominated by the slit-end singularities (informational).  Cortex retention:
    vertices and white-surface area of cortex.label kept in the patch.

    With ``arrays=True`` the private key ``_arrays`` holds the patch arrays used
    by the figure code.
    """
    ff = f"{sd}/surf/{hemi}.{name}.flat.patch.3d"
    idx, _bflag, xyz = read_patch(ff)
    v, f = nib.freesurfer.read_geometry(f"{sd}/surf/{hemi}.white")
    n = len(v)
    fk = patch_faces(idx, f, n)
    p2 = xyz[:, :2]
    used = np.zeros(len(idx), bool)
    used[fk.ravel()] = True
    e = unique_edges(fk)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(idx),) * 2)
    _, cl = connected_components(g + g.T, directed=False)
    ncomp = len(np.unique(cl[used]))
    loops = boundary_loops(fk)
    flips = flip_count(p2, fk)
    Lf = np.linalg.norm(p2[e[:, 0]] - p2[e[:, 1]], axis=1)
    L3 = np.maximum(np.linalg.norm(v[idx][e[:, 0]] - v[idx][e[:, 1]], axis=1), 1e-6)
    r = np.maximum(Lf / L3, 1e-6)
    absl = np.abs(np.log2(r))
    # edge hotspot: flat-space neighbourhood mean of |log2 r| (informational)
    mid = 0.5 * (p2[e[:, 0]] + p2[e[:, 1]])
    lists = cKDTree(mid).query_ball_point(p2, nbhd_mm)
    nbe = np.array([absl[lst].mean() if len(lst) else 0.0 for lst in lists])
    # area-packing hotspot (the swirl detector)
    A3 = face_area3(v[idx].astype(float), fk)
    q0, q1, q2 = p2[fk[:, 0]], p2[fk[:, 1]], p2[fk[:, 2]]
    Af = 0.5 * np.abs((q1[:, 0] - q0[:, 0]) * (q2[:, 1] - q0[:, 1]) - (q1[:, 1] - q0[:, 1]) * (q2[:, 0] - q0[:, 0]))
    flists = cKDTree(p2[fk].mean(1)).query_ball_point(p2, nbhd_mm)
    nb = np.array([A3[lst].sum() / max(Af[lst].sum(), 1e-9) if len(lst) else 1.0 for lst in flists])
    ihot = int(np.argmax(nb))
    bidx = np.concatenate(loops) if loops else np.arange(0)
    btree = cKDTree(p2[bidx]) if len(bidx) else None
    hot_dbnd = float(btree.query(p2[ihot])[0]) if btree is not None else np.nan
    # extreme-edge clusters
    ext = (r > extreme) | (r < 1.0 / extreme)
    hot_cluster = 0
    if ext.any():
        ee = e[ext]
        ge = coo_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])), shape=(len(idx),) * 2)
        _, cle = connected_components(ge + ge.T, directed=False)
        hot_cluster = int(np.bincount(cle[ee[:, 0]]).max())
    # cortex retention (vertices and white-surface area)
    cortex = nib.freesurfer.read_label(f"{sd}/label/{hemi}.cortex.label")
    inc = np.zeros(n, bool)
    inc[cortex] = True
    inp = np.zeros(n, bool)
    inp[idx[used]] = True
    fa = face_area3(v, f)
    va = np.zeros(n)
    for i in range(3):
        np.add.at(va, f[:, i], fa / 3)
    kept_v = int((inc & inp).sum())
    loss_v_pct = 100.0 * (1 - kept_v / max(int(inc.sum()), 1))
    loss_a_pct = 100.0 * (1 - va[inc & inp].sum() / max(va[inc].sum(), 1e-9))
    # orientation diagnostic: display frame vs the white surface's (sign*A, D) plane.
    # The frame itself is anatomical (see anatomical_frame); this only flags a map that
    # ended up far from the usual lateral view.
    disp = np.c_[p2[:, 1], -p2[:, 0]]
    sign = -1.0 if hemi == "lh" else 1.0
    tgt = np.c_[sign * v[idx][:, 1], v[idx][:, 2]]
    dc, tc = disp - disp.mean(0), tgt - tgt.mean(0)
    orient_corr = float(np.corrcoef(dc.ravel(), tc.ravel())[0, 1])
    try:
        _lab0, _names0 = parc.read(sd, hemi)
        frame_axis_sep, frame_updown_sep = frame_separations(p2, _lab0[idx], _names0)
    except (OSError, KeyError, ValueError):
        frame_axis_sep = frame_updown_sep = float("nan")
    try:
        _fr = parcel_fragmentation(idx, fk, parc.read(sd, hemi)[0], parc.read(sd, hemi)[1])
        frag_parcel = max(_fr, key=_fr.get) if _fr else "?"
        frag_max = float(_fr.get(frag_parcel, 0.0))
    except (OSError, KeyError, ValueError):
        frag_parcel, frag_max = "?", float("nan")
    hot_parcel = "?"
    hotT = dict(hot_pack_temporal=np.nan, hot_temporal_vertex=-1, hot_temporal_parcel="?",
                hot_temporal_xyz3d=None, hot_temporal_dbnd=np.nan)
    try:
        lab, names = parc.read(sd, hemi)
    except (OSError, KeyError):
        lab, names = None, []
    if lab is not None:
        hot_parcel = names[lab[idx[ihot]]]
        maskT = np.isin(lab[idx], [names.index(x) for x in TEMPORAL_CORRIDOR_PARCELS if x in names])
        if maskT.any():
            iT = int(np.where(maskT)[0][np.argmax(nb[maskT])])
            hotT = dict(hot_pack_temporal=float(nb[iT]), hot_temporal_vertex=int(idx[iT]),
                        hot_temporal_parcel=names[lab[idx[iT]]],
                        hot_temporal_xyz3d=[float(x) for x in v[idx[iT]]],
                        hot_temporal_dbnd=float(btree.query(p2[iT])[0]) if btree is not None else np.nan)
    q = dict(
        hemi=hemi, name=name, nvert=int(len(idx)), nface=int(len(fk)),
        loops=len(loops), isolated=int((~used).sum()), components=int(ncomp),
        flips=int(flips), orient_corr=orient_corr,
        stretch_med=float(np.median(r)), stretch_p95=float(np.percentile(r, 95)),
        stretch_p99=float(np.percentile(r, 99)),
        hot_pack=float(nb[ihot]), hot_vertex=int(idx[ihot]), hot_parcel=hot_parcel,
        hot_xy=[float(p2[ihot, 0]), float(p2[ihot, 1])],
        hot_xyz3d=[float(x) for x in v[idx[ihot]]], hot_dbnd=hot_dbnd,
        hot_in_temporal=bool(hot_parcel in TEMPORAL_CORRIDOR_PARCELS), **hotT,
        edge_hot=float(nbe.max()), hot_p999=float(np.percentile(absl, 99.9)), hot_cluster=hot_cluster,
        n_extreme_edges=int(ext.sum()),
        cortex_nvert=int(inc.sum()), cortex_kept=kept_v,
        cortex_loss_vert_pct=loss_v_pct, cortex_loss_area_pct=loss_a_pct,
        frag_max=frag_max, frag_parcel=frag_parcel,
        gate_topology=bool(len(loops) == 1 and (~used).sum() == 0 and ncomp == 1),
        gate_flips=bool(flips == 0),
        frame_axis_sep=frame_axis_sep, frame_updown_sep=frame_updown_sep,
        gate_orient=bool(min(frame_axis_sep, frame_updown_sep) > FRAME_SEP_MIN),
        orient_near_scanner_ad=bool(np.isfinite(orient_corr) and orient_corr > ORIENT_CORR_MIN),
    )
    q["gates_ok"] = bool(q["gate_topology"] and q["gate_flips"] and q["gate_orient"])
    if np.isfinite(frag_max) and frag_max > FRAG_WARN:
        log.warning("  [%s] the cuts left %.0f %% of %s detached from the rest of itself; a cut should "
                    "run beside an area, not across it (large folded areas that must be opened are the "
                    "exception)", hemi, 100 * frag_max, frag_parcel)
    if arrays:
        q["_arrays"] = dict(idx=idx, fk=fk, p2=p2, e=e, r=r, nb=nb, nbe=nbe, v=v,
                            loop=display_loop(hemi, p2, fk))
    return q


def passes_gates(q: dict, hot_max: float = HOT_MAX) -> bool:
    """All hemisphere gates including the area-packing gate."""
    return bool(q["gates_ok"] and q["hot_pack"] <= hot_max)


def quality_line(q: dict, hot_max: float = HOT_MAX) -> str:
    """One-line human-readable summary of a :func:`hemi_quality` record."""
    return (f"gates {'OK' if passes_gates(q, hot_max) else 'FAIL'} - loops {q['loops']}, isolated "
            f"{q['isolated']}, components {q['components']}, flips {q['flips']}, "
            f"frame {min(q['frame_axis_sep'], q['frame_updown_sep']):.2f} (corr to scanner A/D "
            f"{q['orient_corr']:.2f}), "
            f"hot_pack {q['hot_pack']:.2f} @ {q['hot_parcel']} (max {hot_max:g}), p95 {q['stretch_p95']:.3f}, "
            f"p99 {q['stretch_p99']:.3f}, cortex loss {q['cortex_loss_area_pct']:.2f} % "
            f"({q['cortex_kept']}/{q['cortex_nvert']} cortex vertices kept)")
