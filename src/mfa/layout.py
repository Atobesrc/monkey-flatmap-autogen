"""Parcel layouts of flat maps: seam sides, reference layouts, cross-hemisphere / cross-subject comparison.

Nothing in this module changes flat coordinates.  Layouts are compared after a
similarity Procrustes alignment (rotation + uniform scale + translation, no
reflection) of normalised parcel centroids; the *side* of the temporal seam a
parcel lies on is a property of the cut alone and is reconstructed from the
patch, so it is identical before and after flattening.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from collections.abc import Mapping, Sequence

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .flatten import HOT_MAX, hemi_quality, silhouette_norm
from .mesh import (
    boundary_loops,
    face_area3,
    majority_face_label,
    patch_faces,
    resample_closed_loop,
    unique_edges,
    vertex_adjacency,
)
from .patch_io import read_patch
from .recipe import SEAM_PARCELS, TEMPORAL_CORRIDOR_PARCELS
from .surface import DEFAULT_PARCELLATION, Hemi, Parcellation
from .utils import json_dump, json_load

log = logging.getLogger("mfa")

LAYOUT_IGNORE = {"Unknown", "unknown", "???", "Medial_wall", "medialwall", "corpuscallosum"}
REFERENCE_FORMAT = "parcel_reference_layout/1"
#: A parcel's seam side counts only if at least 60/40 of its area is on one bank.
SIDE_CONF_MIN = 0.2


# ---------------------------------------------------------------- similarity
def fit_similarity(src: np.ndarray, dst: np.ndarray, w: np.ndarray | None = None, scale: bool = True,
                   reflect: bool = False) -> dict:
    """Umeyama similarity fit ``dst ~ s R src + t`` (optionally weighted).

    Returns ``dict(s, R, t, angle_deg)``; no reflection unless ``reflect=True``.
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if w is None:
        w = np.ones(len(src))
    w = w / w.sum()
    ms, md = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    S, D = src - ms, dst - md
    Hm = (w[:, None] * D).T @ S
    U, sv, Vt = np.linalg.svd(Hm)
    d = np.ones(2)
    if not reflect and np.linalg.det(U @ Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    s = float((sv * d).sum() / (w[:, None] * S**2).sum()) if scale else 1.0
    t = md - s * R @ ms
    return dict(s=s, R=R, t=t, angle_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))


def apply_similarity(T: Mapping, P: np.ndarray) -> np.ndarray:
    """Apply a :func:`fit_similarity` transform to points ``(k, 2)``."""
    R = np.asarray(T["R"], float)
    return (T["s"] * (R @ np.asarray(P, float).T)).T + np.asarray(T["t"], float)


# ---------------------------------------------------------------- seam sides
def slit_sides(v: np.ndarray, f: np.ndarray, iscortex: np.ndarray, corridor: np.ndarray, inpatch: np.ndarray,
               e: np.ndarray | None = None, adj=None) -> dict | None:
    """Temporal slit of a slit patch and the bank of every patch vertex.

    Flattening-independent: needs only the white surface, ``cortex.label``, the
    corridor-parcel mask and the patch vertex set.

    1. removed = cortex vertices not in the patch; their connected strips are the
       slits; the strip with most corridor vertices is the temporal slit;
    2. ``ds`` = geodesic distance along the strip from its rim end; tip = far end;
    3. banks = kept vertices adjacent to the strip.  They form one contiguous arc
       of the patch boundary root_A -> tip -> root_B.  Bank A is the *left* bank
       (outward normal x seam tangent at the nearest slit vertex; majority vote);
    4. side of a kept vertex = +1 if its geodesic distance *within the patch*
       (3D edge lengths; the slit is not crossable) to bank A is smaller than to
       bank B, else -1.  Parcels split by the slit come out about 50/50.

    Returns None if the patch has no identifiable temporal slit.
    """
    n = len(v)
    if e is None:
        e = unique_edges(f)
    if adj is None:
        adj = vertex_adjacency(e, n)
    keepf = np.all(inpatch[f], axis=1)
    used = np.zeros(n, bool)
    used[f[keepf].ravel()] = True
    removed = iscortex & ~used
    rid = np.where(removed)[0]
    if not len(rid):
        return None
    crosses = iscortex[e[:, 0]] != iscortex[e[:, 1]]
    adjwall = np.zeros(n, bool)
    adjwall[e[crosses].ravel()] = True
    ke = removed[e[:, 0]] & removed[e[:, 1]]
    g = coo_matrix((np.ones(ke.sum()), (e[ke, 0], e[ke, 1])), shape=(n, n))
    nc, cl = connected_components(g + g.T, directed=False)
    cnt = np.bincount(cl[rid], weights=corridor[rid].astype(float), minlength=nc)
    best = int(np.argmax(cnt))
    if cnt[best] <= 0:
        return None
    slit = rid[cl[rid] == best]
    src = slit[adjwall[slit]]
    if not len(src):
        rimv = np.where(adjwall & iscortex)[0]
        src = slit[[int(np.argmin(cKDTree(v[rimv]).query(v[slit])[0]))]]
    ks = ke & (cl[e[:, 0]] == best)
    Ls = np.linalg.norm(v[e[ks, 0]] - v[e[ks, 1]], axis=1)
    gs = coo_matrix((np.r_[Ls, Ls], (np.r_[e[ks, 0], e[ks, 1]], np.r_[e[ks, 1], e[ks, 0]])), shape=(n, n)).tocsr()
    ds = dijkstra(gs, indices=src, min_only=True)[slit]
    ds[~np.isfinite(ds)] = 0.0
    dfull = np.full(n, np.inf)
    dfull[slit] = ds
    isslit = np.zeros(n, bool)
    isslit[slit] = True
    bank = np.unique(adj[slit].indices)
    bank = bank[used[bank]]
    near = np.empty(len(bank), int)
    bd = np.empty(len(bank))
    for i, b in enumerate(bank):
        nbv = adj[b].indices
        nbv = nbv[isslit[nbv]]
        j = int(np.argmin(dfull[nbv]))
        near[i], bd[i] = nbv[j], dfull[nbv[j]]
    ki = np.where(inpatch)[0]
    remap = np.full(n, -1)
    remap[ki] = np.arange(len(ki))
    fk = remap[f[keepf]]
    loops = boundary_loops(fk)
    loop = ki[loops[0]]
    pos = np.full(n, -1)
    pos[loop] = np.arange(len(loop))
    lb = pos[bank]
    ok = lb >= 0
    bank, bd, near, lb = bank[ok], bd[ok], near[ok], lb[ok]
    if len(bank) < 4:
        return None
    order = np.argsort(lb)
    lbs = lb[order]
    gaps = np.diff(np.r_[lbs, lbs[0] + len(loop)])
    start = (int(np.argmax(gaps)) + 1) % len(lbs)
    order = np.r_[order[start:], order[:start]]
    bank, bd, near = bank[order], bd[order], near[order]
    tip = int(np.argmax(bd))
    A, B = np.arange(0, tip + 1), np.arange(tip, len(bank))
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros((n, 3))
    for i in range(3):
        np.add.at(vn, f[:, i], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    tang = np.zeros((n, 3))
    for s in slit:
        nbv = adj[s].indices
        nbv = nbv[isslit[nbv]]
        if len(nbv):
            lo, hi = nbv[np.argmin(dfull[nbv])], nbv[np.argmax(dfull[nbv])]
            tang[s] = v[hi] - v[lo] if hi != lo else np.zeros(3)
    tn = np.linalg.norm(tang[slit], axis=1)
    if (tn > 0).sum() >= 2 and (tn == 0).any():  # fill flat spots
        good = slit[tn > 0]
        tang[slit[tn == 0]] = tang[good[cKDTree(v[good]).query(v[slit[tn == 0]])[1]]]
    left = np.cross(vn[near], tang[near])
    vote = np.sign((left * (v[bank] - v[near])).sum(1))
    a_is_left = vote[A].sum() >= vote[B].sum()
    bank_a, bank_b = (bank[A], bank[B]) if a_is_left else (bank[B][::-1], bank[A][::-1])
    kp = used[e[:, 0]] & used[e[:, 1]]
    Lp = np.linalg.norm(v[e[kp, 0]] - v[e[kp, 1]], axis=1)
    gp = coo_matrix((np.r_[Lp, Lp], (np.r_[e[kp, 0], e[kp, 1]], np.r_[e[kp, 1], e[kp, 0]])), shape=(n, n)).tocsr()
    dA = dijkstra(gp, indices=bank_a, min_only=True)
    dB = dijkstra(gp, indices=bank_b, min_only=True)
    vside = np.zeros(n, int)
    vside[used] = np.where(dA[used] <= dB[used], 1, -1)
    return dict(slit=slit, ds=ds, bank=bank, bd=bd, bank_a=bank_a, bank_b=bank_b,
                tip_vertex=int(bank[tip]), root_a=int(bank_a[0]), root_b=int(bank_b[-1]),
                length_mm=float(ds.max()), vside=vside, dA=dA, dB=dB, used=used,
                a_is_left=bool(a_is_left), loop=loop)


def parcel_side_stats(fside: np.ndarray, farea: np.ndarray, m: np.ndarray) -> dict:
    """Side statistics of one parcel (faces ``m``): area fraction on bank A, majority side, confidence."""
    w = farea[m]
    wp, wm = w[fside[m] > 0].sum(), w[fside[m] < 0].sum()
    if wp + wm <= 0:
        return dict(side=0, side_frac=np.nan, side_conf=0.0)
    frac = float(wp / (wp + wm))
    return dict(side=int(1 if frac > 0.5 else -1), side_frac=frac, side_conf=float(abs(2 * frac - 1)))


def flat_layout(sd: str, hemi: str, name: str, parc: Parcellation = DEFAULT_PARCELLATION, n_outline: int = 512) -> dict:
    """Parcel layout of one flattened hemisphere in the normalised canonical display frame.

    Display frame ``X = patch_y, Y = -patch_x`` (what pycortex draws), normalised so
    that ``sqrt(total flat area) = 100`` and the area-weighted patch centroid is at
    the origin.  Per parcel (faces with majority label): normalised centroid, area
    fraction, ``side_frac`` (fraction of its area on bank A of the temporal slit),
    ``side`` (+1 A / -1 B), ``side_conf`` = |2 side_frac - 1|, ``cent_side``,
    ``touches_seam``.  The slit banks and the outline are stored for figures.
    """
    fn = f"{sd}/surf/{hemi}.{name}.flat.patch.3d"
    idx, _bflag, xyz = read_patch(fn)
    v, f = nib.freesurfer.read_geometry(f"{sd}/surf/{hemi}.white")
    n = len(v)
    inpatch = np.zeros(n, bool)
    inpatch[idx] = True
    remap = np.full(n, -1)
    remap[idx] = np.arange(len(idx))
    ff = f[np.all(inpatch[f], axis=1)]
    fk = remap[ff]
    P = np.c_[xyz[:, 1], -xyz[:, 0]]
    a, b, c = P[fk[:, 0]], P[fk[:, 1]], P[fk[:, 2]]
    farea = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    fcent = (a + b + c) / 3.0
    area = float(farea.sum())
    scale = np.sqrt(area) / 100.0
    center = (fcent * farea[:, None]).sum(0) / area

    def N(Q: np.ndarray) -> np.ndarray:
        return (np.asarray(Q, float) - center) / scale

    lab, names = parc.read(sd, hemi)
    maj = majority_face_label(lab[ff])
    cortex = nib.freesurfer.read_label(f"{sd}/label/{hemi}.cortex.label")
    iscortex = np.zeros(n, bool)
    iscortex[cortex] = True
    corridor = np.isin(lab, [names.index(x) for x in TEMPORAL_CORRIDOR_PARCELS if x in names])
    S = slit_sides(v, f, iscortex, corridor, inpatch)
    seam, fside = None, None
    if S is not None:
        vs = S["vside"][ff]
        fside = np.where(vs.sum(1) >= 0, 1, -1)
        PA, PB = P[remap[S["bank_a"]]], P[remap[S["bank_b"]]]
        tipxy = P[remap[S["tip_vertex"]]]
        seam = dict(root=N(0.5 * (PA[0] + PB[-1])), tip=N(tipxy), root_a=N(PA[0]), root_b=N(PB[-1]),
                    length_mm=S["length_mm"], bank_a_xy=N(PA), bank_b_xy=N(PB[::-1]),
                    n_vertices=int(len(S["slit"])), vertices=S["slit"].astype(int),
                    bank_vertices=S["bank"].astype(int), bank_a=S["bank_a"].astype(int),
                    bank_b=S["bank_b"].astype(int), tip_vertex=S["tip_vertex"], a_is_left=S["a_is_left"])
    parcels = {}
    for li in np.unique(maj):
        if li < 0 or names[li] in LAYOUT_IGNORE:
            continue
        m = maj == li
        w = farea[m]
        cent = (fcent[m] * w[:, None]).sum(0) / w.sum()
        rec = dict(cent=N(cent), area_frac=float(w.sum() / area), nface=int(m.sum()),
                   side=0, side_frac=np.nan, side_conf=0.0, cent_side=0, touches_seam=False)
        if seam is not None:
            rec.update(parcel_side_stats(fside, farea, m))
            fi = np.where(m)[0]
            rec["cent_side"] = int(fside[fi[np.argmin(np.linalg.norm(fcent[fi] - cent, axis=1))]])
            pv = np.unique(ff[m].ravel())
            rec["touches_seam"] = bool(np.isin(pv, seam["bank_vertices"]).any())
        parcels[names[li]] = rec
    loop = boundary_loops(fk)[0]
    outline = N(resample_closed_loop(P[loop], n_outline))
    return dict(hemi=hemi, name=name, flat_area=area, sqrt_area=float(np.sqrt(area)), center=center,
                scale_mm_per_unit=float(scale), parcels=parcels, seam=seam, outline=outline,
                nvert=int(len(idx)), nface=int(len(fk)), _face_side=fside)


def patch_parcel_sides(H: Hemi, ni: np.ndarray) -> dict | None:
    """Seam sides of the parcels of a not-yet-flattened slit patch (``ni`` = kept vertex ids).

    Same definition as :func:`flat_layout` (via :func:`slit_sides`), by
    white-surface face area, so a cut design's side signature can be tested
    before spending a flatten on it.
    """
    inpatch = np.zeros(H.n, bool)
    inpatch[ni] = True
    corridor = H.inpar(*[x for x in TEMPORAL_CORRIDOR_PARCELS if x in H.P])
    S = slit_sides(H.v, H.f, H.iscortex, corridor, inpatch, e=H.e, adj=H.adj)
    if S is None:
        return None
    ff = H.f[np.all(inpatch[H.f], axis=1)]
    fa = face_area3(H.v, ff)
    maj = majority_face_label(H.lab[ff])
    fside = np.where(S["vside"][ff].sum(1) >= 0, 1, -1)
    out = {}
    for li in np.unique(maj):
        if li < 0 or H.names[li] in LAYOUT_IGNORE:
            continue
        out[H.names[li]] = parcel_side_stats(fside, fa, maj == li)
    return out


def side_agreement(cp: Mapping, rp: Mapping, seam_parcels: Sequence[str] = SEAM_PARCELS,
                   conf_min: float = SIDE_CONF_MIN) -> dict:
    """Seam-side agreement of candidate parcel stats ``cp`` with reference stats ``rp``.

    Gate parcels = seam parcels whose reference side is defined (confidence >=
    ``conf_min``).  Per gate parcel: 1 = same majority side, 0.5 = the candidate
    parcel is split by the seam, 0 = confidently on the other bank.
    ``ref_side_dist`` = mean |side_frac_cand - side_frac_ref| over the seam parcels.
    """
    gate = [q for q in seam_parcels if q in cp and q in rp and int(rp[q].get("side", 0)) != 0
            and float(rp[q].get("side_conf", 0.0)) >= conf_min]
    agree = {}
    for q in gate:
        same = int(cp[q].get("side", 0)) == int(rp[q]["side"])
        conf = float(cp[q].get("side_conf", 0.0))
        agree[q] = 1.0 if same else (0.5 if conf < conf_min else 0.0)
    dist = [abs(float(cp[q].get("side_frac", np.nan)) - float(rp[q].get("side_frac", np.nan)))
            for q in seam_parcels if q in cp and q in rp]
    dist = [d for d in dist if np.isfinite(d)]
    sides = {q: dict(cand=int(cp[q].get("side", 0)), ref=int(rp[q].get("side", 0)),
                     cand_frac=float(cp[q].get("side_frac", np.nan)), ref_frac=float(rp[q].get("side_frac", np.nan)))
             for q in seam_parcels if q in cp and q in rp}
    return dict(ref_side_agree=float(np.mean(list(agree.values()))) if gate else np.nan,
                ref_side_ok=bool(all(a > 0 for a in agree.values())) if gate else True,
                ref_side_mismatch=[q for q in gate if agree[q] == 0],
                ref_side_split=[q for q in gate if agree[q] == 0.5],
                ref_side_dist=float(np.mean(dist)) if dist else np.nan,
                ref_side_parcels=gate, ref_sides=sides)


def reference_metrics(lay: Mapping, ref: Mapping, seam_parcels: Sequence[str] = SEAM_PARCELS,
                      conf_min: float = SIDE_CONF_MIN) -> dict:
    """Score a :func:`flat_layout` against a reference hemisphere layout.

    Similarity Procrustes of the candidate's normalised centroids onto the
    reference's over all common parcels: ``ref_rms`` (all parcels, reference units
    = 1 % of sqrt(flat area)), ``ref_temporal_rms`` (seam parcels), plus the
    :func:`side_agreement` fields.
    """
    rp, cp = ref["parcels"], lay["parcels"]
    common = sorted(set(rp) & set(cp))
    if len(common) < 3:
        return dict(ref_rms=np.inf, ref_temporal_rms=np.inf, ref_side_agree=0.0, ref_side_ok=False,
                    ref_side_mismatch=list(seam_parcels), ref_side_split=[], ref_side_dist=np.nan,
                    ref_side_parcels=[], ref_resid={}, ref_n=len(common))
    src = np.array([cp[q]["cent"] for q in common], float)
    dst = np.array([rp[q]["cent"] for q in common], float)
    T = fit_similarity(src, dst, scale=True, reflect=False)
    res = np.linalg.norm(apply_similarity(T, src) - dst, axis=1)
    resid = {q: float(r) for q, r in zip(common, res)}
    temp = [q for q in seam_parcels if q in resid]
    out = dict(ref_rms=float(np.sqrt(np.mean(res**2))), ref_mean=float(res.mean()), ref_max=float(res.max()),
               ref_temporal_rms=float(np.sqrt(np.mean([resid[q] ** 2 for q in temp]))) if temp else np.nan,
               ref_resid=resid, ref_n=len(common),
               ref_T=dict(s=T["s"], angle_deg=T["angle_deg"], t=[float(x) for x in T["t"]]))
    out.update(side_agreement(cp, rp, seam_parcels, conf_min))
    return out


def load_reference_layout(fn: str) -> dict:
    """Load a reference layout written by :func:`write_reference_layout`."""
    ref = json_load(fn)
    if ref.get("format") != REFERENCE_FORMAT:
        raise ValueError(f"{fn}: not a {REFERENCE_FORMAT} file")
    for h in ref["hemi"].values():
        for q in h["parcels"].values():
            q["cent"] = np.asarray(q["cent"], float)
        if h.get("seam"):
            for k in ("bank_a_xy", "bank_b_xy"):
                h["seam"][k] = np.asarray(h["seam"][k], float)
    return ref


def write_reference_layout(sd: str, subject: str, name: str, hemis: Sequence[str], out_fn: str,
                           parc: Parcellation = DEFAULT_PARCELLATION, design: Mapping | None = None) -> dict:
    """Store the parcel layout of ``surf/<hemi>.<name>.flat.patch.3d`` as a reference for other subjects.

    The seam is reconstructed from the patch; ``design`` (e.g. the recipe) is
    recorded for documentation only.
    """
    hemi_layouts = {}
    for hemi in hemis:
        lay = flat_layout(sd, hemi, name, parc)
        hemi_layouts[hemi] = lay
        s = lay["seam"]
        log.info("  [%s] layout: %d parcels, flat area %.0f mm2 (1 unit = %.3f mm); temporal slit %d removed "
                 "verts, %.1f mm, bank A = %s of the seam", hemi, len(lay["parcels"]), lay["flat_area"],
                 lay["scale_mm_per_unit"], s["n_vertices"] if s else 0, s["length_mm"] if s else 0,
                 "left" if s and s["a_is_left"] else "right")
        for q in SEAM_PARCELS:
            if q in lay["parcels"]:
                p = lay["parcels"][q]
                log.info("     %-5s side %+d (frac+ %.2f, conf %.2f, centroid side %+d, touches seam %s) cent %s",
                         q, p["side"], p["side_frac"], p["side_conf"], p["cent_side"], p["touches_seam"],
                         np.round(p["cent"], 1))
    ref = dict(format=REFERENCE_FORMAT, subject=subject, name=name, annot=parc.annot,
               created=time.strftime("%Y-%m-%d %H:%M"), design=dict(design) if design else None,
               frame="pycortex display (X = patch_y, Y = -patch_x), normalised: sqrt(flat area) = 100, "
                     "area-weighted centroid at the origin",
               seam_parcels=list(SEAM_PARCELS), side_conf_min=SIDE_CONF_MIN, hemi=hemi_layouts)
    os.makedirs(os.path.dirname(os.path.abspath(out_fn)), exist_ok=True)
    json_dump(ref, out_fn)
    log.info("reference layout -> %s", out_fn)
    return ref


# ---------------------------------------------------------------- cross comparison
def cross_side_disagreement(pa: Mapping, pb: Mapping, seam_parcels: Sequence[str] = SEAM_PARCELS,
                            conf_min: float = SIDE_CONF_MIN) -> dict:
    """Symmetric seam-side disagreement of two hemispheres' parcel side stats.

    ``side_dist`` = mean |side_frac_a - side_frac_b| over the common seam parcels
    (0 = identical bank split, 1 = opposite banks); ``n_mismatch`` = parcels
    confidently on opposite banks.
    """
    d, n_mis, mis = [], 0, []
    for q in seam_parcels:
        if q in pa and q in pb:
            fa, fb = float(pa[q].get("side_frac", np.nan)), float(pb[q].get("side_frac", np.nan))
            if np.isfinite(fa) and np.isfinite(fb):
                d.append(abs(fa - fb))
            if (int(pa[q].get("side", 0)) * int(pb[q].get("side", 0)) < 0
                    and float(pa[q].get("side_conf", 0)) >= conf_min and float(pb[q].get("side_conf", 0)) >= conf_min):
                n_mis += 1
                mis.append(q)
    return dict(side_dist=float(np.mean(d)) if d else np.nan, n_mismatch=int(n_mis), mismatch=mis)


def cross_layout_rms(la: Mapping, lb: Mapping, seam_parcels: Sequence[str] = SEAM_PARCELS) -> dict:
    """Similarity Procrustes of layout ``lb``'s normalised centroids onto ``la``'s.

    RMS residual over all common parcels and over the seam parcels (units:
    sqrt(flat area) = 100), per-parcel residuals and the transform.
    """
    pa, pb = la["parcels"], lb["parcels"]
    common = sorted(set(pa) & set(pb))
    if len(common) < 3:
        return dict(rms=np.inf, temporal_rms=np.inf, resid={}, T=None, common=common)
    src = np.array([pb[q]["cent"] for q in common], float)
    dst = np.array([pa[q]["cent"] for q in common], float)
    T = fit_similarity(src, dst, scale=True, reflect=False)
    res = np.linalg.norm(apply_similarity(T, src) - dst, axis=1)
    resid = {q: float(x) for q, x in zip(common, res)}
    temp = [resid[q] for q in seam_parcels if q in resid]
    return dict(rms=float(np.sqrt(np.mean(res**2))), mean=float(res.mean()), median=float(np.median(res)),
                max=float(res.max()), temporal_rms=float(np.sqrt(np.mean(np.square(temp)))) if temp else np.nan,
                temporal_mean=float(np.mean(temp)) if temp else np.nan,
                temporal_max=float(np.max(temp)) if temp else np.nan, resid=resid, T=T, common=common)


class FlatDesign:
    """One flattened hemisphere in the normalised display frame, with parcel labels per face.

    ``P`` are the normalised coordinates (:func:`flat_layout` frame), ``fk`` the
    patch faces, ``maj`` the majority parcel index per face; :meth:`mirrored`
    flips X (right -> left frame) and swaps the seam banks.
    """

    def __init__(self, sd: str, hemi: str, name: str, parc: Parcellation = DEFAULT_PARCELLATION,
                 lay: dict | None = None):
        self.sd, self.hemi, self.name = sd, hemi, name
        self.lay = lay if lay is not None else flat_layout(sd, hemi, name, parc)
        idx, _, xyz = read_patch(f"{sd}/surf/{hemi}.{name}.flat.patch.3d")
        v, f = nib.freesurfer.read_geometry(f"{sd}/surf/{hemi}.white")
        self.idx = idx
        self.fk = patch_faces(idx, f, len(v))
        self.P = (np.c_[xyz[:, 1], -xyz[:, 0]] - self.lay["center"]) / self.lay["scale_mm_per_unit"]
        lab, self.names = parc.read(sd, hemi)
        self.maj = majority_face_label(lab[idx][self.fk])
        self.mm_per_unit = float(self.lay["scale_mm_per_unit"])
        self.parcels = self.lay["parcels"]
        self.outline = self.lay["outline"]

    def mirrored(self) -> FlatDesign:
        """Mirror image (X -> -X); the seam banks swap, so ``side -> -side``."""
        o = copy.copy(self)
        o.P = self.P.copy()
        o.P[:, 0] = -o.P[:, 0]
        o.lay = dict(self.lay)
        o.lay["parcels"] = {q: dict(p, cent=np.array([-p["cent"][0], p["cent"][1]]), side=-int(p["side"]),
                                    side_frac=(1.0 - p["side_frac"]) if np.isfinite(p["side_frac"]) else np.nan)
                            for q, p in self.lay["parcels"].items()}
        o.parcels = o.lay["parcels"]
        o.outline = self.outline * np.array([-1.0, 1.0])
        return o

    def outline_segments(self, T: Mapping | None = None) -> np.ndarray:
        """``(k, 2, 2)`` line segments of the parcel borders plus the patch boundary."""
        fk, maj = self.fk, self.maj
        P = self.P if T is None else apply_similarity(T, self.P)
        E = np.concatenate([fk[:, [0, 1]], fk[:, [1, 2]], fk[:, [0, 2]]])
        lab = np.concatenate([maj, maj, maj])
        Es = np.sort(E, axis=1)
        order = np.lexsort((Es[:, 1], Es[:, 0]))
        Es, lab = Es[order], lab[order]
        same = np.all(Es[1:] == Es[:-1], axis=1)
        first = np.where(same)[0]
        border = first[lab[first] != lab[first + 1]]
        cnt = np.ones(len(Es), int)
        cnt[first] += 1
        cnt[first + 1] += 1
        bnd = np.where(cnt == 1)[0]
        sel = np.unique(np.r_[border, bnd])
        return P[Es[sel]]

    def label_grid(self, xs: np.ndarray, ys: np.ndarray, T: Mapping | None = None, k: int = 24) -> np.ndarray:
        """Majority parcel index (-1 outside) at every grid point after the optional similarity ``T``."""
        P = self.P if T is None else apply_similarity(T, self.P)
        fk = self.fk
        C = P[fk].mean(1)
        G = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        _, nn = cKDTree(C).query(G, k=k)
        out = np.full(len(G), -1)
        A, B, Cc = P[fk[:, 0]], P[fk[:, 1]], P[fk[:, 2]]
        for j in range(k):
            fi = nn[:, j]
            a, b, c = A[fi], B[fi], Cc[fi]
            v0, v1, v2 = b - a, c - a, G - a
            d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
            d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
            den = d00 * d11 - d01 * d01
            den = np.where(np.abs(den) < 1e-12, 1e-12, den)
            vv = (d11 * d20 - d01 * d21) / den
            ww = (d00 * d21 - d01 * d20) / den
            uu = 1.0 - vv - ww
            inside = (uu >= -1e-6) & (vv >= -1e-6) & (ww >= -1e-6)
            upd = inside & (out < 0)
            out[upd] = self.maj[fi[upd]]
        return out.reshape(len(xs), len(ys))


def compare_flat(FA: FlatDesign, FB: FlatDesign, seam_parcels: Sequence[str] = SEAM_PARCELS,
                 step: float = 0.5) -> dict:
    """Layout comparison of two flattened hemispheres in the same frame.

    Same hemisphere of two subjects, or lh vs mirrored rh: similarity Procrustes
    of B's centroids onto A's, centroid displacement (units and mm in A's map),
    Dice of the parcels after alignment (rasterised at ``step`` units), pixel
    label agreement and seam-side (dis)agreement.
    """
    X = cross_layout_rms(FA.lay, FB.lay, seam_parcels)
    T = X["T"]
    out = dict(n_common=len(X["common"]), rms_units=X["rms"], temporal_rms_units=X["temporal_rms"],
               disp_units=dict(mean=X.get("mean", np.nan), median=X.get("median", np.nan), max=X.get("max", np.nan)),
               mm_per_unit=FA.mm_per_unit, T=T, resid_units=X["resid"])
    mm = FA.mm_per_unit
    r = np.array(list(X["resid"].values())) * mm if X["resid"] else np.array([np.nan])
    temp = np.array([X["resid"][q] for q in seam_parcels if q in X["resid"]]) * mm
    out["disp_mm"] = dict(mean=float(np.mean(r)), median=float(np.median(r)), max=float(np.max(r)),
                          temporal_mean=float(np.mean(temp)) if len(temp) else np.nan,
                          temporal_max=float(np.max(temp)) if len(temp) else np.nan,
                          n_within_3mm=int((r <= 3).sum()), n_within_5mm=int((r <= 5).sum()))
    out["disp_mm_by_parcel"] = {q: float(x * mm) for q, x in X["resid"].items()}
    dice, agree_px = {}, np.nan
    if T is not None:
        PB = apply_similarity(T, FB.P)
        lo = np.minimum(FA.P.min(0), PB.min(0)) - 2
        hi = np.maximum(FA.P.max(0), PB.max(0)) + 2
        xs, ys = np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step)
        ga, gb = FA.label_grid(xs, ys), FB.label_grid(xs, ys, T)
        na = np.array(FA.names + ["_out"])[np.where(ga >= 0, ga, len(FA.names))]
        nb_ = np.array(FB.names + ["_out"])[np.where(gb >= 0, gb, len(FB.names))]
        for q in X["common"]:
            a, b = na == q, nb_ == q
            if a.sum() + b.sum():
                dice[q] = float(2 * (a & b).sum() / (a.sum() + b.sum()))
        both = (ga >= 0) & (gb >= 0)
        agree_px = float((na[both] == nb_[both]).mean()) if both.any() else np.nan
    out["dice"] = dice
    out["dice_mean"] = float(np.mean(list(dice.values()))) if dice else np.nan
    dt = [dice[q] for q in seam_parcels if q in dice]
    out["dice_temporal_mean"] = float(np.mean(dt)) if dt else np.nan
    out["pixel_label_agreement"] = agree_px
    out.update(cross_side_disagreement(FA.parcels, FB.parcels, seam_parcels))
    sa = side_agreement(FB.parcels, FA.parcels, seam_parcels)
    out["side_agree"] = sa["ref_side_agree"]
    out["sides"] = {q: dict(a=float(FA.parcels[q]["side_frac"]), b=float(FB.parcels[q]["side_frac"]))
                    for q in seam_parcels if q in FA.parcels and q in FB.parcels}
    return out


def design_metrics(label: str, names: Mapping[str, str], subjects_dir: str, hemis: Sequence[str] = ("lh", "rh"),
                   hot_max: float = HOT_MAX, parc: Parcellation = DEFAULT_PARCELLATION) -> dict:
    """Every comparison metric of one design ``{subject: patch name}`` across its subjects.

    Per hemisphere: :func:`hemi_quality` gates and distortion; per subject:
    lh-vs-rh silhouette and lh-vs-mirrored-rh layout comparison; per hemisphere:
    the cross-subject comparison (first subject = frame).  The private key
    ``_FD`` holds the :class:`FlatDesign` objects.
    """
    subs = list(names)
    out: dict = dict(label=label, names=dict(names), hemi={}, subject={}, cross={})
    FD = {}
    for s in subs:
        sd = os.path.join(subjects_dir, s)
        for h in hemis:
            q = hemi_quality(sd, h, names[s], parc)
            lay = flat_layout(sd, h, names[s], parc)
            FD[(s, h)] = FlatDesign(sd, h, names[s], parc, lay=lay)
            se = lay["seam"]
            out["hemi"][f"{s}/{h}"] = dict(
                flips=q["flips"], loops=q["loops"], isolated=q["isolated"], components=q["components"],
                p95=q["stretch_p95"], p99=q["stretch_p99"], hot_pack=q["hot_pack"], hot_parcel=q["hot_parcel"],
                loss=q["cortex_loss_area_pct"], gates_ok=bool(q["gates_ok"]), hot_ok=bool(q["hot_pack"] <= hot_max),
                orient_corr=q["orient_corr"], slit_mm=float(se["length_mm"]) if se else np.nan,
                slit_nvert=int(se["n_vertices"]) if se else 0,
                sides={p: float(lay["parcels"][p]["side_frac"]) for p in SEAM_PARCELS if p in lay["parcels"]})
        both = "lh" in hemis and "rh" in hemis
        sil = silhouette_norm(sd, names[s], names[s]) if both else dict(sil_pct=np.nan)
        lr = compare_flat(FD[(s, "lh")], FD[(s, "rh")].mirrored()) if both else None
        out["subject"][s] = dict(sil_pct=sil["sil_pct"],
                                 lh_rh=({k: x for k, x in lr.items() if k not in ("T", "resid_units")} if lr else None))
    for h in hemis:
        for i in range(len(subs)):
            for j in range(i + 1, len(subs)):
                c = compare_flat(FD[(subs[i], h)], FD[(subs[j], h)])
                out["cross"][f"{subs[i]}~{subs[j]}/{h}"] = {k: x for k, x in c.items() if k != "T"}
    out["_FD"] = FD
    return out


def comparison_table(designs: Sequence[Mapping], subjects: Sequence[str], hemis: Sequence[str] = ("lh", "rh")) -> str:
    """Markdown table: rows = metrics, columns = designs (:func:`design_metrics` dicts)."""
    subs = list(subjects)
    cols = [d["label"] for d in designs]
    lines = ["| metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]

    def row(label: str, fn) -> None:
        lines.append(f"| {label} | " + " | ".join(fn(d) for d in designs) + " |")

    row("patch names", lambda d: ", ".join(f"{s}:{n}" for s, n in d["names"].items()))
    for s in subs:
        for h in hemis:
            k = f"{s}/{h}"
            row(f"{k} flips / loops / p95 / p99",
                lambda d, k=k: f"{d['hemi'][k]['flips']} / {d['hemi'][k]['loops']} / {d['hemi'][k]['p95']:.2f} / "
                               f"{d['hemi'][k]['p99']:.2f}")
            row(f"{k} hot_pack @ parcel | cortex loss %",
                lambda d, k=k: f"{d['hemi'][k]['hot_pack']:.2f} @ {d['hemi'][k]['hot_parcel']} | "
                               f"{d['hemi'][k]['loss']:.2f}")
            row(f"{k} temporal slit mm | seam parcels (frac on bank A)",
                lambda d, k=k: f"{d['hemi'][k]['slit_mm']:.0f} | " + " ".join(
                    f"{p}:{d['hemi'][k]['sides'].get(p, np.nan):.2f}" for p in SEAM_PARCELS))
    for s in subs:
        row(f"{s} lh-vs-rh silhouette % of perimeter", lambda d, s=s: f"{d['subject'][s]['sil_pct']:.3f}")
        row(f"{s} lh-vs-mirrored-rh centroid disp mean/median/max mm | Dice all / seam parcels",
            lambda d, s=s: (f"{d['subject'][s]['lh_rh']['disp_mm']['mean']:.1f} / "
                            f"{d['subject'][s]['lh_rh']['disp_mm']['median']:.1f} / "
                            f"{d['subject'][s]['lh_rh']['disp_mm']['max']:.1f} | "
                            f"{d['subject'][s]['lh_rh']['dice_mean']:.2f} / "
                            f"{d['subject'][s]['lh_rh']['dice_temporal_mean']:.2f}")
            if d["subject"][s]["lh_rh"] else "-")
    for key in designs[0]["cross"]:
        row(f"{key} centroid disp mean / median / max (mm)",
            lambda d, key=key: f"{d['cross'][key]['disp_mm']['mean']:.1f} / "
                               f"{d['cross'][key]['disp_mm']['median']:.1f} / "
                               f"{d['cross'][key]['disp_mm']['max']:.1f}")
        row(f"{key} seam-parcel disp mean / max (mm)",
            lambda d, key=key: f"{d['cross'][key]['disp_mm']['temporal_mean']:.1f} / "
                               f"{d['cross'][key]['disp_mm']['temporal_max']:.1f}")
        row(f"{key} parcels within 3 / 5 mm (of {designs[0]['cross'][key]['n_common']})",
            lambda d, key=key: f"{d['cross'][key]['disp_mm']['n_within_3mm']} / "
                               f"{d['cross'][key]['disp_mm']['n_within_5mm']}")
        row(f"{key} Dice mean (all) / seam parcels | pixel agreement",
            lambda d, key=key: f"{d['cross'][key]['dice_mean']:.3f} / {d['cross'][key]['dice_temporal_mean']:.3f} | "
                               f"{100 * d['cross'][key]['pixel_label_agreement']:.1f} %")
        for p in SEAM_PARCELS:
            row(f"{key}   {p} disp mm | Dice",
                lambda d, key=key, p=p: f"{d['cross'][key]['disp_mm_by_parcel'].get(p, np.nan):.1f} | "
                                        f"{d['cross'][key]['dice'].get(p, np.nan):.2f}")
        row(f"{key} layout RMS units (all / seam parcels)",
            lambda d, key=key: f"{d['cross'][key]['rms_units']:.2f} / {d['cross'][key]['temporal_rms_units']:.2f}")
        row(f"{key} seam-side agreement | side dist | opposite-bank parcels",
            lambda d, key=key: f"{d['cross'][key]['side_agree']:.2f} | {d['cross'][key]['side_dist']:.3f} | "
                               f"{d['cross'][key]['n_mismatch']}"
                               + ((" (" + ",".join(d["cross"][key]["mismatch"]) + ")")
                                  if d["cross"][key]["mismatch"] else ""))
    return "\n".join(lines) + "\n"
