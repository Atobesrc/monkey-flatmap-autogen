"""Functional category ROIs on the flat map: consensus circling, exclusivity, tiling, rendering.

Workflow (``mfa roi-consensus`` / ``roi-separate`` / ``roi-tile`` / ``roi-render``):

1. ``roi-consensus`` takes N statistical maps already registered onto the subject's canonical
   anatomical grid (each with its pycortex transform), thresholds each, keeps the vertices
   significant in ALL of them, restricts them to a set of parcels, groups nearby fragments and
   CLOSES each group into one circled region per patch cluster (interior included - what a
   hand-drawn outline produces), names the clusters posterior -> anterior, and writes them as a
   FreeSurfer annot + an ``overlays.svg`` outline layer + a vertex mask (.npy) + .json provenance.
   The mask holds exactly the vertices the figure circles, so analyses and figures agree.
2. ``roi-separate`` makes two ROI groups mutually exclusive by SYMMETRIC dominance arbitration
   of the contested vertices (each side keeps a vertex only where its own contrast beats the
   other side's by a margin; ambiguous vertices are dropped from both seeds), then re-closes
   every category with the other group's final territory forbidden: the groups end up disjoint
   while every ROI remains one connected, closed patch.
3. ``roi-tile`` is the alternative literature-style view: winner-take-all assignment of every
   vertex to the category with the largest sig value, kept only above a threshold and a margin
   over the runner-up (no closing: tiles stay mutually exclusive but can be speckled).
4. ``roi-render`` draws one or several ROI layers as colored outlines on curvature with a
   legend (optionally dashed for an ROI whose quality is in question).

Statistical maps are assumed MION-style by default (activation = signal DECREASE): every map
is sign-flipped before thresholding.  Pass ``--no-mion`` for BOLD.  All commands need pycortex
(the subject's flat surfaces and transforms) and the FreeSurfer subject (white surfaces and
the parcel annotation).
"""

from __future__ import annotations

import json
import logging
import os

import nibabel as nib
import numpy as np

log = logging.getLogger("mfa")

#: Default cluster names per hemisphere, posterior -> anterior (macaque face-patch system).
NAMES_BY_COUNT = {
    1: ["face"], 2: ["PL", "AL"], 3: ["PL", "ML", "AL"], 4: ["PL", "ML", "AL", "AM"],
    5: ["PL", "ML", "MF", "AL", "AM"], 6: ["PL", "ML", "MF", "AL", "AF", "AM"],
}

PALETTE = ["gold", "deepskyblue", "orchid", "tomato", "limegreen", "orange"]


class Mesh:
    """White-surface mesh of one hemisphere with morphological helpers.

    The adjacency keeps the per-face edge multiplicity (an interior edge contributes twice):
    ``grow``/``close``/``components`` only test non-zeros, but the majority boundary smoothing
    of ``roi-separate`` weights neighbours by it, and changing that would move validated ROI
    boundaries.
    """

    def __init__(self, sd: str, hemi: str):
        from scipy.sparse import coo_matrix
        self.V, F = nib.freesurfer.read_geometry(os.path.join(sd, "surf", hemi + ".white"))
        self.e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]])
        n = len(self.V)
        self.A = coo_matrix((np.ones(len(self.e) * 2),
                             (np.r_[self.e[:, 0], self.e[:, 1]],
                              np.r_[self.e[:, 1], self.e[:, 0]])), shape=(n, n)).tocsr()

    def grow(self, m: np.ndarray, k: int) -> np.ndarray:
        for _ in range(k):
            m = m | (self.A.dot(m.astype(np.float32)) > 0)
        return m

    def close(self, m: np.ndarray, k: int) -> np.ndarray:
        return (~self.grow(~self.grow(m, k), k)) | m

    def components(self, m: np.ndarray) -> np.ndarray:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        e2 = self.e[m[self.e[:, 0]] & m[self.e[:, 1]]]
        A2 = coo_matrix((np.ones(len(e2) * 2),
                         (np.r_[e2[:, 0], e2[:, 1]], np.r_[e2[:, 1], e2[:, 0]])),
                        shape=(len(self.V),) * 2)
        _, lab = connected_components(A2, directed=False)
        return np.where(m, lab, -1)


def circled_regions(mesh: Mesh, mask: np.ndarray, merge_rings: int, min_vertices: int,
                    forbidden: np.ndarray | None = None) -> list[tuple[np.ndarray, int, np.ndarray]]:
    """Group mask fragments that lie within 2*merge_rings of each other, drop small groups,
    then CLOSE each group just enough that it is one piece: one smooth circled region per
    patch cluster, interior included - what a hand-drawn outline produces.
    ``forbidden`` vertices are never entered (not by the seeds, not by the closing) - used to
    keep mutually exclusive ROI sets disjoint while both stay closed.
    Returns a list of ``(region_mask, n_original, centroid_xyz)``."""
    if forbidden is not None:
        mask = mask & ~forbidden
    labd = mesh.components(mesh.grow(mask, merge_rings))
    out = []
    taken = forbidden.copy() if forbidden is not None else np.zeros_like(mask)
    groups = [(int((mask & (labd == g)).sum()), g) for g in np.unique(labd[mask])]
    for n_orig, g in sorted(groups, reverse=True):
        if n_orig < min_vertices:
            continue
        orig = mask & (labd == g)
        region = orig
        for k in range(merge_rings, max(9, merge_rings + 6)):
            cl = mesh.close(orig, k) & ~taken
            labc = mesh.components(cl)
            ids = np.unique(labc[orig])
            if len(ids) == 1:
                region = cl & (labc == ids[0])
                break
        else:
            log.warning("[isle] a region stayed multi-piece after closing; kept as is")
            region = cl
        taken |= region
        out.append((region, n_orig, mesh.V[orig].mean(0)))
    return out


def arbitrate(b_sig: np.ndarray, a_max: np.ndarray, b_mask: np.ndarray, a_union: np.ndarray,
              margin: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Symmetric dominance arbitration of the vertices contested between ROI group B and the
    union of group A.  Each side keeps a contested vertex only when ITS contrast dominates the
    other side's by ``margin``; ambiguous vertices are dropped from both seed sets, so a thin
    ambiguous corridor cannot block either side's closing from reconnecting its patch (the
    forbidden-territory mechanism of the re-closing still guarantees disjoint final masks).

    Returns ``(b_seed, a_drop, stats)``: B's seed mask, the mask to remove from every A seed,
    and the contested/dominant/ambiguous counts."""
    b_dom = (b_sig - a_max) >= margin
    a_dom = (a_max - b_sig) >= margin
    contested = b_mask & a_union
    amb = contested & ~b_dom & ~a_dom
    b_seed = (b_mask & ~contested) | (contested & b_dom)
    a_drop = contested & ~a_dom
    stats = dict(contested=int(contested.sum()), b_dominant=int((contested & b_dom).sum()),
                 a_dominant=int((contested & a_dom).sum()), ambiguous=int(amb.sum()))
    return b_seed, a_drop, stats


# ---------------------------------------------------------------- shared pycortex/annot helpers
def sample_volume(cx_subject: str, xfm: str, vol: str, mion: bool) -> np.ndarray:
    """Vertex values of a volume on the subject's canonical grid (line-nearest mapper)."""
    import cortex
    d = np.asanyarray(nib.load(vol).dataobj, float)
    if d.ndim == 4:
        d = d[..., 0]
    if mion:
        d = -d
    return cortex.get_mapper(cx_subject, xfm, "line_nearest")(
        cortex.Volume(np.ascontiguousarray(d.transpose(2, 1, 0)), cx_subject, xfm)).data


def parcel_mask(sd: str, parcel_annot: str, keep: list[str]) -> tuple[np.ndarray, list[int]]:
    """(lh+rh vertex mask of the kept parcels, vertex counts per hemisphere)."""
    parc, nverts = [], []
    for h in ("lh", "rh"):
        lab, _, nm = nib.freesurfer.read_annot(
            os.path.join(sd, "label", "%s.%s.annot" % (h, parcel_annot)))
        ids = [i for i, n in enumerate(nm) if n.decode() in keep]
        parc.append(np.isin(lab, ids))
        nverts.append(len(lab))
    return np.concatenate(parc), nverts


def write_union_annot(sd: str, annot_name: str, hemi_labels: list[np.ndarray],
                      union: list[str]) -> None:
    """Write one annot per hemisphere with a SHARED colortable (index 0 = 'unknown').
    The colortable must be identical across hemispheres or ``overlay-parcels`` refuses it."""
    ctab = np.zeros((len(union), 5), int)
    ctab[:, :3] = [[60, 60, 60]] + [[255, 223, 0]] * (len(union) - 1)
    ctab[1:, 2] = np.arange(1, len(union))    # unique colors so the annot is well-formed
    ctab[:, 4] = ctab[:, 0] + ctab[:, 1] * 256 + ctab[:, 2] * 256 ** 2
    for h, hl in zip(("lh", "rh"), hemi_labels):
        nib.freesurfer.write_annot(os.path.join(sd, "label", "%s.%s.annot" % (h, annot_name)),
                                   np.asarray(hl, int), ctab, union, fill_ctab=False)


def draw_roi_layer(cx_subject: str, fs_subject: str, subjects_dir: str, annot_name: str,
                   layer: str, labels: bool = False, font_size: str = "18.0",
                   smooth: int = 2) -> None:
    """Replace one overlays.svg layer with the outlines of an ROI annot.

    Any existing layer of the same name is removed FIRST: ``draw_parcels`` replaces only
    same-named shape groups, so renamed ROIs (or duplicate layers from old runs) would leave
    stale outlines and labels rendering on every future flatmap."""
    import cortex
    from lxml import etree

    from .overlays import INKNS, draw_parcels
    from .surface import Parcellation

    svgfile = os.path.join(cortex.database.default_filestore, cx_subject, "overlays.svg")
    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    svg = etree.parse(svgfile, parser)
    root = svg.getroot()
    stale = [g for g in root if isinstance(g.tag, str) and g.tag.endswith("g")
             and g.get("{%s}label" % INKNS) == layer]
    for g in stale:
        root.remove(g)
    if stale:
        svg.write(svgfile, xml_declaration=True, encoding="utf-8")
        log.info("[svg] removed %d existing %r layer(s) before redrawing", len(stale), layer)
    draw_parcels(cx_subject, fs_subject, subjects_dir, Parcellation(annot=annot_name),
                 layer=layer, skip=("unknown",), min_faces=4, font_size=font_size,
                 min_label_share=(0.10 if labels else 1.1), smooth=smooth)


def _overlays_svg(cx_subject: str) -> str:
    import cortex
    return os.path.join(cortex.database.default_filestore, cx_subject, "overlays.svg")


# ---------------------------------------------------------------- roi-consensus
def consensus(a) -> None:
    """``mfa roi-consensus``: consensus of N maps -> circled, named ROIs (annot + layer + mask)."""
    import cortex

    cx = a.pycortex_subject or a.subject.replace("sub-", "")
    sd = os.path.join(a.subjects_dir, a.subject)

    # 1. consensus across sessions, on the shared surface
    verts = []
    for spec in a.map:
        xfm, vol = spec.split(":", 1)
        v = sample_volume(cx, xfm, vol, a.mion)
        verts.append(v)
        log.info("[map ] %s: %d vertices >= %g", xfm, int((v >= a.thresh).sum()), a.thresh)
    cons = np.all([v >= a.thresh for v in verts], 0)
    log.info("[cons] consensus (all %d maps): %d vertices", len(verts), int(cons.sum()))
    if a.dominate:
        primary = np.min(np.vstack(verts), 0) if len(verts) > 1 else verts[0]
        for spec in a.dominate:
            xfm, vol = spec.split(":", 1)
            dv = sample_volume(cx, xfm, vol, a.mion)
            cons &= (primary - dv) >= a.dominate_margin
        log.info("[dom ] must beat %d map(s) by >= %g: %d vertices left",
                 len(a.dominate), a.dominate_margin, int(cons.sum()))
    excl = np.zeros_like(cons)
    for fn in (a.exclude_mask or []):
        excl |= np.load(fn).astype(bool)
    if excl.any():
        n0 = int(cons.sum())
        cons &= ~excl
        log.info("[excl] %d consensus vertices inside %d exclusion mask(s) removed",
                 n0 - int(cons.sum()), len(a.exclude_mask))

    # 2. restrict to the allowed parcels
    keep = [s.strip() for s in a.keep_parcels.split(",")]
    inpar, nverts = parcel_mask(sd, a.parcel_annot, keep)
    dropped = int(cons.sum() - (cons & inpar).sum())
    cons &= inpar
    log.info("[cons] outside %s: %d vertices dropped", "/".join(keep), dropped)

    # 3. drop isolated islands (surface-connected components, nearby fragments merged),
    #    name the rest posterior -> anterior; ONE shared colortable for both hemispheres
    t1 = nib.load(os.path.join(sd, "mri", "T1.mgz"))
    cras = (t1.affine @ np.r_[np.array(t1.shape) / 2.0, 1])[:3]
    final = np.zeros_like(cons)
    clusters = []
    regions_h = []
    allc = []
    off = 0
    for hi, h in enumerate(("lh", "rh")):
        m = cons[off:off + nverts[hi]]
        mesh = Mesh(sd, h)
        regs = circled_regions(mesh, m, a.merge_rings, a.min_vertices)
        regs = [(r & ~excl[off:off + nverts[hi]], n, c) for r, n, c in regs]
        log.info("[isle] %s: %d circled regions kept (groups >= %d consensus verts, fragments "
                 "within %d rings grouped, each closed to one piece)",
                 h, len(regs), a.min_vertices, 2 * a.merge_rings)
        for ri, (region, n_orig, c) in enumerate(regs):
            allc.append(dict(hemi=hi, r=ri, n=int(region.sum()), n_orig=n_orig, c=c + cras))
        regions_h.append(list(regs))
        off += nverts[hi]
    cons_any = cons
    if a.lenient:
        consL = np.all([v >= a.lenient for v in verts], 0) & inpar
        cons_any = consL
        ymax = max((d["c"][1] for d in allc), default=-1e9)
        off = 0
        for hi, h in enumerate(("lh", "rh")):
            taken = np.zeros(nverts[hi], bool)
            for region, _, _ in regions_h[hi]:
                taken |= region
            m = consL[off:off + nverts[hi]] & ~taken & ~excl[off:off + nverts[hi]]
            regs = circled_regions(Mesh(sd, h), m, a.merge_rings, a.min_vertices)
            regs = [(r & ~excl[off:off + nverts[hi]], n, c) for r, n, c in regs]
            for region, n_orig, c in regs:
                cw = c + cras
                if cw[1] > ymax + a.lenient_anterior_margin:
                    regions_h[hi].append((region, n_orig, c))
                    allc.append(dict(hemi=hi, r=len(regions_h[hi]) - 1, n=int(region.sum()),
                                     n_orig=n_orig, c=cw, lenient=True))
                    log.info("[len ] %s: anterior cluster admitted at |sig| >= %g (%d consensus "
                             "verts, centroid y %+.1f)", h, a.lenient, n_orig, cw[1])
            off += nverts[hi]
    # name by pooled posterior->anterior BANDS so both hemispheres share names at the same
    # anatomical level; a y-gap > 4 mm starts a new band.  Two clusters of one hemisphere in
    # the same band: the more lateral keeps the band name, a middle-band partner is the
    # fundus patch (MF), otherwise a numeric suffix.
    allc.sort(key=lambda d: d["c"][1])
    want = [s.strip() for s in a.names.split(",")] if a.names else None
    if want and len(want) == 1:
        for d in allc:                                        # one shared name for everything
            d["name"] = want[0]
    else:
        strict = [d for d in allc if not d.get("lenient")]
        bands: list[list[dict]] = []
        for d in strict:
            if not bands or d["c"][1] - bands[-1][-1]["c"][1] > 4.0:
                bands.append([])
            bands[-1].append(d)
        names = (want if want
                 else NAMES_BY_COUNT.get(len(bands),
                                         ["face%d" % (i + 1) for i in range(len(bands))]))
        for bi, band in enumerate(bands):
            base = names[bi] if bi < len(names) else "face%d" % (bi + 1)
            for hi in (0, 1):
                sub = sorted([d for d in band if d["hemi"] == hi],
                             key=lambda d: -abs(d["c"][0]))
                for j, d in enumerate(sub):
                    d["name"] = (base if j == 0 else
                                 ("MF" if base == "ML" and j == 1 else "%s%d" % (base, j + 1)))
        for d in allc:
            if d.get("lenient"):
                d["name"] = "AM_lenient"
    for d in allc:
        h = ("lh", "rh")[d["hemi"]]
        clusters.append(dict(hemi=h, name=d["name"], vertices=d["n"],
                             consensus_vertices=d["n_orig"], lenient=bool(d.get("lenient")),
                             centroid_ras=[round(float(x), 1) for x in d["c"]]))
        log.info("[roi ] %s %-6s %4d verts circled (%4d consensus)%s  centroid RAS "
                 "(%+6.1f, %+6.1f, %+6.1f)",
                 h, d["name"], d["n"], d["n_orig"],
                 "  LENIENT" if d.get("lenient") else "", *d["c"])
    per_hemi = [[(d["name"], regions_h[hi][d["r"]][0]) for d in allc if d["hemi"] == hi]
                for hi in (0, 1)]
    # trim pinched-off lobes that read as detached slivers on the figure: erode each region
    # by one ring, split into pieces, and drop a piece's basin when it holds fewer than
    # --min-piece-consensus CONSENSUS vertices (a lobe with plenty of consensus is a real
    # sub-patch and stays, whatever its shape)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    off = 0
    for hi, h in enumerate(("lh", "rh")):
        mesh = Mesh(sd, h)
        n = nverts[hi]
        hc = cons_any[off:off + n]
        for ri, (nm, region) in enumerate(per_hemi[hi]):
            ero = region & ~(mesh.A.dot((~region).astype(np.float32)) > 0)
            e2 = mesh.e[ero[mesh.e[:, 0]] & ero[mesh.e[:, 1]]]
            A2 = coo_matrix((np.ones(len(e2) * 2),
                             (np.r_[e2[:, 0], e2[:, 1]], np.r_[e2[:, 1], e2[:, 0]])),
                            shape=(n, n))
            _, cl = connected_components(A2, directed=False)
            ids = np.unique(cl[ero])
            if len(ids) < 2:
                continue
            scores = [(int(((region & mesh.grow(ero & (cl == k), 1)) & hc).sum()), k)
                      for k in ids]
            kept = [k for s, k in scores if s >= a.min_piece_consensus]
            if not kept:
                kept = [max(scores)[1]]
            keep_ero = ero & np.isin(cl, kept)
            new = region & mesh.grow(keep_ero, 2)
            cut = int(region.sum() - new.sum())
            if cut:
                log.info("[trim] %s %s: dropped %d vertices in pinched-off slivers (< %d "
                         "consensus verts)", h, nm, cut, a.min_piece_consensus)
                per_hemi[hi][ri] = (nm, new)
        off += n
    for hi in (0, 1):
        ds = [c for c in clusters if c["hemi"] == ("lh", "rh")[hi]]
        for c, (nm, region) in zip(ds, per_hemi[hi]):
            c["vertices"] = int(region.sum())
    union = ["unknown"]
    for rois in per_hemi:
        for nm, _ in rois:
            if nm not in union:
                union.append(nm)
    hemi_labels = []
    off = 0
    for hi in (0, 1):
        hl = np.zeros(nverts[hi], int)
        for nm, region in per_hemi[hi]:
            hl[region] = union.index(nm)
        hemi_labels.append(hl)
        final[off:off + nverts[hi]] = hl > 0
        off += nverts[hi]
    write_union_annot(sd, a.annot_name, hemi_labels, union)
    np.save(a.mask_out, final)
    meta = dict(maps=a.map, thresh=a.thresh, mion=a.mion, keep_parcels=keep,
                min_vertices=a.min_vertices, merge_rings=a.merge_rings,
                min_piece_consensus=a.min_piece_consensus, lenient=a.lenient,
                lenient_anterior_margin=a.lenient_anterior_margin,
                dominate=a.dominate, dominate_margin=a.dominate_margin,
                exclude_masks=a.exclude_mask,
                n_vertices=int(final.sum()),
                definition="circled regions: consensus closed per cluster, slivers trimmed",
                clusters=clusters)
    js = os.path.splitext(a.mask_out)[0] + ".json"
    with open(js, "w") as fp:
        json.dump(meta, fp, indent=1)
    log.info("[mask] %s  (%d vertices)  +  %s", a.mask_out, int(final.sum()), js)

    # 4. outlines (+ labels) into overlays.svg, editable in Inkscape afterwards
    draw_roi_layer(cx, a.subject, a.subjects_dir, a.annot_name, a.layer,
                   labels=a.roi_labels, font_size=str(a.font_size), smooth=a.svg_smooth)

    # 5. final figure: the ROI outlines on curvature (fill and labels optional)
    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        from cortex.quickflat.composite import add_custom
        from matplotlib.colors import ListedColormap
        vals = np.where(final, 1.0, np.nan) if a.fill else np.full(final.shape, np.nan)
        v = cortex.Vertex(vals, cx, vmin=0.5, vmax=1.5, cmap=ListedColormap(["#ffdf00"]))
        fig = cortex.quickflat.make_figure(v, with_curvature=True, with_rois=False,
                                           with_colorbar=False)
        add_custom(fig.axes[0], v, _overlays_svg(cx), a.layer, with_labels=a.roi_labels,
                   labelsize="16pt")
        fig.suptitle("consensus ROIs (layer '%s'): |sig| >= %g in all of %s"
                     % (a.layer, a.thresh, ", ".join(s.split(":")[0] for s in a.map)),
                     fontsize=13)
        fig.savefig(a.out, dpi=150, facecolor="w")
        log.info("[png ] %s", a.out)


# ---------------------------------------------------------------- roi-separate
def _parse_group(spec: str) -> dict:
    name, layer, mask, xfm, vol = spec.split(":", 4)
    return dict(name=name, layer=layer, mask_fn=mask, xfm=xfm, vol=vol)


def _rebuild(sd: str, cx: str, subjects_dir: str, fs_subject: str, nverts: list[int],
             name: str, layer: str, mask: np.ndarray, forbidden: np.ndarray, merge_rings: int,
             min_vertices: int, min_piece: int | None = None, boundary_smooth: int = 0,
             svg_smooth: int = 0, min_flat_piece: int = 0) -> np.ndarray:
    """Re-group + re-close one category with ``forbidden`` territory off limits; write annot
    + svg layer; return the final mask.  The forbidden territory can shatter a region into
    pieces a hand-drawn map would omit: pieces smaller than ``min_piece`` are dropped."""
    min_piece = min_piece or min_vertices
    final = np.zeros_like(mask)
    off = 0
    for hi, h in enumerate(("lh", "rh")):
        mesh = Mesh(sd, h)
        regs = circled_regions(mesh, mask[off:off + nverts[hi]], merge_rings, min_vertices,
                               forbidden=forbidden[off:off + nverts[hi]])
        for region, n_orig, c in regs:
            lab = mesh.components(region)
            ids, cnts = np.unique(lab[region], return_counts=True)
            keep = (ids[cnts >= min_piece] if (cnts >= min_piece).any()
                    else ids[[int(np.argmax(cnts))]])
            drop = int(region.sum() - np.isin(lab, keep)[region].sum())
            if drop:
                log.info("[part] %s %s: dropped %d vertices in pieces < %d (shattered by the "
                         "forbidden territory)", h, name, drop, min_piece)
            final[off:off + nverts[hi]] |= region & np.isin(lab, keep)
        if min_flat_piece:
            # the flat surface has CUTS: a region can be connected on the white surface yet
            # appear as separate islands on the flatmap.  Drop FLAT-components smaller than
            # min_flat_piece - they can never be drawn contiguous with their parent patch.
            import cortex
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            _, fpolys = cortex.db.get_surf(cx, "flat", hemisphere=h)
            e = np.vstack([fpolys[:, [0, 1]], fpolys[:, [1, 2]], fpolys[:, [0, 2]]])
            m = final[off:off + nverts[hi]]
            e2 = e[m[e[:, 0]] & m[e[:, 1]]]
            A2 = coo_matrix((np.ones(len(e2) * 2),
                             (np.r_[e2[:, 0], e2[:, 1]], np.r_[e2[:, 1], e2[:, 0]])),
                            shape=(nverts[hi],) * 2)
            _, fl = connected_components(A2, directed=False)
            ids, cnts = np.unique(fl[m], return_counts=True)
            keepf = (ids[cnts >= min_flat_piece] if (cnts >= min_flat_piece).any()
                     else ids[[int(np.argmax(cnts))]])
            cut = int(m.sum() - np.isin(fl, keepf)[m].sum())
            if cut:
                log.info("[flat] %s %s: dropped %d vertices in flat islands < %d (separated "
                         "from their patch by a cut)", h, name, cut, min_flat_piece)
                final[off:off + nverts[hi]] = m & np.isin(fl, keepf)
        if boundary_smooth:
            # majority smoothing rounds the mesh-edge zigzag of the boundary; it moves a few
            # vertices in or out (explicitly tolerated) and never enters forbidden territory
            m = final[off:off + nverts[hi]]
            deg = np.maximum(np.asarray(mesh.A.sum(1)).ravel(), 1)
            n0 = int(m.sum())
            for _ in range(boundary_smooth):
                frac = mesh.A.dot(m.astype(np.float32)) / deg
                m = (frac >= 0.5) & ~forbidden[off:off + nverts[hi]]
            log.info("[smoo] %s %s: boundary smoothing moved %+d vertices",
                     h, name, int(m.sum()) - n0)
            final[off:off + nverts[hi]] = m
        off += nverts[hi]
    write_union_annot(sd, layer, [final[:nverts[0]].astype(int),
                                  final[nverts[0]:].astype(int)], ["unknown", name])
    draw_roi_layer(cx, fs_subject, subjects_dir, layer, layer, labels=False,
                   font_size="11pt", smooth=svg_smooth)
    return final


def separate(a) -> None:
    """``mfa roi-separate``: make two ROI groups mutually exclusive, keeping all ROIs closed."""
    cx = a.pycortex_subject or a.subject.replace("sub-", "")
    sd = os.path.join(a.subjects_dir, a.subject)
    nverts = [len(nib.freesurfer.read_geometry(os.path.join(sd, "surf", h + ".white"))[0])
              for h in ("lh", "rh")]
    A = [_parse_group(s) for s in a.group_a]
    B = _parse_group(a.group_b)
    for g in A + [B]:
        g["mask"] = np.load(g["mask_fn"]).astype(bool)
        g["sig"] = sample_volume(cx, g["xfm"], g["vol"], a.mion)

    amax = np.max(np.vstack([g["sig"] for g in A]), 0)
    aunion = np.zeros_like(B["mask"])
    for g in A:
        aunion |= g["mask"]
    b_seed, a_drop, stats = arbitrate(B["sig"], amax, B["mask"], aunion, a.margin)
    log.info("[arb ] contested vertices %d: B-dominant %d, A-dominant %d, ambiguous %d "
             "(dropped from both seeds, margin %g)", stats["contested"], stats["b_dominant"],
             stats["a_dominant"], stats["ambiguous"], a.margin)
    a_seeds = [(g, g["mask"] & ~a_drop) for g in A]

    # close B first with the arbitrated A territory forbidden, then close each A category
    # with B's FINAL territory forbidden (A categories may still overlap each other)
    a_union_seed = np.zeros_like(b_seed)
    for _, s in a_seeds:
        a_union_seed |= s
    b_final = _rebuild(sd, cx, a.subjects_dir, a.subject, nverts, B["name"], B["layer"],
                       b_seed, a_union_seed, a.merge_rings_b or a.merge_rings, a.min_vertices,
                       min_piece=a.min_piece, boundary_smooth=a.boundary_smooth,
                       svg_smooth=a.svg_smooth, min_flat_piece=a.min_flat_piece)
    np.save(B["mask_fn"], b_final)
    results = {B["name"]: int(b_final.sum())}
    for g, seed in a_seeds:
        fin = _rebuild(sd, cx, a.subjects_dir, a.subject, nverts, g["name"], g["layer"],
                       seed, b_final, a.merge_rings, a.min_vertices,
                       boundary_smooth=a.boundary_smooth, svg_smooth=a.svg_smooth,
                       min_flat_piece=a.min_flat_piece)
        np.save(g["mask_fn"], fin)
        results[g["name"]] = int(fin.sum())
    for nm, n in results.items():
        log.info("[done] %-8s %6d vertices (disjoint across groups, closed)", nm, n)
    js = os.path.splitext(B["mask_fn"])[0] + "_separation.json"
    with open(js, "w") as fp:
        json.dump(dict(group_a=a.group_a, group_b=a.group_b, margin=a.margin,
                       mion=a.mion, min_vertices=a.min_vertices, merge_rings=a.merge_rings,
                       merge_rings_b=a.merge_rings_b, min_piece=a.min_piece,
                       counts=results, **stats,
                       definition="SYMMETRIC dominance arbitration (each side keeps contested "
                                  "vertices only when it dominates by the margin; ambiguous "
                                  "dropped from both seeds), re-closed disjoint groups"), fp,
                  indent=1)
    log.info("[json] %s", js)


# ---------------------------------------------------------------- roi-tile
def tile(a) -> None:
    """``mfa roi-tile``: winner-take-all tiling of same-session maps into exclusive labels."""
    cx = a.pycortex_subject or a.subject.replace("sub-", "")
    sd = os.path.join(a.subjects_dir, a.subject)
    names, V = [], []
    for spec in a.map:
        nm, xfm, vol = spec.split(":", 2)
        V.append(sample_volume(cx, xfm, vol, a.mion))
        names.append(nm)
    S = np.vstack(V)                                           # categories x vertices
    order = np.argsort(-S, axis=0)
    win = order[0]
    wsig = np.take_along_axis(S, order[:1], 0)[0]
    second = np.take_along_axis(S, order[1:2], 0)[0] if len(names) > 1 else np.zeros_like(wsig)

    keep = [s.strip() for s in a.keep_parcels.split(",")]
    inpar, nverts = parcel_mask(sd, a.parcel_annot, keep)

    print("margin sensitivity (winner >= %g, parcels %s):" % (a.thresh, "/".join(keep)))
    print("margin  " + "".join("%8s" % n for n in names) + "   ambiguous(sup.thr.)")
    base = (wsig >= a.thresh) & inpar
    for m in (0, 1, 2, 3, 4):
        ok = base & (wsig - second >= m)
        row = "".join("%8d" % int((ok & (win == k)).sum()) for k in range(len(names)))
        print("  %4g  %s   %8d" % (m, row, int((base & ~ok).sum())))

    ok = base & (wsig - second >= a.margin)
    labels = np.where(ok, win + 1, 0)
    # island cleanup per category (no closing: tiles must stay mutually exclusive)
    off = 0
    for hi, h in enumerate(("lh", "rh")):
        mesh = Mesh(sd, h)
        for k in range(len(names)):
            m = labels[off:off + nverts[hi]] == k + 1
            if not m.any():
                continue
            grp = mesh.components(mesh.grow(m, a.merge_rings))
            drop = np.zeros_like(m)
            for g in np.unique(grp[m]):
                if (m & (grp == g)).sum() < a.min_vertices:
                    drop |= m & (grp == g)
            if drop.any():
                labels[off:off + nverts[hi]][drop] = 0
        off += nverts[hi]
    for k, nm in enumerate(names):
        log.info("[tile] %-8s %6d vertices (after margin %g + island cleanup)",
                 nm, int((labels == k + 1).sum()), a.margin)
    np.save(a.labels_out, labels)
    meta = dict(maps=a.map, names=names, thresh=a.thresh, margin=a.margin, mion=a.mion,
                keep_parcels=keep, min_vertices=a.min_vertices, merge_rings=a.merge_rings,
                counts={nm: int((labels == k + 1).sum()) for k, nm in enumerate(names)},
                definition="winner-take-all tiling; 0 = unlabeled/ambiguous")
    js = os.path.splitext(a.labels_out)[0] + ".json"
    with open(js, "w") as fp:
        json.dump(meta, fp, indent=1)
    log.info("[mask] %s  +  %s", a.labels_out, js)

    write_union_annot(sd, a.annot_name,
                      [labels[:nverts[0]], labels[nverts[0]:]], ["unknown"] + names)
    draw_roi_layer(cx, a.subject, a.subjects_dir, a.annot_name, a.layer, labels=False,
                   font_size="11pt", smooth=a.svg_smooth)

    if a.out:
        import cortex
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.colors import ListedColormap
        from matplotlib.lines import Line2D
        cat = np.where(labels > 0, labels.astype(float), np.nan)
        v = cortex.Vertex(cat, cx, vmin=0.5, vmax=len(names) + 0.5,
                          cmap=ListedColormap(PALETTE[:len(names)]))
        fig = cortex.quickflat.make_figure(v, with_curvature=True, with_rois=True,
                                           with_labels=True, with_colorbar=False)
        handles = [Line2D([], [], marker="s", linestyle="", markersize=12,
                          markerfacecolor=PALETTE[k], markeredgecolor="none", label=nm)
                   for k, nm in enumerate(names)]
        fig.axes[0].legend(handles=handles, loc="lower center", ncol=len(names),
                           frameon=True, fontsize=14)
        fig.suptitle("winner-take-all tiling: winner >= %g and margin >= %g over runner-up "
                     "(same-session maps: %s)" % (a.thresh, a.margin, ", ".join(names)),
                     fontsize=13)
        fig.savefig(a.out, dpi=150, facecolor="w")
        log.info("[png ] %s", a.out)


# ---------------------------------------------------------------- roi-render
def render_layers(a) -> None:
    """``mfa roi-render``: colored outlines of several ROI layers on curvature, with a legend."""
    import cortex
    import matplotlib
    matplotlib.use("Agg")
    from cortex.quickflat.composite import add_custom
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    cx = a.pycortex_subject or a.subject.replace("sub-", "")
    n = sum(len(cortex.db.get_surf(cx, "flat", hemisphere=h)[0]) for h in ("lh", "rh"))
    v = cortex.Vertex(np.full(n, np.nan), cx, vmin=0, vmax=1, cmap="gray")
    fig = cortex.quickflat.make_figure(v, with_curvature=True, with_rois=a.parcels,
                                       with_labels=a.parcels, with_colorbar=False)
    svg = _overlays_svg(cx)
    handles = []
    for i, spec in enumerate(a.layer):
        parts = spec.split(":")
        name = parts[0]
        color = parts[1] if len(parts) > 1 and parts[1] else PALETTE[i % len(PALETTE)]
        label = parts[2] if len(parts) > 2 and parts[2] else name
        width = float(parts[3]) if len(parts) > 3 and parts[3] else a.linewidth
        style = parts[4].lower() if len(parts) > 4 and parts[4] else "solid"
        if style.startswith("dash"):
            # svg strokes cannot be dashed through add_custom: contour the layer's own
            # annot mask on the flatmap raster instead (same geometry, dashed stroke)
            sdl = os.path.join(a.subjects_dir, a.subject, "label")
            m = [nib.freesurfer.read_annot(os.path.join(sdl, "%s.%s.annot" % (h, name)))[0] > 0
                 for h in ("lh", "rh")]
            vm = cortex.Vertex(np.concatenate(m).astype(np.float32), cx)
            img, extent = cortex.quickflat.utils.make_flatmap_image(vm, height=1024)
            fig.axes[0].contour(np.nan_to_num(np.asarray(img, float)), levels=[0.5],
                                colors=[color], linewidths=width, linestyles="dashed",
                                extent=extent, origin="upper")
        else:
            add_custom(fig.axes[0], v, svg, name, with_labels=False,
                       linecolor=to_rgba(color), linewidth=width)
        handles.append(Line2D([], [], color=color, linewidth=max(1.5, 3 * width / a.linewidth),
                              linestyle="--" if style.startswith("dash") else "-", label=label))
    fig.axes[0].legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True,
                       fontsize=14, borderpad=0.6, handlelength=1.6)
    if a.title:
        fig.suptitle(a.title, fontsize=13)
    fig.savefig(a.out, dpi=150, facecolor="w")
    log.info("[png ] %s", a.out)
