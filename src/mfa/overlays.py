"""Draw a FreeSurfer parcellation into pycortex's ``overlays.svg`` as outlines and labels.

pycortex stores the flat-map drawings in ``<filestore>/<subject>/overlays.svg``.
Each layer (``rois``, ``sulci``, ...) has a ``shapes`` sub-layer with one ``<g>``
per ROI holding ``<path>`` elements and a ``labels`` sub-layer with ``<text>``
elements; ``quickflat`` renders them through Inkscape.  SVG coordinates are the
flat coordinates shifted to the origin, scaled so that the height is 1024 px,
with y flipped (the transform of ``cortex.svgoverlay.make_svg``), reproduced
here so that the paths line up.

For every parcel the flat-surface triangles whose majority label is that parcel
are split into connected pieces, the boundary loops of each piece are traced and
written as one ``<path>`` (holes become extra sub-paths).  The label goes to the
*core* of the piece (deep interior, nearest the core centroid; rotated along elongated
pieces), so
a C-shaped parcel is labelled inside, not on its border.  pycortex attaches
exactly one label per path, re-using a same-named ``<text>`` within 250 px of its
own anchor (the erosion centroid of the path); pieces too small or thin for a
label are therefore folded into the main piece of their hemisphere as extra
sub-paths, so that pycortex does not invent labels for them.  Idempotent: shapes
and labels with the same names are replaced; ``overlays.svg`` is backed up first.
Nothing about the geometry changes.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Sequence

import nibabel as nib
import numpy as np
from lxml import etree
from scipy.spatial import cKDTree

from .mesh import boundary_loops_by_edges, face_components
from .surface import DEFAULT_PARCELLATION, Parcellation
from .utils import timestamp

log = logging.getLogger("mfa")

SVGNS = "http://www.w3.org/2000/svg"
INKNS = "http://www.inkscape.org/namespaces/inkscape"
DEFAULT_SKIP = ("unknown", "Unknown", "???")


def find_layer(parent, label: str):
    """Inkscape layer ``<g inkscape:label=label>`` below ``parent`` (KeyError if absent)."""
    for g in parent.iterfind(".//{%s}g" % SVGNS):
        if g.get("{%s}label" % INKNS) == label:
            return g
    raise KeyError(label)


def svg_coords(pts: np.ndarray) -> np.ndarray:
    """pycortex's flat -> SVG pixel transform (origin at the min corner, height 1024, y flipped)."""
    q = pts[:, :2].astype(float).copy()
    q -= q.min(0)
    q *= 1024.0 / q.max(0)[1]
    q[:, 1] = 1024.0 - q[:, 1]
    return q


def _ensure_layer(svg, layer_name: str):
    root = svg.getroot()
    try:
        return find_layer(root, layer_name)
    except KeyError:
        pass
    layer = etree.SubElement(root, "{%s}g" % SVGNS, id=layer_name, style="display:inline;")
    layer.set("{%s}label" % INKNS, layer_name)
    layer.set("{%s}groupmode" % INKNS, "layer")
    for sub in ("shapes", "labels"):
        g = etree.SubElement(layer, "{%s}g" % SVGNS, id=f"{layer_name}_{sub}", style="display:inline;")
        g.set("{%s}label" % INKNS, sub)
        g.set("{%s}groupmode" % INKNS, "layer")
        if sub == "shapes":
            g.set("clip-path", "url(#edgeclip)")
    log.info("[svg] created new layer %r", layer_name)
    return layer


def label_pose(q, vs, dist, anchor, name, font_px, core_frac=0.6, max_anchor_dist=240.0,
               min_elong=2.2, other_anchors=(), min_other_dist=256.0):
    """Where and how to draw a parcel label inside one piece.

    Position: the parcel's *core* is the set of interior vertices whose distance to the border is
    at least ``core_frac`` of the maximum; the label goes on the core vertex nearest the core's
    centroid. For a compact parcel this is the deepest region; for a long strip it is the middle
    of the strip instead of whichever end happens to be widest. pycortex attaches a label to a
    path only within ~250 px of its own anchor, so candidates are restricted to that radius when
    any exist.

    Rotation: if the piece is elongated (principal-axis ratio >= ``min_elong``) and the text is
    wider than the parcel (estimated text width > 2 x max depth), the label is rotated along the
    piece's major axis, kept within (-90, 90] degrees so it reads left to right.

    pycortex attaches a text to the *first* same-named path whose anchor lies within 250 px of
    it, so a label must also stay farther than ``min_other_dist`` from the anchors of the other
    same-named paths (``other_anchors``); parcels of the two hemispheres that meet at the
    midline (e.g. V1) are the case where this matters.

    Returns (x, y, angle_deg, depth_at_label).
    """
    P = q[vs]
    near = np.hypot(P[:, 0] - anchor[0], P[:, 1] - anchor[1]) < max_anchor_dist
    if not near.any():
        near = np.ones(len(vs), bool)
    far = np.ones(len(vs), bool)
    for oa in other_anchors:
        far &= np.hypot(P[:, 0] - oa[0], P[:, 1] - oa[1]) > min_other_dist
    if (near & far).any():
        near = near & far
    dmax = float(dist[near].max())
    core = near & (dist >= core_frac * dmax)
    cc = P[core].mean(0)
    i = np.where(core)[0][np.argmin(np.hypot(P[core, 0] - cc[0], P[core, 1] - cc[1]))]
    x, y = P[i]
    angle = 0.0
    X = P - P.mean(0)
    cov = X.T @ X / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(cov)
    elong = np.sqrt(max(w[1], 1e-12) / max(w[0], 1e-12))
    text_w = 0.6 * font_px * len(name)
    if elong >= min_elong and text_w > 2.0 * dmax:
        ax, ay = V[:, 1]
        angle = float(np.degrees(np.arctan2(ay, ax)))
        if angle > 90.0:
            angle -= 180.0
        elif angle <= -90.0:
            angle += 180.0
    return float(x), float(y), angle, float(dist[i])


def draw_parcels(cx_subject: str, fs_subject: str, subjects_dir: str, parc: Parcellation = DEFAULT_PARCELLATION,
                 layer: str = "rois", skip: Sequence[str] = DEFAULT_SKIP, min_faces: int = 40,
                 font_size: str = "11pt", min_label_depth: float = 14.0, min_label_share: float = 0.10) -> dict:
    """Write parcel outlines and labels of ``label/<hemi>.<annot>.annot`` into ``overlays.svg``.

    Parameters
    ----------
    cx_subject, fs_subject, subjects_dir
        pycortex subject, FreeSurfer subject and ``$SUBJECTS_DIR``.
    parc : Parcellation
        Annotation to draw.
    layer : str
        overlays.svg layer (created if absent; ``rois`` is pycortex's default layer).
    skip : names not drawn.
    min_faces : drop parcel islands smaller than this (faces).
    font_size : label font size in the SVG.
    min_label_depth, min_label_share
        A piece thinner than ``min_label_depth`` px or holding less than
        ``min_label_share`` of the parcel gets an outline but no label.

    Returns
    -------
    dict
        ``svg`` path, ``backup`` path, ``n_paths``, ``n_labels``, ``skipped`` (unlabelled pieces),
        ``close_pairs`` (label pairs closer than three text heights).
    """
    import cortex
    from cortex.svgoverlay import _center_pts, _parse_svg_pts  # pycortex's own label-anchor logic

    pts, polys = cortex.db.get_surf(cx_subject, "flat", merge=True, nudge=True)
    q = svg_coords(pts)
    sd = os.path.join(subjects_dir, fs_subject)
    labels, names = [], None
    for h in ("lh", "rh"):
        lab, n = parc.read(sd, h)
        if names is None:
            names = n
        if n != names:
            raise ValueError("lh/rh annotation colortables differ")
        labels.append(lab)
    lab = np.concatenate(labels)
    n_lh = len(labels[0])
    if len(lab) != len(pts):
        raise ValueError(f"annotation has {len(lab)} vertices, the pycortex flat surface {len(pts)}")
    fl = lab[polys]
    face_lab = np.where(fl[:, 0] == fl[:, 1], fl[:, 0], np.where(fl[:, 1] == fl[:, 2], fl[:, 1], fl[:, 0]))

    svgfile = os.path.join(cortex.database.default_filestore, cx_subject, "overlays.svg")
    bak = f"{svgfile}.preparcels.{timestamp()}.bak"
    shutil.copy2(svgfile, bak)
    log.info("[svg] backup -> %s", bak)
    parser = etree.XMLParser(remove_blank_text=True, huge_tree=True)
    svg = etree.parse(svgfile, parser)
    layer_el = _ensure_layer(svg, layer)
    shapes = find_layer(layer_el, "shapes")
    textlayer = find_layer(layer_el, "labels")
    style_txt = ("font-family:Helvetica, sans-serif;font-size:%s;font-weight:bold;font-style:italic;"
                 "fill:white;fill-opacity:1;text-anchor:middle;filter:url(#dropshadow)" % font_size)
    font_px = float(font_size.rstrip("pt")) * 96.0 / 72.0
    placed, skipped = [], []
    n_paths = n_labels = 0
    for li, name in enumerate(names):
        if name in skip:
            continue
        fidx = np.where(face_lab == li)[0]
        if len(fidx) == 0:
            continue
        for g in shapes.findall("{%s}g" % SVGNS):
            if g.get("{%s}label" % INKNS) == name:
                shapes.remove(g)
        for t in textlayer.findall("{%s}text" % SVGNS):
            if t.text == name:
                textlayer.remove(t)
        g = etree.SubElement(shapes, "{%s}g" % SVGNS)
        g.set("{%s}label" % INKNS, name)
        g.set("{%s}groupmode" % INKNS, "layer")
        g.set("id", "roi_%s" % name.replace("/", "_").replace(" ", "_"))
        pieces = []
        for comp in face_components(polys[fidx]):
            if len(comp) < min_faces:
                continue
            cf = polys[fidx[comp]]
            loops = boundary_loops_by_edges(cf)
            if not loops:
                continue
            vs = np.unique(cf)
            bverts = np.unique(np.concatenate([np.asarray(lp) for lp in loops]))
            dist, _ = cKDTree(q[bverts]).query(q[vs])
            v0 = vs[np.argmax(dist)]
            hemi = "lh" if v0 < n_lh else "rh"
            pieces.append(dict(comp=comp, loops=loops, v0=v0, depth=float(dist.max()), hemi=hemi,
                               center=q[bverts].mean(0), share=len(comp) / float(len(fidx))))
        largest: dict[str, int] = {}
        for i, pc in enumerate(pieces):
            if pc["hemi"] not in largest or len(pc["comp"]) > len(pieces[largest[pc["hemi"]]]["comp"]):
                largest[pc["hemi"]] = i
        groups = {i: [i] for i in largest.values()}
        for i, pc in enumerate(pieces):
            if i in groups:
                continue
            if pc["depth"] < min_label_depth or pc["share"] < min_label_share:
                skipped.append(f"{pc['hemi']} {name} ({100 * pc['share']:.0f}% of parcel, {pc['depth']:.0f} px deep)")
                groups[largest[pc["hemi"]]].append(i)
            else:
                groups[i] = [i]
        made = []
        for k, (main, members) in enumerate(groups.items()):
            d = "".join("M" + " L".join("%.2f %.2f" % tuple(q[v]) for v in loop) + " Z "
                        for i in members for loop in pieces[i]["loops"])
            p = etree.SubElement(g, "{%s}path" % SVGNS)
            p.set("d", d.strip())
            p.set("id", f"{g.get('id')}_{k}")
            n_paths += 1
            made.append((k, main, _center_pts(_parse_svg_pts(p.get("d")))))
        for k, main, (ax_, ay_) in made:
            pc = pieces[main]
            others = [a for (kk, _, a) in made if kk != k]
            cf = polys[fidx[pc["comp"]]]
            vs = np.unique(cf)
            bverts = np.unique(np.concatenate([np.asarray(lp) for lp in pc["loops"]]))
            dist, _ = cKDTree(q[bverts]).query(q[vs])
            x, y, angle, depth_here = label_pose(q, vs, dist, (ax_, ay_), name, font_px, other_anchors=others)
            dy = 0.35 * font_px
            xb = x - dy * np.sin(np.radians(angle))   # baseline offset along the text's 'down' direction
            yb = y + dy * np.cos(np.radians(angle))
            t = etree.SubElement(textlayer, "{%s}text" % SVGNS)
            t.text = name
            t.set("x", "%.2f" % xb)
            t.set("y", "%.2f" % yb)
            if abs(angle) > 1.0:
                t.set("transform", "rotate(%.1f %.2f %.2f)" % (angle, xb, yb))
            t.set("style", style_txt)
            t.set("id", f"label_{g.get('id')}_{k}")
            placed.append((name, x, y, float(dist.max())))
            n_labels += 1
    svg.write(svgfile, pretty_print=True, xml_declaration=True, encoding="utf-8")
    log.info("[svg] wrote %d outlines and %d labels into layer %r of %s", n_paths, n_labels, layer, svgfile)
    if skipped:
        log.info("[labels] not labelled (too small/thin, outline kept): %s", "; ".join(skipped))
    P = np.array([[x, y] for _, x, y, _ in placed])
    close = []
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            dd = float(np.hypot(*(P[i] - P[j])))
            if dd < 3 * font_px:
                close.append((placed[i][0], placed[j][0], dd))
                log.info("[labels] close pair: %s / %s  (%.0f px)", placed[i][0], placed[j][0], dd)
    cache = os.path.join(cortex.database.default_filestore, cx_subject, "cache")
    if os.path.isdir(cache):
        for f in os.listdir(cache):
            if layer in f or "overlay" in f or f.endswith(".svg") or f.endswith(".png"):
                os.remove(os.path.join(cache, f))
    return dict(svg=svgfile, backup=bak, n_paths=n_paths, n_labels=n_labels, skipped=skipped, close_pairs=close)


def render_layer(cx_subject: str, fs_subject: str, subjects_dir: str, layer: str, out_png: str,
                 labelsize: str = "14pt", title: str | None = None) -> str:
    """Render curvature + the outlines and labels of one overlays.svg layer to a PNG."""
    import cortex
    import matplotlib.pyplot as plt

    sd = os.path.join(subjects_dir, fs_subject)
    curv = np.concatenate([nib.freesurfer.read_morph_data(f"{sd}/surf/{h}.curv") for h in ("lh", "rh")])
    v = cortex.Vertex(np.clip(-curv, -0.5, 0.5), cx_subject, cmap="gray", vmin=-1.0, vmax=1.0)
    fig = cortex.quickshow(v, with_curvature=False, with_rois=(layer == "rois"), with_labels=True,
                           with_colorbar=False, height=1024)
    if layer != "rois":
        from cortex.quickflat.composite import add_custom

        svgfile = os.path.join(cortex.database.default_filestore, cx_subject, "overlays.svg")
        add_custom(fig.axes[0], v, svgfile, layer, with_labels=True, labelsize=labelsize)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(out_png, dpi=120, facecolor="w", bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    log.info("[render] %s", out_png)
    return out_png
