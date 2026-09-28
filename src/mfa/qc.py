"""Figures: per-hemisphere QC, butterflies, seam routes on the inflated surface, layout overlays, tuner report."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from .flatten import flat_metrics, hemi_quality  # noqa: E402
from .layout import FlatDesign, apply_similarity, compare_flat, flat_layout  # noqa: E402
from .mesh import boundary_loops, boundary_vertices, majority_face_label, patch_faces  # noqa: E402
from .patch_io import read_patch  # noqa: E402
from .recipe import SEAM_PARCELS  # noqa: E402
from .surface import DEFAULT_PARCELLATION, Parcellation  # noqa: E402

log = logging.getLogger("mfa")

V1_TINT = np.array([0.2, 0.9, 0.3])
BOUNDARY_COLOR = "#ff8800"


def _curvature_colors(sd: str, hemi: str, idx: np.ndarray, parc: Parcellation, tint_v1: bool = True) -> np.ndarray:
    curv = nib.freesurfer.read_morph_data(f"{sd}/surf/{hemi}.curv")
    base = np.clip(-curv[idx] * 5 + 0.55, 0.15, 0.85)
    col = np.stack([base] * 3, -1)
    if tint_v1:
        try:
            lab, names = parc.read(sd, hemi)
            v1 = lab[idx] == names.index("V1")
            col[v1] = 0.6 * col[v1] + 0.4 * V1_TINT
        except (OSError, ValueError, KeyError):
            pass
    return col


def render_qc(sd: str, hemi: str, name: str, out_png: str, parc: Parcellation = DEFAULT_PARCELLATION,
              title_extra: str = "") -> dict:
    """Curvature map (V1 tinted, boundary in orange) and flipped-triangle map of one flat patch."""
    m, (idx, fk, p2, flipped, _v) = flat_metrics(sd, hemi, name)
    col = _curvature_colors(sd, hemi, idx, parc)
    bvert = boundary_vertices(fk)
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), facecolor="w")
    axes[0].add_collection(PolyCollection(p2[fk], facecolors=col[fk].mean(1), edgecolors="none"))
    axes[0].plot(p2[bvert, 0], p2[bvert, 1], ".", ms=1.2, color=BOUNDARY_COLOR)
    axes[0].set_title(f"{hemi}.{name} - curvature (orange = boundary) | flipped {m['flipped_pct']:.2f}% | "
                      f"stretch med {m['stretch_med']:.2f} p95 {m['stretch_p95']:.2f} {title_extra}")
    facec = np.where(flipped[:, None], [[1.0, 0.1, 0.1]], [[1.0, 1.0, 1.0]])
    axes[1].add_collection(PolyCollection(p2[fk], facecolors=facec, edgecolors="none"))
    axes[1].plot(p2[bvert, 0], p2[bvert, 1], ".", ms=0.8, color="#bbbbbb")
    axes[1].set_title(f"overlap check: RED = flipped ({m['flipped']} tris)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.autoscale()
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=100, facecolor="w")
    plt.close(fig)
    return m


def draw_butterfly(ax, sd: str, names: Mapping[str, str], parc: Parcellation = DEFAULT_PARCELLATION, title: str = "",
                   tint_v1: bool = True) -> None:
    """Both hemispheres of one subject in the canonical display frame (lh left, rh right).

    ``names = {hemi: patch name}``; V1 is tinted and the boundary drawn in orange.
    """
    for hemi in ("lh", "rh"):
        if hemi not in names:
            continue
        idx, _, xyz = read_patch(f"{sd}/surf/{hemi}.{names[hemi]}.flat.patch.3d")
        v, f = nib.freesurfer.read_geometry(f"{sd}/surf/{hemi}.white")
        fk = patch_faces(idx, f, len(v))
        p2 = xyz[:, :2]
        disp = np.c_[p2[:, 1], -p2[:, 0]]
        disp = disp - disp.mean(0)
        width = disp[:, 0].max() - disp[:, 0].min()
        disp[:, 0] += -(width / 2 + 5) if hemi == "lh" else (width / 2 + 5)
        col = _curvature_colors(sd, hemi, idx, parc, tint_v1)
        ax.add_collection(PolyCollection(disp[fk], facecolors=col[fk].mean(1), edgecolors="none"))
        L = disp[boundary_loops(fk)[0]]
        ax.plot(np.r_[L[:, 0], L[0, 0]], np.r_[L[:, 1], L[0, 1]], "-", lw=0.8, color=BOUNDARY_COLOR)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def render_butterfly(sd: str, name: str, out_png: str, parc: Parcellation = DEFAULT_PARCELLATION,
                     title: str = "") -> str:
    """Butterfly figure of ``<hemi>.<name>`` for both hemispheres."""
    fig, ax = plt.subplots(figsize=(18, 9), facecolor="w")
    draw_butterfly(ax, sd, {"lh": name, "rh": name}, parc, title=title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=100, facecolor="w")
    plt.close(fig)
    return out_png


def render_seam3d(specs: Sequence[tuple[str, str, str]], out_png: str,
                  parc: Parcellation = DEFAULT_PARCELLATION) -> str:
    """Seam routes on the inflated surfaces: one row per ``(sd, name, label)``.

    Medial and ventral view per hemisphere; black = removed vertices (all slits),
    red = the temporal slit, white square = its rim entry; temporal parcels
    tinted, dark = medial wall.
    """
    COL = {"Amy": "#d62728", "MTL": "#1f77b4", "MPal": "#9467bd", "TG": "#ff7f0e", "ITC": "#2ca02c",
           "STG/STSd": "#bcbd22", "OFC": "#8c564b", "V1": "#7fdc7f"}

    def draw(ax, V, F, cols, view, pts=None, title=""):
        ia, ib, idp, sgn = view
        C = V[F].mean(1)
        order = np.argsort(sgn * C[:, idp])
        Q = V[:, [ia, ib]]
        ax.add_collection(PolyCollection(Q[F[order]], facecolors=cols[order], edgecolors="none"))
        if pts:
            thr = np.percentile(sgn * V[:, idp], 40)
            for vi, kw in pts:
                if not len(vi):
                    continue
                keep = sgn * V[vi, idp] >= thr
                ax.plot(Q[vi[keep], 0], Q[vi[keep], 1], **kw)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.axis("off")
        ax.set_title(title, fontsize=9)

    fig, axes = plt.subplots(len(specs), 4, figsize=(26, 6.2 * len(specs)), facecolor="w", squeeze=False)
    for i, (sd, name, label) in enumerate(specs):
        for j, hemi in enumerate(("lh", "rh")):
            Vi, F = nib.freesurfer.read_geometry(f"{sd}/surf/{hemi}.inflated")
            curv = nib.freesurfer.read_morph_data(f"{sd}/surf/{hemi}.curv")
            lab, names = parc.read(sd, hemi)
            cortex = nib.freesurfer.read_label(f"{sd}/label/{hemi}.cortex.label")
            isc = np.zeros(len(Vi), bool)
            isc[cortex] = True
            lay = flat_layout(sd, hemi, name, parc)
            idx, _, _ = read_patch(f"{sd}/surf/{hemi}.{name}.flat.patch.3d")
            inpatch = np.zeros(len(Vi), bool)
            inpatch[idx] = True
            removed = np.where(isc & ~inpatch)[0]
            s = lay["seam"]
            slit = s["vertices"] if s else np.zeros(0, int)
            base = np.clip(-curv[F].mean(1) * 4 + 0.6, 0.25, 0.9)
            cols = np.stack([base] * 3, -1)
            maj = majority_face_label(lab[F])
            for q, c in COL.items():
                if q in names:
                    mm = maj == names.index(q)
                    cols[mm] = 0.45 * cols[mm] + 0.55 * np.array(matplotlib.colors.to_rgb(c))
            wall = ~np.all(isc[F], axis=1)
            cols[wall] = 0.35 * cols[wall] + 0.65 * np.array([0.15, 0.15, 0.2])
            pts = [(removed, dict(ls="none", marker=".", ms=2.5, color="k")),
                   (slit, dict(ls="none", marker=".", ms=4, color="#ff1744"))]
            if s:
                pts += [(np.array([s["vertices"][0]]), dict(ls="none", marker="s", ms=8, color="w", mec="k"))]
            med = (1, 2, 0, 1.0) if hemi == "lh" else (1, 2, 0, -1.0)
            ven = (0, 1, 2, -1.0)
            draw(axes[i, 2 * j], Vi, F, cols, med, pts,
                 f"{label} {hemi} medial view (inflated): black = removed (all slits), red = temporal slit "
                 f"({len(slit)} verts, {s['length_mm'] if s else 0:.0f} mm); dark = medial wall")
            draw(axes[i, 2 * j + 1], Vi, F, cols, ven, pts,
                 f"{label} {hemi} ventral view: Amy red, MTL blue, MPal purple, TG orange, ITC green, OFC brown")
    plt.tight_layout()
    plt.savefig(out_png, dpi=80, facecolor="w")
    plt.close(fig)
    return out_png


def overlay_figure(FA: FlatDesign, FB: FlatDesign, label_a: str, label_b: str, title: str, out_png: str,
                   cmp: dict | None = None, seam_parcels: Sequence[str] = SEAM_PARCELS) -> str:
    """Parcel outlines of A (blue) and of B after similarity alignment onto A (red), with centroid arrows."""
    if cmp is None:
        cmp = compare_flat(FA, FB, seam_parcels)
    T = cmp["T"]
    fig, ax = plt.subplots(figsize=(13, 11), facecolor="w")
    ax.add_collection(LineCollection(FA.outline_segments(), colors="#1f5fbf", lw=0.9, label=label_a))
    ax.add_collection(LineCollection(FB.outline_segments(T), colors="#d62728", lw=0.9, alpha=0.85, label=label_b))
    for q, p in FA.parcels.items():
        c1 = p["cent"]
        ax.text(c1[0], c1[1], q, fontsize=9, ha="center", va="center",
                fontweight="bold" if q in seam_parcels else "normal")
        if q in FB.parcels and T is not None:
            c2 = apply_similarity(T, FB.parcels[q]["cent"][None])[0]
            ax.annotate("", xy=c2, xytext=c1, arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    d = cmp["disp_mm"]
    mis = (" (" + ",".join(cmp["mismatch"]) + ")") if cmp["mismatch"] else ""
    ax.set_title(f"{title}\n{label_a} (blue) vs {label_b} (red; similarity-aligned onto blue: rotation + uniform "
                 f"scale, no warp); arrows = parcel centroid displacement\n"
                 f"displacement mean {d['mean']:.1f} / median {d['median']:.1f} / max {d['max']:.1f} mm "
                 f"(seam parcels mean {d['temporal_mean']:.1f} mm) | Dice mean {cmp['dice_mean']:.2f} "
                 f"(seam parcels {cmp['dice_temporal_mean']:.2f}) | pixel agreement "
                 f"{100 * cmp['pixel_label_agreement']:.1f} % | seam-side dist {cmp['side_dist']:.3f}, "
                 f"{cmp['n_mismatch']} parcel(s) on opposite banks{mis}", fontsize=10)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=100, facecolor="w")
    plt.close(fig)
    return out_png


def render_hotspot_map(sd: str, hemi: str, name: str, out_png: str, parc: Parcellation = DEFAULT_PARCELLATION,
                       hot_pack_mm: float = 2.0, hot_max: float = 3.0) -> dict:
    """Local area-packing map (swirl detector) of one flat patch, hotspot circled."""
    q = hemi_quality(sd, hemi, name, parc, nbhd_mm=hot_pack_mm, arrays=True)
    A = q["_arrays"]
    fig, ax = plt.subplots(figsize=(10, 9), facecolor="w")
    _draw_hotspot(ax, q, A, hot_pack_mm, hot_max)
    plt.tight_layout()
    plt.savefig(out_png, dpi=100, facecolor="w")
    plt.close(fig)
    return {k: x for k, x in q.items() if not k.startswith("_")}


def _draw_hotspot(ax, q: dict, A: dict, hot_pack_mm: float, hot_max: float) -> None:
    fk, p2, nb = A["fk"], A["p2"], A["nb"]
    disp = np.c_[p2[:, 1], -p2[:, 0]]
    pc = PolyCollection(disp[fk], edgecolors="none", cmap="viridis",
                        norm=matplotlib.colors.Normalize(0.5, max(2.5, hot_max)))
    pc.set_array(nb[fk].mean(1))
    ax.add_collection(pc)
    plt.colorbar(pc, ax=ax, shrink=0.6, label=f"local area packing 3D/flat in {hot_pack_mm:g} mm flat disk")
    hx = q["hot_xy"]
    ax.plot(hx[1], -hx[0], "o", mfc="none", mec="r", ms=18, mew=2)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    ax.set_title(f"{q['hemi']}.{q['name']} area-packing hotspot map (swirl detector)\n"
                 f"max {q['hot_pack']:.2f} @ {q['hot_parcel']} (red, {q['hot_dbnd']:.1f} mm from boundary) | "
                 f"temporal max {q['hot_pack_temporal']:.2f} @ {q['hot_temporal_parcel']}\n"
                 f"edge p99.9 |log2 r| {q['hot_p999']:.2f} | extreme cluster {q['hot_cluster']} edges", fontsize=10)


def render_joint_report(subjects: Sequence[str], sd_map: Mapping[str, str], code: str, ranked: Sequence[Mapping],
                        best: Mapping, hemi_q: Mapping[str, Mapping[str, Mapping]], weights: Mapping[str, float],
                        hot_max: float, n_recipes: int, out_png: str, parc: Parcellation = DEFAULT_PARCELLATION) -> str:
    """Joint-search report: butterfly of every subject under the chosen recipe plus a top-5 table."""
    n = len(subjects)
    fig = plt.figure(figsize=(20, 9 * n + 5), facecolor="w")
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[1.0] * n + [0.45])
    for i, s in enumerate(subjects):
        ax = fig.add_subplot(gs[i, 0])
        qs = hemi_q[s]
        draw_butterfly(ax, sd_map[s], {"lh": code, "rh": code}, parc, title=(
            f"{s}: recipe {best['recipe_str']}\n"
            f"lh-vs-rh silhouette {best['sil'][s]:.3f} % | p99 {qs['lh']['stretch_p99']:.2f}/"
            f"{qs['rh']['stretch_p99']:.2f} "
            f"| hot_pack {qs['lh']['hot_pack']:.2f}/{qs['rh']['hot_pack']:.2f} | loss "
            f"{qs['lh']['cortex_loss_area_pct']:.2f}/"
            f"{qs['rh']['cortex_loss_area_pct']:.2f} % | flips {qs['lh']['flips']}/{qs['rh']['flips']} | temporal "
            f"extension {qs['lh'].get('extend_mm', np.nan):.1f}/{qs['rh'].get('extend_mm', np.nan):.1f} mm "
            f"(green = V1, orange = boundary)"))
    axt = fig.add_subplot(gs[n, 0])
    axt.axis("off")
    cols = ["rank", "recipe (w / temporal route src dst corridor extend / frontal route src dst corridor)", "score",
            "qual", "sil % " + "/".join(subjects),
            "x-layout rms", "x-side dist (opp.)", "hot worst", "p99 mean", "loss %", "gates"]
    cells = []
    for i, R in enumerate(ranked[:5], 1):
        if R["status"] != "ok":
            continue
        r = R["recipe"]
        cells.append([str(i), f"w{r['slit_width']} {r.get('temporal_route', 'path')} "
                      f"{str(r['temporal_src']).replace('nearest_', '')} {r['temporal_dst']} "
                      f"{r['temporal_corridor']} {r['temporal_extend']} / {r.get('frontal_route', 'path')} "
                      f"{str(r.get('frontal_src', '')).replace('nearest_', '')} {r.get('frontal_dst', '')} "
                      f"{r.get('frontal_corridor', '')}",
                      f"{R['score']:.3f}", f"{R['qual']:.2f}", "/".join(f"{R['sil'][s]:.2f}" for s in subjects),
                      f"{R['xlay_mean']:.2f}", f"{R['xside_mean']:.3f} ({R['n_mismatch']})", f"{R['hot_worst']:.2f}",
                      f"{R['p99_mean']:.2f}", f"{R['loss_mean']:.2f}",
                      "OK" if (R["gates_ok"] and R["hot_ok"]) else "FAIL"])
    if cells:
        tb = axt.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(8)
        tb.auto_set_column_width(list(range(len(cols))))
        tb.scale(1.0, 1.6)
    W = weights
    axt.set_title(f"JOINT RECIPE SEARCH: top 5 of {len(ranked)} flattened recipes ({n_recipes} in the "
                  f"product); score = "
                  f"mean_hemi[{W['w_p99']:g}(p99-1) + {W['w_loss']:g} loss% + {W['w_hot']:g}(hot-1)] + "
                  f"{W['w_sil']:g} sil% "
                  f"+ {W['w_x']:g} x-layout rms + {W['w_x2']:g} x-side dist; gates on every hemisphere first",
                  fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=90, facecolor="w")
    plt.close(fig)
    return out_png
