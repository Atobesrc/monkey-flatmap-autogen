"""One hemisphere of a FreeSurfer subject: mesh, cortex label, parcellation, seam routes, slit patches."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .mesh import unique_edges, vertex_adjacency
from .recipe import BORDER_PENALTY, PARCEL_ALIASES, default_recipe, route_kwargs

log = logging.getLogger("mfa")

#: Default annotation name (``label/<hemi>.<annot>.annot``): CHARM level-2 (ARM2) parcels.
DEFAULT_ANNOT = "aparc.ARM2atlas.mapped"
#: Parcels the default recipe's landmark rules need.
REQUIRED_PARCELS = ("V1", "V2-V4", "OFC", "lat_PFC", "ACgG", "MTL", "Amy", "TG", "ITC")
#: Parcels used by optional recipe options, the cingulate cut and the layout metrics.
OPTIONAL_PARCELS = ("MPal", "PMC", "STG/STSd", "MT")
#: Cuts computed by default.
DEFAULT_CUTS = ("calcarine", "frontal", "temporal")
ALL_CUTS = DEFAULT_CUTS + ("cingulate",)


@dataclass(frozen=True)
class Parcellation:
    """Which annotation to read and how its names map onto the names the recipes use.

    Parameters
    ----------
    annot : str
        Annotation name: ``label/<hemi>.<annot>.annot``.
    parcel_map : tuple of (canonical, alias)
        Pairs ``(name used by the recipes, name in the annotation)``; the alias is
        renamed to the canonical name when the annotation is read.
    """

    annot: str = DEFAULT_ANNOT
    parcel_map: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_spec(cls, annot: str | None = None, parcel_map: str | None = None) -> Parcellation:
        """Build from command-line strings.

        ``parcel_map`` is ``'canonical=alias,canonical=alias,...'`` or the path of a
        JSON file holding ``{"canonical": "alias", ...}``.
        """
        pairs: list[tuple[str, str]] = []
        if parcel_map:
            if os.path.exists(parcel_map):
                with open(parcel_map) as fp:
                    pairs = [(str(k), str(v)) for k, v in json.load(fp).items()]
            else:
                for part in parcel_map.replace(";", ",").split(","):
                    if part.strip():
                        k, v = part.split("=", 1)
                        pairs.append((k.strip(), v.strip()))
        return cls(annot=annot or DEFAULT_ANNOT, parcel_map=tuple(pairs))

    def read(self, sd: str, hemi: str) -> tuple[np.ndarray, list[str]]:
        """Read ``label/<hemi>.<annot>.annot`` and return ``(labels, names)`` with aliases renamed."""
        lab, _ctab, raw = nib.freesurfer.read_annot(f"{sd}/label/{hemi}.{self.annot}.annot")
        names = [x.decode() if isinstance(x, bytes) else x for x in raw]
        for canonical, alias in self.parcel_map:
            if alias not in names:
                raise KeyError(f"[{hemi}] parcel-map alias {alias!r} (for {canonical!r}) is not in "
                               f"{self.annot}: {names}")
            if canonical in names and names.index(canonical) != names.index(alias):
                raise KeyError(f"[{hemi}] parcel-map: {canonical!r} already exists in the annotation")
            names[names.index(alias)] = canonical
        return lab, names


#: The default parcellation (CHARM level-2 names, no renaming).
DEFAULT_PARCELLATION = Parcellation()


class Hemi:
    """White surface, cortex label and parcellation of one hemisphere, with the cut machinery.

    Parameters
    ----------
    sd : str
        Subject directory ``$SUBJECTS_DIR/<subject>``.
    hemi : {'lh', 'rh'}
    parc : Parcellation
        Annotation to read.

    Attributes
    ----------
    v, f : ndarray
        White-surface vertices ``(n, 3)`` and faces ``(m, 3)``.
    curv : ndarray
        ``surf/<hemi>.curv`` (positive in sulci).
    iscortex : bool ndarray
        Membership of ``label/<hemi>.cortex.label``; its complement is the medial wall.
    lab, names, P : parcel labels per vertex, names, and ``{name: index}``.
    e, w : unique edges and their sulcus-preferring Dijkstra costs
        (edge length, x0.3 when both ends have curvature > 0.05).
    adjwall : bool ndarray
        Vertices with an edge crossing the cortex / medial-wall boundary.
    rim : int ndarray
        Cortex vertices adjacent to the medial wall (the medial-wall rim).
    adj : csr_matrix
        Vertex adjacency (for ring dilation).
    seam_info : dict
        Anchors and realised extension lengths of the last :meth:`seam_paths` call.
    """

    def __init__(self, sd: str, hemi: str, parc: Parcellation = DEFAULT_PARCELLATION):
        self.sd, self.hemi, self.parc = sd, hemi, parc
        self.v, self.f = nib.freesurfer.read_geometry(f"{sd}/surf/{hemi}.white")
        self.curv = nib.freesurfer.read_morph_data(f"{sd}/surf/{hemi}.curv")
        cortex = nib.freesurfer.read_label(f"{sd}/label/{hemi}.cortex.label")
        self.n = len(self.v)
        self.iscortex = np.zeros(self.n, bool)
        self.iscortex[cortex] = True
        self.lab, self.names = parc.read(sd, hemi)
        self.P = {x: i for i, x in enumerate(self.names)}
        self.e = unique_edges(self.f)
        elen = np.linalg.norm(self.v[self.e[:, 0]] - self.v[self.e[:, 1]], axis=1)
        sulcal = (self.curv[self.e[:, 0]] > 0.05) & (self.curv[self.e[:, 1]] > 0.05)
        self.w = elen * np.where(sulcal, 0.3, 1.0)
        crosses = self.iscortex[self.e[:, 0]] != self.iscortex[self.e[:, 1]]
        adjwall = np.zeros(self.n, bool)
        adjwall[self.e[crosses][:, 0]] = True
        adjwall[self.e[crosses][:, 1]] = True
        self.adjwall = adjwall
        self.rim = np.where(adjwall & self.iscortex)[0]
        self.adj: csr_matrix = vertex_adjacency(self.e, self.n)
        self.seam_info: dict[str, dict] = {}

    # ------------------------------------------------------------ parcels
    def _pid(self, name: str) -> int:
        name = PARCEL_ALIASES.get(name, name)
        try:
            return self.P[name]
        except KeyError:
            raise KeyError(f"[{self.hemi}] parcel {name!r} is not in {self.parc.annot} "
                           f"(available: {self.names}); map it with --parcel-map") from None

    def inpar(self, *ns: str) -> np.ndarray:
        """Boolean vertex mask of the union of parcels ``ns``."""
        return np.isin(self.lab, [self._pid(x) for x in ns])

    def inpar_c(self, *ns: str) -> np.ndarray:
        """Parcel mask restricted to cortex.label territory (the patch territory)."""
        return self.inpar(*ns) & self.iscortex

    # ------------------------------------------------------------ graph
    def graph(self, allowed: np.ndarray | None = None, wmul: np.ndarray | None = None) -> csr_matrix:
        """Weighted vertex graph restricted to edges with both ends in ``allowed``."""
        keep = (np.ones(len(self.e), bool) if allowed is None
                else (allowed[self.e[:, 0]] & allowed[self.e[:, 1]]))
        E, W = self.e[keep], self.w[keep]
        if wmul is not None:
            W = W * wmul[keep]
        return coo_matrix((np.r_[W, W], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                          shape=(self.n, self.n)).tocsr()

    def path(self, src: int, dst: int, allowed: np.ndarray | None = None,
             wmul: np.ndarray | None = None) -> np.ndarray:
        """Sulcus-preferring shortest path ``src -> dst`` inside ``allowed``.

        If ``dst`` is unreachable inside the corridor the path is computed on the
        unrestricted surface (the corridor is a preference, the anchors are the rule).
        """
        d, pred = dijkstra(self.graph(allowed, wmul), indices=src, return_predecessors=True)
        if np.isinf(d[dst]):
            d, pred = dijkstra(self.graph(None, wmul), indices=src, return_predecessors=True)
        p = [dst]
        while p[-1] != src and pred[p[-1]] >= 0:
            p.append(pred[p[-1]])
        return np.array(p[::-1])

    # ------------------------------------------------------------ landmarks
    def anchor_vertex(self, spec: str) -> int:
        """Vertex of a landmark: ``'<P>_ant'`` / ``'<P>_post'`` (most anterior / posterior
        vertex of parcel P) or ``'<A>_<B>_border'`` (most anterior vertex on the A/B border)."""
        if spec.endswith("_border"):
            a, b = spec[: -len("_border")].split("_", 1)
            ma, mb = self.inpar_c(a), self.inpar_c(b)
            E = self.e
            bx = (ma[E[:, 0]] & mb[E[:, 1]]) | (mb[E[:, 0]] & ma[E[:, 1]])
            bv = np.unique(E[bx].ravel())
            if not len(bv):
                raise ValueError(f"[{self.hemi}] no {a}/{b} border vertices")
            return int(bv[np.argmax(self.v[bv][:, 1])])
        if spec.endswith("_ant"):
            m = self.inpar_c(spec[: -len("_ant")])
            if not m.any():
                raise ValueError(f"[{self.hemi}] empty parcel in anchor {spec}")
            return int(np.where(m)[0][np.argmax(self.v[m][:, 1])])
        if spec.endswith("_post"):
            m = self.inpar_c(spec[: -len("_post")])
            if not m.any():
                raise ValueError(f"[{self.hemi}] empty parcel in anchor {spec}")
            return int(np.where(m)[0][np.argmin(self.v[m][:, 1])])
        raise ValueError(f"unknown anchor {spec!r}")

    def landmark_point(self, spec: str) -> np.ndarray:
        """3D point of a landmark defined by parcel membership and A-P ordering only.

        ``'<P>_ant'`` / ``'<P>_post'``: most anterior / posterior vertex of P;
        ``'<A>_<B>_border'``: centroid of the A/B border vertices;
        ``'<P1>_<P2>..._centroid'``: centroid of the union of the parcels.
        """
        if spec.endswith("_centroid"):
            names = spec[: -len("_centroid")].split("_")
            m = self.inpar_c(*names)
            if not m.any():
                raise ValueError(f"[{self.hemi}] empty landmark {spec}")
            return self.v[m].mean(0)
        if spec.endswith("_border"):
            a, b = spec[: -len("_border")].split("_", 1)
            ma, mb = self.inpar_c(a), self.inpar_c(b)
            E = self.e
            bx = (ma[E[:, 0]] & mb[E[:, 1]]) | (mb[E[:, 0]] & ma[E[:, 1]])
            bv = np.unique(E[bx].ravel())
            if not len(bv):
                raise ValueError(f"[{self.hemi}] no {a}/{b} border vertices")
            return self.v[bv].mean(0)
        if spec.endswith("_ant") or spec.endswith("_post"):
            return self.v[self.anchor_vertex(spec)]
        raise ValueError(f"unknown landmark {spec!r}")

    def temporal_source(self, rule: str) -> int:
        """Rim entry vertex of the temporal seam: ``'nearest_<landmark>'`` = the medial-wall
        rim vertex nearest :meth:`landmark_point`."""
        if not rule.startswith("nearest_"):
            raise ValueError(f"unknown temporal_src rule {rule!r}")
        pt = self.landmark_point(rule[len("nearest_"):])
        return int(self.rim[np.argmin(np.linalg.norm(self.v[self.rim] - pt, axis=1))])

    def extend_target(self, spec: str, corridor: np.ndarray) -> np.ndarray:
        """Target vertex set (inside ``corridor``) of an anatomical seam extension.

        ``'<P>_post'`` / ``'<P>_ant'`` / ``'<A>_<B>_border'``: corridor vertex nearest that
        landmark; ``'ITC_frac_<f>'``: ITC vertex nearest the centroid of the ITC vertices in
        a 1 mm A-P band at fraction f of the ITC A-P extent (0 = anterior end);
        ``'V2V4_border'``: every corridor vertex adjacent to V2-V4 (the seam stops at the
        first one it reaches).
        """
        v = self.v
        cv = np.where(corridor)[0]
        if not len(cv):
            raise ValueError(f"[{self.hemi}] empty extension corridor")
        if spec.startswith("ITC_frac_"):
            fr = float(spec[len("ITC_frac_"):])
            itc = np.where(self.inpar_c("ITC") & corridor)[0]
            if not len(itc):
                itc = np.where(self.inpar_c("ITC"))[0]
            y = v[itc, 1]
            yt = y.max() - fr * (y.max() - y.min())
            half = 0.5
            band = itc[np.abs(y - yt) <= half]
            while len(band) < 5 and half < 10:
                half *= 2
                band = itc[np.abs(y - yt) <= half]
            c = v[band].mean(0)
            return np.array([int(band[np.argmin(np.linalg.norm(v[band] - c, axis=1))])])
        if spec == "V2V4_border":
            m = self.inpar_c("V2-V4")
            E = self.e
            bx = (corridor[E[:, 0]] & m[E[:, 1]]) | (m[E[:, 0]] & corridor[E[:, 1]])
            bv = np.unique(E[bx].ravel())
            bv = bv[corridor[bv]]
            if not len(bv):
                raise ValueError(f"[{self.hemi}] corridor has no V2-V4 border")
            return bv.astype(int)
        pt = self.landmark_point(spec)
        return np.array([int(cv[np.argmin(np.linalg.norm(v[cv] - pt, axis=1))])])

    def extend_to_landmark(self, path: np.ndarray, mask: np.ndarray,
                           targets: np.ndarray) -> tuple[np.ndarray, float]:
        """Continue ``path`` from its tip to the nearest of ``targets`` inside ``mask``.

        Sulcus-preferring Dijkstra that never doubles back onto the seam.  Returns
        ``(extension vertices without the tip, 3D length in mm)``.  Raises
        ``ValueError`` if no target is reachable inside the corridor: the recipe is
        then not realisable on this hemisphere (there is no fallback route).
        """
        end = int(path[-1])
        allowed = mask.copy()
        onpath = np.zeros(self.n, bool)
        onpath[path] = True
        allowed &= ~onpath
        allowed[end] = True
        targets = np.asarray(targets, int)
        if onpath[targets].all():
            return np.array([], int), 0.0  # the landmark is already on the seam
        targets = targets[allowed[targets] | (targets == end)]
        if not len(targets):
            raise ValueError(f"[{self.hemi}] extension targets lie outside the corridor")
        d, pred = dijkstra(self.graph(allowed), indices=end, return_predecessors=True)
        dt = d[targets]
        if not np.isfinite(dt).any():
            raise ValueError(f"[{self.hemi}] extension target unreachable inside the corridor")
        t = int(targets[np.argmin(dt)])
        p = [t]
        while p[-1] != end and pred[p[-1]] >= 0:
            p.append(pred[p[-1]])
        p = np.array(p[::-1])
        mm = float(np.linalg.norm(np.diff(self.v[p], axis=0), axis=1).sum()) if len(p) > 1 else 0.0
        return p[1:], mm

    # ------------------------------------------------------------ seam paths
    def seam_paths(self, cuts: Iterable[str], temporal_src: str | None = None,
                   temporal_dst: str | None = None, temporal_corridor: str | None = None,
                   temporal_extend: str = "none", calcarine_extend: str = "none") -> dict[str, np.ndarray]:
        """Seam routes of the requested cuts as vertex paths on the white surface.

        Anchors use only anterior (y) / dorsal (z) coordinates and parcel
        membership, so they are mirror-symmetric.  The temporal-seam fields default
        to the packaged recipe (:func:`mfa.recipe.default_recipe`); a recipe sets
        them all at once through :func:`mfa.recipe.route_kwargs`.

        Parameters
        ----------
        cuts : iterable of {'calcarine', 'frontal', 'temporal', 'cingulate'}
        temporal_src, temporal_dst, temporal_corridor, temporal_extend, calcarine_extend
            Recipe route fields (see :mod:`mfa.recipe`).

        Returns
        -------
        dict
            ``{cut: vertex path}``.  Anchors and realised extension lengths (mm) are
            stored in :attr:`seam_info`.
        """
        rk = route_kwargs(default_recipe())
        temporal_src = temporal_src or rk["temporal_src"]
        temporal_dst = temporal_dst or rk["temporal_dst"]
        temporal_corridor = temporal_corridor or rk["temporal_corridor"]
        v, rim = self.v, self.rim
        out: dict[str, np.ndarray] = {}
        self.seam_info = {}
        cuts = list(cuts)
        if "calcarine" in cuts:
            v1 = self.inpar_c("V1")
            src = rim[np.argmin(np.linalg.norm(v[rim] - v[v1].mean(0), axis=1))]
            dst = np.where(v1)[0][np.argmin(v[v1][:, 1])]  # most posterior V1
            corr = v1 | self.inpar_c("V2-V4") | self.adjwall
            p = self.path(src, dst, corr)
            cext_mm = 0.0
            if calcarine_extend != "none":
                occ = v1 | self.inpar_c("V2-V4")
                ext, cext_mm = self.extend_to_landmark(p, occ, self.extend_target(calcarine_extend, occ))
                log.info("  [%s] calcarine extension -> %s: +%d verts (%.1f mm)", self.hemi,
                         calcarine_extend, len(ext), cext_mm)
                p = np.concatenate([p, ext])
            out["calcarine"] = p
            self.seam_info["calcarine"] = dict(src=int(src), dst=int(dst), n=int(len(p)),
                                               extend=str(calcarine_extend), extend_mm=cext_mm)
        if "frontal" in cuts:
            src = rim[np.argmax(v[rim][:, 1])]  # most anterior rim vertex
            dst = int(np.argmax(v[:, 1]))       # most anterior vertex of the hemisphere
            out["frontal"] = self.path(src, dst, self.inpar_c("OFC", "lat_PFC", "ACgG") | self.adjwall)
            self.seam_info["frontal"] = dict(src=int(src), dst=int(dst), n=int(len(out["frontal"])))
        if "temporal" in cuts:
            mt = self.inpar_c("MTL", "Amy")
            tg = self.inpar_c("TG")
            itc = self.inpar_c("ITC")
            src = self.temporal_source(temporal_src)
            dst = self.anchor_vertex(temporal_dst)
            wmul = None
            if temporal_corridor == "lateral":
                allowed = tg | itc | self.adjwall
            elif temporal_corridor == "border":
                allowed = mt | tg | itc | self.adjwall
                inner = self.inpar_c("Amy", "MTL")
                wmul = np.where(inner[self.e[:, 0]] & inner[self.e[:, 1]], BORDER_PENALTY, 1.0)
            elif temporal_corridor == "medial":
                allowed = mt | tg | itc | self.adjwall
            else:
                raise ValueError(f"unknown temporal corridor {temporal_corridor!r}")
            allowed = allowed.copy()
            allowed[src] = allowed[dst] = True
            p = self.path(src, dst, allowed, wmul)
            log.info("  [%s] temporal seam: src %s (%d), dst %s (%d), corridor %s: %d verts", self.hemi,
                     temporal_src, int(src), temporal_dst, int(dst), temporal_corridor, len(p))
            n_route = int(len(p))
            ext_mm = 0.0
            if temporal_extend != "none":
                # anatomical extension: posterior through TG|ITC to a landmark (never into
                # STS/STG); no fallback if the landmark is unreachable
                corr = tg | itc
                ext, ext_mm = self.extend_to_landmark(p, corr, self.extend_target(temporal_extend, corr))
                log.info("  [%s] temporal extension -> %s: +%d verts (%.1f mm posterior along TG/ITC)",
                         self.hemi, temporal_extend, len(ext), ext_mm)
                p = np.concatenate([p, ext])
            out["temporal"] = p
            self.seam_info["temporal"] = dict(src=int(src), dst=int(dst), n_route=n_route, n=int(len(p)),
                                              extend=str(temporal_extend), extend_mm=ext_mm,
                                              src_rule=temporal_src, dst_rule=temporal_dst,
                                              corridor=temporal_corridor)
        if "cingulate" in cuts:
            # dorsomedial cut: from the rim, dorsally along the ACgG/PMC border
            a, pm = self.inpar_c("ACgG"), self.inpar_c("PMC")
            bx = (a[self.e[:, 0]] & pm[self.e[:, 1]]) | (pm[self.e[:, 0]] & a[self.e[:, 1]])
            bverts = np.unique(self.e[bx].ravel())
            if len(bverts):
                src = rim[np.argmin(np.linalg.norm(v[rim] - v[bverts].mean(0), axis=1))]
                dst = int(bverts[np.argmax(v[bverts][:, 2])])  # most dorsal
                out["cingulate"] = self.path(src, dst, a | pm | self.adjwall)
                self.seam_info["cingulate"] = dict(src=int(src), dst=int(dst), n=int(len(out["cingulate"])))
        return out

    # ------------------------------------------------------------ patch build
    def dilate(self, mask: np.ndarray, rings: int) -> np.ndarray:
        """Grow a vertex mask by ``rings`` vertex rings."""
        m = mask.copy()
        for _ in range(rings):
            m = m | (self.adj.dot(m.astype(np.int8)) > 0)
        return m

    def build_slit_patch(self, paths: Mapping[str, np.ndarray],
                         slit_width: int = 1) -> tuple[np.ndarray, np.ndarray, dict]:
        """Slit patch: cortex.label territory minus the seams, opened to a disk.

        The seam vertices (dilated to ``slit_width`` rings) are removed from the
        cortex mask; the largest connected component is kept; isolated vertices are
        dropped; and every interior boundary loop (a slit that did not reach the
        outer rim) is opened by removing the shortest corridor of kept vertices to
        the nearest non-kept vertex, until one boundary loop remains.

        Returns
        -------
        ni : int ndarray
            Kept vertex ids.
        border : bool ndarray
            Border flag per surface vertex (boundary vertices of the patch).
        stats : dict
            ``nvert, nface, nbedge, loops, isolated, seam_removed, auto_corridors``.
        """
        seam = np.zeros(self.n, bool)
        for p in paths.values():
            seam[p] = True
        if slit_width > 1:
            seam = self.dilate(seam, slit_width - 1)
        keep = self.iscortex & ~seam

        def largest_component(keep: np.ndarray) -> np.ndarray:
            ki = np.where(keep)[0]
            remap = np.full(self.n, -1)
            remap[ki] = np.arange(len(ki))
            fk = remap[self.f[np.all(keep[self.f], axis=1)]]
            e2 = unique_edges(fk)
            g = coo_matrix((np.ones(len(e2)), (e2[:, 0], e2[:, 1])), shape=(len(ki),) * 2)
            _nc, cl = connected_components(g + g.T, directed=False)
            main = np.argmax(np.bincount(cl[np.unique(fk)]))
            k2 = np.zeros(self.n, bool)
            k2[ki[cl == main]] = True
            return k2

        def analyze(keep: np.ndarray) -> tuple:
            ni = np.where(keep)[0]
            remap = np.full(self.n, -1)
            remap[ni] = np.arange(len(ni))
            fr = remap[self.f[np.all(keep[self.f], axis=1)]]
            ec = Counter(tuple(sorted(x)) for tri in fr
                         for x in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])))
            bedges = np.array([k for k, c in ec.items() if c == 1])
            bg = coo_matrix((np.ones(len(bedges)), (bedges[:, 0], bedges[:, 1])), shape=(len(ni),) * 2)
            _nb, bl = connected_components(bg + bg.T, directed=False)
            bvert = np.unique(bedges)
            loopids = np.unique(bl[bvert])
            used = np.zeros(len(ni), bool)
            used[np.unique(fr)] = True
            return ni, fr, bedges, bl, bvert, loopids, used

        keep = largest_component(keep)
        n_auto = 0
        for _it in range(12):
            ni, fr, bedges, bl, bvert, loopids, used = analyze(keep)
            if (~used).sum():  # isolated vertices (in the patch but in no kept face)
                keep[ni[~used]] = False
                continue
            if len(loopids) <= 1:
                break
            sizes = {lid: (bl[bvert] == lid).sum() for lid in loopids}
            outer = max(sizes, key=sizes.get)
            inner = [lid for lid in loopids if lid != outer]
            hole = ni[bvert[bl[bvert] == inner[0]]]
            # multi-source Dijkstra from the hole to the nearest non-kept vertex; remove the corridor
            target = ~keep
            allowed = keep | target
            res = dijkstra(self.graph(allowed), indices=hole, return_predecessors=True, min_only=True)
            d, pred = res[0], res[1]
            tcand = np.where(target & np.isfinite(d))[0]
            if not len(tcand):
                break
            t = tcand[np.argmin(d[tcand])]
            corridor = [t]
            while pred[corridor[-1]] >= 0:
                corridor.append(pred[corridor[-1]])
            corridor = np.array(corridor)
            keep[corridor[keep[corridor]]] = False
            n_auto += 1
            log.info("  [%s] extended slit: removed a corridor of %d verts to open an interior loop",
                     self.hemi, int(keep[corridor].size))
            keep = largest_component(keep)
        ni, fr, bedges, bl, bvert, loopids, used = analyze(keep)
        border = np.zeros(self.n, bool)
        border[ni[bvert]] = True
        stats = dict(nvert=len(ni), nface=len(fr), nbedge=len(bedges), loops=len(loopids),
                     isolated=int((~used).sum()), seam_removed=int((seam & self.iscortex).sum()),
                     auto_corridors=int(n_auto))
        log.info("  [%s] patch: %d verts, %d faces, boundary loops %d, isolated %d", self.hemi,
                 stats["nvert"], stats["nface"], stats["loops"], stats["isolated"])
        return ni, border, stats


def check_parcels(names: Sequence[str], required: Iterable[str] = REQUIRED_PARCELS) -> list[str]:
    """Names of ``required`` parcels missing from an annotation's name list."""
    return [p for p in required if p not in names]
