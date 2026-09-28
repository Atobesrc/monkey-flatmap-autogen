"""One hemisphere of a FreeSurfer subject: mesh, cortex label, parcellation, seam routes, slit patches."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .mesh import unique_edges, vertex_adjacency
from .recipe import (
    BAND,
    BAND_RINGS,
    BORDER_END,
    BORDER_PENALTY,
    BORDER_PULL_DEFAULT,
    BORDER_RIM,
    BRIDGE_PENALTY,
    CALCARINE_DST,
    CINGULATE_DST,
    FRONTAL_CORRIDORS,
    PAIR,
    PARCEL_ALIASES,
    ROUTE_PATH,
    border_pair,
    default_recipe,
    is_anchored_route,
    is_trace_route,
    route_kwargs,
)

#: Distance-from-border assigned to corridor vertices the border cannot reach (mm; effectively excluded).
FAR_FROM_BORDER = 50.0

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
        self.elen = elen
        self.w = elen * np.where(sulcal, 0.3, 1.0)
        crosses = self.iscortex[self.e[:, 0]] != self.iscortex[self.e[:, 1]]
        adjwall = np.zeros(self.n, bool)
        adjwall[self.e[crosses][:, 0]] = True
        adjwall[self.e[crosses][:, 1]] = True
        self.adjwall = adjwall
        self.rim = np.where(adjwall & self.iscortex)[0]
        self.adj: csr_matrix = vertex_adjacency(self.e, self.n)
        self.seam_info: dict[str, dict] = {}
        self._border_dist: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

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
    def graph(self, allowed: np.ndarray | None = None, wmul: np.ndarray | None = None,
              base: np.ndarray | None = None) -> csr_matrix:
        """Weighted vertex graph restricted to edges with both ends in ``allowed``.

        ``base`` = per-edge costs (default the sulcus-preferring :attr:`w`; :attr:`elen`
        for plain lengths), optionally multiplied by ``wmul``.
        """
        keep = (np.ones(len(self.e), bool) if allowed is None
                else (allowed[self.e[:, 0]] & allowed[self.e[:, 1]]))
        E, W = self.e[keep], (self.w if base is None else base)[keep]
        if wmul is not None:
            W = W * wmul[keep]
        return coo_matrix((np.r_[W, W], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                          shape=(self.n, self.n)).tocsr()

    def path(self, src: int, dst: int, allowed: np.ndarray | None = None,
             wmul: np.ndarray | None = None, base: np.ndarray | None = None, strict: bool = False) -> np.ndarray:
        """Sulcus-preferring shortest path ``src -> dst`` inside ``allowed``.

        If ``dst`` is unreachable inside the corridor the path is computed on the
        unrestricted surface (the corridor is a preference, the anchors are the
        rule), unless ``strict`` is set, which raises ``ValueError`` instead.
        """
        d, pred = dijkstra(self.graph(allowed, wmul, base), indices=src, return_predecessors=True)
        if np.isinf(d[dst]):
            if strict:
                raise ValueError(f"[{self.hemi}] vertex {dst} is unreachable from {src} inside the corridor")
            d, pred = dijkstra(self.graph(None, wmul, base), indices=src, return_predecessors=True)
        p = [dst]
        while p[-1] != src and pred[p[-1]] >= 0:
            p.append(pred[p[-1]])
        return np.array(p[::-1])

    def path_mm(self, p: np.ndarray) -> float:
        """3D length (mm) of a vertex path."""
        return float(np.linalg.norm(np.diff(self.v[p], axis=0), axis=1).sum()) if len(p) > 1 else 0.0

    # ------------------------------------------------------------ landmarks
    def split_pair(self, spec: str) -> tuple[str, str]:
        """Split ``'A_B'`` into two parcel names (names may themselves contain ``_``, e.g. ``lat_PFC``)."""
        parts = spec.split("_")
        for i in range(1, len(parts)):
            a, b = "_".join(parts[:i]), "_".join(parts[i:])
            if PARCEL_ALIASES.get(a, a) in self.P and PARCEL_ALIASES.get(b, b) in self.P:
                return a, b
        raise KeyError(f"[{self.hemi}] {spec!r} is not a pair of parcels of {self.parc.annot} "
                       f"(available: {self.names}); map them with --parcel-map")

    def split_names(self, spec: str) -> list[str]:
        """Split ``'A_B_C'`` into parcel names, longest match first (a name may itself
        contain ``_``, e.g. ``lat_PFC``, and may be an alias, e.g. ``V2V4``)."""
        parts = spec.split("_")
        out: list[str] = []
        i = 0
        while i < len(parts):
            for j in range(len(parts), i, -1):
                cand = "_".join(parts[i:j])
                if PARCEL_ALIASES.get(cand, cand) in self.P:
                    out.append(cand)
                    i = j
                    break
            else:
                raise KeyError(f"[{self.hemi}] {spec!r} is not a list of parcels of {self.parc.annot} "
                               f"(available: {self.names}); map them with --parcel-map")
        return out

    def border_vertices(self, a: str, b: str) -> np.ndarray:
        """Cortex vertices on the A/B parcel border (both ends of an edge joining A and B)."""
        ma, mb = self.inpar_c(a), self.inpar_c(b)
        E = self.e
        bx = (ma[E[:, 0]] & mb[E[:, 1]]) | (mb[E[:, 0]] & ma[E[:, 1]])
        bv = np.unique(E[bx].ravel())
        if not len(bv):
            raise ValueError(f"[{self.hemi}] no {a}/{b} border vertices")
        return bv

    def geodesic_from(self, seeds: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Geodesic distance (mm) from `seeds`, propagated only through vertices of `mask`.

        Uses edge lengths on the white surface, so the result depends on the mesh and the
        parcellation only -- never on the position or orientation of the head.
        """
        from scipy.sparse.csgraph import dijkstra
        d = dijkstra(self.graph(mask), indices=np.asarray(seeds, dtype=int), min_only=True)
        out = np.asarray(d, dtype=float)
        out[~np.isfinite(out)] = -1.0
        return out

    def anchor_vertex(self, spec: str) -> int:
        """Vertex of a landmark defined by the parcellation and surface geodesics only.

        ``'<P1>_<P2>.._tip'``
            the vertex of that region geodesically farthest (inside it) from *every* border it
            has with another cortical parcel -- the point of the region deepest away from the
            rest of cortex, i.e. the tip of a pole (e.g. ``TG_tip`` = the temporal pole,
            ``OFC_lat_PFC_tip`` = the frontal pole).  Preferred over ``<P>_far_<Q>`` for a
            pole, which measures away from one named neighbour only and therefore slides
            along the pole when that neighbour's border moves.
        ``'<P>_far_<Q>'``
            the vertex of parcel P geodesically farthest (inside P) from the P/Q border -- the
            tip of P pointing away from Q (e.g. ``V1_far_V2-V4`` = the occipital pole).
        ``'<A>_<B>_border_far_wall'``
            the A/B border vertex geodesically farthest from the medial-wall boundary -- the
            outer end of that border (e.g. ``OFC_lat_PFC_border_far_wall`` = the frontal pole
            where the orbital and lateral surfaces meet).
        ``'<A>_<B>_border_far_<Q>'``
            the A/B border vertex geodesically farthest from parcel Q.

        No rule uses a coordinate axis, so every anchor is invariant to how the head lies in
        the scanner and is mirror-symmetric between hemispheres by construction.
        """
        m = re.match(r"^(.+)_tip$", spec)
        if m:
            names = self.split_names(m.group(1))
            mp = self.inpar_c(*names)
            if not mp.any():
                raise ValueError(f"[{self.hemi}] empty region in anchor {spec}")
            other = self.iscortex & ~mp
            E = self.e
            bx = (mp[E[:, 0]] & other[E[:, 1]]) | (other[E[:, 0]] & mp[E[:, 1]])
            bv = np.unique(E[bx].ravel())
            bv = bv[mp[bv]]
            if not len(bv):
                raise ValueError(f"[{self.hemi}] {spec}: {'+'.join(names)} borders no other cortical parcel")
            d = self.geodesic_from(bv, mp)
            d[~mp] = -1.0
            return int(np.argmax(d))
        m = re.match(r"^(.*)_border_far_(.+)$", spec)
        if m:
            a, b = self.split_pair(m.group(1))
            bv = self.border_vertices(a, b)
            pair = self.inpar_c(a) | self.inpar_c(b)
            if m.group(2) == "wall":
                seeds, region = self.rim, pair | self.adjwall
            else:
                seeds, region = self.border_vertices(*self.split_pair(f"{a}_{m.group(2)}")), pair
            d = self.geodesic_from(seeds, region)
            cand = d[bv]
            if not (cand > 0).any():
                raise ValueError(f"[{self.hemi}] {spec}: the {a}/{b} border is not reachable")
            return int(bv[int(np.argmax(cand))])
        m = re.match(r"^(.+)_far_(.+)$", spec)
        if m:
            p, q = self.split_pair(f"{m.group(1)}_{m.group(2)}")
            mp = self.inpar_c(p)
            if not mp.any():
                raise ValueError(f"[{self.hemi}] empty parcel in anchor {spec}")
            bv = self.border_vertices(p, q)
            bv = bv[mp[bv]]
            if not len(bv):
                raise ValueError(f"[{self.hemi}] {spec}: no {p}/{q} border inside {p}")
            d = self.geodesic_from(bv, mp)
            d[~mp] = -1.0
            return int(np.argmax(d))
        raise ValueError(f"unknown anchor {spec!r}: use '<P1>_<P2>.._tip', '<P>_far_<Q>' or "
                         "'<A>_<B>_border_far_<Q|wall>' (coordinate-based anchors are not supported)")

    def landmark_point(self, spec: str) -> np.ndarray:
        """3D point of a landmark defined by the parcellation and surface geodesics only.

        ``'<P1>_<P2>.._centroid'`` the centroid of the union of those parcels;
        ``'<A>_<B>_border'`` the centroid of the A/B border;
        anything else is resolved by :meth:`anchor_vertex` (``'<P>_far_<Q>'``,
        ``'<A>_<B>_border_far_<Q|wall>'``).
        """
        if spec.endswith("_centroid"):
            names = spec[: -len("_centroid")].split("_")
            m = self.inpar_c(*names)
            if not m.any():
                raise ValueError(f"[{self.hemi}] empty landmark {spec}")
            return self.v[m].mean(0)
        if spec.endswith("_border"):
            bv = self.border_vertices(*self.split_pair(spec[: -len("_border")]))
            return self.v[bv].mean(0)
        return self.v[self.anchor_vertex(spec)]

    def rim_source(self, rule: str, cut: str = "seam") -> int:
        """Rim entry vertex of a seam: ``'nearest_<landmark>'`` = the medial-wall rim vertex
        nearest :meth:`landmark_point` (the landmark is parcel-anchored; the rim vertex is the
        one closest to it)."""
        if not rule.startswith("nearest_"):
            raise ValueError(f"unknown {cut} rim-entry rule {rule!r} (expected 'nearest_<landmark>')")
        pt = self.landmark_point(rule[len("nearest_"):])
        return int(self.rim[np.argmin(np.linalg.norm(self.v[self.rim] - pt, axis=1))])

    def temporal_source(self, rule: str) -> int:
        """Rim entry vertex of the temporal seam (:meth:`rim_source`)."""
        return self.rim_source(rule, "temporal")

    def border_distance(self, pair: str, whole: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Geodesic distance (mm) of every vertex from the A/B border.

        Multi-source Dijkstra from the border vertices with plain edge lengths,
        computed once per hemisphere and pair: over the corridor A | B (+ the
        wall-adjacent vertices; the anchored-path cost) or, with ``whole``, over
        the whole cortex (the reporting metric, valid for any seam).  Returns
        ``(d, on_border, corridor)``; ``d`` is inf outside the graph's reach.
        """
        a, b = self.split_pair(pair)
        key = f"{a}_{b}" + ("|whole" if whole else "")
        if key not in self._border_dist:
            bv = self.border_vertices(a, b)
            onb = np.zeros(self.n, bool)
            onb[bv] = True
            ab = self.inpar_c(a) | self.inpar_c(b) | self.adjwall
            d = dijkstra(self.graph(self.iscortex if whole else ab, base=self.elen), indices=bv, min_only=True)
            self._border_dist[key] = (d, onb, ab)
        return self._border_dist[key]

    def seam_border_stats(self, seam: np.ndarray, pair: str) -> dict:
        """How closely a seam follows the A/B border: mean / max geodesic distance (mm, whole-cortex
        distance field), fraction within 1 mm, fraction of vertices on the border, seam length (mm)."""
        d, onb, _ab = self.border_distance(pair, whole=True)
        ds = d[seam]
        ds = np.where(np.isfinite(ds), ds, FAR_FROM_BORDER)
        return dict(border=pair, d_mean=float(ds.mean()), d_max=float(ds.max()),
                    frac_1mm=float((ds <= 1.0).mean()), on_border=float(onb[seam].mean()),
                    seam_mm=self.path_mm(seam))

    def anchored_seam(self, pair: str, src_rule: str, dst: str, pull: float = BORDER_PULL_DEFAULT,
                      cut: str = "seam") -> tuple[np.ndarray, dict]:
        """Shortest path between two landmarks, pulled onto the A/B border (route mode ``border:<A>_<B>``).

        Edge cost = ``length * (1 + pull * d)`` with ``d`` the mean distance-from-border
        (:meth:`border_distance`) of the edge's endpoints, inside A | B plus the
        wall-adjacent vertices, so the seam can never leave the two parcels that
        define the border.  The sulcus term of the plain path is dropped here: the
        border is the anatomical target, and a second pull (x0.3 in sulci) would
        make the seam depend on which pull wins locally.  Endpoints: ``src_rule``
        (``nearest_<landmark>`` rim vertex), ``dst`` (:meth:`anchor_vertex`, or for
        ``border_end`` the border vertex farthest from the source along this cost).
        No fallback: an unreachable destination makes the recipe unrealisable.
        """
        a, b = self.split_pair(pair)
        d, onb, ab = self.border_distance(pair)
        dd = np.where(np.isfinite(d), d, FAR_FROM_BORDER)
        base = self.elen * (1.0 + pull * 0.5 * (dd[self.e[:, 0]] + dd[self.e[:, 1]]))
        src = self.rim_source(src_rule, cut)
        allowed = ab.copy()
        allowed[src] = True
        if dst == BORDER_END:
            dsrc = dijkstra(self.graph(allowed, base=base), indices=src)
            bv = np.where(onb)[0]
            reach = bv[np.isfinite(dsrc[bv])]
            if not len(reach):
                raise ValueError(f"[{self.hemi}] the {a}/{b} border is unreachable from {src_rule} inside {a} | {b}")
            tip = int(reach[np.argmax(dsrc[reach])])
        else:
            tip = self.anchor_vertex(dst)
        allowed[tip] = True
        seam = self.path(src, tip, allowed, base=base, strict=True)
        info = dict(route=f"border:{pair}", parcels=(a, b), border_pull=float(pull),
                    **self.seam_border_stats(seam, pair))
        log.info("  [%s] %s seam anchored to the %s/%s border (pull %g/mm): src %s (%d, %s) -> dst %s (%d, %s): "
                 "%d verts, %.1f mm; distance to border mean %.2f max %.2f mm, %.0f %% within 1 mm, %.0f %% on it",
                 self.hemi,
                 cut, a, b, pull, src_rule, src, self.names[self.lab[src]], dst, tip, self.names[self.lab[tip]],
                 len(seam), info["seam_mm"], info["d_mean"], info["d_max"], 100 * info["frac_1mm"],
                 100 * info["on_border"])
        return seam, info

    def border_seam(self, pair: str, dst: str, cut: str = "seam") -> tuple[np.ndarray, dict]:
        """Seam that traces the A/B parcel border (route mode ``border_trace:<A>_<B>``).

        1. *Rim end*: the border vertex nearest the medial-wall rim (Euclidean; a
           border that touches the rim has distance 0).  If it is not itself on
           the rim, it is joined to the nearest rim vertex by the sulcus-preferring
           shortest path inside A | B (plus wall-adjacent vertices); no fallback.
        2. *Tip end*: the border vertex nearest the ``dst`` landmark
           (:meth:`landmark_point`), or for ``dst = 'border_end'`` the border
           vertex farthest along the border from the rim end.
        3. *Chain*: shortest path from the rim end to the tip end with edge cost =
           length, multiplied by :data:`mfa.recipe.BRIDGE_PENALTY` for every edge
           that leaves the border, restricted to A | B: it runs along the border
           and bridges gaps (islands, holes in the parcellation) with the shortest
           detour inside the two parcels.
        4. *Smoothing*: the seam is the plain shortest path (edge length only) from
           the rim vertex to the tip inside a :data:`mfa.recipe.BAND_RINGS`-ring
           band around link + chain, so mesh-scale zigzag of the border is removed
           while the seam stays within one vertex ring of it.

        Returns ``(seam vertices, info)`` with ``info``: ``on_border`` (fraction of
        seam vertices adjacent to both parcels), ``n_bridge`` (chain vertices off
        the border), ``link_n`` (vertices joining rim and border), ``chain_mm``,
        ``seam_mm``, ``rim_vertex``, ``rim_end``, ``tip``.
        """
        a, b = self.split_pair(pair)
        bv = self.border_vertices(a, b)
        onb = np.zeros(self.n, bool)
        onb[bv] = True
        ab = self.inpar_c(a) | self.inpar_c(b) | self.adjwall
        v, rim = self.v, self.rim
        # 1. rim end: the border vertex geodesically nearest the rim inside A | B (multi-source
        #    Dijkstra from every rim vertex); a border vertex on the rim has distance 0
        d_rim, pred, src_of = dijkstra(self.graph(ab, base=self.elen), indices=rim, return_predecessors=True,
                                       min_only=True)
        reach = bv[np.isfinite(d_rim[bv])]
        if not len(reach):
            raise ValueError(f"[{self.hemi}] the {a}/{b} border cannot be reached from the medial-wall rim "
                             f"inside {a} | {b}")
        rim_end = int(reach[np.argmin(d_rim[reach])])
        rim_v = int(src_of[rim_end])
        if rim_v == rim_end or d_rim[rim_end] == 0:
            rim_v, link = rim_end, np.array([rim_end])
        else:
            link = [rim_end]
            while link[-1] != rim_v and pred[link[-1]] >= 0:
                link.append(int(pred[link[-1]]))
            link = np.array(link[::-1])
        # 3. chain along the border (also used to find the far end)
        wmul = np.where(onb[self.e[:, 0]] & onb[self.e[:, 1]], 1.0, BRIDGE_PENALTY)
        if dst == BORDER_END:
            dd = dijkstra(self.graph(ab, wmul, self.elen), indices=rim_end)
            reach = bv[np.isfinite(dd[bv])]
            tip = int(reach[np.argmax(dd[reach])])
        else:
            pt = self.landmark_point(dst)
            tip = int(bv[np.argmin(np.linalg.norm(v[bv] - pt, axis=1))])
        chain = self.path(rim_end, tip, ab, wmul=wmul, base=self.elen, strict=True)
        full = np.concatenate([link[:-1], chain]) if len(link) > 1 else chain
        # 4. smoothing inside the band
        m = np.zeros(self.n, bool)
        m[full] = True
        band = self.dilate(m, BAND_RINGS) & (ab | m)
        seam = self.path(rim_v, tip, band, base=self.elen, strict=True)
        # the seam vertices that correspond to the chain (not the rim link): those at or beyond the
        # first seam vertex within one ring of the chain's rim end
        near_end = self.dilate(np.isin(np.arange(self.n), [rim_end]), BAND_RINGS)
        hits = np.where(near_end[seam])[0]
        i0 = int(hits[0]) if len(hits) else 0
        seam_chain = seam[i0:]
        info = dict(route=f"border_trace:{pair}", parcels=(a, b), rim_vertex=rim_v, rim_end=rim_end, tip=tip,
                    link_n=int(len(link) - 1), link_mm=self.path_mm(link), n_bridge=int((~onb[chain]).sum()),
                    chain_n=int(len(chain)), chain_mm=self.path_mm(chain),
                    on_border_chain=float(onb[seam_chain].mean()), chain_frac=float(len(seam_chain) / len(seam)),
                    n_border_vertices=int(len(bv)), **self.seam_border_stats(seam, pair))
        log.info("  [%s] %s seam on the %s/%s border: rim end %d (%s), tip %d (%s, dst %s), chain %d verts "
                 "(%d bridging, link %d verts %.1f mm) -> smoothed seam %d verts, %.1f mm, %.0f %% on the border "
                 "(%.0f %% of the chain part)", self.hemi, cut, a, b, rim_end,
                 "on rim" if rim_v == rim_end else f"linked from rim {rim_v}", tip, self.names[self.lab[tip]], dst,
                 len(chain), info["n_bridge"], info["link_n"], info["link_mm"], len(seam), info["seam_mm"],
                 100 * info["on_border"], 100 * info["on_border_chain"])
        return seam, info

    def extend_target(self, spec: str, corridor: np.ndarray) -> np.ndarray:
        """Target vertex set (inside ``corridor``) of an anatomical seam extension.

        ``'<A>_<B>_border'`` / ``'<P>_far_<Q>'``: the corridor vertex nearest that landmark;
        ``'V2V4_border'``: every corridor vertex adjacent to V2-V4 (the seam stops at the
        first one it reaches).  Every form is coordinate-free.
        """
        v = self.v
        cv = np.where(corridor)[0]
        if not len(cv):
            raise ValueError(f"[{self.hemi}] empty extension corridor")
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
                   temporal_extend: str = "none", calcarine_extend: str = "none",
                   frontal_src: str | None = None, frontal_dst: str | None = None,
                   frontal_corridor: str | None = None, temporal_route: str | None = None,
                   frontal_route: str | None = None,
                   border_pull: float | None = None) -> dict[str, np.ndarray]:
        """Seam routes of the requested cuts as vertex paths on the white surface.

        Every anchor is defined by parcel membership and geodesic distance on the
        surface, never by a coordinate axis, so it is mirror-symmetric and
        subject-independent (see :meth:`anchor_vertex`):

        - calcarine: rim vertex nearest the V1 centroid -> the occipital pole
          (:data:`mfa.recipe.CALCARINE_DST`), corridor V1 | V2-V4;
        - frontal: ``frontal_src`` (rim vertex nearest a frontal landmark) ->
          ``frontal_dst`` (a geodesic anchor), corridor
          ``frontal_corridor`` (:data:`mfa.recipe.FRONTAL_CORRIDORS`);
        - temporal: ``temporal_src`` -> ``temporal_dst`` inside ``temporal_corridor``,
          optionally extended to ``temporal_extend``;
        - route mode (``frontal_route`` / ``temporal_route``): ``path`` = the
          corridor-restricted sulcus-preferring shortest path between the two
          landmarks; ``border:<A>_<B>`` = the same two landmarks joined by a path
          anchored to the A/B border (:meth:`anchored_seam`, strength
          ``border_pull``); ``border_trace:<A>_<B>`` = the A/B border chain itself
          (:meth:`border_seam`), from its end nearest the rim to the border vertex
          nearest ``*_dst`` (or its far end for ``border_end``);
        - cingulate: rim vertex nearest the ACgG/PMC border centroid -> the outer end
          of that border (:data:`mfa.recipe.CINGULATE_DST`), corridor ACgG | PMC.

        The route fields default to the packaged recipe
        (:func:`mfa.recipe.default_recipe`); a recipe sets them all at once
        through :func:`mfa.recipe.route_kwargs`.

        Parameters
        ----------
        cuts : iterable of {'calcarine', 'frontal', 'temporal', 'cingulate'}
        temporal_src, temporal_dst, temporal_corridor, temporal_extend, calcarine_extend,
        frontal_src, frontal_dst, frontal_corridor, temporal_route, frontal_route, border_pull
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
        frontal_src = frontal_src or rk["frontal_src"]
        frontal_dst = frontal_dst or rk["frontal_dst"]
        frontal_corridor = frontal_corridor or rk["frontal_corridor"]
        # the route fields default to the packaged recipe too: taking the corridor from the
        # recipe but the route from a hard-coded default mixes two recipes and raises further down
        temporal_route = temporal_route or rk["temporal_route"]
        frontal_route = frontal_route or rk["frontal_route"]
        border_pull = rk["border_pull"] if border_pull is None else float(border_pull)
        v, rim = self.v, self.rim
        out: dict[str, np.ndarray] = {}
        self.seam_info = {}
        cuts = list(cuts)
        if "calcarine" in cuts:
            v1 = self.inpar_c("V1")
            src = rim[np.argmin(np.linalg.norm(v[rim] - v[v1].mean(0), axis=1))]
            # occipital pole: the V1 vertex geodesically farthest from the V1/V2-V4 border
            dst = self.anchor_vertex(CALCARINE_DST)
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
        if "frontal" in cuts and (is_trace_route(frontal_route) or is_anchored_route(frontal_route)):
            if is_trace_route(frontal_route):
                if frontal_src != BORDER_RIM or frontal_corridor != BAND:
                    raise ValueError(f"frontal_route {frontal_route!r} requires frontal_src={BORDER_RIM} and "
                                     f"frontal_corridor={BAND}")
                p, binfo = self.border_seam(border_pair(frontal_route), frontal_dst, "frontal")
            else:
                if frontal_corridor != PAIR:
                    raise ValueError(f"frontal_route {frontal_route!r} requires frontal_corridor={PAIR}")
                p, binfo = self.anchored_seam(border_pair(frontal_route), frontal_src, frontal_dst, border_pull,
                                              "frontal")
            out["frontal"] = p
            self.seam_info["frontal"] = dict(src=int(p[0]), dst=int(p[-1]), n=int(len(p)), src_rule=frontal_src,
                                             dst_rule=frontal_dst, corridor=frontal_corridor,
                                             src_parcel=self.names[self.lab[p[0]]],
                                             dst_parcel=self.names[self.lab[p[-1]]], **binfo)
        elif "frontal" in cuts:
            if frontal_corridor not in FRONTAL_CORRIDORS:
                raise ValueError(f"unknown frontal corridor {frontal_corridor!r}")
            src = self.rim_source(frontal_src, "frontal")
            dst = self.anchor_vertex(frontal_dst)
            allowed = self.inpar_c(*FRONTAL_CORRIDORS[frontal_corridor]) | self.adjwall
            allowed[src] = allowed[dst] = True
            out["frontal"] = self.path(src, dst, allowed)
            log.info("  [%s] frontal seam: src %s (%d, %s), dst %s (%d, %s), corridor %s: %d verts", self.hemi,
                     frontal_src, src, self.names[self.lab[src]], frontal_dst, dst, self.names[self.lab[dst]],
                     frontal_corridor, len(out["frontal"]))
            self.seam_info["frontal"] = dict(src=int(src), dst=int(dst), n=int(len(out["frontal"])),
                                             src_rule=frontal_src, dst_rule=frontal_dst, corridor=frontal_corridor,
                                             src_parcel=self.names[self.lab[src]], dst_parcel=self.names[self.lab[dst]],
                                             route=ROUTE_PATH, seam_mm=self.path_mm(out["frontal"]),
                                             on_border=float("nan"))
        if "temporal" in cuts:
            mt = self.inpar_c("MTL", "Amy")
            tg = self.inpar_c("TG")
            itc = self.inpar_c("ITC")
            binfo: dict = dict(route=ROUTE_PATH, on_border=float("nan"))
            if is_trace_route(temporal_route):
                if temporal_src != BORDER_RIM or temporal_corridor != BAND:
                    raise ValueError(f"temporal_route {temporal_route!r} requires temporal_src={BORDER_RIM} and "
                                     f"temporal_corridor={BAND}")
                p, binfo = self.border_seam(border_pair(temporal_route), temporal_dst, "temporal")
                src, dst = int(p[0]), int(p[-1])
            elif is_anchored_route(temporal_route):
                if temporal_corridor != PAIR:
                    raise ValueError(f"temporal_route {temporal_route!r} requires temporal_corridor={PAIR}")
                p, binfo = self.anchored_seam(border_pair(temporal_route), temporal_src, temporal_dst, border_pull,
                                              "temporal")
                src, dst = int(p[0]), int(p[-1])
            else:
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
                binfo["seam_mm"] = self.path_mm(p)
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
                                              corridor=temporal_corridor, **binfo)
        if "cingulate" in cuts:
            # dorsomedial cut: from the rim, dorsally along the ACgG/PMC border
            a, pm = self.inpar_c("ACgG"), self.inpar_c("PMC")
            bx = (a[self.e[:, 0]] & pm[self.e[:, 1]]) | (pm[self.e[:, 0]] & a[self.e[:, 1]])
            bverts = np.unique(self.e[bx].ravel())
            if len(bverts):
                src = rim[np.argmin(np.linalg.norm(v[rim] - v[bverts].mean(0), axis=1))]
                # outer end of the ACgG/PMC border (farthest from the medial-wall boundary)
                dst = self.anchor_vertex(CINGULATE_DST)
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
