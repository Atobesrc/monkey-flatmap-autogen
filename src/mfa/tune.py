"""Joint search of ONE cut recipe over every hemisphere of several subjects.

Nothing here warps coordinates or excludes cortex: every candidate is an
honest SLIM flattening of an honest slit patch, and the search only decides
*which* recipe every subject is cut with.

1. pre-screen (no flattening, :func:`prescreen_hemi` in parallel processes):
   every recipe of the discrete product on every hemisphere of every subject; a
   recipe survives only if every hemisphere gives one boundary loop, no isolated
   vertices, an identifiable temporal slit and a non-degenerate route; recipes
   with identical patches everywhere are merged;
2. staged flattening (``mfa flatten`` subprocesses, cached per subject /
   hemisphere / recipe as ``<hemi>.<name>_<code>.*``): all survivors if there are
   few, otherwise one stage per field group in ``stage_order`` (default
   ``temporal, frontal, extend, width``): the first group is varied with the
   other groups held at their start values (the packaged recipe's values when
   they are in the option sets, else the first option; ``start_width`` /
   ``start_extend``), ranked by the pre-screen cross-subject side disagreement
   and capped at ``stage_cap``; every later group is varied at the best recipe
   so far; then (either way) a refinement round: every one-field neighbour of
   the best recipe over the complete option sets of every field (pre-screened
   first);
3. ranking: hard gates on *every* hemisphere (topology, 0 flips, orientation,
   ``hot_pack <= hot_max``); then
   ``score = mean_hemi[w_p99 (p99-1) + w_loss loss% + w_hot (hot_pack-1)]
   + w_sil * mean_subject(lh-vs-mirrored-rh silhouette %)
   + w_x * mean_hemi(cross-subject layout RMS, units)
   + w_x2 * mean_hemi(cross-subject seam-side disagreement, 0..1)``.

The search uses one shared width so that routes are compared like with like;
the width actually applied to a subject is chosen per hemisphere afterwards
(``mfa flatten --slit-width auto``).
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .flatten import (
    DEFAULT_SLIM_ITERS,
    DEFAULT_SLIM_TOL,
    HOT_EXTREME,
    HOT_MAX,
    HOT_PACK_MM,
    hemi_quality,
    silhouette_from_loops,
)
from .layout import (
    FlatDesign,
    comparison_table,
    cross_layout_rms,
    cross_side_disagreement,
    design_metrics,
    flat_layout,
    patch_parcel_sides,
)
from .mesh import boundary_loops, patch_faces
from .qc import overlay_figure, render_joint_report, render_seam3d
from .recipe import (
    AUTO_SLIT_WIDTHS,
    BORDER_END,
    BORDER_PULL_DEFAULT,
    FIELD_OPTIONS,
    FRONTAL_CORRIDOR_OPTIONS,
    FRONTAL_DST_OPTIONS,
    FRONTAL_FIELDS,
    FRONTAL_ROUTE_OPTIONS,
    FRONTAL_SRC_OPTIONS,
    RECIPE_FIELDS,
    RECIPE_FORMAT,
    SEAM_PARCELS,
    TEMPORAL_CORRIDORS,
    TEMPORAL_DST_OPTIONS,
    TEMPORAL_EXTEND_OPTIONS,
    TEMPORAL_ROUTE_OPTIONS,
    TEMPORAL_SRC_OPTIONS,
    Recipe,
    canonicalise_routes,
    default_recipe,
    is_anchored_route,
    is_border_route,
    is_coordinate_free,
    is_trace_route,
    recipe_code,
    recipe_key,
    recipe_str,
    validate_recipe,
)
from .surface import DEFAULT_CUTS, DEFAULT_PARCELLATION, Hemi, Parcellation
from .utils import fmt_g, json_dump, json_load, run_jobs

log = logging.getLogger("mfa")

JOINT_DSTS = TEMPORAL_DST_OPTIONS
#: Field groups of the staged search.
STAGE_GROUPS = {
    "temporal": ("temporal_src", "temporal_dst", "temporal_corridor", "temporal_route"),
    "frontal": ("frontal_src", "frontal_dst", "frontal_corridor", "frontal_route"),
    "extend": ("temporal_extend", "calcarine_extend"),
    "width": ("slit_width",),
}
DEFAULT_STAGE_ORDER = ("temporal", "frontal", "extend", "width")
#: A temporal route shorter than this many vertices is degenerate.
MIN_ROUTE = 10


@dataclass
class JointOptions:
    """Options of the joint recipe search (see :class:`JointTuner`)."""

    subjects: Sequence[str]
    subjects_dir: str
    out_dir: str
    name: str = "uc"
    cuts: Sequence[str] = DEFAULT_CUTS
    parc: Parcellation = DEFAULT_PARCELLATION
    slim_iters: int = DEFAULT_SLIM_ITERS
    slim_tol: float = DEFAULT_SLIM_TOL
    widths: Sequence[int] = (1, 2, 3)
    srcs: Sequence[str] = TEMPORAL_SRC_OPTIONS
    dsts: Sequence[str] = JOINT_DSTS
    corridors: Sequence[str] = TEMPORAL_CORRIDORS
    extends: Sequence[str] = TEMPORAL_EXTEND_OPTIONS
    calcarine: Sequence[str] = ("none",)
    frontal_srcs: Sequence[str] = FRONTAL_SRC_OPTIONS
    frontal_dsts: Sequence[str] = FRONTAL_DST_OPTIONS
    frontal_corridors: Sequence[str] = FRONTAL_CORRIDOR_OPTIONS
    temporal_routes: Sequence[str] = TEMPORAL_ROUTE_OPTIONS
    frontal_routes: Sequence[str] = FRONTAL_ROUTE_OPTIONS
    border_end: bool = True  # add the 'border_end' destination for border routes
    border_pulls: Sequence[float] = (BORDER_PULL_DEFAULT,)
    stage_order: Sequence[str] = DEFAULT_STAGE_ORDER
    refine_widths: Sequence[int] = AUTO_SLIT_WIDTHS
    stage_cap: int = 20
    full_cap: int = 40
    start_width: int | None = None
    start_extend: str = "none"
    w_sil: float = 2.5
    w_p99: float = 1.0
    w_loss: float = 0.4
    w_hot: float = 1.0
    w_x: float = 0.5
    w_x2: float = 3.0
    hot_max: float = HOT_MAX
    hot_pack_mm: float = HOT_PACK_MM
    hot_extreme: float = HOT_EXTREME
    max_parallel: int = 6
    threads: int = 4
    prescreen_workers: int = 12
    no_refine: bool = False
    compare: list[tuple[str, dict[str, str]]] = field(default_factory=list)

    @property
    def tdir(self) -> str:
        return os.path.join(self.out_dir, "joint_tune")

    @property
    def product_options(self) -> dict[str, Sequence]:
        """Options of every recipe field in the product."""
        return dict(slit_width=tuple(self.widths), temporal_src=tuple(self.srcs), temporal_dst=tuple(self.dsts),
                    temporal_corridor=tuple(self.corridors), temporal_extend=tuple(self.extends),
                    calcarine_extend=tuple(self.calcarine), frontal_src=tuple(self.frontal_srcs),
                    frontal_dst=tuple(self.frontal_dsts), frontal_corridor=tuple(self.frontal_corridors),
                    temporal_route=tuple(self.temporal_routes), frontal_route=tuple(self.frontal_routes),
                    border_pull=tuple(self.border_pulls))

    @property
    def refine_options(self) -> dict[str, Sequence]:
        """Options of every field for the refinement round: the product options plus every
        landmark option set (:data:`mfa.recipe.FIELD_OPTIONS`) and ``refine_widths``."""
        out: dict[str, Sequence] = {}
        po = self.product_options
        for f in RECIPE_FIELDS:
            extra = tuple(self.refine_widths) if f == "slit_width" else FIELD_OPTIONS.get(f, ())
            seen: list = []
            for x in tuple(po[f]) + tuple(extra):
                if x not in seen:
                    seen.append(x)
            out[f] = tuple(seen)
        return out


def _seam_hash(a: np.ndarray) -> str:
    return hashlib.md5(np.asarray(a, np.int64).tobytes()).hexdigest()


def prescreen_hemi(sd: str, subject: str, hemi: str, recipes: Sequence[Recipe], cuts: Sequence[str] = DEFAULT_CUTS,
                   parc: Parcellation = DEFAULT_PARCELLATION, min_route: int = MIN_ROUTE) -> dict[tuple, dict]:
    """Pre-screen (no flattening) every recipe on one hemisphere.

    Seam routes (cached by route / extension), slit-patch topology (one loop, no
    isolated vertices), the number of automatic rim corridors the patch builder
    had to add, the realised extension length (mm) and the seam-side signature of
    the seam parcels.  A recipe whose landmark is unreachable, whose route is
    degenerate (< ``min_route`` vertices) or whose patch is not a disk fails
    (``topo_ok`` False).  Returns ``{recipe_key: record}``.
    """
    level = log.level
    log.setLevel(logging.WARNING)
    try:
        H = Hemi(sd, hemi, parc)
        tgitc = H.inpar_c("TG") | H.inpar_c("ITC")
        base_cache: dict = {}
        route_cache: dict = {}
        ext_cache: dict = {}
        patch_cache: dict = {}
        out = {}
        for r in recipes:
            key = recipe_key(r)
            rec: dict = dict(subject=subject, hemi=hemi, code=recipe_code(r), status="ok", error="")
            try:
                ck = str(r["calcarine_extend"])
                pull = float(r["border_pull"])
                bk = (ck,) + tuple(str(r[f]) for f in FRONTAL_FIELDS) + (f"{pull:g}",)
                if bk not in base_cache:
                    paths = H.seam_paths([c for c in cuts if c != "temporal"], calcarine_extend=ck,
                                         frontal_src=bk[1], frontal_dst=bk[2], frontal_corridor=bk[3],
                                         frontal_route=bk[4], border_pull=pull)
                    fi = H.seam_info.get("frontal", {})
                    base_cache[bk] = (paths, float(H.seam_info.get("calcarine", {}).get("extend_mm", 0.0)),
                                      dict(frontal_src_vertex=fi.get("src"), frontal_dst_vertex=fi.get("dst"),
                                           frontal_src_parcel=fi.get("src_parcel"),
                                           frontal_dst_parcel=fi.get("dst_parcel"),
                                           n_frontal=int(len(paths.get("frontal", []))),
                                           frontal_mm=float(fi.get("seam_mm", np.nan)),
                                           frontal_on_border=float(fi.get("on_border", np.nan)),
                                           frontal_on_border_chain=float(fi.get("on_border_chain", np.nan)),
                                           frontal_d_mean=float(fi.get("d_mean", np.nan)),
                                           frontal_d_max=float(fi.get("d_max", np.nan)),
                                           frontal_frac_1mm=float(fi.get("frac_1mm", np.nan)),
                                           frontal_bridge=int(fi.get("n_bridge", 0)),
                                           frontal_link=int(fi.get("link_n", 0))))
                base, calc_mm, frontal_info = base_cache[bk]
                rk = (str(r["temporal_src"]), str(r["temporal_dst"]), str(r["temporal_corridor"]),
                      str(r["temporal_route"]), f"{pull:g}")
                if rk not in route_cache:
                    p_ = H.seam_paths(["temporal"], temporal_src=rk[0], temporal_dst=rk[1], temporal_corridor=rk[2],
                                      temporal_route=rk[3], border_pull=pull, temporal_extend="none")["temporal"]
                    ti = H.seam_info.get("temporal", {})
                    route_cache[rk] = (p_, dict(temporal_mm=float(ti.get("seam_mm", np.nan)),
                                                temporal_on_border=float(ti.get("on_border", np.nan)),
                                                temporal_on_border_chain=float(ti.get("on_border_chain", np.nan)),
                                                temporal_d_mean=float(ti.get("d_mean", np.nan)),
                                                temporal_d_max=float(ti.get("d_max", np.nan)),
                                                temporal_frac_1mm=float(ti.get("frac_1mm", np.nan)),
                                                temporal_bridge=int(ti.get("n_bridge", 0)),
                                                temporal_link=int(ti.get("link_n", 0)),
                                                temporal_src_parcel=H.names[H.lab[p_[0]]],
                                                temporal_dst_parcel=H.names[H.lab[p_[-1]]]))
                p0, temporal_info = route_cache[rk]
                ek = (_seam_hash(p0), str(r["temporal_extend"]))
                if ek not in ext_cache:
                    if str(r["temporal_extend"]) != "none":
                        ext, mm = H.extend_to_landmark(p0, tgitc, H.extend_target(str(r["temporal_extend"]), tgitc))
                    else:
                        ext, mm = np.array([], int), 0.0
                    ext_cache[ek] = (np.concatenate([p0, ext]).astype(int), float(mm))
                p, ext_mm = ext_cache[ek]
                ph = _seam_hash(p)
                pk = (ph, int(r["slit_width"]), bk)
                if pk not in patch_cache:
                    ni, _border, stats = H.build_slit_patch(dict(base, temporal=p), int(r["slit_width"]))
                    ps = patch_parcel_sides(H, ni)
                    # loops counted like hemi_quality (loop walker): a width-1 pinch point is one
                    # boundary-edge component but two loops
                    n_loops = len(boundary_loops(patch_faces(ni, H.f, H.n)))
                    patch_cache[pk] = dict(
                        patch_hash=_seam_hash(ni), loops=int(max(stats["loops"], n_loops)),
                        isolated=int(stats["isolated"]), auto_corridors=int(stats["auto_corridors"]),
                        seam_removed=int(stats["seam_removed"]), nvert=int(stats["nvert"]),
                        sides=({q: dict(side=int(ps[q]["side"]), side_frac=float(ps[q]["side_frac"]),
                                        side_conf=float(ps[q]["side_conf"])) for q in SEAM_PARCELS if q in ps}
                               if ps else None))
                rec.update(patch_cache[pk])
                rec.update(seam_hash=ph, n_seam=int(len(p)), n_route=int(len(p0)), extend_mm=ext_mm,
                           calcarine_extend_mm=calc_mm, src_vertex=int(p0[0]), dst_vertex=int(p0[-1]),
                           **frontal_info, **temporal_info)
                rec["degenerate"] = bool(len(p0) < min_route)
                rec["topo_ok"] = bool(rec["loops"] == 1 and rec["isolated"] == 0 and rec["sides"] is not None
                                      and not rec["degenerate"])
            except (ValueError, KeyError, IndexError) as ex:
                rec.update(status="error", error=str(ex), topo_ok=False, loops=-1, isolated=-1, auto_corridors=-1,
                           patch_hash="", seam_hash="", n_seam=0, n_route=0, extend_mm=np.nan, sides=None,
                           degenerate=False)
            out[key] = rec
        return out
    finally:
        log.setLevel(level)


def _prescreen_task(task: tuple) -> tuple[tuple[str, str], dict]:
    sd, subject, hemi, recipes, cuts, parc = task
    return (subject, hemi), prescreen_hemi(sd, subject, hemi, recipes, cuts, parc)


def parse_compare_specs(specs: Sequence[str] | None) -> list[tuple[str, dict[str, str]]]:
    """Parse ``'label:subA=name,subB=name'`` design specifications."""
    out = []
    for sp in specs or []:
        label, rest = sp.split(":", 1)
        names = {}
        for part in rest.split(","):
            s, n = part.split("=")
            names[s.strip()] = n.strip()
        out.append((label, names))
    return out


class JointTuner:
    """Joint multi-hemisphere search of one cut recipe (module docstring)."""

    def __init__(self, opts: JointOptions):
        self.o = opts
        self.subjects = list(opts.subjects)
        self.hemis = ["lh", "rh"]
        self.base = opts.name
        self.tdir = opts.tdir
        for d in ("logs", "quality", "runs"):
            os.makedirs(os.path.join(self.tdir, d), exist_ok=True)
        self.sd = {s: os.path.join(opts.subjects_dir, s) for s in self.subjects}
        self.W = dict(w_sil=opts.w_sil, w_p99=opts.w_p99, w_loss=opts.w_loss, w_hot=opts.w_hot, w_x=opts.w_x,
                      w_x2=opts.w_x2)
        self.hot_max = opts.hot_max
        self.recipes: dict[tuple, Recipe] = {}
        self.pre: dict[tuple[str, str], dict] = {}
        self.dup_of: dict[tuple, tuple] = {}
        self.q: dict[tuple, dict] = {}
        self.lay: dict[tuple, dict] = {}
        self.loops: dict[tuple, np.ndarray] = {}
        self.R: dict[tuple, dict] = {}
        self.sig_seen: dict[tuple, tuple] = {}
        self.stages: list[dict] = []
        self.t0 = time.time()
        for f, opts_ in list(opts.product_options.items()) + list(opts.refine_options.items()):
            for o in opts_:
                if not is_coordinate_free(f, o):
                    raise ValueError(f"tune-joint: {f} option {str(o)!r} is not coordinate-free; cut rules must "
                                     "use the parcellation and surface geodesics only (parcel / border centroids, "
                                     "'<P>_far_<Q>', '<A>_<B>_border_far_<Q|wall>')")

    # ------------------------------------------------------------ recipes
    def register(self, r: Mapping[str, object]) -> tuple:
        """Validate a recipe, register it if new (landmark rules only); returns its key.

        A border route fixes its rim entry and corridor (:func:`canonicalise_routes`).
        """
        r = validate_recipe(canonicalise_routes(r))
        key = recipe_key(r)
        if key not in self.recipes:
            self.recipes[key] = r
        return key

    def _variants(self, base: Mapping[str, object], field: str, options: Sequence) -> list[tuple]:
        """Register ``base`` with ``field`` set to each option; combinations a border route makes
        meaningless (a rim-entry or corridor option under a border route, ``border_end`` under a
        path route) are skipped."""
        out = []
        cuts = {"temporal_src": "temporal", "temporal_corridor": "temporal", "temporal_dst": "temporal",
                "frontal_src": "frontal", "frontal_corridor": "frontal", "frontal_dst": "frontal"}
        for x in options:
            r = dict(base, **{field: x})
            cut = cuts.get(field)
            if cut and field.endswith("_corridor") and is_border_route(r[f"{cut}_route"]):
                continue  # implied by the border route
            if cut and field.endswith("_src") and is_trace_route(r[f"{cut}_route"]):
                continue  # trace: the rim end is the border end
            if cut and field.endswith("_dst") and str(x) == BORDER_END and not is_border_route(r[f"{cut}_route"]):
                continue
            if field == "border_pull" and not any(is_anchored_route(r[f]) for f in ("temporal_route",
                                                                                    "frontal_route")):
                continue  # only anchored routes use it
            try:
                k = self.register(r)
            except ValueError:
                continue
            if k not in out:
                out.append(k)
        return out

    def _dst_options(self, cut: str, route: str, base_options: Sequence) -> tuple:
        opts = tuple(base_options)
        if self.o.border_end and is_border_route(route) and BORDER_END not in opts:
            opts += (BORDER_END,)
        return opts

    def product(self) -> list[tuple]:
        """Register every recipe of the option product; returns their keys.

        Under a border route the rim-entry and corridor options collapse to the
        implied values and ``border_end`` is added to the destinations.
        """
        po = dict(self.o.product_options)
        out = []
        for vals in product(*(po[f] for f in RECIPE_FIELDS if not f.endswith("_dst"))):
            base = dict(zip([f for f in RECIPE_FIELDS if not f.endswith("_dst")], vals))
            for td in self._dst_options("temporal", base["temporal_route"], po["temporal_dst"]):
                for fd in self._dst_options("frontal", base["frontal_route"], po["frontal_dst"]):
                    r = dict(base, temporal_dst=td, frontal_dst=fd)
                    for cut in ("temporal", "frontal"):
                        if not is_border_route(r[f"{cut}_route"]) and r[f"{cut}_dst"] == BORDER_END:
                            r = None
                            break
                    if r is None:
                        continue
                    k = self.register(r)
                    if k not in out:
                        out.append(k)
        return out

    def neighbours(self, key: tuple) -> list[tuple]:
        """Register every one-field neighbour of ``key`` over the refinement option sets."""
        base = self.recipes[key]
        ro = dict(self.o.refine_options)
        out = []
        for f in RECIPE_FIELDS:
            opts = ro[f]
            if f.endswith("_dst"):
                opts = self._dst_options(f.split("_")[0], str(base[f.split("_")[0] + "_route"]), opts)
            for k in self._variants(base, f, [x for x in opts if str(x) != str(base[f])]):
                if k != key and k not in out:
                    out.append(k)
        return out

    def _cname(self, key: tuple) -> str:
        return f"{self.base}_{recipe_code(self.recipes[key])}"

    # ------------------------------------------------------------ pre-screen
    def prescreen(self, keys: Sequence[tuple], tag: str = "") -> list[tuple]:
        """Pre-screen the recipes on every hemisphere (cached); returns the distinct passing keys."""
        o = self.o
        t0 = time.time()
        cache_fn = os.path.join(self.tdir, "prescreen_cache.json")
        cached: dict[tuple[str, str], dict] = {}
        raw: dict = {}
        if os.path.exists(cache_fn):
            raw = json_load(cache_fn)
            for sh, recs in raw.items():
                s_, h_ = sh.split("/")
                cached[(s_, h_)] = recs
        need = [k for k in keys if any("|".join(k) not in cached.get((s, h), {})
                                       for s in self.subjects for h in self.hemis)]
        for s in self.subjects:
            for h in self.hemis:
                for k in keys:
                    rec = cached.get((s, h), {}).get("|".join(k))
                    if rec is not None:
                        self.pre.setdefault((s, h), {})[k] = rec
        if len(need) < len(keys):
            log.info("[joint] pre-screen cache %s: %d recipes reused", cache_fn, len(keys) - len(need))
        recipes = [self.recipes[k] for k in need]
        widths = sorted({int(r["slit_width"]) for r in recipes})
        tasks = []
        for s in self.subjects:
            for h in self.hemis:
                for w in widths:  # one task per (subject, hemi, width): separate caches
                    rs = [r for r in recipes if r["slit_width"] == w]
                    tasks.append((self.sd[s], s, h, rs, tuple(o.cuts), o.parc))
        n_workers = max(1, min(o.prescreen_workers, len(tasks)))
        log.info("[joint] pre-screen: %d recipes x %d hemispheres (%d tasks, %d processes)...", len(need),
                 len(self.subjects) * len(self.hemis), len(tasks), n_workers)
        results: dict[tuple[str, str], dict] = {}
        if tasks and need:
            if n_workers == 1:
                for t in tasks:
                    k, res = _prescreen_task(t)
                    results.setdefault(k, {}).update(res)
            else:
                ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
                with ctx.Pool(n_workers) as pool:
                    for k, res in pool.imap_unordered(_prescreen_task, tasks):
                        results.setdefault(k, {}).update(res)
                        log.info("  [joint] pre-screen %s %s: +%d recipes (%.1f min)", k[0], k[1], len(res),
                                 (time.time() - t0) / 60)
        for k, res in results.items():
            self.pre.setdefault(k, {}).update(res)
        for s in self.subjects:
            for h in self.hemis:
                if (s, h) in self.pre:
                    raw.setdefault(f"{s}/{h}", {}).update({"|".join(k): rec for k, rec in self.pre[(s, h)].items()})
        json_dump(raw, cache_fn)
        sig_seen = self.sig_seen  # persistent: a later round cannot re-flatten a patch set already known
        n_ok = n_dup = 0
        distinct = []
        for key in keys:
            recs = [self.pre[(s, h)][key] for s in self.subjects for h in self.hemis]
            if all(r["topo_ok"] for r in recs):
                sig = tuple(r["patch_hash"] for r in recs)
                if sig in sig_seen and sig_seen[sig] != key:
                    self.dup_of[key] = sig_seen[sig]
                    n_dup += 1
                else:
                    sig_seen[sig] = key
                    n_ok += 1
                    distinct.append(key)
        mins = (time.time() - t0) / 60
        log.info("[joint] pre-screen done in %.1f min: %d distinct recipes pass on every hemisphere, %d duplicates "
                 "(identical patches), %d fail somewhere", mins, n_ok, n_dup, len(keys) - n_ok - n_dup)
        self.stages.append(dict(stage=f"prescreen{tag}", n_recipes=len(keys), n_flattens=0, minutes=mins,
                                n_pass=n_ok, n_dup=n_dup))
        self.write_prescreen()
        return distinct

    def pre_ok(self, key: tuple) -> bool:
        return key not in self.dup_of and all(self.pre.get((s, h), {}).get(key, {}).get("topo_ok", False)
                                              for s in self.subjects for h in self.hemis)

    def pre_ok_known(self, key: tuple) -> bool:
        """True if the pre-screen record of ``key`` exists on every hemisphere."""
        return all(key in self.pre.get((s, h), {}) for s in self.subjects for h in self.hemis)

    def pre_side(self, key: tuple) -> dict:
        """Pre-flatten cross-subject seam-side disagreement, opposite-bank parcels, automatic corridors."""
        d, n, auto = [], 0, 0
        for h in self.hemis:
            for i in range(len(self.subjects)):
                for j in range(i + 1, len(self.subjects)):
                    a = self.pre[(self.subjects[i], h)][key]["sides"]
                    b = self.pre[(self.subjects[j], h)][key]["sides"]
                    x = cross_side_disagreement(a, b)
                    d.append(x["side_dist"])
                    n += x["n_mismatch"]
            for s in self.subjects:
                auto += max(0, self.pre[(s, h)][key].get("auto_corridors", 0))
        return dict(side_dist=float(np.nanmean(d)) if d else np.nan, n_mismatch=n, auto_corridors=auto)

    def write_prescreen(self) -> str:
        cols = ["code", "recipe", "pre_ok", "duplicate_of", "pre_side_dist", "pre_n_mismatch", "auto_corridors"]
        per = ["status", "topo_ok", "loops", "isolated", "auto_corridors", "n_route", "n_seam", "extend_mm",
               "degenerate", "temporal_mm", "temporal_d_mean", "temporal_d_max", "temporal_frac_1mm",
               "temporal_on_border", "temporal_on_border_chain", "temporal_bridge", "temporal_link",
               "temporal_src_parcel", "temporal_dst_parcel", "frontal_src_parcel", "frontal_dst_parcel", "n_frontal",
               "frontal_mm", "frontal_d_mean", "frontal_d_max", "frontal_frac_1mm", "frontal_on_border",
               "frontal_on_border_chain", "frontal_bridge", "frontal_link", "error"]
        for s in self.subjects:
            for h in self.hemis:
                cols += [f"{s}/{h}:{c}" for c in per]
        fn = os.path.join(self.tdir, "prescreen.tsv")
        first = (self.subjects[0], self.hemis[0])
        with open(fn, "w") as fp:
            fp.write("\t".join(cols) + "\n")
            for key, r in self.recipes.items():
                if first not in self.pre or key not in self.pre[first]:
                    continue
                have_sides = all(self.pre[(s, h)][key].get("sides") for s in self.subjects for h in self.hemis)
                ps = self.pre_side(key) if have_sides else dict(side_dist=np.nan, n_mismatch=-1, auto_corridors=-1)
                row = [recipe_code(r), recipe_str(r), str(self.pre_ok(key)),
                       recipe_code(self.recipes[self.dup_of[key]]) if key in self.dup_of else "",
                       fmt_g(ps["side_dist"], 3), str(ps["n_mismatch"]), str(ps["auto_corridors"])]
                for s in self.subjects:
                    for h in self.hemis:
                        rec = self.pre[(s, h)][key]
                        row += [fmt_g(rec.get(c, ""), 1) for c in per]
                fp.write("\t".join(row) + "\n")
        return fn

    # ------------------------------------------------------------ flattening
    def _flat_exists(self, s: str, h: str, key: tuple) -> bool:
        c = self._cname(key)
        return (os.path.exists(f"{self.sd[s]}/surf/{h}.{c}.flat.patch.3d")
                and os.path.exists(f"{self.sd[s]}/surf/{h}.{c}.patch.3d"))

    def _job(self, s: str, h: str, key: tuple) -> tuple[str, list[str], str]:
        o = self.o
        c = self._cname(key)
        argv = [sys.executable, "-m", "mfa.cli", "flatten", "--subjects-dir", o.subjects_dir, "--subject", s,
                "--hemis", h, "--name", c, "--cut-recipe", recipe_str(self.recipes[key]),
                "--slim-iters", str(o.slim_iters), "--slim-tol", f"{o.slim_tol:g}", "--cuts", *o.cuts,
                "--no-render", "--no-orient-backup", "--out-dir", os.path.join(self.tdir, "runs"),
                "--run-json", os.path.join(self.tdir, "quality", f"{s}_{h}_{c}.run.json"),
                "--annot", o.parc.annot]
        if o.parc.parcel_map:
            argv += ["--parcel-map", ",".join(f"{k}={v}" for k, v in o.parc.parcel_map)]
        return f"{s}:{h}:{c}", argv, os.path.join(self.tdir, "logs", f"{s}_{h}_{c}.log")

    def ensure(self, keys: Sequence[tuple], tag: str = "") -> None:
        """Flatten every (subject, hemisphere, recipe) without a flat yet, then evaluate and aggregate."""
        t0 = time.time()
        todo = [(s, h, k) for k in keys for s in self.subjects for h in self.hemis
                if not self._flat_exists(s, h, k) and (s, h, k) not in self.q]
        jobs = [self._job(*x) for x in todo]
        if jobs:
            log.info("[joint] %s: flattening %d hemispheres of %d recipes (<= %d in parallel)...", tag, len(jobs),
                     len(keys), self.o.max_parallel)
            rc = run_jobs(jobs, self.o.max_parallel, self.o.threads)
            for label, r in rc.items():
                if r != 0:
                    log.warning("  [joint] %s failed (rc=%d); see %s/logs/", label, r, self.tdir)
        for k in keys:
            for s in self.subjects:
                for h in self.hemis:
                    self.evaluate(s, h, k)
            self.aggregate(k)
        mins = (time.time() - t0) / 60
        self.stages.append(dict(stage=tag, n_recipes=len(keys), n_flattens=len(jobs), minutes=mins))
        log.info("[joint] %s: %d flattens, %.1f min", tag, len(jobs), mins)

    def evaluate(self, s: str, h: str, key: tuple) -> dict:
        """Quality and layout of one flattened (subject, hemisphere, recipe) (cached)."""
        kk = (s, h, key)
        if kk in self.q:
            return self.q[kk]
        c = self._cname(key)
        if not self._flat_exists(s, h, key):
            self.q[kk] = dict(subject=s, hemi=h, name=c, status="failed", gates_ok=False, hot_pack=np.inf,
                              stretch_p99=np.inf, stretch_p95=np.inf, cortex_loss_area_pct=np.inf, flips=-1,
                              loops=-1, isolated=-1, components=-1, hot_parcel="?", orient_corr=np.nan)
            log.info("  [joint] %s %s %s: FAILED (no flat patch)", s, h, c)
            return self.q[kk]
        t0 = time.time()
        qa = hemi_quality(self.sd[s], h, c, self.o.parc, nbhd_mm=self.o.hot_pack_mm, extreme=self.o.hot_extreme,
                          arrays=True)
        self.loops[kk] = qa["_arrays"]["loop"]
        q = {k: x for k, x in qa.items() if not k.startswith("_")}
        lay = flat_layout(self.sd[s], h, c, self.o.parc)
        self.lay[kk] = lay
        q["seam_mm"] = float(lay["seam"]["length_mm"]) if lay["seam"] else np.nan
        q["sides"] = {p: dict(side=int(lay["parcels"][p]["side"]), side_frac=float(lay["parcels"][p]["side_frac"]),
                              side_conf=float(lay["parcels"][p]["side_conf"]))
                      for p in SEAM_PARCELS if p in lay["parcels"]}
        rj = os.path.join(self.tdir, "quality", f"{s}_{h}_{c}.run.json")
        pre = self.pre.get((s, h), {}).get(key, {})
        if os.path.exists(rj):
            run = json_load(rj).get(h, {})
            si = run.get("seam_info", {})
            q["extend_mm"] = float(si.get("temporal", {}).get("extend_mm", np.nan))
            q["slim_iters"] = run.get("slim", {}).get("iters")
            q["slim_secs"] = run.get("slim", {}).get("secs")
            for cut in ("temporal", "frontal"):
                for k_ in ("on_border", "on_border_chain", "d_mean", "d_max", "frac_1mm"):
                    q[f"{cut}_{k_}"] = float(si.get(cut, {}).get(k_, np.nan))
                q[f"{cut}_route_mm"] = float(si.get(cut, {}).get("seam_mm", np.nan))
        else:
            q["extend_mm"] = float(pre.get("extend_mm", np.nan))
            for cut in ("temporal", "frontal"):
                for k_ in ("on_border", "on_border_chain", "d_mean", "d_max", "frac_1mm"):
                    q[f"{cut}_{k_}"] = float(pre.get(f"{cut}_{k_}", np.nan))
                q[f"{cut}_route_mm"] = float(pre.get(f"{cut}_mm", np.nan))
        q.update(status="ok", subject=s, eval_secs=time.time() - t0, recipe=self.recipes[key])
        json_dump(q, os.path.join(self.tdir, "quality", f"{s}_{h}_{c}.json"))
        self.q[kk] = q
        log.info("  [joint] %s %s %s: gates=%s flips=%d p95=%.3f p99=%.3f pack=%.2f@%s loss=%.2f%% ext=%.1fmm (%.1fs)",
                 s, h, c, "OK" if q["gates_ok"] else "FAIL", q["flips"], q["stretch_p95"], q["stretch_p99"],
                 q["hot_pack"], q["hot_parcel"], q["cortex_loss_area_pct"], q["extend_mm"], q["eval_secs"])
        return q

    # ------------------------------------------------------------ scoring
    def aggregate(self, key: tuple) -> dict:
        """Aggregate the per-hemisphere records of a recipe into gates and the score."""
        W = self.W
        qs = {(s, h): self.q[(s, h, key)] for s in self.subjects for h in self.hemis}
        ok_all = all(q.get("status") == "ok" for q in qs.values())
        R: dict = dict(key=key, code=self._cname(key), recipe=self.recipes[key],
                       recipe_str=recipe_str(self.recipes[key]),
                       flattened=True, status="ok" if ok_all else "failed")
        gates = ok_all and all(q["gates_ok"] for q in qs.values())
        hot_ok = ok_all and all(q["hot_pack"] <= self.hot_max for q in qs.values())
        R.update(gates_ok=bool(gates), hot_ok=bool(hot_ok))
        for (s, h), q in qs.items():
            R[f"{s}/{h}"] = dict(flips=q.get("flips"), loops=q.get("loops"), p95=q.get("stretch_p95"),
                                 p99=q.get("stretch_p99"), hot=q.get("hot_pack"), hot_parcel=q.get("hot_parcel"),
                                 loss=q.get("cortex_loss_area_pct"), gates_ok=q.get("gates_ok"),
                                 extend_mm=q.get("extend_mm", np.nan), seam_mm=q.get("seam_mm", np.nan),
                                 temporal_on_border=q.get("temporal_on_border", np.nan),
                                 temporal_d_mean=q.get("temporal_d_mean", np.nan),
                                 temporal_d_max=q.get("temporal_d_max", np.nan),
                                 temporal_frac_1mm=q.get("temporal_frac_1mm", np.nan),
                                 frontal_on_border=q.get("frontal_on_border", np.nan),
                                 frontal_d_mean=q.get("frontal_d_mean", np.nan),
                                 frontal_d_max=q.get("frontal_d_max", np.nan),
                                 frontal_frac_1mm=q.get("frontal_frac_1mm", np.nan),
                                 frontal_mm=q.get("frontal_route_mm", np.nan), name=q.get("name"))
        if not ok_all:
            R.update(qual=np.inf, sil_mean=np.inf, xlay_mean=np.inf, xside_mean=np.inf, score=np.inf, n_mismatch=-1,
                     rank_key=(1, np.inf))
            self.R[key] = R
            return R
        R["qual"] = float(np.mean([W["w_p99"] * (q["stretch_p99"] - 1) + W["w_loss"] * q["cortex_loss_area_pct"]
                                   + W["w_hot"] * (q["hot_pack"] - 1) for q in qs.values()]))
        R["p99_mean"] = float(np.mean([q["stretch_p99"] for q in qs.values()]))
        R["loss_mean"] = float(np.mean([q["cortex_loss_area_pct"] for q in qs.values()]))
        R["hot_mean"] = float(np.mean([q["hot_pack"] for q in qs.values()]))
        R["hot_worst"] = float(max(q["hot_pack"] for q in qs.values()))
        sil = {s: silhouette_from_loops(self.loops[(s, "lh", key)], self.loops[(s, "rh", key)])["sil_pct"]
               for s in self.subjects}
        R["sil"] = sil
        R["sil_mean"] = float(np.mean(list(sil.values())))
        xl, xs, nm, mis = {}, {}, 0, []
        for h in self.hemis:
            for i in range(len(self.subjects)):
                for j in range(i + 1, len(self.subjects)):
                    a, b = self.subjects[i], self.subjects[j]
                    X = cross_layout_rms(self.lay[(a, h, key)], self.lay[(b, h, key)])
                    S = cross_side_disagreement(qs[(a, h)]["sides"], qs[(b, h)]["sides"])
                    xl[f"{a}~{b}/{h}"] = dict(rms=X["rms"], temporal_rms=X["temporal_rms"])
                    xs[f"{a}~{b}/{h}"] = S
                    nm += S["n_mismatch"]
                    mis += [f"{h}:{q}" for q in S["mismatch"]]
        R["xlay"] = xl
        R["xside"] = xs
        R["xlay_mean"] = float(np.mean([x["rms"] for x in xl.values()])) if xl else 0.0
        R["xlay_temporal_mean"] = float(np.nanmean([x["temporal_rms"] for x in xl.values()])) if xl else 0.0
        R["xside_mean"] = float(np.nanmean([x["side_dist"] for x in xs.values()])) if xs else 0.0
        R["n_mismatch"] = int(nm)
        R["mismatch"] = mis
        R["score"] = float(R["qual"] + W["w_sil"] * R["sil_mean"] + W["w_x"] * R["xlay_mean"]
                           + W["w_x2"] * R["xside_mean"])
        R["terms"] = dict(qual=R["qual"], sil=W["w_sil"] * R["sil_mean"], xlay=W["w_x"] * R["xlay_mean"],
                          xside=W["w_x2"] * R["xside_mean"])
        R["rank_key"] = (0 if (gates and hot_ok) else 1, R["score"])
        self.R[key] = R
        return R

    def ranked(self) -> list[dict]:
        return sorted(self.R.values(), key=lambda R: R["rank_key"])

    def best(self) -> dict | None:
        P = self.ranked()
        return P[0] if P else None

    def _report(self, tag: str) -> None:
        P = self.ranked()
        log.info("[joint] %s: top 5 of %d flattened recipes", tag, len(P))
        for R in P[:5]:
            if R["status"] != "ok":
                log.info("   %s: FAILED", R["code"])
                continue
            ext = "/".join(f"{R[f'{s}/{h}']['extend_mm']:.0f}" for s in self.subjects for h in self.hemis)
            log.info("   %-34s score %.3f = qual %.2f + sil %.2f + xlay %.2f + xside %.2f | sil%% %s | xlay %.2f | "
                     "side dist %.3f (%d opp.) | hot worst %.2f | ext mm %s | gates %s hot %s", R["code"], R["score"],
                     R["qual"], R["terms"]["sil"], R["terms"]["xlay"], R["terms"]["xside"],
                     "/".join(f"{x:.2f}" for x in R["sil"].values()), R["xlay_mean"], R["xside_mean"], R["n_mismatch"],
                     R["hot_worst"], ext, "OK" if R["gates_ok"] else "FAIL", "OK" if R["hot_ok"] else "FAIL")

    # ------------------------------------------------------------ search
    def search(self, prescreen_only: bool = False) -> list[dict]:
        """Run the staged search; returns the ranked recipes."""
        o = self.o
        keys = self.product()
        log.info("[joint] recipe product: %d recipes = %s; subjects %s", len(keys),
                 " x ".join(f"{f} {len(v)}" for f, v in o.product_options.items()), self.subjects)
        distinct = self.prescreen(keys)
        if not distinct:
            raise SystemExit("[joint] no recipe passes the pre-screen on every hemisphere")
        if prescreen_only:
            log.info("[joint] prescreen only: %d distinct passing recipes; see %s", len(distinct),
                     os.path.join(self.tdir, "prescreen.tsv"))
            return []

        def fld(k: tuple, f: str) -> str:
            return str(self.recipes[k][f])

        def same(k: tuple, b: Mapping, fields: Sequence[str]) -> bool:
            return all(fld(k, f) == str(b["recipe"][f]) for f in fields)

        if len(distinct) <= o.full_cap:
            log.info("[joint] %d distinct recipes <= full-cap %d: flattening all", len(distinct), o.full_cap)
            self.ensure(distinct, "full")
        else:
            order = [g for g in o.stage_order if g in STAGE_GROUPS]
            if not order:
                raise ValueError(f"tune-joint: stage_order must name groups of {list(STAGE_GROUPS)}")
            # start values of the groups not varied in the first stage
            po = o.product_options
            dflt = default_recipe()
            start = {f: (str(dflt[f]) if str(dflt[f]) in [str(x) for x in po[f]] else str(po[f][0]))
                     for f in RECIPE_FIELDS}
            start["temporal_extend"] = o.start_extend if o.start_extend in o.extends else str(o.extends[0])
            w0 = o.start_width
            if w0 is None:
                for w in sorted(o.widths):
                    if any(fld(k, "slit_width") == str(w) and all(fld(k, f) == start[f] for f in RECIPE_FIELDS
                                                                   if f != "slit_width") for k in distinct):
                        w0 = w
                        break
            if w0 is None:
                cnt = {w: sum(1 for k in distinct if fld(k, "slit_width") == str(w)) for w in o.widths}
                w0 = max(cnt, key=cnt.get)
            start["slit_width"] = str(w0)
            for i, g in enumerate(order):
                fields = STAGE_GROUPS[g]
                if i == 0:
                    fixed = [f for f in RECIPE_FIELDS if f not in fields]
                    S = [k for k in distinct if all(fld(k, f) == start[f] for f in fixed)]
                    S.sort(key=lambda k: (self.pre_side(k)["n_mismatch"], self.pre_side(k)["side_dist"],
                                          self.pre_side(k)["auto_corridors"]))
                    log.info("[joint] stage %d (%s) at %s: %d distinct recipes pass the pre-screen; flattening "
                             "the %d with the smallest pre-screen cross-subject side disagreement", i + 1, g,
                             ", ".join(f"{f}={start[f]}" for f in fixed), len(S), min(len(S), o.stage_cap))
                    for k in S[:o.stage_cap]:
                        ps = self.pre_side(k)
                        log.info("     %-44s pre side dist %.3f opp %d auto %d", self._cname(k), ps["side_dist"],
                                 ps["n_mismatch"], ps["auto_corridors"])
                else:
                    b = self.best()
                    fixed = [f for f in RECIPE_FIELDS if f not in fields]
                    S = [k for k in distinct if k not in self.R and same(k, b, fixed)]
                    log.info("[joint] stage %d (%s): %d variants of %s", i + 1, g, len(S), b["code"])
                if S:
                    self.ensure(S[:o.stage_cap], f"stage{i + 1}-{g}")
                    self._report(f"stage{i + 1}-{g}")
        if not o.no_refine:
            b = self.best()
            if b is not None:
                cand = self.neighbours(b["key"])
                new = [k for k in cand if k not in self.R and k not in self.dup_of and not self.pre_ok_known(k)]
                if new:
                    self.prescreen(new, "-refine")
                S4 = [k for k in cand if k not in self.R and self.pre_ok(k)]
                log.info("[joint] refine: %d one-field neighbours of %s over the full option sets (%d registered, "
                         "%d pass the pre-screen and are new)", len(cand), b["code"], len(cand), len(S4))
                if S4:
                    self.ensure(S4, "refine-neighbours")
                    self._report("refine-neighbours")
        P = self.ranked()
        self._report("final")
        return P

    # ------------------------------------------------------------ outputs
    def write_tables(self) -> str:
        """``recipe_results.tsv`` (every recipe) and ``results.json``."""
        P = self.ranked()
        rank = {R["key"]: i + 1 for i, R in enumerate(P)}
        cols = ["rank", "code", "recipe", "flattened", "pre_ok", "duplicate_of", "gates_ok", "hot_ok", "score", "qual",
                "sil_mean", "xlay_mean", "xlay_temporal_mean", "xside_mean", "n_mismatch", "mismatch", "p99_mean",
                "loss_mean", "hot_mean", "hot_worst"]
        cols += [f"sil:{s}" for s in self.subjects]
        for s in self.subjects:
            for h in self.hemis:
                cols += [f"{s}/{h}:{c}" for c in ("flips", "loops", "p95", "p99", "hot", "hot_parcel", "loss",
                                                  "extend_mm", "seam_mm", "temporal_d_mean", "temporal_d_max",
                                                  "temporal_frac_1mm", "temporal_on_border", "frontal_mm",
                                                  "frontal_d_mean", "frontal_d_max", "frontal_frac_1mm",
                                                  "frontal_on_border", "gates_ok", "name")]
        fn = os.path.join(self.tdir, "recipe_results.tsv")
        with open(fn, "w") as fp:
            fp.write("\t".join(cols) + "\n")
            order = sorted(self.recipes, key=lambda k: (rank.get(k, 10**6), self._cname(k)))
            for key in order:
                r = self.recipes[key]
                R = self.R.get(key)
                row: dict = dict(rank=rank.get(key, ""), code=self._cname(key), recipe=recipe_str(r),
                                 flattened=bool(R is not None), pre_ok=self.pre_ok(key),
                                 duplicate_of=(self._cname(self.dup_of[key]) if key in self.dup_of else ""))
                if R is not None:
                    row.update({k: R.get(k, "") for k in ("gates_ok", "hot_ok", "score", "qual", "sil_mean",
                                                          "xlay_mean",
                                                          "xlay_temporal_mean", "xside_mean", "n_mismatch", "p99_mean",
                                                          "loss_mean", "hot_mean", "hot_worst")})
                    row["mismatch"] = ",".join(R.get("mismatch", []))
                    for s in self.subjects:
                        row[f"sil:{s}"] = R.get("sil", {}).get(s, "")
                        for h in self.hemis:
                            for c, x in R.get(f"{s}/{h}", {}).items():
                                row[f"{s}/{h}:{c}"] = x
                fp.write("\t".join(fmt_g(row.get(c, ""), 4) for c in cols) + "\n")
        o = self.o
        json_dump(dict(recipes={self._cname(k): dict(recipe=r, rank=rank.get(k), pre_ok=self.pre_ok(k),
                                                     duplicate_of=(self._cname(self.dup_of[k])
                                                                   if k in self.dup_of else None))
                                for k, r in self.recipes.items()},
                       results=[{k: x for k, x in R.items() if k not in ("key", "rank_key")} for R in P],
                       weights=self.W, hot_max=self.hot_max, stages=self.stages, subjects=self.subjects,
                       options={f: [str(x) for x in v] for f, v in o.product_options.items()},
                       refine_options={f: [str(x) for x in v] for f, v in o.refine_options.items()}),
                  os.path.join(self.tdir, "results.json"))
        return fn

    def export_chosen(self, best: Mapping) -> dict:
        """Copy the winning recipe's patches to ``<hemi>.<name>.*`` in every subject and write the records.

        Writes ``joint_tune/chosen_recipe.json``, ``joint_tune/cut_recipe.json`` (the
        recipe alone, usable as ``--cut-recipe``) and a per-subject
        ``<out_dir>/<subject>_<name>/chosen.json`` for ``mfa apply``.
        """
        key = best["key"]
        r = self.recipes[key]
        c = self._cname(key)
        files, subs = {}, {}
        for s in self.subjects:
            for h in self.hemis:
                for suf in ("patch.3d", "flat.patch.3d"):
                    src = f"{self.sd[s]}/surf/{h}.{c}.{suf}"
                    dst = f"{self.sd[s]}/surf/{h}.{self.base}.{suf}"
                    shutil.copy2(src, dst)
                    files[f"{s}/{h}.{suf}"] = dst
            qs = {h: self.q[(s, h, key)] for h in self.hemis}
            subs[s] = dict(gates_ok=bool(all(q["gates_ok"] for q in qs.values())),
                           hot_ok=bool(all(q["hot_pack"] <= self.hot_max for q in qs.values())),
                           sil_pct=best["sil"][s],
                           hemi={h: {k: x for k, x in q.items() if k not in ("recipe", "sides")}
                                 for h, q in qs.items()},
                           slit_width={h: int(r["slit_width"]) for h in self.hemis},
                           extend_mm={h: q.get("extend_mm") for h, q in qs.items()},
                           reproduce=(f"mfa flatten --subjects-dir {self.o.subjects_dir} --subject {s} --hemis lh rh "
                                      f"--name {self.base} --cut-recipe {os.path.join(self.tdir, 'cut_recipe.json')} "
                                      f"--out-dir {self.o.out_dir}"),
                           apply=(f"mfa apply --subjects-dir {self.o.subjects_dir} --subject {s} --name {self.base} "
                                  f"--out-dir {self.o.out_dir}"))
        recipe_file = dict(format=RECIPE_FORMAT, recipe=r, code=recipe_code(r), created=time.strftime("%Y-%m-%d %H:%M"),
                           source="mfa tune-joint", score=best["score"],
                           note="ONE anatomical cut route for every hemisphere of every subject; every endpoint is "
                                "anchored to a parcel or a parcel border. Apply with `mfa flatten --cut-recipe "
                                "<this file>`. slit_width is only the default/fallback: use --slit-width auto "
                                "(narrowest passing width per hemisphere).")
        json_dump(recipe_file, os.path.join(self.tdir, "cut_recipe.json"))
        chosen = dict(design="unified_recipe", recipe=r, recipe_str=recipe_str(r), code=c, name=self.base,
                      route={k: r[k] for k in RECIPE_FIELDS if k != "slit_width"}, subjects=subs,
                      metrics={k: x for k, x in best.items() if k not in ("key", "rank_key", "recipe")},
                      weights=self.W, hot_max=self.hot_max, files=files, stages=self.stages,
                      minutes=(time.time() - self.t0) / 60, recipe_file=os.path.join(self.tdir, "cut_recipe.json"),
                      new_subject=(f"mfa flatten --subject <SUBJECT> --hemis lh rh --name {self.base} --cut-recipe "
                                   f"{os.path.join(self.tdir, 'cut_recipe.json')} --slit-width auto"))
        json_dump(chosen, os.path.join(self.tdir, "chosen_recipe.json"))
        for s in self.subjects:
            d = os.path.join(self.o.out_dir, f"{s}_{self.base}")
            os.makedirs(d, exist_ok=True)
            json_dump(dict(chosen, subject=s, slit_width=subs[s]["slit_width"],
                           metrics=dict(chosen["metrics"], gates_ok=subs[s]["gates_ok"], hot_ok=subs[s]["hot_ok"]),
                           reproduce=subs[s]["reproduce"], apply=subs[s]["apply"]), os.path.join(d, "chosen.json"))
        return chosen

    def render(self, best: Mapping) -> tuple[str, str, dict[str, str]]:
        """report.png, seam3d.png and the overlay figures (cross-subject per hemisphere, lh vs rh per subject)."""
        key = best["key"]
        c = self._cname(key)
        hemi_q = {s: {h: self.q[(s, h, key)] for h in self.hemis} for s in self.subjects}
        rep = render_joint_report(self.subjects, self.sd, c, self.ranked(), best, hemi_q, self.W, self.hot_max,
                                  len(self.recipes), os.path.join(self.tdir, "report.png"), self.o.parc)
        seam = render_seam3d([(self.sd[s], c, f"{s} {c}") for s in self.subjects],
                             os.path.join(self.tdir, "seam3d.png"), self.o.parc)
        FD = {(s, h): FlatDesign(self.sd[s], h, c, self.o.parc, lay=self.lay[(s, h, key)])
              for s in self.subjects for h in self.hemis}
        over = {}
        for h in self.hemis:
            for i in range(len(self.subjects)):
                for j in range(i + 1, len(self.subjects)):
                    a, b = self.subjects[i], self.subjects[j]
                    fn = os.path.join(self.tdir, f"overlay_{a}_vs_{b}_{h}.png")
                    overlay_figure(FD[(a, h)], FD[(b, h)], f"{a} {h}", f"{b} {h}",
                                   f"Same recipe {c}, {h}: cross-subject parcel layout", fn)
                    over[f"{a}~{b}/{h}"] = fn
        for s in self.subjects:
            fn = os.path.join(self.tdir, f"overlay_lh_vs_rh_{s}.png")
            overlay_figure(FD[(s, "lh")], FD[(s, "rh")].mirrored(), f"{s} lh", f"{s} rh (mirrored)",
                           f"Same recipe {c}, {s}: lh vs mirrored rh", fn)
            over[f"{s}/lh~rh"] = fn
        return rep, seam, over

    def compare(self, best: Mapping) -> str:
        """Comparison table of the chosen recipe vs the ``compare`` designs (markdown + json)."""
        c = self._cname(best["key"])
        designs = [design_metrics(f"unified recipe {c}", {s: c for s in self.subjects}, self.o.subjects_dir,
                                  self.hemis, self.hot_max, self.o.parc)]
        for label, names in self.o.compare:
            missing = [s for s, nm in names.items() if not os.path.exists(f"{self.sd[s]}/surf/lh.{nm}.flat.patch.3d")]
            if missing or set(names) != set(self.subjects):
                log.warning("[joint] compare: skipping %s (missing patches for %s)", label, missing or "some subjects")
                continue
            designs.append(design_metrics(label, names, self.o.subjects_dir, self.hemis, self.hot_max, self.o.parc))
        md = comparison_table(designs, self.subjects, self.hemis)
        fn = os.path.join(self.tdir, "comparison.md")
        with open(fn, "w") as fp:
            fp.write(f"# Chosen recipe vs other designs ({time.strftime('%Y-%m-%d %H:%M')})\n\n"
                     f"recipe: `{recipe_str(self.recipes[best['key']])}`\n\n" + md)
        json_dump([{k: x for k, x in d.items() if not k.startswith("_")} for d in designs],
                  os.path.join(self.tdir, "comparison.json"))
        log.info("%s", md)
        return fn


def run_joint(opts: JointOptions, figures_only: bool = False, prescreen_only: bool = False) -> dict | None:
    """Driver of ``mfa tune-joint``."""
    if opts.name == "flatten":
        raise SystemExit("tune-joint: use a scratch --name; 'flatten' is reserved for the promoted patches")
    T = JointTuner(opts)
    if figures_only:
        ch = json_load(os.path.join(T.tdir, "chosen_recipe.json"))
        r = validate_recipe(ch["recipe"])
        key = recipe_key(r)
        T.recipes[key] = r
        for s in T.subjects:
            for h in T.hemis:
                T.evaluate(s, h, key)
        best = T.aggregate(key)
        rep, seam, over = T.render(best)
        T.compare(best)
        log.info("[joint] figures: %s %s %s", rep, seam, list(over.values()))
        return None
    P = T.search(prescreen_only=prescreen_only)
    if prescreen_only:
        return None
    best = P[0]
    tf = T.write_tables()
    chosen = T.export_chosen(best)
    rep, seam, over = T.render(best)
    cmp_fn = T.compare(best)
    chosen.update(report_png=rep, seam3d_png=seam, overlays=over, tables=dict(recipes=tf), comparison=cmp_fn,
                  minutes=(time.time() - T.t0) / 60)
    json_dump(chosen, os.path.join(T.tdir, "chosen_recipe.json"))
    log.info("[joint] CHOSEN recipe: %s -> <hemi>.%s.* in %s (gates %s, hot %s, score %.3f)",
             recipe_str(best["recipe"]),
             opts.name, ", ".join(T.subjects), "OK" if best["gates_ok"] else "FAIL", "OK" if best["hot_ok"] else "FAIL",
             best["score"])
    for s in T.subjects:
        log.info("[joint] %s: extension %s | reproduce: %s", s,
                 ", ".join(f"{h} {chosen['subjects'][s]['extend_mm'][h]:.1f} mm" for h in T.hemis),
                 chosen["subjects"][s]["reproduce"])
    log.info("[joint] recipe file: %s\n[joint] report: %s\n[joint] seam3d: %s\n[joint] comparison: %s\n"
             "[joint] total %.1f min", chosen["recipe_file"], rep, seam, cmp_fn, chosen["minutes"])
    if not (best["gates_ok"] and best["hot_ok"]):
        log.warning("[joint] no recipe passes every gate on every hemisphere; the best-scoring recipe is reported "
                    "(see recipe_results.tsv for the best per slit width)")
        for w in sorted({int(R["recipe"]["slit_width"]) for R in P}):
            Rw = [R for R in P if int(R["recipe"]["slit_width"]) == w]
            log.warning("   best w=%d: %s score %.3f gates %s hot %s", w, Rw[0]["code"], Rw[0]["score"],
                        Rw[0]["gates_ok"], Rw[0]["hot_ok"])
    return chosen
