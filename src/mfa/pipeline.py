"""Per-subject pipeline: recipe -> seam routes -> slit patch -> SLIM -> orientation -> gates -> QC.

The seam *route* comes from the recipe and is identical for every hemisphere of
every subject.  The slit *width* is a technical parameter: with the ``auto``
policy the narrowest width in :data:`mfa.recipe.AUTO_SLIT_WIDTHS` whose
flattened patch passes every gate is kept (least cortex removed).
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .flatten import (
    DEFAULT_SLIM_ITERS,
    DEFAULT_SLIM_TOL,
    HOT_EXTREME,
    HOT_MAX,
    HOT_PACK_MM,
    canonical_orient,
    hemi_quality,
    passes_gates,
    quality_line,
    silhouette_norm,
    slim_flatten,
)
from .patch_io import write_patch
from .qc import render_qc
from .recipe import AUTO_SLIT_WIDTHS, ROUTE_FIELDS, Recipe, recipe_str, route_kwargs
from .surface import DEFAULT_CUTS, DEFAULT_PARCELLATION, Hemi, Parcellation
from .utils import json_dump, json_load

log = logging.getLogger("mfa")

#: Patch name reserved for the promoted (canonical) flat map of a subject.
PROMOTED_NAME = "flatten"


class RecipeNotRealisable(RuntimeError):
    """A landmark of the recipe is unreachable on this hemisphere (no fallback route is taken)."""


class PatchTopologyError(RuntimeError):
    """The slit patch is not a disk (several boundary loops or isolated vertices) at a fixed width."""


class NoPassingWidth(RuntimeError):
    """No slit width in :data:`mfa.recipe.AUTO_SLIT_WIDTHS` passes the gates with this route."""


@dataclass
class FlattenOptions:
    """Options of :func:`flatten_subject`.

    Attributes
    ----------
    subjects_dir, subject : str
        ``$SUBJECTS_DIR`` and the FreeSurfer subject name.
    name : str
        Scratch patch name: ``surf/<hemi>.<name>.{patch,flat.patch}.3d`` (``flatten`` is reserved).
    recipe : dict
        Validated cut recipe (the route plus the default slit width).
    slit_width : {hemi: int | 'auto' | None}
        Per-hemisphere width; None = the recipe value.
    hemis, cuts : sequences
    parc : Parcellation
    slim_iters, slim_tol : SLIM stopping rule.
    orient, orient_backup : canonical orientation and its ``.precanon`` backup.
    patch_only : build and verify the patch, do not flatten.
    render : write the QC figures.
    out_dir : root of the run outputs (``<out_dir>/<subject>_<name>/``).
    run_json : optional path for the per-hemisphere run record.
    hot_pack_mm, hot_extreme, hot_max : hotspot detector settings and gate.
    recipe_source : where the recipe came from (documentation only).
    """

    subjects_dir: str
    subject: str
    name: str
    recipe: Recipe
    slit_width: dict[str, int | str | None] = field(default_factory=lambda: {"lh": None, "rh": None})
    hemis: Sequence[str] = ("lh", "rh")
    cuts: Sequence[str] = DEFAULT_CUTS
    parc: Parcellation = DEFAULT_PARCELLATION
    slim_iters: int = DEFAULT_SLIM_ITERS
    slim_tol: float = DEFAULT_SLIM_TOL
    orient: bool = True
    orient_backup: bool = True
    patch_only: bool = False
    render: bool = True
    out_dir: str = "mfa_out"
    run_json: str | None = None
    hot_pack_mm: float = HOT_PACK_MM
    hot_extreme: float = HOT_EXTREME
    hot_max: float = HOT_MAX
    recipe_source: str = "packaged"

    @property
    def sd(self) -> str:
        return os.path.join(self.subjects_dir, self.subject)

    @property
    def run_dir(self) -> str:
        return run_dir(self.out_dir, self.subject, self.name)


def run_dir(out_dir: str, subject: str, name: str) -> str:
    """Directory of one subject/name run: ``<out_dir>/<subject>_<name>``."""
    return os.path.join(out_dir, f"{subject}_{name}")


def flatten_hemisphere(H: Hemi, opts: FlattenOptions) -> dict:
    """Cut, flatten, orient and gate one hemisphere; returns the run record.

    With ``opts.slit_width[hemi] == 'auto'`` the widths of
    :data:`mfa.recipe.AUTO_SLIT_WIDTHS` are tried in order on the same route (the
    attempts stay as ``<hemi>.<name>_autow<w>.*``) and the first passing one is
    copied to ``<hemi>.<name>.*``.
    """
    hemi, sd = H.hemi, opts.sd
    log.info("[%s] computing seam paths...", hemi)
    log.info("  [%s] cut recipe (%s): %s", hemi, opts.recipe_source, recipe_str(opts.recipe))
    try:
        paths = H.seam_paths(opts.cuts, **route_kwargs(opts.recipe))
    except ValueError as ex:
        raise RecipeNotRealisable(f"[{hemi}] recipe {recipe_str(opts.recipe)} is not realisable on this "
                                  f"hemisphere: {ex}") from ex
    for k, p in paths.items():
        log.info("  [%s] %s: %d vertices", hemi, k, len(p))
    si = dict(H.seam_info)
    if si.get("temporal", {}).get("extend_mm", 0.0):
        log.info("  [%s] temporal extension realised: %.1f mm (%s)", hemi, si["temporal"]["extend_mm"],
                 si["temporal"]["extend"])
    w_opt = opts.slit_width.get(hemi)
    auto_w = w_opt == "auto"
    widths = list(AUTO_SLIT_WIDTHS) if auto_w else [int(w_opt if w_opt is not None else opts.recipe["slit_width"])]
    attempts: list[dict] = []
    for w in widths:
        nm = f"{opts.name}_autow{w}" if auto_w else opts.name
        if auto_w:
            log.info("[%s] slit width auto: trying w=%d -> %s.%s.*", hemi, w, hemi, nm)
        ni, border, stats = H.build_slit_patch(paths, w)
        info = dict(patch=stats, slit_width=w, name=nm, recipe=dict(opts.recipe, slit_width=w),
                    recipe_source=opts.recipe_source, slit_width_mode="auto" if auto_w else "fixed",
                    seam_info=si, seam_lengths={k: int(len(p)) for k, p in paths.items()},
                    temporal_seam_vertices=[int(x) for x in paths.get("temporal", [])])
        attempts.append(info)
        if not (stats["loops"] == 1 and stats["isolated"] == 0):
            info["gates_ok"] = False
            msg = (f"[{hemi}] w={w}: patch has {stats['loops']} boundary loops / {stats['isolated']} "
                   f"isolated vertices")
            if auto_w:
                log.info("  %s -> next width", msg)
                continue
            raise PatchTopologyError(msg + " (try another --slit-width or 'auto')")
        pf = f"{sd}/surf/{hemi}.{nm}.patch.3d"
        write_patch(pf, ni, H.v, border)
        log.info("  [%s] wrote %s", hemi, pf)
        if opts.patch_only:
            info["gates_ok"] = True
            break
        log.info("[%s] SLIM flatten (iters<=%d, tol=%g)...", hemi, opts.slim_iters, opts.slim_tol)
        t0 = time.time()
        trace = slim_flatten(sd, hemi, nm, iters=opts.slim_iters, tol=opts.slim_tol)
        info["slim"] = dict(iters=len(trace) - 1, energy=trace[-1][1], flips=trace[-1][2], secs=time.time() - t0)
        if opts.orient:
            R, fit = canonical_orient(sd, hemi, nm, backup=opts.orient_backup)
            info["orient"] = dict(corr=fit, det=float(np.linalg.det(R)))
        q = hemi_quality(sd, hemi, nm, opts.parc, nbhd_mm=opts.hot_pack_mm, extreme=opts.hot_extreme)
        info["quality"] = q
        ok = passes_gates(q, opts.hot_max)
        info["gates_ok"] = ok
        log.info("  [%s] w=%d: %s", hemi, w, quality_line(q, opts.hot_max))
        if ok or not auto_w:
            break
    runinfo = dict(attempts[-1])
    if auto_w:
        keys = ("loops", "isolated", "components", "flips", "orient_corr", "hot_pack", "hot_parcel",
                "stretch_p95", "stretch_p99", "cortex_loss_area_pct", "gates_ok")
        runinfo["attempts"] = [dict(a, quality={k: a["quality"][k] for k in keys} if "quality" in a else None)
                               for a in attempts]
        last = attempts[-1]
        if not last.get("gates_ok"):
            raise NoPassingWidth(f"[{hemi}] --slit-width auto: no width in {AUTO_SLIT_WIDTHS} passes the gates "
                                 f"with this route (see the per-width lines above); the attempts are kept as "
                                 f"{hemi}.{opts.name}_autow*.*")
        for suf in ("patch.3d",) + (() if opts.patch_only else ("flat.patch.3d",)):
            shutil.copy2(f"{sd}/surf/{hemi}.{last['name']}.{suf}", f"{sd}/surf/{hemi}.{opts.name}.{suf}")
        log.info("[%s] slit width auto: chose w=%d (%s.%s.* -> %s.%s.*; tried %s)", hemi, last["slit_width"],
                 hemi, last["name"], hemi, opts.name, ", ".join(str(a["slit_width"]) for a in attempts))
    return runinfo


def flatten_subject(opts: FlattenOptions) -> dict:
    """Run the whole per-subject pipeline for ``opts.hemis``.

    Writes ``surf/<hemi>.<name>.{patch,flat.patch}.3d``, the run record
    (``<run_dir>/run.json`` and ``opts.run_json``), ``<run_dir>/chosen.json`` (read
    by ``mfa apply``) and the QC figures ``<run_dir>/<hemi>_qc.png``.
    """
    if opts.name == PROMOTED_NAME:
        raise ValueError(f"--name {PROMOTED_NAME!r} is reserved for the promoted patches; use a scratch name, "
                         "then `mfa apply` promotes it")
    os.makedirs(opts.run_dir, exist_ok=True)
    runinfo: dict[str, dict] = {}
    for hemi in opts.hemis:
        H = Hemi(opts.sd, hemi, opts.parc)
        runinfo[hemi] = flatten_hemisphere(H, opts)
    if not opts.patch_only:
        quals = {h: runinfo[h]["quality"] for h in opts.hemis}
        for h in opts.hemis:
            log.info("[%s] %s (slit width %d): %s", h, opts.name, runinfo[h]["slit_width"],
                     quality_line(quals[h], opts.hot_max))
        write_chosen(opts, runinfo, quals)
        if opts.render:
            for h in opts.hemis:
                out = os.path.join(opts.run_dir, f"{h}_qc.png")
                render_qc(opts.sd, h, opts.name, out, opts.parc)
                log.info("[%s] QC figure -> %s", h, out)
    json_dump(runinfo, os.path.join(opts.run_dir, "run.json"))
    if opts.run_json:
        json_dump(runinfo, opts.run_json)
    return runinfo


def write_chosen(opts: FlattenOptions, runinfo: Mapping[str, Mapping], quals: Mapping[str, Mapping]) -> str:
    """Write ``<run_dir>/chosen.json``: route, realised slit width and gates per hemisphere, silhouette.

    ``mfa apply`` reads it to refuse the promotion of a design that failed a
    gate.  A hemisphere run separately is merged into an existing file for the
    same name and route.
    """
    fn = os.path.join(opts.run_dir, "chosen.json")
    route = {k: opts.recipe[k] for k in ROUTE_FIELDS}
    hemi = {h: dict(quals[h], slit_width=int(runinfo[h]["slit_width"]),
                    slit_width_mode=runinfo[h].get("slit_width_mode", "fixed"),
                    extend_mm=float(runinfo[h].get("seam_info", {}).get("temporal", {}).get("extend_mm", 0.0)),
                    slim_iters=runinfo[h].get("slim", {}).get("iters"),
                    attempts=[dict(slit_width=a["slit_width"], gates_ok=a.get("gates_ok"), quality=a.get("quality"))
                              for a in runinfo[h].get("attempts", [])])
            for h in opts.hemis}
    if os.path.exists(fn):
        try:
            old = json_load(fn)
            if old.get("design") == "unified_recipe" and old.get("route") == route:
                for h, q in old.get("subjects", {}).get(opts.subject, {}).get("hemi", {}).items():
                    hemi.setdefault(h, q)
        except (OSError, ValueError):
            pass
    hs = sorted(hemi)
    widths = {h: hemi[h]["slit_width"] for h in hs}
    sub = dict(gates_ok=bool(all(hemi[h]["gates_ok"] for h in hs)),
               hot_ok=bool(all(hemi[h]["hot_pack"] <= opts.hot_max for h in hs)),
               slit_width=widths, hemi=hemi, extend_mm={h: hemi[h]["extend_mm"] for h in hs},
               reproduce=(f"mfa flatten --subjects-dir {opts.subjects_dir} --subject {opts.subject} --hemis "
                          f"{' '.join(hs)} --name {opts.name} --cut-recipe "
                          f"\"{recipe_str(dict(opts.recipe, slit_width=widths[hs[0]]))}\" "
                          f"--slit-width {','.join(f'{h}={widths[h]}' for h in hs)} --out-dir {opts.out_dir}"),
               apply=(f"mfa apply --subjects-dir {opts.subjects_dir} --subject {opts.subject} --name {opts.name} "
                      f"--out-dir {opts.out_dir}"))
    if hs == ["lh", "rh"]:
        try:
            sub["sil_pct"] = float(silhouette_norm(opts.sd, opts.name, opts.name)["sil_pct"])
        except (OSError, ValueError, IndexError) as ex:  # informational only
            sub["sil_pct"] = None
            log.warning("[recipe] silhouette not computed: %s", ex)
    chosen = dict(design="unified_recipe", route=route, recipe=dict(opts.recipe), recipe_str=recipe_str(opts.recipe),
                  recipe_source=opts.recipe_source, slit_width=widths,
                  slit_width_mode={h: hemi[h]["slit_width_mode"] for h in hs},
                  slit_width_policy=("route = shared anatomical standard (recipe); width = narrowest passing width "
                                     f"per hemisphere ({AUTO_SLIT_WIDTHS}; gates: 1 loop, 0 isolated, 1 component, "
                                     f"0 flips, orientation, hot_pack <= {opts.hot_max:g})"),
                  name=opts.name, subject=opts.subject, subjects={opts.subject: sub},
                  metrics=dict(gates_ok=sub["gates_ok"], hot_ok=sub["hot_ok"], sil_pct=sub.get("sil_pct")),
                  hot_max=opts.hot_max, created=time.strftime("%Y-%m-%d %H:%M"), source="mfa flatten",
                  reproduce=sub["reproduce"], apply=sub["apply"])
    json_dump(chosen, fn)
    log.info("[recipe] %s %s: slit width %s; gates %s, hot %s%s -> %s", opts.subject, opts.name,
             ", ".join(f"{h}={widths[h]}" for h in hs), "OK" if sub["gates_ok"] else "FAIL",
             "OK" if sub["hot_ok"] else "FAIL",
             f", lh-vs-rh silhouette {sub['sil_pct']:.3f} %" if sub.get("sil_pct") is not None else "", fn)
    log.info("[recipe] promote with: %s", sub["apply"])
    return fn
