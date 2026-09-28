"""Command-line interface: the ``mfa`` entry point and its subcommands."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence

from . import __version__
from .flatten import DEFAULT_SLIM_ITERS, DEFAULT_SLIM_TOL, HOT_EXTREME, HOT_MAX, HOT_PACK_MM
from .recipe import (
    AUTO_SLIT_WIDTHS,
    CALCARINE_EXTEND_OPTIONS,
    RECIPE_FIELDS,
    TEMPORAL_CORRIDORS,
    TEMPORAL_EXTEND_OPTIONS,
    TEMPORAL_SRC_OPTIONS,
    default_recipe,
    packaged_recipe_path,
    parse_recipe,
    parse_slit_width,
    recipe_code,
    recipe_str,
)
from .surface import ALL_CUTS, DEFAULT_ANNOT, DEFAULT_CUTS, Parcellation

log = logging.getLogger("mfa")

DEFAULT_OUT_DIR = "mfa_out"


# ---------------------------------------------------------------- helpers
def _add_common(p: argparse.ArgumentParser, subject: bool = True, out_dir: bool = True) -> None:
    p.add_argument("--subjects-dir", default=os.environ.get("SUBJECTS_DIR"),
                   help="FreeSurfer subjects directory (default: $SUBJECTS_DIR)")
    if subject:
        p.add_argument("--subject", required=True, help="FreeSurfer subject name ($SUBJECTS_DIR/<subject>)")
    p.add_argument("--annot", default=DEFAULT_ANNOT,
                   help=f"annotation name label/<hemi>.<annot>.annot (default {DEFAULT_ANNOT})")
    p.add_argument("--parcel-map", default=None, metavar="SPEC",
                   help="rename parcels of the annotation to the names the recipes use: "
                        "'Amy=Amygdala,TG=TemporalPole,...' or a JSON file {canonical: alias}")
    if out_dir:
        p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                       help=f"root of the outputs (figures, records; default ./{DEFAULT_OUT_DIR})")
    p.add_argument("--quiet", action="store_true", help="only warnings and errors")


def _require_subjects_dir(a: argparse.Namespace) -> str:
    if not a.subjects_dir:
        raise SystemExit("--subjects-dir is required (or set $SUBJECTS_DIR)")
    return a.subjects_dir


def _parc(a: argparse.Namespace) -> Parcellation:
    return Parcellation.from_spec(a.annot, a.parcel_map)


def _hot_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hot-pack-mm", type=float, default=HOT_PACK_MM,
                   help=f"radius (flat mm) of the local area-packing disk (default {HOT_PACK_MM:g})")
    p.add_argument("--hot-extreme", type=float, default=HOT_EXTREME,
                   help=f"extreme-edge stretch threshold for the cluster metric (default {HOT_EXTREME:g})")
    p.add_argument("--hot-max", type=float, default=HOT_MAX,
                   help=f"gate: reject a hemisphere whose max local area-packing factor exceeds this "
                        f"(default {HOT_MAX:g})")


def _slim_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--slim-iters", type=int, default=DEFAULT_SLIM_ITERS,
                   help=f"SLIM iteration cap (default {DEFAULT_SLIM_ITERS})")
    p.add_argument("--slim-tol", type=float, default=DEFAULT_SLIM_TOL,
                   help=f"SLIM stopping rule: stop when |dE| < tol * E (default {DEFAULT_SLIM_TOL:g})")
    p.add_argument("--cuts", nargs="+", default=list(DEFAULT_CUTS), choices=list(ALL_CUTS),
                   help=f"cuts to compute (default {' '.join(DEFAULT_CUTS)}; 'cingulate' adds a dorsomedial cut)")


# ---------------------------------------------------------------- subcommands
def cmd_flatten(a: argparse.Namespace) -> None:
    from .pipeline import FlattenOptions, flatten_subject

    subjects_dir = _require_subjects_dir(a)
    try:
        recipe = parse_recipe(a.cut_recipe) if a.cut_recipe else default_recipe()
    except (ValueError, OSError) as ex:
        raise SystemExit(f"--cut-recipe: {ex}") from None
    try:
        slit = parse_slit_width(a.slit_width)
    except ValueError as ex:
        raise SystemExit(f"--slit-width: {ex}") from None
    source = a.cut_recipe or packaged_recipe_path()
    if a.print_recipe:
        print(f"recipe source: {source}")
        for h in a.hemis:
            w = slit.get(h)
            r = dict(recipe, slit_width=(recipe["slit_width"] if w in (None, "auto") else w))
            print(f"  {h}: {recipe_str(r)}  (code {recipe_code(r)})"
                  + (f"  [slit_width auto: narrowest passing of {AUTO_SLIT_WIDTHS}]" if w == "auto" else ""))
        return
    opts = FlattenOptions(subjects_dir=subjects_dir, subject=a.subject, name=a.name, recipe=recipe, slit_width=slit,
                          hemis=tuple(a.hemis), cuts=tuple(a.cuts), parc=_parc(a), slim_iters=a.slim_iters,
                          slim_tol=a.slim_tol, orient=not a.no_orient, orient_backup=not a.no_orient_backup,
                          patch_only=a.patch_only, render=not a.no_render, out_dir=a.out_dir, run_json=a.run_json,
                          hot_pack_mm=a.hot_pack_mm, hot_extreme=a.hot_extreme, hot_max=a.hot_max,
                          recipe_source=source)
    try:
        flatten_subject(opts)
    except (RuntimeError, ValueError) as ex:
        raise SystemExit(str(ex)) from None


def cmd_apply(a: argparse.Namespace) -> None:
    from .pycortex_io import promote

    promote(_require_subjects_dir(a), a.subject, a.name, a.out_dir, cx_subject=a.pycortex_subject,
            hemis=tuple(a.hemis), force=a.force, dry_run=a.dry_run, render=not a.no_render)


def cmd_import_subject(a: argparse.Namespace) -> None:
    from .pycortex_io import import_subject

    cx = a.pycortex_subject or a.subject.replace("sub-", "")
    import_subject(a.subject, cx, _require_subjects_dir(a))


def cmd_tune_joint(a: argparse.Namespace) -> None:
    from .tune import JointOptions, parse_compare_specs, run_joint

    opts = JointOptions(subjects=list(a.subjects), subjects_dir=_require_subjects_dir(a), out_dir=a.out_dir,
                        name=a.name, cuts=tuple(a.cuts), parc=_parc(a), slim_iters=a.slim_iters, slim_tol=a.slim_tol,
                        widths=tuple(a.widths), srcs=tuple(a.srcs), dsts=tuple(a.dsts), corridors=tuple(a.corridors),
                        extends=tuple(a.extends), calcarine=tuple(a.calcarine), stage_cap=a.stage_cap,
                        full_cap=a.full_cap, start_width=a.start_width, start_extend=a.start_extend, w_sil=a.w_sil,
                        w_p99=a.w_p99, w_loss=a.w_loss, w_hot=a.w_hot, w_x=a.w_x, w_x2=a.w_x2, hot_max=a.hot_max,
                        hot_pack_mm=a.hot_pack_mm, hot_extreme=a.hot_extreme, max_parallel=a.max_parallel,
                        threads=a.threads, prescreen_workers=a.prescreen_workers, no_refine=a.no_refine,
                        compare=parse_compare_specs(a.compare))
    run_joint(opts, figures_only=a.figures_only, prescreen_only=a.prescreen_only)


def cmd_reference(a: argparse.Namespace) -> None:
    from .layout import write_reference_layout

    sd = os.path.join(_require_subjects_dir(a), a.subject)
    write_reference_layout(sd, a.subject, a.name, tuple(a.hemis), a.out, _parc(a))


def cmd_compare(a: argparse.Namespace) -> None:
    from .layout import FlatDesign, comparison_table, design_metrics
    from .qc import overlay_figure
    from .tune import parse_compare_specs
    from .utils import json_dump

    subjects_dir = _require_subjects_dir(a)
    parc = _parc(a)
    designs_spec = []
    if a.subjects:
        designs_spec.append((a.name, {s: a.name for s in a.subjects}))
    designs_spec += parse_compare_specs(a.designs)
    if not designs_spec:
        raise SystemExit("give --subjects (and optionally --name), or --designs")
    subjects = list(designs_spec[0][1])
    out_dir = os.path.join(a.out_dir, "compare")
    os.makedirs(out_dir, exist_ok=True)
    designs = []
    for label, names in designs_spec:
        if set(names) != set(subjects):
            raise SystemExit(f"design {label!r} must list the same subjects as the first design ({subjects})")
        designs.append(design_metrics(label, names, subjects_dir, tuple(a.hemis), a.hot_max, parc))
    md = comparison_table(designs, subjects, tuple(a.hemis))
    with open(os.path.join(out_dir, "comparison.md"), "w") as fp:
        fp.write("# Layout comparison\n\n" + md)
    json_dump([{k: x for k, x in d.items() if not k.startswith("_")} for d in designs],
              os.path.join(out_dir, "comparison.json"))
    print(md)
    for d in designs:
        FD: dict[tuple[str, str], FlatDesign] = d["_FD"]
        tag = d["label"].replace(" ", "_").replace("/", "-")
        for h in a.hemis:
            for i in range(len(subjects)):
                for j in range(i + 1, len(subjects)):
                    s1, s2 = subjects[i], subjects[j]
                    overlay_figure(FD[(s1, h)], FD[(s2, h)], f"{s1} {h}", f"{s2} {h}",
                                   f"{d['label']}, {h}: cross-subject parcel layout",
                                   os.path.join(out_dir, f"overlay_{tag}_{s1}_vs_{s2}_{h}.png"))
        if "lh" in a.hemis and "rh" in a.hemis:
            for s in subjects:
                overlay_figure(FD[(s, "lh")], FD[(s, "rh")].mirrored(), f"{s} lh", f"{s} rh (mirrored)",
                               f"{d['label']}, {s}: lh vs mirrored rh",
                               os.path.join(out_dir, f"overlay_{tag}_lh_vs_rh_{s}.png"))
    log.info("[compare] outputs in %s", out_dir)


def cmd_overlay_parcels(a: argparse.Namespace) -> None:
    from .overlays import draw_parcels, render_layer

    subjects_dir = _require_subjects_dir(a)
    cx = a.pycortex_subject or a.subject.replace("sub-", "")
    draw_parcels(cx, a.subject, subjects_dir, _parc(a), layer=a.layer, skip=tuple(a.skip), min_faces=a.min_faces,
                 font_size=a.font_size, min_label_depth=a.min_label_depth, min_label_share=a.min_label_share)
    if a.render:
        title = a.render_title if a.render_title is not None else f"{cx}: {a.annot} parcels (layer {a.layer})"
        render_layer(cx, a.subject, subjects_dir, a.layer, a.render, labelsize=a.render_labelsize, title=title or None)


def cmd_atlas_annot(a: argparse.Namespace) -> None:
    from .atlas import atlas_to_annot

    cx = a.pycortex_subject or a.subject.replace("sub-", "")
    atlas_to_annot(a.volume, a.lookup, cx, a.subject, _require_subjects_dir(a), a.out_name, id_col=a.id_col,
                   name_col=a.name_col, region_col=(a.region_col or None), color_col=(a.color_col or None),
                   valid_regions=tuple(a.valid_regions), extra_valid=tuple(a.extra_valid), depths=tuple(a.depths),
                   hemis=tuple(a.hemis))


def cmd_qc(a: argparse.Namespace) -> None:
    from .flatten import hemi_quality, passes_gates, quality_line, silhouette_norm
    from .layout import SIDE_CONF_MIN, flat_layout, load_reference_layout, reference_metrics
    from .pipeline import run_dir
    from .qc import render_butterfly, render_hotspot_map, render_qc
    from .recipe import SEAM_PARCELS
    from .utils import json_dump

    sd = os.path.join(_require_subjects_dir(a), a.subject)
    parc = _parc(a)
    rdir = run_dir(a.out_dir, a.subject, a.name)
    os.makedirs(rdir, exist_ok=True)
    ref = load_reference_layout(a.reference_layout) if a.reference_layout else None
    out: dict = dict(subject=a.subject, name=a.name, hemi={})
    for h in a.hemis:
        q = hemi_quality(sd, h, a.name, parc, nbhd_mm=a.hot_pack_mm, extreme=a.hot_extreme)
        print(f"[{h}] {a.name}: {quality_line(q, a.hot_max)}")
        if ref is not None:
            lay = flat_layout(sd, h, a.name, parc)
            if h in ref.get("hemi", {}):
                q.update(reference_metrics(lay, ref["hemi"][h], tuple(ref.get("seam_parcels", SEAM_PARCELS)),
                                           float(ref.get("side_conf_min", SIDE_CONF_MIN))))
                sides = " ".join(f"{p}:{s['cand']:+d}{'' if s['cand'] == s['ref'] else '!'}"
                                 for p, s in q["ref_sides"].items())
                print(f"[{h}] vs reference: rms {q['ref_rms']:.2f} (mean {q['ref_mean']:.2f}, max {q['ref_max']:.2f}; "
                      f"1 unit = {lay['scale_mm_per_unit']:.3f} mm here), seam parcels rms "
                      f"{q['ref_temporal_rms']:.2f}, "
                      f"side agreement {q['ref_side_agree']:.2f} [{sides}] (! = differs from reference)")
            else:
                print(f"[{h}] reference layout has no {h} entry")
        out["hemi"][h] = q
        out["hemi"][h]["passes"] = passes_gates(q, a.hot_max)
        if not a.no_render:
            render_qc(sd, h, a.name, os.path.join(rdir, f"{h}_qc.png"), parc)
            render_hotspot_map(sd, h, a.name, os.path.join(rdir, f"{h}_hotspot.png"), parc, a.hot_pack_mm, a.hot_max)
    if "lh" in a.hemis and "rh" in a.hemis:
        sil = silhouette_norm(sd, a.name, a.name)
        out["silhouette"] = sil
        print(f"lh-vs-mirrored-rh silhouette: {sil['sil_mm']:.3f} mm = {sil['sil_pct']:.3f} % of the mean perimeter")
        if not a.no_render:
            render_butterfly(sd, a.name, os.path.join(rdir, "butterfly.png"), parc,
                             title=f"{a.subject} {a.name}: silhouette {sil['sil_pct']:.3f} % (green = V1)")
    json_dump(out, os.path.join(rdir, "qc.json"))
    log.info("[qc] outputs in %s", rdir)


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """The ``mfa`` argument parser."""
    ap = argparse.ArgumentParser(prog="mfa",
                                 description="Automatic macaque cortical flatmaps from FreeSurfer subjects: "
                                 "landmark-defined cuts, SLIM flattening, quality gates, pycortex import, overlays.")
    ap.add_argument("--version", action="version", version=f"mfa {__version__}")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("flatten", help="cut, flatten, orient and gate one subject with the shared recipe")
    _add_common(p)
    p.add_argument("--hemis", nargs="+", default=["lh", "rh"], choices=["lh", "rh"])
    p.add_argument("--name", default="mfa", help="scratch patch name surf/<hemi>.<name>.* (default mfa; "
                                                  "'flatten' is reserved for `mfa apply`)")
    p.add_argument("--cut-recipe", default=None, metavar="SPEC",
                   help="cut recipe: a JSON file, 'key=value,...' or 'FILE,key=value,...' (fields: "
                        + ", ".join(RECIPE_FIELDS) + f"); default: the packaged recipe {packaged_recipe_path()}")
    p.add_argument("--slit-width", default=None, metavar="W",
                   help="vertex rings removed around the seam route: N (both hemispheres), 'auto' (per hemisphere the "
                        f"narrowest width in {AUTO_SLIT_WIDTHS} whose flattened patch passes every gate; recommended), "
                        "or 'lh=1,rh=auto'. Default: the recipe's value")
    p.add_argument("--print-recipe", action="store_true", help="print the resolved recipe and exit")
    _slim_args(p)
    _hot_args(p)
    p.add_argument("--patch-only", action="store_true", help="build and verify the patch, do not flatten")
    p.add_argument("--no-orient", action="store_true", help="skip the canonical orientation")
    p.add_argument("--no-orient-backup", action="store_true", help="orient without the .precanon backup copy")
    p.add_argument("--no-render", action="store_true", help="skip the QC figures")
    p.add_argument("--run-json", default=None, metavar="FILE", help="also write the run record to this file")
    p.set_defaults(func=cmd_flatten)

    p = sp.add_parser("apply", help="promote <hemi>.<name>.* to flatten.*, import into pycortex, redraw overlays.svg")
    _add_common(p)
    p.add_argument("--name", default="mfa", help="scratch patch name to promote (default mfa)")
    p.add_argument("--hemis", nargs="+", default=["lh", "rh"], choices=["lh", "rh"])
    p.add_argument("--pycortex-subject", default=None, help="pycortex subject (default: --subject without 'sub-')")
    p.add_argument("--force", action="store_true", help="promote even without a passing gate record")
    p.add_argument("--dry-run", action="store_true", help="only print the plan")
    p.add_argument("--no-render", action="store_true", help="skip the final flat-map render")
    p.set_defaults(func=cmd_apply)

    p = sp.add_parser("import-subject", help="create the pycortex subject from a FreeSurfer subject")
    _add_common(p, out_dir=False)
    p.add_argument("--pycortex-subject", default=None, help="pycortex subject (default: --subject without 'sub-')")
    p.set_defaults(func=cmd_import_subject)

    p = sp.add_parser("tune-joint", help="joint search of ONE recipe over every hemisphere of several subjects")
    _add_common(p, subject=False)
    p.add_argument("--subjects", nargs="+", required=True, help="subjects of the joint search")
    p.add_argument("--name", default="uc", help="scratch patch name of the chosen recipe (default uc)")
    _slim_args(p)
    _hot_args(p)
    p.add_argument("--widths", nargs="+", type=int, default=[1, 2, 3], help="slit widths searched (one shared value)")
    p.add_argument("--srcs", nargs="+", default=list(TEMPORAL_SRC_OPTIONS), help="temporal_src options")
    p.add_argument("--dsts", nargs="+", default=["TG_ant", "TG_ITC_border", "Amy_ant", "MTL_ant"],
                   help="temporal_dst options")
    p.add_argument("--corridors", nargs="+", default=list(TEMPORAL_CORRIDORS), help="temporal_corridor options")
    p.add_argument("--extends", nargs="+", default=list(TEMPORAL_EXTEND_OPTIONS), help="temporal_extend options")
    p.add_argument("--calcarine", nargs="+", default=["none"],
                   help=f"calcarine_extend options (default none; options {CALCARINE_EXTEND_OPTIONS})")
    p.add_argument("--stage-cap", type=int, default=20, help="recipes flattened in stage 1 (routes)")
    p.add_argument("--full-cap", type=int, default=40,
                   help="if at most this many distinct recipes pass the pre-screen, flatten them all")
    p.add_argument("--start-width", type=int, default=None, help="slit width of stage 1 (default: automatic)")
    p.add_argument("--start-extend", default="TG_post", help="temporal_extend of stage 1 (default TG_post)")
    p.add_argument("--w-sil", type=float, default=2.5, help="weight of the lh-vs-rh silhouette %%")
    p.add_argument("--w-p99", type=float, default=1.0, help="weight of (p99 stretch - 1)")
    p.add_argument("--w-loss", type=float, default=0.4, help="weight of the cortex area loss %%")
    p.add_argument("--w-hot", type=float, default=1.0, help="weight of (hot_pack - 1)")
    p.add_argument("--w-x", type=float, default=0.5, help="weight of the cross-subject layout RMS (units)")
    p.add_argument("--w-x2", type=float, default=3.0, help="weight of the cross-subject seam-side disagreement")
    p.add_argument("--max-parallel", type=int, default=6, help="concurrent SLIM flattens")
    p.add_argument("--threads", type=int, default=4, help="OMP/BLAS threads per flatten job")
    p.add_argument("--prescreen-workers", type=int, default=12, help="parallel pre-screen processes")
    p.add_argument("--no-refine", action="store_true", help="skip the one-field neighbourhood refinement")
    p.add_argument("--compare", nargs="*", default=None, metavar="LABEL:sub=name,...",
                   help="extra designs for the comparison table")
    p.add_argument("--figures-only", action="store_true", help="only regenerate figures/tables of an existing search")
    p.add_argument("--prescreen-only", action="store_true", help="run the pre-screen, write prescreen.tsv, exit")
    p.set_defaults(func=cmd_tune_joint)

    p = sp.add_parser("reference", help="write a reference parcel layout from a flattened subject")
    _add_common(p, out_dir=False)
    p.add_argument("--name", default="flatten", help="flat patch name (default flatten)")
    p.add_argument("--hemis", nargs="+", default=["lh", "rh"], choices=["lh", "rh"])
    p.add_argument("--out", required=True, metavar="FILE", help="output JSON")
    p.set_defaults(func=cmd_reference)

    p = sp.add_parser("compare", help="cross-hemisphere / cross-subject layout comparison of one or more designs")
    _add_common(p, subject=False)
    p.add_argument("--subjects", nargs="+", metavar="SUBJECT",
                   help="subjects to compare; with --name this compares one design across them")
    p.add_argument("--name", default="flatten",
                   help="patch name of the design to compare across --subjects (default: flatten, the promoted maps)")
    p.add_argument("--designs", nargs="*", default=[], metavar="LABEL:sub=name,...",
                   help="advanced: additional designs whose patch name differs per subject, "
                        "e.g. 'candidate:sub-01=uc,sub-02=uc2' (may be repeated)")
    p.add_argument("--hemis", nargs="+", default=["lh", "rh"], choices=["lh", "rh"])
    p.add_argument("--hot-max", type=float, default=HOT_MAX)
    p.set_defaults(func=cmd_compare)

    p = sp.add_parser("overlay-parcels", help="draw an annotation into pycortex's overlays.svg (outlines + labels)")
    _add_common(p, out_dir=False)
    p.add_argument("--pycortex-subject", default=None, help="pycortex subject (default: --subject without 'sub-')")
    p.add_argument("--layer", default="rois", help="overlays.svg layer to draw into (default rois)")
    p.add_argument("--skip", nargs="*", default=["unknown", "Unknown", "???"], help="label names to skip")
    p.add_argument("--min-faces", type=int, default=40, help="drop parcel islands smaller than this (faces)")
    p.add_argument("--font-size", default="11pt")
    p.add_argument("--min-label-depth", type=float, default=14.0,
                   help="skip the label of a piece thinner than this (px from border to its deepest point)")
    p.add_argument("--min-label-share", type=float, default=0.10,
                   help="skip the label of a piece holding less than this fraction of the parcel")
    p.add_argument("--render", default=None, metavar="PNG", help="also render curvature + outlines + labels")
    p.add_argument("--render-labelsize", default="14pt")
    p.add_argument("--render-title", default=None, help="figure title ('' for none)")
    p.set_defaults(func=cmd_overlay_parcels)

    p = sp.add_parser("atlas-annot", help="resample a label volume onto the surface -> .annot files")
    _add_common(p, out_dir=False)
    p.add_argument("--pycortex-subject", default=None, help="pycortex subject (default: --subject without 'sub-')")
    p.add_argument("--volume", required=True, help="integer label volume (NIfTI/MGZ) in the subject's scanner space")
    p.add_argument("--lookup", required=True, help="TSV lookup table (id, name[, region, color] columns)")
    p.add_argument("--out-name", required=True, help="annot name: label/<hemi>.<out-name>.annot")
    p.add_argument("--id-col", default="ID")
    p.add_argument("--name-col", default="name")
    p.add_argument("--region-col", default="region", help="'' if the table has no region column")
    p.add_argument("--color-col", default="color", help="'' if the table has no colour column")
    p.add_argument("--valid-regions", nargs="*", default=["cortex"], help="regions kept")
    p.add_argument("--extra-valid", nargs="*", default=[],
                   help="label names kept regardless of region (allocortex that reaches the surface)")
    p.add_argument("--depths", nargs="*", type=float, default=[0.2, 0.35, 0.5, 0.65, 0.8])
    p.add_argument("--hemis", nargs="+", default=["lh", "rh"], choices=["lh", "rh"])
    p.set_defaults(func=cmd_atlas_annot)

    p = sp.add_parser("qc", help="metrics and figures of existing flat patches")
    _add_common(p)
    p.add_argument("--name", default="flatten", help="flat patch name (default flatten)")
    p.add_argument("--hemis", nargs="+", default=["lh", "rh"], choices=["lh", "rh"])
    p.add_argument("--reference-layout", default=None, metavar="FILE", help="score against a reference layout")
    _hot_args(p)
    p.add_argument("--no-render", action="store_true")
    p.set_defaults(func=cmd_qc)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``mfa`` command."""
    ap = build_parser()
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if getattr(a, "quiet", False) else logging.INFO,
                        format="%(message)s", stream=sys.stdout)
    a.func(a)


if __name__ == "__main__":
    main()
