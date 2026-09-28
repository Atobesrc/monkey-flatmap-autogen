"""pycortex plumbing: subject import, flat import, overlays.svg regeneration, promotion with backups.

pycortex is imported lazily so that the cutting / flattening / QC parts of the
package work without it.
"""

from __future__ import annotations

import filecmp
import logging
import os
import shutil
from collections.abc import Sequence

import numpy as np

from .pipeline import PROMOTED_NAME, run_dir
from .recipe import recipe_str
from .utils import json_load, timestamp

log = logging.getLogger("mfa")


def _cortex():
    try:
        import cortex
    except ImportError as ex:  # pragma: no cover - depends on the environment
        raise ImportError("pycortex is required for this step: pip install "
                          "'git+https://github.com/gallantlab/pycortex.git' (see README)") from ex
    return cortex


def filestore() -> str:
    """pycortex's database directory (``filestore`` in its options.cfg)."""
    return _cortex().database.default_filestore


def import_subject(fs_subject: str, cx_subject: str, subjects_dir: str) -> str:
    """Create the pycortex subject (white / pial / inflated as GIFTI) from a FreeSurfer subject."""
    cortex = _cortex()
    cortex.freesurfer.import_subj(fs_subject, pycortex_subject=cx_subject, freesurfer_subject_dir=subjects_dir)
    db = os.path.join(filestore(), cx_subject)
    log.info("[pycortex] imported %s -> %s", fs_subject, db)
    return db


def import_flat(fs_subject: str, name: str, cx_subject: str, subjects_dir: str) -> None:
    """Import ``surf/<hemi>.<name>.flat.patch.3d`` of both hemispheres as the pycortex flat surface."""
    cortex = _cortex()
    cortex.freesurfer.import_flat(fs_subject, name, cx_subject=cx_subject, freesurfer_subject_dir=subjects_dir,
                                  auto_overwrite=True)
    log.info("[pycortex] imported flat %s.%s -> %s", fs_subject, name, cx_subject)


def clear_cache(cx_subject: str) -> None:
    """Delete pycortex's cached flat-pixel maps of a subject (stale after any geometry change)."""
    shutil.rmtree(os.path.join(filestore(), cx_subject, "cache"), ignore_errors=True)


def regenerate_overlays(cx_subject: str, backup_tag: str) -> list[str]:
    """Rebuild ``overlays.svg`` for the current flat geometry (the old file is kept as a backup)."""
    cortex = _cortex()
    svg = os.path.join(filestore(), cx_subject, "overlays.svg")
    if os.path.exists(svg):
        bak = f"{svg}.{backup_tag}.bak"
        shutil.move(svg, bak)
        log.info("[pycortex] overlays.svg backed up -> %s", bak)
    ov = cortex.db.get_overlay(cx_subject)
    layers = list(ov.layers.keys())
    log.info("[pycortex] overlays.svg layers: %s", layers)
    return layers


def render_curvature(cx_subject: str, out_png: str) -> str:
    """Render the flat map of a pycortex subject with curvature shading only."""
    cortex = _cortex()
    import matplotlib.pyplot as plt

    c = cortex.db.get_surfinfo(cx_subject, "curvature")
    c.cmap = "gray"
    vm = float(np.nanpercentile(np.abs(c.data), 98))  # a single outlier must not flatten the grey scale
    c.vmin, c.vmax = -vm, vm
    fig = cortex.quickshow(c, with_curvature=False, with_rois=False, with_labels=False, with_colorbar=False)
    fig.savefig(out_png, dpi=120, facecolor="w")
    plt.close(fig)
    log.info("[pycortex] render -> %s", out_png)
    return out_png


def check_chosen(chosen_fn: str, subject: str, hemis: Sequence[str], force: bool) -> None:
    """Refuse the promotion of a design that failed a gate (unless ``force``)."""
    if not os.path.exists(chosen_fn):
        if force:
            log.warning("[apply] no gate record %s; --force given, promoting unchecked", chosen_fn)
            return
        raise SystemExit(f"[apply] no gate record {chosen_fn}: run `mfa flatten` for this name first "
                         "(or pass --force to promote unchecked patches)")
    ch = json_load(chosen_fn)
    subs = ch.get("subjects", {})
    if subject not in subs:
        if not force:
            raise SystemExit(f"{chosen_fn}: no record for {subject}; pass --force to promote anyway")
        return
    sm = subs[subject]
    if not (sm.get("gates_ok", True) and sm.get("hot_ok", True)) and not force:
        raise SystemExit(f"{chosen_fn}: {subject} failed a gate under the chosen recipe; re-run or pass --force")
    missing = [h for h in hemis if h not in sm.get("hemi", {})]
    if missing and not force:
        raise SystemExit(f"{chosen_fn}: no gate record for {missing} (run the recipe on that hemisphere first, "
                         "or pass --force)")
    log.info("[apply] gate record %s: recipe %s; slit width %s", chosen_fn, recipe_str(ch["recipe"]),
             ch.get("slit_width"))


def promote(subjects_dir: str, subject: str, name: str, out_dir: str, cx_subject: str | None = None,
            hemis: Sequence[str] = ("lh", "rh"), force: bool = False, dry_run: bool = False,
            render: bool = True) -> list[tuple[str, str, str]]:
    """Promote ``<hemi>.<name>.*`` to the canonical ``<hemi>.flatten.*``, import into pycortex, redraw overlays.

    Every overwritten file gets a timestamped backup: the previous patches
    (``*.pre<name>.<stamp>.bak``), the pycortex subject folder
    (``<filestore>/<cx>.pre<name>.<stamp>.bak``) and ``overlays.svg``.  The
    pycortex subject is created if it does not exist.  Returns the executed plan.
    """
    sd = os.path.join(subjects_dir, subject)
    cx = cx_subject or subject.replace("sub-", "")
    stamp = timestamp()
    rdir = run_dir(out_dir, subject, name)
    check_chosen(os.path.join(rdir, "chosen.json"), subject, hemis, force)
    steps: list[tuple[str, str, str]] = []
    for hemi in hemis:
        for suf in ("patch.3d", "flat.patch.3d"):
            src = f"{sd}/surf/{hemi}.{name}.{suf}"
            dst = f"{sd}/surf/{hemi}.{PROMOTED_NAME}.{suf}"
            if not os.path.exists(src):
                raise SystemExit(f"missing {src} (run `mfa flatten` first)")
            if os.path.exists(dst):
                steps.append(("backup", dst, f"{dst}.pre{name}.{stamp}.bak"))
            steps.append(("copy", src, dst))
    db = os.path.join(filestore(), cx)
    if os.path.isdir(db):
        steps.append(("backup-dir", db, f"{db}.pre{name}.{stamp}.bak"))
    else:
        steps.append(("import_subj", subject, cx))
    steps.append(("import_flat", f"{subject}:{PROMOTED_NAME}", cx))
    steps.append(("clear-cache", f"{db}/cache", ""))
    steps.append(("regen-overlays", f"{db}/overlays.svg", ""))
    if render:
        steps.append(("render", os.path.join(rdir, f"final_flatmap_{cx}.png"), ""))
    log.info("[apply] %splan:", "DRY RUN - " if dry_run else "")
    for s in steps:
        log.info("   %-14s %s  ->  %s", s[0], s[1], s[2])
    if dry_run:
        return steps
    for kind, a, b in steps:
        if kind in ("backup", "copy"):
            shutil.copy2(a, b)
            if not filecmp.cmp(a, b, shallow=False):
                raise SystemExit(f"[apply] copy verification failed: {a} != {b}")
        elif kind == "backup-dir":
            shutil.copytree(a, b, symlinks=True)
        elif kind == "import_subj":
            import_subject(a, b, subjects_dir)
        elif kind == "import_flat":
            import_flat(subject, PROMOTED_NAME, b, subjects_dir)
        elif kind == "clear-cache":
            shutil.rmtree(a, ignore_errors=True)
        elif kind == "regen-overlays":
            regenerate_overlays(cx, f"pre{name}.{stamp}")
        elif kind == "render":
            os.makedirs(os.path.dirname(a), exist_ok=True)
            render_curvature(cx, a)
    log.info("[apply] done: %s promoted to %s for %s / pycortex %s", name, PROMOTED_NAME, subject, cx)
    return steps
