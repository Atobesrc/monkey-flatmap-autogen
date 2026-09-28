"""Resample a volumetric atlas onto the cortical surface -> FreeSurfer ``.annot`` files.

A single nearest-voxel sample per vertex leaves a few percent of vertices in
white matter or CSF, which show up as holes in the outline drawing.  Instead,
every vertex samples the label volume at several depths between the white and
the pial surface (nearest voxel), keeps only *valid* labels (cortical regions
plus explicitly listed allocortical labels that genuinely reach the surface),
takes the majority, and vertices without any valid sample are filled from
their mesh neighbours (iterated majority vote) so the surface is labelled
everywhere.  Names and colours come from a lookup table; both hemispheres share
one colortable.

Vertex positions are taken from the pycortex subject, which is in the scanner
RAS frame of the FreeSurfer volumes (pycortex adds the ``c_ras`` offset on
import), so a label volume aligned with the subject's T1 can be sampled directly
through its affine.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import Counter
from collections.abc import Sequence

import nibabel as nib
import numpy as np

log = logging.getLogger("mfa")

DEFAULT_DEPTHS = (0.2, 0.35, 0.5, 0.65, 0.8)


def _neighbours(polys: np.ndarray, n: int) -> list[set[int]]:
    nb: list[set[int]] = [set() for _ in range(n)]
    for a, b, c in polys:
        nb[a].update((b, c))
        nb[b].update((a, c))
        nb[c].update((a, b))
    return nb


def read_lookup(tsv: str, id_col: str = "ID", name_col: str = "name", region_col: str | None = "region",
                color_col: str | None = "color") -> dict[int, dict]:
    """Read a tab-separated lookup table into ``{id: {"name", "region", "color"}}``.

    ``region_col`` / ``color_col`` may be None when the table has no such column
    (then every label is a candidate and colours are generated).
    """
    with open(tsv) as fp:
        rows = list(csv.DictReader(fp, delimiter="\t"))
    info = {}
    for r in rows:
        if not r.get(id_col, "").strip():
            continue
        info[int(r[id_col])] = dict(name=r[name_col].strip(),
                                    region=(r.get(region_col, "") or "").strip() if region_col else "",
                                    color=(r.get(color_col, "") or "").strip() if color_col else "")
    return info


def atlas_to_annot(volume: str, lookup: str, cx_subject: str, fs_subject: str, subjects_dir: str, out_name: str,
                   id_col: str = "ID", name_col: str = "name", region_col: str | None = "region",
                   color_col: str | None = "color", valid_regions: Sequence[str] = ("cortex",),
                   extra_valid: Sequence[str] = (), depths: Sequence[float] = DEFAULT_DEPTHS,
                   hemis: Sequence[str] = ("lh", "rh")) -> dict[str, str]:
    """Write ``label/<hemi>.<out_name>.annot`` from a label volume and its lookup table.

    Parameters
    ----------
    volume : str
        Integer label volume (NIfTI/MGZ) in the subject's scanner space.
    lookup : str
        TSV with at least an id and a name column (``id_col``, ``name_col``); an
        optional ``region_col`` (e.g. 'cortex') selects valid labels and an
        optional ``color_col`` (``#rrggbb``) gives the colortable.
    valid_regions : labels whose region is in this list are kept.
    extra_valid : label *names* kept regardless of region (surface allocortex).
    depths : sampling depths between white (0) and pial (1).

    Returns
    -------
    dict
        ``{hemi: annot path}``.
    """
    import cortex

    info = read_lookup(lookup, id_col, name_col, region_col, color_col)
    valid_ids = {i for i, r in info.items()
                 if i != 0 and (region_col is None or r["region"] in valid_regions or r["name"] in extra_valid)}
    if not valid_ids:
        raise ValueError("no valid labels: check --valid-regions / --extra-valid and the lookup columns")
    names = ["unknown"] + sorted({info[i]["name"] for i in valid_ids})
    name_idx = {n: k for k, n in enumerate(names)}
    id_to_idx = {i: name_idx[info[i]["name"]] for i in valid_ids}

    img = nib.load(volume)
    vol = np.asanyarray(img.dataobj).astype(int)
    inv = np.linalg.inv(img.affine)
    log.info("[atlas] %s: %d valid labels -> %d names; volume %s @ %.2f mm", os.path.basename(volume),
             len(valid_ids), len(names) - 1, vol.shape, img.header.get_zooms()[0])

    ctab = np.zeros((len(names), 5), int)
    used: set[tuple[int, ...]] = set()
    for i in sorted(valid_ids):
        k = id_to_idx[i]
        if ctab[k, :3].any():
            continue
        h = info[i]["color"].lstrip("#")
        if len(h) == 6:
            rgb = [int(h[j:j + 2], 16) for j in (0, 2, 4)]
        else:
            rgb = [(37 * k) % 256, (91 * k) % 256, (151 * k) % 256]
        while tuple(rgb) in used or tuple(rgb) == (0, 0, 0):  # FreeSurfer encodes the label by its colour
            rgb = [(c + 7) % 256 for c in rgb]
        used.add(tuple(rgb))
        ctab[k, :3] = rgb
    ctab[:, 4] = ctab[:, 0] + ctab[:, 1] * 256 + ctab[:, 2] * 65536

    out_paths = {}
    for hemi in hemis:
        cxh = {"lh": "left", "rh": "right"}[hemi]
        wm, polys = cortex.db.get_surf(cx_subject, "wm", hemisphere=cxh)
        pia, _ = cortex.db.get_surf(cx_subject, "pia", hemisphere=cxh)
        n = len(wm)
        votes = np.zeros((n, len(names)), np.int16)
        for d in depths:
            pts = wm * (1 - d) + pia * d
            ijk = np.round(inv[:3, :3].dot(pts.T).T + inv[:3, 3]).astype(int)
            ok = np.all((ijk >= 0) & (ijk < np.array(vol.shape)), 1)
            lab = np.zeros(n, int)
            lab[ok] = vol[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
            for vid, k in id_to_idx.items():
                votes[lab == vid, k] += 1
        out = np.where(votes[:, 1:].sum(1) > 0, votes[:, 1:].argmax(1) + 1, 0)
        n_direct = int((out > 0).sum())
        nb = _neighbours(polys, n)
        it = 0
        while (out == 0).any() and it < 200:
            todo = np.where(out == 0)[0]
            new = out.copy()
            for v in todo:
                c = Counter(out[u] for u in nb[v] if out[u] > 0)
                if c:
                    new[v] = c.most_common(1)[0][0]
            if (new == out).all():
                break
            out = new
            it += 1
        n_filled = int((out > 0).sum()) - n_direct
        path = os.path.join(subjects_dir, fs_subject, "label", f"{hemi}.{out_name}.annot")
        nib.freesurfer.write_annot(path, out, ctab, names, fill_ctab=False)
        cnt = Counter(names[k] for k in out)
        log.info("[%s] %d vertices: %.1f%% sampled directly, %.1f%% filled from neighbours (%d iterations), "
                 "%d unlabelled; %d parcels present -> %s", hemi, n, 100.0 * n_direct / n, 100.0 * n_filled / n,
                 it, int((out == 0).sum()), len(cnt) - (1 if "unknown" in cnt else 0), path)
        log.info("      %s", ", ".join("%s(%d)" % kv for kv in sorted(cnt.items(), key=lambda kv: -kv[1])
                                       if kv[0] != "unknown"))
        out_paths[hemi] = path
    return out_paths
