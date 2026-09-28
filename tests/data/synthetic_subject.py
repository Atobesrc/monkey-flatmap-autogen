"""Generate a small synthetic FreeSurfer-style subject (sphere-like mesh with fake labels).

The mesh is an icosphere; the medial wall is a cap on the medial side and the
parcels are angular regions named like the CHARM level-2 parcels the recipes
use.  It is only meant to exercise the code paths (landmarks, seam routes, slit
patches, flattening, metrics), not to look like a brain.
"""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np

PARCEL_NAMES = ["unknown", "ACgG", "OFC", "lat_PFC", "MTL", "TG", "ITC", "STG/STSd", "MT", "V2-V4", "V1", "MPal",
                "Amy", "PMC", "other"]


def icosphere(subdivisions: int = 4, radius: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Icosphere vertices (n, 3) and consistently oriented faces (m, 3)."""
    t = (1.0 + 5**0.5) / 2.0
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                  [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], int)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(subdivisions):
        verts = list(V)
        cache: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                p = verts[a] + verts[b]
                verts.append(p / np.linalg.norm(p))
                cache[key] = len(verts) - 1
            return cache[key]

        newF = []
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V, F = np.array(verts), np.array(newF, int)
    return V * radius, F


def label_vertices(V: np.ndarray, radius: float, medial_sign: float) -> tuple[np.ndarray, np.ndarray]:
    """Fake parcel labels and the cortex mask.  ``medial_sign`` = +1 for lh (medial is +x), -1 for rh."""
    x, y, z = V[:, 0] * medial_sign / radius, V[:, 1] / radius, V[:, 2] / radius
    lab = np.full(len(V), PARCEL_NAMES.index("other"))
    cortex = x <= 0.55
    lab[~cortex] = PARCEL_NAMES.index("unknown")
    P = PARCEL_NAMES.index
    occ = y < -0.55
    lab[occ] = P("V1")
    lab[(y >= -0.55) & (y < -0.3)] = P("V2-V4")
    frontal = y > 0.5
    lab[frontal & (z < -0.2)] = P("OFC")
    lab[frontal & (z >= -0.2) & (x > 0.3)] = P("ACgG")
    lab[frontal & (z >= -0.2) & (x <= 0.3)] = P("lat_PFC")
    mid = (y >= -0.3) & (y <= 0.5)
    ventral = mid & (z < -0.35)
    lab[ventral] = P("ITC")
    lab[ventral & (y > 0.3)] = P("TG")
    lab[ventral & (x > 0.2) & (y > 0.0) & (y <= 0.3)] = P("Amy")
    lab[ventral & (x > 0.2) & (y <= 0.0)] = P("MTL")
    lab[ventral & (x > 0.42) & (y <= 0.0)] = P("MPal")
    lab[mid & (z >= -0.35) & (z < 0.0) & (x < -0.2)] = P("STG/STSd")
    lab[(y > -0.35) & (y < -0.2) & (x < -0.3) & (z > -0.35) & (z < -0.1)] = P("MT")
    lab[mid & (z > 0.2) & (x > 0.3)] = P("PMC")
    lab[~cortex] = P("unknown")
    return lab, cortex


def write_label(fn: str, idx: np.ndarray, V: np.ndarray) -> None:
    """Write a FreeSurfer ASCII label file."""
    with open(fn, "w") as fp:
        fp.write("#!ascii label  , from subject synthetic vox2ras=TkReg\n")
        fp.write(f"{len(idx)}\n")
        for i in idx:
            fp.write(f"{int(i)}  {V[i, 0]:.3f}  {V[i, 1]:.3f}  {V[i, 2]:.3f} 0.0000000000\n")


def make_subject(root: str, subject: str = "sub-synth", subdivisions: int = 4, radius: float = 30.0,
                 annot: str = "aparc.ARM2atlas.mapped") -> str:
    """Create ``root/subject/{surf,label}`` with both hemispheres; returns the subject directory."""
    sd = os.path.join(root, subject)
    os.makedirs(os.path.join(sd, "surf"), exist_ok=True)
    os.makedirs(os.path.join(sd, "label"), exist_ok=True)
    V0, F0 = icosphere(subdivisions, radius)
    rng = np.random.default_rng(0)
    ctab = np.zeros((len(PARCEL_NAMES), 5), int)
    ctab[:, :3] = rng.integers(1, 255, size=(len(PARCEL_NAMES), 3))
    ctab[:, 4] = ctab[:, 0] + ctab[:, 1] * 256 + ctab[:, 2] * 65536
    for hemi, sign in (("lh", 1.0), ("rh", -1.0)):
        V = V0.copy()
        F = F0.copy()
        if hemi == "rh":
            V[:, 0] = -V[:, 0]
            F = F[:, [0, 2, 1]]
        # gentle undulations so that the mesh is not exactly a sphere and has 'sulcal' edges
        theta = np.arctan2(V[:, 1], V[:, 0])
        phi = np.arccos(np.clip(V[:, 2] / radius, -1, 1))
        bump = 1.0 + 0.04 * np.sin(5 * theta) * np.cos(4 * phi)
        V = V * bump[:, None]
        curv = (0.3 * np.sin(5 * theta) * np.cos(4 * phi)).astype(np.float32)
        nib.freesurfer.write_geometry(f"{sd}/surf/{hemi}.white", V, F)
        nib.freesurfer.write_geometry(f"{sd}/surf/{hemi}.pial", V * 1.05, F)
        nib.freesurfer.write_geometry(f"{sd}/surf/{hemi}.inflated", V0 * (1 if hemi == "lh" else [-1, 1, 1]), F)
        nib.freesurfer.write_morph_data(f"{sd}/surf/{hemi}.curv", curv)
        nib.freesurfer.write_morph_data(f"{sd}/surf/{hemi}.sulc", curv)
        lab, cortex = label_vertices(V, radius, sign)
        write_label(f"{sd}/label/{hemi}.cortex.label", np.where(cortex)[0], V)
        nib.freesurfer.write_annot(f"{sd}/label/{hemi}.{annot}.annot", lab, ctab, PARCEL_NAMES, fill_ctab=False)
    return sd


if __name__ == "__main__":
    import sys

    print(make_subject(sys.argv[1] if len(sys.argv) > 1 else "."))
