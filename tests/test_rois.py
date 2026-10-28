"""Tests of the ROI machinery: mesh morphology, circled regions, dominance arbitration, CLI."""

import nibabel as nib
import numpy as np
import pytest

from mfa.cli import build_parser
from mfa.rois import Mesh, arbitrate, circled_regions


@pytest.fixture(scope="module")
def grid_subject(tmp_path_factory):
    """An n x n triangulated planar grid saved as ``surf/lh.white`` of a fake subject."""
    n = 20
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    V = np.c_[x.ravel(), y.ravel(), np.zeros(n * n)]
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            F += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    sd = tmp_path_factory.mktemp("roi_subject") / "sub-grid"
    (sd / "surf").mkdir(parents=True)
    nib.freesurfer.write_geometry(str(sd / "surf" / "lh.white"), V, np.asarray(F))
    return str(sd), n


def test_mesh_grow_close_components(grid_subject):
    sd, n = grid_subject
    mesh = Mesh(sd, "lh")
    m = np.zeros(n * n, bool)
    m[5 * n + 5] = True
    g1 = mesh.grow(m, 1)
    assert m.sum() == 1 and g1.sum() > 1 and g1[5 * n + 5]
    two = np.zeros(n * n, bool)
    two[2 * n + 2] = two[15 * n + 15] = True
    lab = mesh.components(two)
    assert lab[2 * n + 2] != lab[15 * n + 15]
    assert (lab[~two] == -1).all()
    ring = np.zeros(n * n, bool)        # closing a ring fills its interior
    for i in range(4, 9):
        ring[4 * n + i] = ring[8 * n + i] = ring[i * n + 4] = ring[i * n + 8] = True
    closed = mesh.close(ring, 3)
    assert closed[6 * n + 6]
    assert closed[ring].all()


def test_circled_regions_merges_nearby_and_drops_specks(grid_subject):
    sd, n = grid_subject
    mesh = Mesh(sd, "lh")
    m = np.zeros(n * n, bool)
    for i in range(3, 6):               # two 3x3 blobs two columns apart ...
        for j in list(range(3, 6)) + list(range(8, 11)):
            m[i * n + j] = True
    m[18 * n + 18] = True               # ... and one far one-vertex speck
    regs = circled_regions(mesh, m, merge_rings=3, min_vertices=5)
    assert len(regs) == 1               # blobs grouped into ONE circled region, speck dropped
    region, n_orig, _ = regs[0]
    assert n_orig == 18
    assert region[m & (np.arange(n * n) != 18 * n + 18)].all()
    lab = mesh.components(region)
    assert len(np.unique(lab[region])) == 1   # closed into a single connected piece


def test_circled_regions_never_enters_forbidden(grid_subject):
    sd, n = grid_subject
    mesh = Mesh(sd, "lh")
    m = np.zeros(n * n, bool)
    for i in range(3, 6):
        for j in list(range(3, 6)) + list(range(8, 11)):
            m[i * n + j] = True
    forbidden = np.zeros(n * n, bool)
    forbidden[np.arange(n) * n + 7] = True    # a full wall between the blobs
    regs = circled_regions(mesh, m, merge_rings=3, min_vertices=5, forbidden=forbidden)
    assert regs
    for region, _, _ in regs:
        assert not (region & forbidden).any()


def test_arbitrate_symmetric_margins():
    b_sig = np.array([10.0, 5.0, 5.0, 9.0, 0.0])
    a_max = np.array([5.0, 5.0, 10.0, 0.0, 9.0])
    b_mask = np.array([True, True, True, True, False])
    a_union = np.array([True, True, True, False, True])
    b_seed, a_drop, stats = arbitrate(b_sig, a_max, b_mask, a_union, margin=2.0)
    assert stats == dict(contested=3, b_dominant=1, a_dominant=1, ambiguous=1)
    # vertex 0: B dominates -> B keeps it, A loses it
    assert b_seed[0] and a_drop[0]
    # vertex 1: ambiguous -> dropped from BOTH seeds (so neither side's closing is blocked)
    assert not b_seed[1] and a_drop[1]
    # vertex 2: A dominates -> A keeps it, B loses it
    assert not b_seed[2] and not a_drop[2]
    # vertex 3: B-only territory is untouched; vertex 4: A-only territory is untouched
    assert b_seed[3] and not a_drop[3]
    assert not b_seed[4] and not a_drop[4]


def test_cli_roi_commands_parse():
    ap = build_parser()
    a = ap.parse_args(["roi-consensus", "--subjects-dir", "/sd", "--subject", "sub-x",
                       "--map", "xfm:vol.nii.gz", "--mask-out", "m.npy"])
    assert a.func.__name__ == "cmd_roi_consensus" and a.mion and a.thresh == 4.0
    a = ap.parse_args(["roi-separate", "--subjects-dir", "/sd", "--subject", "sub-x",
                       "--group-a", "face:facepatches:f.npy:xfm:v.nii",
                       "--group-b", "object:objectpatches:o.npy:xfm:w.nii"])
    assert a.func.__name__ == "cmd_roi_separate" and a.margin == 2.0
    a = ap.parse_args(["roi-tile", "--subjects-dir", "/sd", "--subject", "sub-x",
                       "--map", "face:xfm:v.nii", "--labels-out", "l.npy"])
    assert a.func.__name__ == "cmd_roi_tile"
    a = ap.parse_args(["roi-render", "--subjects-dir", "/sd", "--subject", "sub-x",
                       "--layer", "facepatches:gold:face", "--out", "x.png"])
    assert a.func.__name__ == "cmd_roi_render" and a.linewidth == 4.0
