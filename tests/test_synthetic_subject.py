"""Integration tests on the synthetic subject: landmarks, seam routes, slit patches, flattening, gates."""

import os

import numpy as np
import pytest

from conftest import needs_igl
from mfa.mesh import boundary_loops, patch_faces
from mfa.patch_io import read_patch, write_patch
from mfa.recipe import default_recipe, parse_recipe, route_kwargs
from mfa.surface import REQUIRED_PARCELS, Hemi, Parcellation, check_parcels

pytestmark = pytest.mark.slow


def test_hemi_loads_and_landmarks(synthetic_subject):
    subjects_dir, subject = synthetic_subject
    H = Hemi(os.path.join(subjects_dir, subject), "lh")
    assert check_parcels(H.names, REQUIRED_PARCELS) == []
    assert H.iscortex.sum() < H.n and H.rim.size > 0
    # geodesic anchors: always inside the named parcel, defined without any coordinate axis
    occ = H.anchor_vertex("V1_far_V2-V4")
    assert H.lab[occ] == H.P["V1"]
    tip = H.anchor_vertex("TG_far_ITC")
    assert H.lab[tip] == H.P["TG"]
    b = H.anchor_vertex("TG_ITC_border_far_wall")
    assert H.lab[b] in (H.P["TG"], H.P["ITC"])
    c = H.landmark_point("MTL_Amy_centroid")
    assert c.shape == (3,)
    src = H.temporal_source("nearest_Amy_TG_border")
    assert src in H.rim
    with pytest.raises(ValueError):          # coordinate-based anchors are gone
        H.anchor_vertex("TG_ant")
    with pytest.raises(KeyError):
        H.inpar("NotAParcel")


def test_parcel_map_renames(synthetic_subject):
    subjects_dir, subject = synthetic_subject
    sd = os.path.join(subjects_dir, subject)
    parc = Parcellation.from_spec(None, "Amygdala=Amy")   # pretend the annot calls it 'Amy' and we want 'Amygdala'
    lab, names = parc.read(sd, "lh")
    assert "Amygdala" in names and "Amy" not in names
    with pytest.raises(KeyError):
        Parcellation.from_spec(None, "Amy=NoSuchName").read(sd, "lh")


@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_seam_paths_and_slit_patch(synthetic_subject, hemi):
    subjects_dir, subject = synthetic_subject
    H = Hemi(os.path.join(subjects_dir, subject), hemi)
    # the synthetic parcels are coarse blobs, so use a temporal entry that is not adjacent to the
    # destination on this geometry; the rules are the packaged coordinate-free ones
    recipe = parse_recipe("temporal_src=nearest_MTL_Amy_centroid", base=default_recipe())
    paths = H.seam_paths(("calcarine", "frontal", "temporal"), **route_kwargs(recipe))
    assert set(paths) == {"calcarine", "frontal", "temporal"}
    for k, p in paths.items():
        assert len(p) >= 3, k
        assert len(np.unique(p)) == len(p), f"{k} path revisits a vertex"
    # seams start on the medial-wall rim (the temporal one at the recipe's entry vertex)
    assert H.adjwall[paths["calcarine"][0]] and H.adjwall[paths["frontal"][0]]
    assert paths["temporal"][0] == H.temporal_source(recipe["temporal_src"])
    assert paths["temporal"][-1] == H.anchor_vertex(recipe["temporal_dst"])
    assert H.seam_info["temporal"]["extend_mm"] == 0.0
    for w in (1, 2):
        ni, border, stats = H.build_slit_patch(paths, w)
        assert stats["loops"] == 1 and stats["isolated"] == 0
        assert stats["nvert"] < H.iscortex.sum()
        assert border[ni].sum() == len(np.unique(np.concatenate(boundary_loops(patch_faces(ni, H.f, H.n)))))
        # no medial-wall vertex and no seam vertex is in the patch
        assert H.iscortex[ni].all()
        for p in paths.values():
            assert not np.isin(p, ni).any()
    n1 = H.build_slit_patch(paths, 1)[2]["seam_removed"]
    n2 = H.build_slit_patch(paths, 2)[2]["seam_removed"]
    assert n2 > n1  # wider slits remove more cortex


def test_temporal_extension_and_corridors(synthetic_subject):
    subjects_dir, subject = synthetic_subject
    H = Hemi(os.path.join(subjects_dir, subject), "lh")
    base = route_kwargs(default_recipe())
    p0 = H.seam_paths(("temporal",), **base)["temporal"]
    p1 = H.seam_paths(("temporal",), **dict(base, temporal_extend="V2V4_border"))["temporal"]
    assert len(p1) > len(p0) and np.array_equal(p1[: len(p0)], p0)
    assert H.seam_info["temporal"]["extend_mm"] > 0
    for corridor in ("medial", "border"):
        p = H.seam_paths(("temporal",), **dict(base, temporal_corridor=corridor))["temporal"]
        assert p[0] == p0[0] and p[-1] == p0[-1]


@needs_igl
def test_flatten_orient_and_gates(synthetic_subject, tmp_path):
    from mfa.flatten import canonical_orient, hemi_quality, silhouette_norm, slim_flatten
    from mfa.layout import flat_layout

    subjects_dir, subject = synthetic_subject
    sd = os.path.join(subjects_dir, subject)
    for hemi in ("lh", "rh"):
        H = Hemi(sd, hemi)
        paths = H.seam_paths(("calcarine", "frontal", "temporal"), **route_kwargs(default_recipe()))
        ni, border, stats = H.build_slit_patch(paths, 1)
        write_patch(f"{sd}/surf/{hemi}.t.patch.3d", ni, H.v, border)
        trace = slim_flatten(sd, hemi, "t", iters=40, tol=1e-6)
        assert trace[-1][2] == 0
        idx, _, xyz = read_patch(f"{sd}/surf/{hemi}.t.flat.patch.3d")
        assert np.array_equal(idx, ni) and np.allclose(xyz[:, 2], 0)
        R, corr = canonical_orient(sd, hemi, "t", backup=False)
        assert abs(abs(np.linalg.det(R)) - 1) < 1e-9
        q = hemi_quality(sd, hemi, "t")
        assert q["loops"] == 1 and q["isolated"] == 0 and q["components"] == 1
        assert q["flips"] == 0 and q["gate_flips"]
        assert q["gate_orient"] and min(q["frame_axis_sep"], q["frame_updown_sep"]) > 0.05
        assert 0 < q["cortex_loss_area_pct"] < 20
        lay = flat_layout(sd, hemi, "t")
        assert "V1" in lay["parcels"] and "TG" in lay["parcels"]
        assert abs(lay["sqrt_area"] - np.sqrt(lay["flat_area"])) < 1e-9
        if lay["seam"] is not None:
            assert lay["seam"]["n_vertices"] > 0
    sil = silhouette_norm(sd, "t", "t")
    assert 0 <= sil["sil_pct"] < 20  # the two synthetic hemispheres are mirror images up to the undulation
