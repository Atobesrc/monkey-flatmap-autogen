import numpy as np

from mfa.layout import apply_similarity, fit_similarity, side_agreement


def test_fit_similarity_recovers_rotation_scale_translation():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(12, 2))
    ang = np.deg2rad(37.0)
    Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = 1.7 * (Rm @ src.T).T + np.array([5.0, -2.0])
    T = fit_similarity(src, dst)
    assert abs(T["s"] - 1.7) < 1e-9
    assert abs(T["angle_deg"] - 37.0) < 1e-9
    assert np.allclose(apply_similarity(T, src), dst, atol=1e-9)


def test_fit_similarity_no_reflection_by_default():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(10, 2))
    dst = src * np.array([-1.0, 1.0])  # a pure reflection
    T = fit_similarity(src, dst)
    assert np.linalg.det(T["R"]) > 0
    T2 = fit_similarity(src, dst, reflect=True)
    assert np.linalg.det(T2["R"]) < 0
    assert np.allclose(apply_similarity(T2, src), dst, atol=1e-9)


def test_fit_similarity_without_scale():
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dst = 3.0 * src
    T = fit_similarity(src, dst, scale=False)
    assert T["s"] == 1.0


def test_side_agreement():
    ref = {"Amy": dict(side=1, side_frac=0.9, side_conf=0.8), "TG": dict(side=-1, side_frac=0.1, side_conf=0.8),
           "ITC": dict(side=1, side_frac=0.55, side_conf=0.1)}   # ITC: reference side not confident -> not a gate
    cand = {"Amy": dict(side=1, side_frac=0.8, side_conf=0.6), "TG": dict(side=1, side_frac=0.7, side_conf=0.4),
            "ITC": dict(side=-1, side_frac=0.2, side_conf=0.6)}
    sa = side_agreement(cand, ref, seam_parcels=("Amy", "TG", "ITC"))
    assert sa["ref_side_parcels"] == ["Amy", "TG"]
    assert sa["ref_side_mismatch"] == ["TG"]
    assert not sa["ref_side_ok"]
    assert abs(sa["ref_side_agree"] - 0.5) < 1e-12
