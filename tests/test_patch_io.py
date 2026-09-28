import numpy as np

from mfa.patch_io import read_patch, write_patch


def test_patch_round_trip(tmp_path):
    n = 50
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(n, 3)).astype(np.float32).astype(float)
    idx = np.array([3, 7, 11, 20, 49, 0])
    border = np.array([True, False, False, True, False, False])
    fn = str(tmp_path / "lh.test.patch.3d")
    write_patch(fn, idx, xyz, border)
    idx2, border2, xyz2 = read_patch(fn)
    assert np.array_equal(idx, idx2)
    assert np.array_equal(border, border2)
    assert np.allclose(xyz[idx], xyz2)
    # the file has an 8-byte header and 16 bytes per vertex
    assert (tmp_path / "lh.test.patch.3d").stat().st_size == 8 + 16 * len(idx)
