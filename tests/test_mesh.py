import numpy as np

from mfa.mesh import (
    boundary_loops,
    boundary_loops_by_edges,
    boundary_vertices,
    face_components,
    flip_count,
    majority_face_label,
    patch_faces,
    resample_closed_loop,
    unique_edges,
)


def grid_mesh(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Regular triangulated grid in the unit square."""
    xs, ys = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
    V = np.c_[xs.ravel(), ys.ravel(), np.zeros(nx * ny)]
    F = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            F += [[a, b, c], [a, c, d]]
    return V, np.array(F)


def test_boundary_loops_disk_and_annulus():
    V, F = grid_mesh(9, 9)
    loops = boundary_loops(F)
    assert len(loops) == 1
    assert len(loops[0]) == 4 * 8  # perimeter vertices of a 9x9 grid
    # punch a hole: remove the faces touching the centre vertex -> two loops
    centre = 4 * 9 + 4
    Fh = F[~np.any(F == centre, axis=1)]
    loops = boundary_loops(Fh)
    assert len(loops) == 2
    assert len(loops[0]) == 32 and len(loops[1]) == 6
    assert len(boundary_vertices(Fh)) == 32 + 6
    by_edges = boundary_loops_by_edges(Fh)
    assert sorted(len(lp) for lp in by_edges) == [6, 32]


def test_patch_faces_and_edges():
    V, F = grid_mesh(5, 5)
    idx = np.arange(10)  # first two rows
    fk = patch_faces(idx, F, len(V))
    assert len(fk) == 8 and fk.max() < 10
    e = unique_edges(F)
    assert e.shape[1] == 2 and len(e) == 4 * 4 * 3 + 2 * 4  # interior + boundary edges of a 5x5 grid


def test_flip_count_and_components():
    V, F = grid_mesh(4, 4)
    p2 = V[:, :2]
    assert flip_count(p2, F) == 0
    p2f = p2.copy()
    p2f[5] = [5.0, 5.0]  # drag one interior vertex far away: some triangles flip
    assert flip_count(p2f, F) > 0
    comps = face_components(F)
    assert len(comps) == 1
    Fsplit = np.vstack([F[:4], F[-4:]])
    assert len(face_components(Fsplit)) == 2


def test_majority_face_label_and_resample():
    L = np.array([[1, 1, 2], [1, 2, 2], [1, 2, 3]])
    assert majority_face_label(L).tolist() == [1, 2, 1]
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    Q = resample_closed_loop(square, 8)
    assert Q.shape == (8, 2)
    assert np.allclose(Q[0], [0, 0]) and np.allclose(Q[2], [1, 0]) and np.allclose(Q[4], [1, 1])
