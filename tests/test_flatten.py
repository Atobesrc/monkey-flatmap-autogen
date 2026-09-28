import numpy as np
import pytest

from conftest import needs_igl
from mfa.flatten import silhouette_from_loops, symdir_energy
from mfa.mesh import flip_count


def hemisphere_mesh(n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated open hemisphere (a disk-topology surface with curvature)."""
    r, th = np.meshgrid(np.linspace(0.05, 1.0, n), np.linspace(0, 2 * np.pi, 3 * n, endpoint=False), indexing="ij")
    x, y = (r * np.cos(th)).ravel(), (r * np.sin(th)).ravel()
    z = np.sqrt(np.clip(1.0 - x**2 - y**2, 0, None))
    V = np.c_[x, y, z]
    nth = 3 * n
    F = []
    for i in range(n - 1):
        for j in range(nth):
            a, b = i * nth + j, i * nth + (j + 1) % nth
            c, d = (i + 1) * nth + (j + 1) % nth, (i + 1) * nth + j
            F += [[a, b, c], [a, c, d]]
    # cap the inner ring with a centre vertex
    V = np.vstack([V, [[0.0, 0.0, 1.0]]])
    ctr = len(V) - 1
    for j in range(nth):
        F.append([ctr, (j + 1) % nth, j])
    return V, np.array(F, np.int64)


@needs_igl
def test_tutte_then_slim_on_synthetic_disk():
    from mfa.flatten import init_tutte_uniform, run_slim

    V, F = hemisphere_mesh(10)
    uv0 = init_tutte_uniform(V, F)
    assert uv0.shape == (len(V), 2)
    assert flip_count(uv0, F) == 0  # Tutte with uniform weights is bijective
    E0, fl0 = symdir_energy(V, F, uv0)
    assert fl0 == 0
    uv, trace = run_slim(V, F, uv0, iters=30, tol=1e-9)
    E1, fl1 = symdir_energy(V, F, uv)
    assert fl1 == 0
    assert E1 < E0  # SLIM decreases the symmetric-Dirichlet energy
    assert trace[0][0] == 0 and trace[-1][0] >= 1
    # deterministic: a second run gives the identical result
    uv_again, _ = run_slim(V, F, init_tutte_uniform(V, F), iters=30, tol=1e-9)
    assert np.array_equal(uv, uv_again)


def test_silhouette_metric_on_synthetic_loops():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.c_[np.cos(t), np.sin(t)] * 50
    same = silhouette_from_loops(circle, circle + 10.0)  # translation is removed by the centroid alignment
    assert same["sil_mm"] < 1e-9 and same["sil_pct"] < 1e-9
    ellipse = np.c_[60 * np.cos(t), 40 * np.sin(t)]
    diff = silhouette_from_loops(circle, ellipse)
    assert diff["sil_mm"] > 1.0
    assert diff["sil_pct"] == pytest.approx(100 * diff["sil_mm"] / (0.5 * (diff["perim_lh"] + diff["perim_rh"])))
    assert diff["perim_lh"] == pytest.approx(2 * np.pi * 50, rel=1e-3)
