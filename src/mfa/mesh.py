"""Triangle-mesh helpers shared by the cutting, flattening and layout modules."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def unique_edges(F: np.ndarray) -> np.ndarray:
    """Sorted unique undirected edges ``(m, 2)`` of a face array ``(f, 3)``."""
    return np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), axis=1), axis=0)


def vertex_adjacency(e: np.ndarray, n: int) -> csr_matrix:
    """Symmetric ``(n, n)`` 0/1 vertex adjacency matrix from an edge list."""
    return coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()


def patch_faces(idx: np.ndarray, f: np.ndarray, n: int) -> np.ndarray:
    """Faces of the parent surface whose vertices are all in the patch, re-indexed to patch-local ids.

    Parameters
    ----------
    idx : (k,) int array
        Patch vertex ids (parent indexing).
    f : (m, 3) int array
        Faces of the parent surface.
    n : int
        Number of vertices of the parent surface.
    """
    keep = np.zeros(n, bool)
    keep[idx] = True
    remap = np.full(n, -1)
    remap[idx] = np.arange(len(idx))
    return remap[f[np.all(keep[f], axis=1)]]


def face_area3(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-face area of a 3D triangle mesh."""
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return 0.5 * np.linalg.norm(nrm, axis=1)


def signed_area2(p2: np.ndarray, fk: np.ndarray) -> np.ndarray:
    """Twice the signed area of every 2D triangle."""
    a, b, c = p2[fk[:, 0]], p2[fk[:, 1]], p2[fk[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def flip_count(p2: np.ndarray, fk: np.ndarray) -> int:
    """Number of triangles whose 2D orientation disagrees with the median orientation."""
    area2 = signed_area2(p2, fk)
    sgn = np.sign(np.median(area2))
    return int(((area2 * sgn) < 0).sum())


def boundary_loops(fk: np.ndarray) -> list[np.ndarray]:
    """All boundary loops of a triangle patch (patch-local vertex indices), largest first.

    Boundary edges are the edges used by exactly one face.  The loops are walked
    from an arbitrary start preferring unvisited neighbours, so touch-point
    junctions (a vertex shared by two loops) are tolerated.
    """
    ec = Counter(tuple(sorted(x)) for tri in fk
                 for x in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])))
    nbr: dict[int, list[int]] = {}
    for (a, b), c in ec.items():
        if c == 1:
            nbr.setdefault(a, []).append(b)
            nbr.setdefault(b, []).append(a)
    seen: set[int] = set()
    loops = []
    for s in nbr:
        if s in seen:
            continue
        L, prev, cur = [s], -1, s
        seen.add(s)
        while True:
            cand = [x for x in nbr[cur] if x != prev and x not in seen]
            if not cand:
                break
            prev, cur = cur, cand[0]
            L.append(cur)
            seen.add(cur)
        loops.append(np.array(L))
    return sorted(loops, key=len, reverse=True)


def face_components(faces: np.ndarray) -> list[np.ndarray]:
    """Connected components (via shared edges) of a face array; lists of row indices."""
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_faces[(min(a, b), max(a, b))].append(fi)
    parent = list(range(len(faces)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for fl in edge_faces.values():
        for j in fl[1:]:
            ra, rb = root(fl[0]), root(j)
            if ra != rb:
                parent[rb] = ra
    comps = defaultdict(list)
    for i in range(len(faces)):
        comps[root(i)].append(i)
    return [np.array(v) for v in comps.values()]


def boundary_loops_by_edges(faces: np.ndarray) -> list[list[int]]:
    """Ordered vertex loops of the boundary of a face set (global vertex ids).

    Unlike :func:`boundary_loops` this consumes boundary *edges* one by one,
    which is the form the SVG path writer needs (holes become separate loops).
    """
    cnt: dict[tuple[int, int], int] = defaultdict(int)
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            cnt[(min(a, b), max(a, b))] += 1
    nbr = defaultdict(list)
    for (a, b), c in cnt.items():
        if c == 1:
            nbr[a].append(b)
            nbr[b].append(a)
    unused = {e for e, c in cnt.items() if c == 1}
    loops = []
    while unused:
        a, b = next(iter(unused))
        unused.discard((a, b))
        loop = [a, b]
        cur = b
        while cur != a:
            nxt = None
            for c in nbr[cur]:
                e = (min(cur, c), max(cur, c))
                if e in unused:
                    nxt = c
                    unused.discard(e)
                    break
            if nxt is None:
                break
            loop.append(nxt)
            cur = nxt
        if loop[-1] == a:
            loop.pop()
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def resample_closed_loop(P: np.ndarray, N: int) -> np.ndarray:
    """Resample a closed 2D polyline to ``N`` points, uniform in arc length, starting at ``P[0]``."""
    Pc = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(Pc, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0.0, s[-1], N, endpoint=False)
    return np.stack([np.interp(t, s, Pc[:, 0]), np.interp(t, s, Pc[:, 1])], 1)


def poly_centroid(Q: np.ndarray) -> np.ndarray:
    """Area centroid of a closed polygon given by its vertices in order."""
    x, y = Q[:, 0], Q[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    a = x * y1 - x1 * y
    A = a.sum() / 2
    return np.array([((x + x1) * a).sum() / (6 * A), ((y + y1) * a).sum() / (6 * A)])


def loop_perimeter(P: np.ndarray) -> float:
    """Perimeter of a closed polyline."""
    Pc = np.vstack([P, P[:1]])
    return float(np.linalg.norm(np.diff(Pc, axis=0), axis=1).sum())


def majority_face_label(L: np.ndarray) -> np.ndarray:
    """Majority label of each face from its three vertex labels ``(f, 3)``; ties go to vertex 0."""
    maj = L[:, 0].copy()
    m12 = L[:, 1] == L[:, 2]
    maj[m12] = L[m12, 1]
    return maj


def boundary_vertices(fk: np.ndarray) -> np.ndarray:
    """Vertices (patch-local) of the boundary edges of a triangle patch."""
    ec = Counter(tuple(sorted(x)) for tri in fk
                 for x in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])))
    bedges = np.array([k for k, c in ec.items() if c == 1])
    return np.unique(bedges.ravel()) if len(bedges) else np.arange(0)
