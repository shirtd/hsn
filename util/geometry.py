import numpy.linalg as la
from scipy import spatial
import numpy as np

def dist_to_line(a, b, v):
    ab, av, bv = b - a, v - a, v - b
    if av.dot(ab) <= 0.0: return la.norm(av)
    if bv.dot(ab) >= 0.0: return la.norm(bv)
    return la.norm(np.cross(ab, av)) / la.norm(ab)

def dist_to_curve(C, p):
    if isinstance(C, list) or (isinstance(C, np.ndarray) and C.ndim > 2):
        return min(dist_to_curve(c, p) for c in C)
    E = zip(np.concatenate((C[-1][None], C[:-1])), C)
    return min(dist_to_line(u, v, p) for u, v in E)

def close_points(A, B, t):
    T = spatial.KDTree(A)
    C = T.query_ball_point(B, t)
    return np.unique(np.concatenate(C))
