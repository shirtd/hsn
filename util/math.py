import numpy.linalg as la
from scipy import spatial
import numpy as np

def distance_matrix(x, axis=2, **kw):
    return la.norm(x[np.newaxis] - x[:, np.newaxis], axis=axis, **kw)

def normalize(x, axis=None):
    x = np.array(x) if isinstance(x, list) else x
    if x.max(axis) != x.min(axis):
        return (x - x.min(axis)) / (x.max(axis) - x.min(axis))
    return x

def distance_grid(X, G):
    T = spatial.KDTree(X)
    return np.array([[T.query(x)[0] for x in row] for row in G])
