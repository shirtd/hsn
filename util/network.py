from shapely.geometry import Polygon, LinearRing
from skimage.measure import find_contours
from scipy.interpolate import griddata
from scipy import spatial
import numpy.linalg as la
import dionysus as dio
import numpy as np

''''''''''''''''''''''''''''''
''' GAUSSIAN NETWORK UTIL  '''
''''''''''''''''''''''''''''''

def in_hull(p, n):
    return 0 in p or n - 1 in p

def find_contours_ext(M, t, k=4, **kwargs):
    return [c - k for c in find_contours(M, t, fully_connected='high', **kwargs)]

def possible_pairs(l, i):
    return [j for j in range(len(l)) if i != j and l[j].is_ccw != l[i].is_ccw]
    # return filter(lambda j: i != j and l[j].is_ccw != l[i].is_ccw, range(len(l)))

def pair_contour(C):
    lines = [LinearRing(c) for c in C]
    polys = [Polygon(c) for c in C]
    possible = {i : possible_pairs(lines, i) for i in range(len(lines))}
    paired, pairs = [], {}
    for k, v in possible.items():
        p = polys[k]
        contains = [i for i in v if polys[i].contains(p)]
        for i in contains:
            q = contains[np.argmin([lines[k].distance(lines[i]) for i in contains])]
            if not q in pairs:
                pairs[q] = []
            pairs[q].append(k)
            paired.append(k)
    return pairs

def poly_contour(C):
    if len(C) == 1:
        return [Polygon(C[0])]
    pairs = pair_contour(C)
    if len(pairs):
        return [Polygon(C[k], [C[j] for j in v]) for k, v in pairs.items()]
    else:
        return [Polygon(c) for c in C]


''''''''''''''''''''
''' GRID NETWORK '''
''''''''''''''''''''

def grid_distance(m, s=1.):
    return np.sqrt(2 * (s / (np.sqrt(m) - 1)) ** 2)

def grid_network(m, dim=None, noise=0.):
    n = int(np.sqrt(m))
    x = np.linspace(0, 1, n)
    y = np.vstack(map(lambda a: a.flatten(), np.meshgrid(x, x))).T
    e = noise * np.random.rand(len(y), 2) - np.array([noise, noise]) / 2.
    return y + e


'''''''''''''''''''''''''''''
'''''' RANDOM NETWORKS ''''''
'''''''''''''''''''''''''''''

def random_network(n, dim=2, noise=None):
    return np.random.rand(n, dim)

''''''''''''''''''''''''
''' uniform network  '''
def make_boundary(t=0.1, x=[0., 1.], y=[0., 1.]):
    outer = np.vstack([x + x[::-1], np.vstack((y, y)).reshape((-1,), order='F')]).T
    inner = np.vectorize(lambda x: t if x == 0 else 1 - t)(outer[::-1])
    return outer, inner

def sample_boundary(outer, inner, n=128):
    low, high = outer.flatten(), np.roll(inner[::-1], 3, axis=0).flatten()
    return np.random.uniform(low, high, (n // 4, 8)).reshape(-1, 2)

def sample_interior(inner, n=256):
    return np.random.uniform(inner[1], inner[-1], (n, 2))

def uniform_network(n=256, t=0.125):
    m = int(n * t)
    outer, inner = make_boundary(t)
    bdy = sample_boundary(outer, inner, m)
    inter = sample_interior(inner, n - m)
    return np.vstack((bdy, inter)), range(m), bdy
