from itertools import combinations
from util.plot import get_axes
import dionysus as dio
from tqdm import tqdm
import numpy as np

def dgm_lim(d):
    if any(p.death < np.inf for p in d):
        return max(p.death if p.death < np.inf else p.birth for p in d)
    return -np.inf

def clean_dgm(d):
    return np.array([[p.birth, p.death if p.death < np.inf else 0] for p in d])

def remove_inf(D):
    # return [dio.Diagram([(p.birth, p.death if p.death < np.inf else 0) for p in d]) for d in D]
    return [dio.Diagram([(p.birth, p.death) for p in d if p.death < np.inf]) for d in D]

def bneck_mat(ds):
    n = len(ds)
    Ds = [remove_inf(d) for d in ds]
    dmat = np.zeros((n, n), dtype=float)
    for i, j in tqdm(list(combinations(range(n), 2)), '[ bottleneck'):
        d = max(dio.bottleneck_distance(a, b) for a, b in zip(Ds[i], Ds[j]))
        dmat[i, j] = dmat[j, i] = d
    return dmat
