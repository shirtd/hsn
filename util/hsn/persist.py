from itertools import combinations
import dionysus as dio
from tqdm import tqdm
import numpy as np

def dgm_lim(d):
    if any(p.death < np.inf for p in d):
        return max(p.death if p.death < np.inf else p.birth for p in d)
    return -np.inf

def _zero_inf(d):
    return [(p.birth, p.death if p.death < np.inf else 0) for p in d]

def _remove_inf(d):
    return [(p.birth, p.death) for p in d if p.death < np.inf]

def clean_dgm(d, unpaired=True, dtype=np.array):
    return dtype((_zero_inf if unpaired else _remove_inf)(d))

def clean_dgms(D, unpaired=True, dtype=dio.Diagram):
    return [clean_dgm(d, unpaired, dtype) for d in D]

# def max_bneck(Ds, ij):
#     return max(dio.bottleneck_distance(a, b) for a, b in zip(Ds[ij[0]], Ds[ij[1]]))

def bneck_mat(ds, unpaired=False):
    n = len(ds)
    Ds = [clean_dgms(d, unpaired) for d in ds]
    dmat = np.zeros((n, n), dtype=float)
    # pmap(max_bneck, list(combinations(range(n), 2)), Ds)
    for i, j in tqdm(list(combinations(range(n), 2)), '[ bottleneck'):
        d = max(dio.bottleneck_distance(a, b) for a, b in zip(Ds[i], Ds[j]))
        dmat[i, j] = dmat[j, i] = d
    return dmat
