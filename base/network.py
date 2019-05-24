from util.math.geometry import dist_to_curve, close_points, distance_grid
from util.hsn.network import find_contours_ext, poly_contour
from util.tools.io import delete_line
from util.data import grf, grid_idx
from abc import ABC
import numpy as np

PAD = 8

class _GaussianNetwork(ABC):

    def test_size(self):
        C0 = find_contours_ext(self.MC.T, self.alpha + self.beta, PAD)
        C1 = find_contours_ext(self.MC.T, 2 * self.alpha, PAD)
        poly0, poly1 = list(map(poly_contour, (C0, C1)))
        self.poly['size'] = (poly0, poly1)
        self.contour['size'] = (C0, C1)
        return (len(poly0) + len(poly1) and len(poly0) >= len(poly1)
            and all(any(p.within(q) for p in poly0) for q in poly1))

    def test_close(self):
        C0 = find_contours_ext(self.MC.T, 2 * self.alpha, PAD)
        C1 = find_contours_ext(self.MD.T, 2 * self.alpha, PAD)
        P0, P1 = list(map(poly_contour, (C0, C1)))
        self.poly['close'], self.contour['close'] = (P0, P1), (C0, C1)
        return len(P0) <= len(P1) and all(any(p.within(q) for q in P1) for p in P0)


class GaussianNetwork(_GaussianNetwork):

    def __init__(self, net_size, exp, bound, noise, delta=0.02):
        self.net_size, self.exp = net_size, exp
        self.noise = noise * 10. / self.net_size
        self.X = grf(self.exp, self.net_size ** 2)
        self.alpha = np.sqrt(2)  + self.noise
        self.beta = 3 * self.alpha
        self.contour, self.poly = {}, {}
        self.bound = self.find_boundary(bound, delta)

    def add_noise(self, x):
        x = np.array(x, dtype=float)
        y = self.noise * np.random.rand(len(x), 2)
        return x + y - np.array([self.noise, self.noise]) / 2.

    def find_boundary(self, bound, delta=0.02):
        bound -= delta
        while bound < 1:
            bound += delta
            print('[ bound = %0.2f' % bound)
            self.get_domain(bound)
            if (self.test_size() and self.test_close()):
                break
            delete_line()
        return bound

    def get_domain(self, bound):
        _n, n = len(self.X), len(self.X) + 2 * PAD
        G0_idx, G_idx = grid_idx(_n), grid_idx(range(-PAD, _n + PAD))
        G, G0 = G_idx.reshape(-1, 2), G0_idx.reshape(-1, 2)
        D0_idx = np.vstack([(i, j) for i,j in G0 if self.X[i,j] <= bound])
        self.M, _G = distance_grid(D0_idx, G_idx), self.add_noise(G)
        self.contour['B'] = find_contours_ext(self.M.T, self.noise, PAD)
        domain_idx = {l for l, (i, j) in enumerate(G + PAD) if self.M.T[i, j] <= self.noise}
        cand_idx = close_points(_G, np.vstack(self.contour['B']), np.sqrt(3/2) * self.alpha)
        ext_idx = {i for i in cand_idx if dist_to_curve(self.contour['B'], _G[i]) <= self.alpha}
        bnd_idx = ext_idx.intersection(domain_idx)
        int_idx = {i for i in domain_idx if not i in bnd_idx}
        comp_idx = [i for i in range(len(G)) if not (i in int_idx or i in bnd_idx)]
        self.INT, self.B = _G[list(int_idx)], _G[list(bnd_idx)]
        self.C, self.DOM = _G[comp_idx], np.vstack((self.B, self.INT))
        n_int, n_bdy, n_dom = len(self.INT), len(self.B), len(self.DOM)
        self.indices = set(range(n_dom))
        self.B_indices = set(range(n_bdy))
        self.DOM_indices = set(range(n_bdy, n_dom))
        self.MC, self.MD = distance_grid(self.C, G_idx), distance_grid(self.DOM, G_idx)

    def get_dict(self):
        keys = ['X', 'net_size', 'exp',
                'alpha', 'beta', 'noise', 'bound',
                'M', 'MC', 'MD', 'contour', 'poly',
                'DOM', 'INT', 'B', 'C']
        return {k : self.__dict__[k] for k in keys}
