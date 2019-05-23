from util.network import find_contours_ext, poly_contour
from util.geometry import dist_to_curve, close_points
from util.math import distance_grid
from util.data import grf, grid_idx
from util.io import delete_line
from abc import ABC
import numpy as np

PAD = 8

class _GaussianNetwork(ABC):
    ''''''''''''''''''''
    ''' assumption 1 '''
    def test_size(self):
        C0 = find_contours_ext(self.MC.T, self.alpha + self.beta, PAD)
        C1 = find_contours_ext(self.MC.T, 2 * self.alpha, PAD)
        poly0, poly1 = list(map(poly_contour, (C0, C1)))
        self.poly['size'] = (poly0, poly1)
        self.contour['size'] = (C0, C1)
        return (len(poly0) + len(poly1) and len(poly0) >= len(poly1)
            and all(any(p.within(q) for p in poly0) for q in poly1))
    ''''''''''''''''''''
    ''' assumption 2 '''
    def test_close(self):
        C0 = find_contours_ext(self.MC.T, 2 * self.alpha, PAD)
        C1 = find_contours_ext(self.MD.T, 2 * self.alpha, PAD)
        P0, P1 = list(map(poly_contour, (C0, C1)))
        self.poly['close'], self.contour['close'] = (P0, P1), (C0, C1)
        return len(P0) <= len(P1) and all(any(p.within(q) for q in P1) for p in P0)

class GaussianNetwork(_GaussianNetwork):
    def __init__(self, net_size, exp, alpha, beta, bound, noise, delta=0.02):
        self.net_size, self.exp = net_size, exp
        self.X = grf(self.exp, self.net_size ** 2)
        self.alpha, self.beta, self.noise = alpha, beta, noise
        self.contour, self.poly = {}, {}
        self.bound = self.find_boundary(bound, delta)
    def add_noise(self, x):
        x = np.array(x, dtype=float)
        y = self.noise * np.random.rand(len(x), 2)
        return x + y - np.array([self.noise, self.noise]) / 2.
    ''''''''''''''''''''''''''''''''''''''''''''''''
    '''  generate domain satisfying assumptions  '''
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
    ''''''''''''''''''''''''''''''''''''
    '''  generate gaussian network   '''
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
        self.MC, self.MD = distance_grid(self.C, G_idx), distance_grid(self.DOM, G_idx)
    def get_dict(self):
        keys = ['X', 'net_size', 'exp',
                'alpha', 'beta', 'noise', 'bound',
                'M', 'MC', 'MD', 'contour', 'poly',
                'DOM', 'INT', 'B', 'C']
        return {k : self.__dict__[k] for k in keys}

# from util.plot import *
# class PlotGaussianNetwork(_GaussianNetwork):
#     imkw = {'interpolation' : 'bilinear', 'origin' : 'lower'}
#     def __init__(self, net, fig, ax):
#         for k, v in dir(net):
#             setattr(self, k, v)
#         self.fig, self.ax = fig, ax
#         self.extent = (-1 * (PAD + 0.5), self.net_size + PAD - 0.5,
#                         -1 * (PAD + 0.5),self.net_size + PAD - 0.5)
#     ''''''''''''''''''''''''''''''''''''
#     ''' plot network and assumptions '''
#     def clear_ax(self):
#         list(map(lambda x: x.cla(), self.ax))
#         list(map(lambda x: x.axis('equal'), self.ax))
#         list(map(lambda x: x.axis('off'), self.ax))
#     def plot_X(self, axis): axis.imshow(self.X.T, **self.imkw)
#     def plot_M(self, axis): axis.imshow(self.M, extent=self.extent, **self.imkw)
#     def plot_MC(self, axis): axis.imshow(self.MC, extent=self.extent, **self.imkw)
#     def plot_MD(self, axis): axis.imshow(self.MD, extent=self.extent, **self.imkw)
#     def domain_plot(self, shadow=True):
#         self.clear_ax()
#         self.plot_X(self.ax[0])
#         self.plot_M(self.ax[1])
#         plot_contours(self.ax[1], self.contour['B'], shadow, c='black')
#         self.plot_all(self.ax[2], shadow)
#         plot_contours(self.ax[2], self.contour['B'], shadow, c='black')
#     def size_plot(self, shadow=True):
#         self.clear_ax()
#         self.plot_all(self.ax[0], shadow)
#         self.plot_MC(self.ax[1])
#         self.plot_MC(self.ax[2])
#         p0, p1 = self.poly['size']
#         plot_poly(self.ax[1], p0, shadow, c='red', alpha=0.3)
#         plot_poly(self.ax[2], p1, shadow, c='blue', alpha=0.3)
#     def close_plot(self, shadow=True):
#         self.clear_ax()
#         self.plot_all(self.ax[0], shadow)
#         self.plot_MC(self.ax[1])
#         self.plot_MD(self.ax[2])
#         p0, p1 = self.poly['close']
#         plot_poly(self.ax[1], p0, shadow, c='red', alpha=0.3)
#         plot_poly(self.ax[2], p1, shadow, c='blue', alpha=0.15)
#     def plot_domain(self, axis, shadow=True, **kw):
#         if 'color' in kw:
#             kw['c'] = kw['color']
#             del kw['color']
#         kw['c'] = 'blue' if not 'c' in kw else kw['c']
#         kw['zorder'] = 1 if not 'zorder' in kw else kw['zorder']
#         kw['markersize'] = 1 if not 'markersize' in kw else kw['markersize']
#         if shadow:
#             kw['path_effects'] = [pfx.withSimplePatchShadow()]
#         axis.plot(self.INT[:,0], self.INT[:,1], 'o', **kw)
#     def plot_boundary(self, axis, shadow=True, **kw):
#         if 'color' in kw:
#             kw['c'] = kw['color']
#             del kw['color']
#         kw['c'] = 'red' if not 'c' in kw else kw['c']
#         kw['zorder'] = 0 if not 'zorder' in kw else kw['zorder']
#         kw['markersize'] = 1.5 if not 'markersize' in kw else kw['markersize']
#         if shadow:
#             kw['path_effects'] = [pfx.withSimplePatchShadow()]
#         axis.plot(self.B[:,0], self.B[:,1], 'o', **kw)
#     def plot_complement(self, axis, shadow=False, **kw):
#         if 'color' in kw:
#             kw['c'] = kw['color']
#             del kw['color']
#         kw['c'] = 'black' if not 'c' in kw else kw['c']
#         kw['zorder'] = 2 if not 'zorder' in kw else kw['zorder']
#         kw['markersize'] = 0.5 if 'markersize' not in kw else kw['markersize']
#         if shadow:
#             kw['path_effects'] = [pfx.withSimplePatchShadow()]
#         axis.plot(self.C[:,0], self.C[:,1], 'o', **kw)
#     def plot_all(self, axis, shadow=True):
#         self.plot_domain(axis, shadow, c='blue', zorder=1)
#         self.plot_boundary(axis, shadow, c='red', zorder=0)
