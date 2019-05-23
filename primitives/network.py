from collections import defaultdict
from itertools import combinations
from primitives.persist import *
from functools import reduce
from util.network import grf
from util.network import *
from util.plot import *
from tqdm import tqdm
from util import *

PAD = 8

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

class GaussianNetwork:
    imkw = {'interpolation' : 'bilinear', 'origin' : 'lower'}
    def __init__(self, net_size, exp, alpha, beta, bound, noise, delta=0.02, fig=None, ax=[]):
        self.net_size, self.exp = net_size, exp
        self.X = grf(self.exp, self.net_size ** 2)
        self.alpha, self.beta, self.noise = alpha, beta, noise
        self.ax = ax
        if len(self.ax):
            self.fig = fig
        self.contour, self.poly = {}, {}
        self.bound = self.find_boundary(bound, len(ax) > 0, delta)
        self.extent = (-1 * (PAD + 0.5), self.net_size + PAD - 0.5,
                        -1 * (PAD + 0.5), self.net_size + PAD - 0.5)
    def add_noise(self, x):
        x = np.array(x, dtype=float)
        y = self.noise * np.random.rand(len(x), 2)
        return x + y - np.array([self.noise, self.noise]) / 2.
    ''''''''''''''''''''''''''''''''''''''''''''''''
    '''  generate domain satisfying assumptions  '''
    def find_boundary(self, bound, plot=False, delta=0.02):
        bound -= delta
        while bound < 1:
            bound += delta
            print('[ bound = %0.2f' % bound)
            self.get_domain(bound, plot)
            if self.test_size(plot) and self.test_close(plot): break
            delete_line()
        return bound
    ''''''''''''''''''''''''''''''''''''
    '''  generate gaussian network   '''
    def get_domain(self, bound, plot=False):
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
    ''''''''''''''''''''
    ''' assumption 1 '''
    def test_size(self, plot=False):
        C0 = find_contours_ext(self.MC.T, self.alpha + self.beta, PAD)
        C1 = find_contours_ext(self.MC.T, 2 * self.alpha, PAD)
        poly0, poly1 = list(map(poly_contour, (C0, C1)))
        self.poly['size'] = (poly0, poly1)
        self.contour['size'] = (C0, C1)
        return (len(poly0) + len(poly1) and len(poly0) >= len(poly1)
            and all(any(p.within(q) for p in poly0) for q in poly1))
    ''''''''''''''''''''
    ''' assumption 2 '''
    def test_close(self, plot=False):
        C0 = find_contours_ext(self.MC.T, 2 * self.alpha, PAD)
        C1 = find_contours_ext(self.MD.T, 2 * self.alpha, PAD)
        poly0, poly1 = list(map(poly_contour, (C0, C1)))
        self.poly['close'] = (poly0, poly1)
        self.contour['close'] = (C0, C1)
        return len(poly0) <= len(poly1) and all(any(p.within(q) for q in poly1) for p in poly0)
    ''''''''''''''''''''''''''''''''''''
    ''' plot network and assumptions '''
    def clear_ax(self):
        list(map(lambda x: x.cla(), self.ax))
        list(map(lambda x: x.axis('equal'), self.ax))
        list(map(lambda x: x.axis('off'), self.ax))
    def plot_X(self, axis): axis.imshow(self.X.T, **self.imkw)
    def plot_M(self, axis): axis.imshow(self.M, extent=self.extent, **self.imkw)
    def plot_MC(self, axis): axis.imshow(self.MC, extent=self.extent, **self.imkw)
    def plot_MD(self, axis): axis.imshow(self.MD, extent=self.extent, **self.imkw)
    def domain_plot(self, shadow=True):
        self.clear_ax()
        self.plot_X(self.ax[0])
        self.plot_M(self.ax[1])
        plot_contours(self.ax[1], self.contour['B'], shadow, c='black')
        self.plot_all(self.ax[2], shadow)
        plot_contours(self.ax[2], self.contour['B'], shadow, c='black')
    def size_plot(self, shadow=True):
        self.clear_ax()
        self.plot_all(self.ax[0], shadow)
        self.plot_MC(self.ax[1])
        self.plot_MC(self.ax[2])
        p0, p1 = self.poly['size']
        plot_poly(self.ax[1], p0, shadow, c='red', alpha=0.3)
        plot_poly(self.ax[2], p1, shadow, c='blue', alpha=0.3)
    def close_plot(self, shadow=True):
        self.clear_ax()
        self.plot_all(self.ax[0], shadow)
        self.plot_MC(self.ax[1])
        self.plot_MD(self.ax[2])
        p0, p1 = self.poly['close']
        plot_poly(self.ax[1], p0, shadow, c='red', alpha=0.3)
        plot_poly(self.ax[2], p1, shadow, c='blue', alpha=0.15)
    def plot_domain(self, axis, shadow=True, **kw):
        if 'color' in kw:
            kw['c'] = kw['color']
            del kw['color']
        kw['c'] = 'blue' if not 'c' in kw else kw['c']
        kw['zorder'] = 1 if not 'zorder' in kw else kw['zorder']
        kw['markersize'] = 1 if not 'markersize' in kw else kw['markersize']
        if shadow:
            kw['path_effects'] = [pfx.withSimplePatchShadow()]
        axis.plot(self.INT[:,0], self.INT[:,1], 'o', **kw)
    def plot_boundary(self, axis, shadow=True, **kw):
        if 'color' in kw:
            kw['c'] = kw['color']
            del kw['color']
        kw['c'] = 'red' if not 'c' in kw else kw['c']
        kw['zorder'] = 0 if not 'zorder' in kw else kw['zorder']
        kw['markersize'] = 1.5 if not 'markersize' in kw else kw['markersize']
        if shadow:
            kw['path_effects'] = [pfx.withSimplePatchShadow()]
        axis.plot(self.B[:,0], self.B[:,1], 'o', **kw)
    def plot_complement(self, axis, shadow=False, **kw):
        if 'color' in kw:
            kw['c'] = kw['color']
            del kw['color']
        kw['c'] = 'black' if not 'c' in kw else kw['c']
        kw['zorder'] = 2 if not 'zorder' in kw else kw['zorder']
        kw['markersize'] = 0.5 if 'markersize' not in kw else kw['markersize']
        if shadow:
            kw['path_effects'] = [pfx.withSimplePatchShadow()]
        axis.plot(self.C[:,0], self.C[:,1], 'o', **kw)
    def plot_all(self, axis, shadow=True):
        self.plot_domain(axis, shadow, c='blue', zorder=1)
        self.plot_boundary(axis, shadow, c='red', zorder=0)
    def get_dict(self):
        keys = ['X', 'net_size', 'exp',
                'alpha', 'beta', 'noise', 'bound',
                'M', 'MC', 'MD', 'contour', 'poly',
                'DOM', 'INT', 'B', 'C']
        return {k : self.__dict__[k] for k in keys}

class _HSN:
    def in_int(self, s):
        return all(not v in self.Q for v in s)
    def in_bdy(self, s):
        return all(v in self.Q for v in s)
    def get_cycle(self, pt):
        c = self.cycle(pt)
        return self.np_cycle(c) if c != None else np.empty((0,2))
    def get_cycles(self):
        return [self.get_cycle(pt) for pt in pts if self.is_paired(pt)]
    def restrict_domain(self, t):
        TB = spatial.KDTree(self.B)
        fun = lambda i: TB.query(self.data[i])[0] > t
        return list(filter(fun, range(len(self.data))))
    def get_net_filt(self, alpha, dim, Q):
        return self.lfilt(lambda s: (
                s.dimension() < dim
                and s.data <= alpha
                and not any(v in Q for v in s)))
    def get_net_simplices(self, alpha, dim, Q):
        return [(list(s), s.data) for s in self.F if (
                        s.dimension() < dim
                        and s.data <= alpha
                        and not any(v in Q for v in s))]
    def plot(self, axes, shadow=True, clear=True):
        self.clear_ax()
        A = self.lfilt(lambda s: s.data <= self.alpha)
        Q = lfilt(lambda s: all(v in self.Q for v in s), A)
        INT = lfilt(lambda s: not any(v in self.Q for v in s), A)
        self.plot_simplices(axes[0], A, shadow, c='black')
        xs, ys = axes[0].get_ylim(), axes[0].get_ylim()
        self.plot_simplices(axes[1], Q, shadow, c='red')
        self.plot_simplices(axes[2], INT, shadow, c='blue')
        list(map(lambda x: x.set_xlim(*xs), axes))
        list(map(lambda x: x.set_ylim(*ys), axes))

class HSN(RipsHomology, GaussianNetwork, _HSN):
    def __init__(self, net_size, exp, dim, bound, alpha, beta, noise=0., fig=None, ax=[], delta=0.02, prime=2):
        GaussianNetwork.__init__(self, net_size, exp, alpha, beta, bound, noise, delta, fig, ax)
        print('[ %d point interior, %d point boundary' % (len(self.INT), len(self.B)))
        data, Q = np.vstack((self.B, self.INT)), set(range(len(self.B)))
        RipsHomology.__init__(self, data, dim + 1, beta, prime, Q)
        self.Salpha = self.get_net_simplices(self.alpha, self.dim, self.Q)
        self.dim = self.dim - 1
        self.covered, _, _ = self.tcc()
        res_str = '' if self.covered else ' not'
        print(" | D \ B^%0.2f is%s covered"\
                "" % (2 * self.alpha, res_str))
    def components(self, t):
        print('[ finding connected components of D \ B^%0.4f' % t)
        F = dio.Filtration([dio.Simplex(*s) for s in self.Salpha])
        H = dio.homology_persistence(F, 2, 'clearing', True)
        D = list(map(self.sort_dgm, dio.init_diagrams(H, F)))
        return [p for p in D[0] if H.pair(p.data) == H.unpaired]
    def get_unpaired_points(self, dim, t=np.Inf):
        return [p for p in self.D[dim] if not self.is_paired(p) and p.birth <= t]
    def tcc(self):
        HRD = self.get_unpaired_points(self.dim, self.alpha)
        H0 = self.components(self.alpha)
        return len(H0) == len(HRD), HRD, H0
    def plot_network(self, axis):
        axis.scatter(self.INT[:,0], self.INT[:,1], s=5, c='blue', zorder=3)
        axis.scatter(self.B[:,0], self.B[:,1], s=10, c='red', zorder=4)
    def save(self, fname=None, dir='data'):
        if not os.path.exists(dir):
            print('[ creating directory %s' % dir)
            os.mkdir(dir)
        fpath = lambda l: os.path.exists(os.path.join(dir, l))
        if fname is None:
            name, i = '%d%d' % (self.net_size, self.exp), 0
            name = name if self.covered else '%s_nocover' % name
            while fpath('.'.join(('_'.join((name, str(i))), 'pkl'))):
                i += 1
            name = '_'.join((name, str(i)))
            fname = os.path.join(dir, '.'.join((name, 'pkl')))
        print('[ saving net as %s' % fname)
        with open(fname, 'wb') as f:
            pkl.dump({'S' : self.Salpha, 'net_dict' : self.get_dict(),
                    'dim' : self.dim, 'prime' : self.prime}, f)

class LoadHSN(DioFilt, _HSN):
    def __init__(self, net_dict, S, dim, prime=2):
        for k, v in net_dict.items():
            setattr(self, k, v)
        self.prime, self.dim, self.t, self.Salpha = prime, dim+1, self.beta, S
        DioFilt.__init__(self, lambda: None, lambda: None, self.DOM)
