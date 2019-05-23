from primitives.persist import RipsHomology, DioFilt, lfilt, dio
from primitives.network import GaussianNetwork
from scipy import spatial
import pickle as pkl
import numpy as np
import os

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
    def __init__(self, net_size, exp, dim, bound, alpha, beta, noise=0., delta=0.02, prime=2):
        GaussianNetwork.__init__(self, net_size, exp, alpha, beta, bound, noise, delta)
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
