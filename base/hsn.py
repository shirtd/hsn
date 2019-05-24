from base.persist import RipsHomology, DioFilt, lfilt, dio
from util.tools.io import load_pkl, pkl_dict
from base.network import GaussianNetwork
from scipy import spatial
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


class HSN(RipsHomology, GaussianNetwork, _HSN):

    def __init__(self, net_size, exp, bound, noise=0., delta=0.02, dim=2, prime=2):
        GaussianNetwork.__init__(self, net_size, exp, bound, noise, delta)
        print('[ %d point interior, %d point boundary' % (len(self.INT), len(self.B)))
        RipsHomology.__init__(self, self.DOM, dim + 1, self.beta, prime, self.B_indices)
        self.Salpha = self.get_net_simplices(self.alpha, self.dim, self.Q)
        self.dim = self.dim - 1
        self.covered = self.is_covered()

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

    def is_covered(self):
        res, HRD, H0 = self.tcc()
        res_str = '' if res else ' not'
        print(" | D \ B^%0.2f is%s covered"\
                "" % (2 * self.alpha, res_str))
        return res

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
        return pkl_dict(fname, net_dict=self.get_dict(),
                                prime=self.prime,
                                S=self.Salpha,
                                dim=self.dim)


class LoadHSN(DioFilt, _HSN):
    def __init__(self, fname):
        self.init_attr(**load_pkl(fname))
        DioFilt.__init__(self, lambda: None, lambda: None, self.DOM)

    def init_attr(self, net_dict, S, dim, prime=2):
        for k, v in net_dict.items():
            setattr(self, k, v)
        self.prime = prime
        self.dim = dim + 1
        self.t = self.beta
        self.Salpha = S
