from matplotlib.pyplot import Circle, Polygon
from matplotlib.colors import Normalize
import matplotlib.patheffects as pfx
from scipy.sparse import csc_matrix
import matplotlib.cm as cm
import dionysus as dio
import numpy as np

def lfilt(f, F):
    return dio.Filtration(list(filter(f, F)))

def sort_dgm(dgm):
    return sorted(dgm, key=lambda p: p.death - p.birth, reverse=True)

def np_dgm(dgm):
    return np.array([[p.birth, p.death] for p in dgm])

class DioWrap:
    def __init__(self, ffun, pfun, data):
        self.ffun, self.pfun, self.data = ffun, pfun, data
    def build_filt(self, *args, **kwargs):
        self.F = self.ffun(self.data, *args, **kwargs)
    def build_pers(self, *args, **kwargs):
        self.H = self.pfun(self.F, *args, **kwargs)
        self.D = dio.init_diagrams(self.H, self.F)
        # self.dgms = map(sort_dgm, self.D)

class DioFilt(DioWrap):
    def __init__(self, ffun, pfun, data):
        DioWrap.__init__(self, ffun, pfun, data)
    def __setitem__(self, i, s):
        self.F[i] = s
    def __getitem__(self, i): return self.F[i]
    def __contains__(self, s): return s in self.F
    def __iter__(self): return self.F.__iter__()
    def __len__(self): return self.F.__len__()
    def __repr__(self): return self.F.__repr__()
    def __str__(self): return self.F.__str__()
    def add(self, s): return self.F.add(s)
    def append(self, s): return self.F.append(s)
    def index(self, s): return self.F.index(s)
    def rearrange(self, I): return self.F.rearrange(I)
    def sort(self): return self.F.sort()
    def lfilt(self, f): return lfilt(f, self.F)
    def threshold(self, t): return self.lfilt(lambda s: s.data <= t)
    def np_filt(self, dim):
        return np.array([list(s) for s in self if s.dimension() == dim])
    def sdict(self, i, s):
        v, d, b = list(s), s.data, list(map(self.index, s.boundary()))
        return {'index' : i, 'vertices' : v, 'data' : d, 'boundary' : b}
    def fdict(self):
        return [self.sdict(i, s) for i, s in enumerate(self)]
    def restrict(self, V):
        return self.lfilt(lambda s: all(v in V for v in s))
    def plot_cover(self, axis, r, I=[], **kw):
        if 'c' in kw:
            kw['color'] = kw['c']
            del kw['c']
        kw['alpha'] = kw['alpha'] if 'alpha' in kw else 0.12
        kw['zorder'] = kw['zorder'] if 'zorder' in kw else 0
        kw['color'] = kw['color'] if 'color' in kw else 'black'
        I = I if len(I) else range(len(self.data))
        list(map(lambda i: axis.add_artist(Circle(self.data[i], r, **kw)), I))
    def plot_vertex(self, axis, s, shadow=True, **kw):
        p = self.data[list(s)]
        if 'color' in kw:
            kw['c'] = kw['color']
            del kw['color']
        kw['c'] = 'black' if not 'c' in kw else kw['c']
        kw['alpha'] = 1 if not 'alpha' in kw else kw['alpha']
        kw['zorder'] = 2 if not 'zorder' in kw else kw['zorder']
        kw['markersize'] = 1 if not 'markersize' in kw else kw['markersize']
        if shadow:
            kw['path_effects'] = [pfx.withSimplePatchShadow()]
        return axis.plot(p[:,0], p[:,1], 'o', **kw)
    def plot_edge(self, axis, s, shadow=True, **kw):
        p = self.data[list(s)]
        if 'color' in kw:
            kw['c'] = kw['color']
            del kw['color']
        kw['c'] = 'black' if not 'c' in kw else kw['c']
        kw['alpha'] = 0.5 if not 'alpha' in kw else kw['alpha']
        kw['zorder'] = 1 if not 'zorder' in kw else kw['zorder']
        kw = {'color' : 'black', 'alpha' : 0.5, 'zorder' : 1}
        if shadow:
            kw['path_effects'] = [pfx.SimpleLineShadow(), pfx.Normal()]
        return axis.plot(p[:,0], p[:,1], **kw)
    def plot_triangle(self, axis, s, shadow=True, **kw):
        p = self.data[list(s)]
        if 'c' in kw:
            kw['color'] = kw['c']
            del kw['c']
        kw['color'] = 'black' if not 'color' in kw else kw['color']
        kw['alpha'] = 0.1 if not 'alpha' in kw else kw['alpha']
        kw['zorder'] = 0 if not 'zorder' in kw else kw['zorder']
        if shadow:
            kw['path_effects'] = [pfx.withSimplePatchShadow()]
        return axis.add_patch(Polygon(p, **kw))
    def plot_simplex(self, axis, i, shadow=True, **kw):
        s = i if isinstance(i, dio.Simplex) else self.F[i]
        if s.dimension() == 0:
            return self.plot_vertex(axis, s, shadow, **kw)
        elif s.dimension() == 1:
            return self.plot_edge(axis, s, shadow, **kw)
        elif s.dimension() == 2:
            return self.plot_triangle(axis, s, shadow, **kw)
    def plot_simplices(self, axis, S, shadow=True, **kw):
        list(map(lambda s: self.plot_simplex(axis, s, shadow, **kw), S))

class DioPersist(DioFilt):
    def __init__(self, ffun, pfun, data, prime):
        DioFilt.__init__(self, ffun, pfun, data)
        self.prime = prime
    def dgm_dim(self, dim): return self.sort_dgm(self.D[dim])
    def get_point(self, dim, i=0): return self.dgm_dim(dim)[i]
    def unpaired(self): return self.H.unpaired
    def pair(self, pt): return self.H.pair(pt.data)
    def is_paired(self, pt): return self.pair(pt) != self.unpaired()
    def sort_dgm(self, dgm): return sort_dgm(dgm)
    def dgm_paired(self, dgm): return np.array(list(filter(self.is_paired, dgm)))
    def sort_dgm_paired(self, dgm): return self.dgm_paired(self.sort_dgm(dgm))
    def sort_dgms_paired(self): return list(map(self.sort_dgm_paired, self.D))
    def np_cycle(self, c): return np.array([list(self.F[s.index]) for s in c])
    def to_np_dgm(self, dgm): return np_dgm(dgm)
    def np_dgm_dim(self, dim): return np_dgm(self.dgm_dim(dim))
    def np_dgm(self): return list(map(self.np_dgm_dim, range(len(self.D))))
    def np_cycle_thresh(self, c, t):
        return np.array([list(self.F[s.index]) for s in c if self.F[s.index].data <= t])
    def dgm_thresh(self):
        l = [x for dgm in self.np_dgm() if len(dgm) for x in dgm[:, 1]]
        deaths = np.array([d for d in l if d < np.inf])
        return deaths.max() if len(deaths) else 0.
    def plot_dgm(self, axis):
        axis.cla()
        t = self.dgm_thresh()
        axis.plot([0, t], [0, t], c='black', alpha=0.5, zorder=0)
        for dgm in self.np_dgm():
            if len(dgm):
                axis.scatter(dgm[:,0], dgm[:,1], s=10, zorder=1)

class DioHomology(DioPersist):
    def __init__(self, ffun, data, prime, Q, *args, **kwargs):
        DioPersist.__init__(self, ffun, dio.homology_persistence, data, prime)
        self.build_filt(*args, **kwargs)
        if len(Q):
            self.Q, self.relative = Q, self.restrict(Q)
            print('[ computing persistent relative homology in Z%d' % self.prime)
            self.build_pers(self.relative, self.prime, progress=True)
        else:
            print('[ computing persistent homology in Z%d' % self.prime)
            self.build_pers(self.prime, progress=True)
    def is_paired(self, pt):
        return self.pair(pt) != self.H.unpaired
    def cycle(self, pt):
        if self.is_paired(pt):
            return self.H[self.H.pair(pt.data)]
        return None

class DioCohomology(DioPersist):
    def __init__(self, ffun, data, prime, *args, **kwargs):
        DioPersist.__init__(self, ffun, dio.cohomology_persistence, data, prime)
        self.build_filt(*args, **kwargs)
        print('[ computing persistent cohomology in Z%d' % self.prime)
        self.build_pers(self.prime, True)
    def cocycle(self, pt):
        return self.H.cocycle(pt.data)
    def coords(self, pt):
        c, f = self.cocycle(pt), self.threshold((pt.death + pt.birth) / 2.)
        return np.array(dio.smooth(f, c, self.prime))
    def plot_coords(self, axis, v, shadow=True):
        color = cm.ScalarMappable(Normalize(v.min(), v.max()), cm.rainbow)
        kw = {'markersize' : 3}
        if shadow:
            kw['path_effects'] = [pfx.withSimplePatchShadow()]
        for c, (x, y) in zip(v, self.data):
            axis.plot(x, y, 'o', color=color.to_rgba(c), **kw)
        # axis.scatter(self.data[:,0], self.data[:,1], s=20, c=map(color.to_rgba, v))
    # TODO generalize to dim > 1
    def lift_simplex(self, f, s):
        i, bdy = f.index(s), s.boundary()
        return [(1. - 2. * (j % 2.), i, f.index(t)) for j, t in enumerate(bdy)]
    def lift_boundary(self, f, dim=1):
        data, row, col = list(zip(*np.concatenate([self.lift_simplex(f, s) for s in f if s.dimension() == dim])))
        row, col = list(map(lambda x: np.array(x, dtype=int), (row, col)))
        k = max(max(row), max(col)) + 1
        return csc_matrix((np.array(data), (row, col)), shape=(k, k))
    def lift_cocycle(self, c, bdy):
        row_max, k = bdy.indices.max(), bdy.shape[0]
        z = [x for x in c if x.index <= row_max]
        z_col = np.zeros(len(z), dtype=int)
        z_row, elem = list(zip(*list(map(lambda x: (x.index, x.element), z))))
        z_data = [e if e < self.prime / 2. else e - self.prime for e in elem]
        return csc_matrix((z_data, (z_row, z_col)), shape=(k, 1)).toarray()
    def to_integer(self, f, c):
        bdy = self.lift_boundary(f)
        row_max = bdy.indices.max()
        z_bdy = self.lift_cocycle(c, bdy)
        F = dio.Filtration([s for i, s in enumerate(f) if i <= bdy.indices.max()])
        return F, bdy, z_bdy
    def get_lift(self, pt):
        c, f = self.cocycle(pt), self.threshold((pt.death + pt.birth) / 2.)
        return self.to_integer(f, c)


class SimplicialHomology(DioHomology):
    def __init__(self, F, data, prime, Q=[], **kw):
        DioHomology.__init__(self, lambda x, y: y, data, prime, Q, F)
        list(map(lambda kv: setattr(self, *kv), kw.iteritems()))

class SimplicialCohomology(DioCohomology):
    def __init__(self, F, data, prime, **kw):
        DioCohomology.__init__(self, lambda x, y: y, data, prime, F)
        list(map(lambda kv: setattr(self, *kv), kw.iteritems()))

class RipsHomology(DioHomology):
    def __init__(self, data, dim, t, prime, Q=[]):
        print('[ %d point %dD rips filtration at %0.2f' % (len(data), dim, t))
        DioHomology.__init__(self, dio.fill_rips, data, prime, Q, dim, t)
        self.dim, self.t = dim, t

class RipsCohomology(DioCohomology):
    def __init__(self, data, dim, t, prime):
        print('[ %d point %dD rips filtration at %0.2f' % (len(data), dim, t))
        DioCohomology.__init__(self, dio.fill_rips, data, prime, dim, t)
        self.dim, self.t = dim, t

class StarHomology(DioHomology):
    def __init__(self, data, prime, Q=[], reverse=False):
        print('[ %d point lower star filtration' % len(data.flatten()))
        DioHomology.__init__(self, dio.fill_freudenthal, data, prime, Q, reverse)
        self.reverse = reverse

class StarCohomology(DioCohomology):
    def __init__(self, data, prime, reverse=False):
        print('[ %d point lower star filtration' % len(data.flatten()))
        DioCohomology.__init__(self, dio.fill_freudenthal, data, prime, reverse)
        self.reverse = reverse
