from matplotlib.patches import PathPatch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.patheffects as pfx
from primitives.persist import *
from matplotlib.path import Path
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from shapely.geometry import *
import matplotlib.cm as cm
from operator import mul
import numpy as np
from . import *
import sys

plt.ion()

''''''''''''
''' UTIL '''
''''''''''''

def eqax(axis, lim=1., is_3d=False):
    axis.set_xlim(-lim, lim)
    axis.set_ylim(-lim, lim)
    if is_3d:
        axis.set_zlim(-lim, lim)
    return axis

def get_axes(row=1, col=1, is_3d=False, **kw):
    if is_3d: kw['projection'] = '3d'
    fig, ax = plt.subplots(row, col, **kw)
    plt.tight_layout()
    return fig, ax

def plot_edges(axis, E, z=[], thresh=-np.Inf, **kw):
    print('[ plotting %d edges' % len(E))
    if len(z):
        color = cm.ScalarMappable(Normalize(z.min(), z.max()), cm.rainbow)
    c = list(map(color.to_rgba, z) if len(z) else ['black'] * len(E))
    for e, a in zip(E, c):
        kw['c'], dims = a, [e[:,i] for i in range(e.shape[1])]
        axis.plot(*dims, **kw)
    if len(z):
        color.set_array(list(filter(lambda a: a >= thresh, z)))
        plt.colorbar(color, ax=axis)

def ring_coding(ob):
    n = len(ob.coords)
    codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO
    return codes

def pathify(p):
    vertices = np.concatenate([np.asarray(p.exterior)]
                + [np.asarray(r) for r in p.interiors])
    codes = np.concatenate([ring_coding(p.exterior)]
                + [ring_coding(r) for r in p.interiors])
    return Path(vertices, codes)

def plot_poly(axis, poly, shadow=True, **kw):
    if 'c' in kw:
        kw['color'] = kw['c']
        del kw['c']
    if shadow:
        kw['path_effects'] = [pfx.withSimplePatchShadow()]
    list(map(lambda p: axis.add_patch(PathPatch(pathify(p), **kw)), poly))

def plot_contours(axis, C, shadow=True, **kw):
    if shadow:
        kw['path_effects'] = [pfx.SimpleLineShadow(), pfx.Normal()]
    return list(map(lambda c: axis.plot(c[:,0], c[:,1], **kw), C))

def query_axis(fig, ax):
    i = raw_input(': select axis to save '+str(list(range(len(ax))))+': ')
    try:
        i = int(i)
    except:
        print(' ! invalid entry')
        i = query_axis(fig, ax)
    fname = raw_input(': save as ')
    save_axis(fig, ax[i], fname)

def save_axis(fig, ax, fname):
    fpath = os.path.dirname(fname)
    if fpath and not os.path.exists(fpath):
        print(' ! creating folders %s' % fpath)
        os.makedirs(fpath)
    print(' | saving %s' % fname)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(fname, bbox_inches=extent)


''''''''''''''''''
'''  INTERACT  '''
''''''''''''''''''

class Interact:
    # def __init__(self, classes, F, data, prime, *args, **kw):
    def __init__(self, fig, ax, classes, **kw):
        self.fig, self.ax = fig, ax
        list(map(lambda x: x.axis('equal'), self.ax))
        list(map(lambda x: x.axis('off'), self.ax))
        self.classes, self.kw = classes, kw
        # self.F, self.data, self.prime = F, data, prime
        self.cache = {}
    def get_obj(self, fun, **kw):
        if not fun in self.cache: # and self.cache[fun]
            self.cache[fun] = self.classes[fun](self.fig, self.ax, **kw)
        self.cache[fun].init_plot()
        return self.cache[fun]
    # def anevent(self, fun):
    def anevent(self, fun, **kw):
        _kw = self.kw.copy()
        for k, v in kw.iteritems():
            _kw[k] = v
        # self.OBJ = self.get_obj(fun, self.F, self.data, self.prime, *args, **kw)
        self.OBJ = self.get_obj(fun, **_kw)
        def event(e):
            sys.stdout.write('\r')
            sys.stdout.flush()
            key = self.OBJ.query(e)
            if key: self.OBJ.plot(key)
            plt.show(False)
            sys.stdout.write('> ')
            sys.stdout.flush()
        return event
    def addevent(self, fun, **kw):
        fun = self.anevent(fun, **kw)
        return self.fig.canvas.mpl_connect('button_release_event', fun)

class PersistencePlot(DioPersist):
    def __init__(self, fig, ax, fun, birth=True):
        self.fig, self.ax, self.fun = fig, ax, fun
        self.birth, self.cycle_thresh = birth, np.Inf
    def clear_ax(self, axis):
        axis.cla()
        axis.axis('equal')
        axis.axis('off')
    def clear(self):
        list(map(self.clear_ax, self.ax))
    def initialize(self, dims):
        self.dims = dims
        self.dgms = self.sort_dgms_paired()
        dgms = list(map(self.to_np_dgm, self.dgms))
        self.kd = {d : KDTree(dgms[d]) for d in dims if len(dgms[d]) > 0}
        self.cache, self.last = {}, None
        self.init_plot()
    def init_plot(self):
        self.plot_data(self.ax[0], c='black')
        self.plot_dgm(self.ax[2])
    def query(self, e):
        self.last = e
        p = np.array([e.xdata, e.ydata])
        if e.inaxes == self.ax[2]:
            ds = [(k, v.query(p)) for k, v in self.kd.iteritems()]
            dim, (d, idx) = min(ds, key=lambda kd: kd[1][1])
            pt = self.get_point(dim, idx)
            self.plot_dgm(self.ax[2])
            self.ax[2].scatter(pt.birth, pt.death, s=20, zorder=2, c='black')
            return pt
        return None
    def thresh_cycle(self, key):
        c = self.cache[key]
        t = key.birth if self.birth else self.cycle_thresh
        cycle = sorted(c, key=lambda e: self.F[e.index].data, reverse=True)
        v, idx = list(zip(*[(x.element, x.index) for x in cycle]))
        i, n = 0, len(cycle)
        ft = lambda l: self.F[idx[l]].data > t
        fp = lambda l: sum(v[l:]) % self.prime != 0
        while i < n and (ft(i) or fp(i)): i += 1
        while i >= n or fp(i): i -= 1
        return [self.F[j] for j in idx[i:]]
    def replot(self):
        if self.last != None:
            key = self.query(self.last)
            if key: self.plot(key)
    def plot_edges(self, axis, E, shadow=True, **kw):
        kw['c'] = kw['c'] if 'c' in kw else 'black'
        kw['alpha'] = kw['alpha'] if 'alpha' in kw else 0.6
        kw['zorder'] = kw['zorder'] if 'zorder' in kw else 1
        if shadow:
            kw['path_effects'] = [pfx.SimpleLineShadow(), pfx.Normal()]
        list(map(lambda e: axis.plot(e[:,0], e[:,1], **kw), E))
    def plot_data(self, axis, x=[], shadow=True, **kw):
        x = self.data if not len(x) else x
        kw['markersize'] = kw['s'] if 's' in kw else 3
        kw['zorder'] = kw['zorder'] if 'zorder' in kw else 2
        if shadow:
            kw['path_effects'] = [pfx.withSimplePatchShadow()]
        axis.plot(self.data[:,0], self.data[:,1], 'o', **kw)
