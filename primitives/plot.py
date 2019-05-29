from util.hsn.persist import dgm_lim, clean_dgm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import dionysus as dio
from tqdm import tqdm
from abc import ABC
import numpy as np

plt.ion()

def get_axes(row=1, col=1, is_3d=False, **kw):
    if is_3d: kw['projection'] = '3d'
    fig, ax = plt.subplots(row, col, **kw)
    plt.tight_layout()
    return fig, ax

def plot_model(P, **kw):
    fig, ax = get_axes(1, 1)
    ax.axis('equal')
    aplot = ax.scatter(P[:,0], P[:,1], **kw)
    plt.colorbar(aplot)
    plt.tight_layout()
    return fig, ax

def plot_dgm(ax, D, max_dim=np.inf):
    lim = max(dgm_lim(d) for i, d in enumerate(D) if i < max_dim)
    dgms = [clean_dgm(d) for i, d in enumerate(D) if i < max_dim]
    diag = ax.plot([0, lim], [0, lim], c='black', alpha=0.5)
    return diag + [ax.scatter(d[:,0], d[:,1], s=5) for d in dgms if len(d)]

def plot_dgms(ax, Ds):
    for D in Ds:
        fig2, ax2 = get_axes(1, 1)
        plot_dgm(ax2, D)
        plt.show(False)
        input('...')
        plt.close(fig2)


''' ------------ *
 | ABSTRACT TYPE |
 * ------------ '''

class NetPlot(ABC):
    fig, plots = None, {}

    def __init__(self, *args, **kw):
        self._args, self._kw = args, kw

    def get_color(self, x):
        return (np.array(x) - min(x)) / (max(x) - min(x))

    def get_mappable(self, x, cmap=cm.coolwarm):
        if isinstance(x, dict):
            x = [x[k] for k in sorted(x.keys())]
        return cm.ScalarMappable(Normalize(min(x), max(x)), cmap)

    def get_cmap(self, x, cmap=cm.coolwarm):
        if isinstance(x, dict):
            x = [x[k] for k in sorted(x.keys())]
        c = cm.ScalarMappable(Normalize(min(x), max(x)), cmap)
        return np.array([list(c.to_rgba(v))[:3] for v in x])

    def colorbar(self, axis, x, cmap=cm.coolwarm):
        norm = Normalize(min(x), max(x))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm._A = x
        return self.fig.colorbar(sm, ax=axis)
        # return ColorbarBase(axis, cmap=cmap, norm=norm)

    def init_plot(self):
        if self.fig is not None: plt.close(self.fig)
        self.fig, self.ax = get_axes(*self._args, **self._kw)
        self.fig.subplots_adjust(wspace=0.125)
        self.handle, self.plots = None, {}

    def init_net(self, axis, net):
        list(map(axis.axis, ('equal', 'off')))
        self.plots['net'] = {}
        for i, s in tqdm(list(enumerate(net.Salpha)), '[ plotting network'):
            s = dio.Simplex(*s)
            if s.dimension() == 0:
                x = net.plot_vertex(self.ax[0], s, False)
                self.plots['net'][i] = x[0] if isinstance(x, list) else x
            elif s.dimension() == 1:
                x = net.plot_edge(self.ax[0], s, False)
                self.plots['net'][i] = x[0] if isinstance(x, list) else x
            elif s.dimension() == 2:
                x = net.plot_triangle(self.ax[0], s, False, alpha=0.5)
                self.plots['net'][i] = x[0] if isinstance(x, list) else x
