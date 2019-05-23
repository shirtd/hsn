from primitives.plot import NetPlot, plot_dgm, plt, cm
from primitives.model import StaticError
import numpy.linalg as la
import numpy as np

class Interact(NetPlot):
    handle = None
    def __init__(self, *args, **kw):
        NetPlot.__init__(self, *args, **kw)
    def plot_embedding(self, axis, net, P, c, **kw):
        list(map(axis.axis, ('equal', 'off')))
        kw = {'c' : self.get_color(c), **kw}
        aplot = axis.scatter(P[:,0], P[:,1], **kw)
        plt.colorbar(aplot)
        self.connect(axis)
    def add_action(self, action, event):
        return self.fig.canvas.mpl_connect(action, event)
    def connect(self, axis, action='button_release_event'):
        if self.handle is not None: self.disconnect()
        self.handle = self.add_action(action, self.anevent(axis))
    def disconnect(self):
        if self.handle is not None:
            self.fig.canvas.mpl_disconnect(self.handle)
            self.handle = None
    def query(self, axis, e):
        if e.inaxes == axis:
            p = np.array([e.xdata, e.ydata])
            return self.find_key(p)
        return None
    def anevent(self, axis):
        def event(e):
            key = self.query(axis, e)
            if key:
                self.plot_key(key)
                plt.show(False)
        return event
    def plot_key(self, key): pass
    def find_key(self, p): pass

class StaticErrorInteract(StaticError, Interact):
    def __init__(self, net, error, std=5, fp=None, embedding='tsne', **kw):
        Interact.__init__(self, 1, 3, figsize=(12, 4))
        StaticError.__init__(self, net, error, std)
        if fp is not None:
            self.run(fp, embedding, **kw)
    def init_embedding(self, embedding='tsne', **kw):
        super(StaticErrorInteract, self).init_embedding(embedding, **kw)
        self.init_net(self.ax[0], self.f.net)
        self.plot_embedding(self.ax[2], self.f.net, self.P, self.error)
    def run(self, fp, embedding='tsne', **kw):
        super(StaticErrorInteract, self).run(fp)
        self.init_plot()
        self.init_embedding(embedding, **kw)
    def find_key(self, p):
        return min(range(len(self.P)), key=lambda i: la.norm(self.P[i] - p))
    def plot_key(self, key):
        self.plot_net(key)
        self.plot_dgm(key)
    def plot_net(self, key, shadow=False):
        colors = self.get_cmap(self.function[key], cm.coolwarm)
        for i, c in enumerate(colors):
            self.plots['net'][i].set_color(c)
    def plot_dgm(self, key):
        self.ax[1].cla()
        dgm = self.diagram[key]
        return plot_dgm(self.ax[1], dgm, len(dgm) - 1)
