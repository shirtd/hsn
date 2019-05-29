from primitives.plot import NetPlot, plot_dgm, plt, cm
from primitives.model import Model, StaticError, DynamicError
from collections import defaultdict
import numpy.linalg as la
import numpy as np
import time

''' -------- *
 | BASE TYPE |
 * -------- '''

class Interact(NetPlot):
    handle = None

    def __init__(self, *args, **kw):
        NetPlot.__init__(self, *args, **kw)
        self.selected = {'embedding_points' : []}
        self.select_mod = {'embedding_points' : {'s' : 30}}
        self.plot_args = {'embedding_points' : {'zorder' : 1, 's' : 10}}
        self.fselect = {'embedding_points' : {'s' : self._set_point_size}}

    def _set_point_size(self, x, y):
        self.plots['embedding_points'][x].set_sizes([y])

    def plot_embedding(self, axis, net, P, c, **kw):
        print('[ plotting embedding')
        list(map(axis.axis, ('equal', 'off')))
        self.colors = self.get_cmap(c, cmap=cm.viridis)
        kw = {**self.plot_args['embedding_points'], **kw}
        aplot = {i : axis.scatter(p[0], p[1], c=[self.colors[i]], **kw) for i,p in enumerate(P)}
        self.plots['embedding_points'] = aplot
        # self.fig.colorbar(self.get_mappable(c, cm.viridis), ax=axis)
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

class ModelInteract(Interact, Model):
    def __init__(self):
        Interact.__init__(self, 1, 3, figsize=(12, 4))
        self.cur_key = None

    def find_key(self, p):
        return min(range(len(self.P)), key=lambda i: la.norm(self.P[i] - p))

    def select_point(self, key):
        for l, f in self.fselect['embedding_points'].items():
            f(key, self.select_mod['embedding_points'][l])
        self.selected['embedding_points'].append(key)

    def clear_selection(self):
        for k, v in self.selected.items():
            for l, f in self.fselect[k].items():
                list(map(lambda x: f(x, self.plot_args[k][l]), v))

    def select_key(self, key):
        self.clear_selection()
        self.select_point(key)

    def plot_key(self, key):
        self.select_key(key)
        self.plot_net(key)
        self.plot_dgm(key)

    def plot_net(self, key, shadow=False):
        fk = self.function[key]
        if self.cur_key is not None:
            idx = np.where(abs(self.function[self.cur_key] - fk) > 1e-2)[0]
        else:
            idx = list(range(len(fk)))
        colors = self.get_cmap(fk, cm.coolwarm)
        for i in idx:
            self.plots['net'][i].set_color(colors[i])
        self.cur_key = key

    def plot_dgm(self, key):
        self.ax[1].cla()
        dgm = self.diagram[key]
        return plot_dgm(self.ax[1], dgm, len(dgm) - 1)

''' --------------- *
 | CONCRETE OBJECTS |
 * --------------- '''

class StaticErrorInteract(StaticError, ModelInteract):

    def __init__(self, net, error, std=5, fp=None, embedding='tsne', **kw):
        ModelInteract.__init__(self)
        StaticError.__init__(self, net, error, std)
        if fp is not None:
            self.run(fp, embedding, **kw)

    def init_embedding(self, embedding='tsne', **kw):
        super().init_embedding(embedding, **kw)
        self.init_net(self.ax[0], self.f.net)
        self.plot_embedding(self.ax[2], self.f.net, self.P, self.error, **kw)

    def run(self, fp, embedding='tsne', **kw):
        super().run(fp)
        self.init_plot()
        self.init_embedding(embedding, **kw)

class DynamicErrorInteract(DynamicError, ModelInteract):

    def __init__(self, net, error, fp=None, embedding='tsne', **kw):
        ModelInteract.__init__(self)
        DynamicError.__init__(self, net, error)
        self.selected['embedding_curves'] = []
        self.select_mod['embedding_curves'] = {'alpha' : 0.75}
        self.plot_args['embedding_curves'] = {'alpha' : 0.1, 'zorder' : 0}
        self.fselect['embedding_curves'] = {'alpha' : self._set_curve_alpha}
        self.fselect['embedding_points'] = {'s' : self._set_point_sizes}

        if fp is not None:
            self.run(fp, embedding, **kw)

    def _set_curve_alpha(self, x, y):
        for c in self.plots['embedding_curves'][self.label[x]]:
            c.set_alpha(y)

    def _set_point_sizes(self, x, y):
        for i in self.curves[self.label[x]]:
            self.plots['embedding_points'][i].set_sizes([y])

    def init_embedding(self, embedding='tsne', **kw):
        super().init_embedding(embedding, **kw)
        self.init_net(self.ax[0], self.f.net)
        self.plot_embedding(self.ax[2], self.f.net, self.P, self.error, **kw)

    def run(self, fp, embedding='tsne', **kw):
        super().run(fp)
        self.init_plot()
        self.init_embedding(embedding, **kw)

    def get_curves(self):
        curves = defaultdict(list)
        for i, l in enumerate(self.label):
            curves[l].append(i)
        return curves

    def plot_embedding(self, axis, net, P, c, **kw):
        super().plot_embedding(axis, net, P, c, **kw)
        curve_plots = {}# defaultdict(list)
        self.curves = self.get_curves()
        kw = self.plot_args['embedding_curves']
        for k, v in self.curves.items():
            kw['c'] = self.colors[max(v, key=lambda i: self.error[i])]
            curve_plots[k] = self.ax[2].plot(self.P[v,0], self.P[v,1], **kw)
        self.plots['embedding_curves'] = curve_plots

    def select_curve(self, key):
        for l, f in self.fselect['embedding_curves'].items():
            f(key, self.select_mod['embedding_curves'][l])
        self.selected['embedding_curves'].append(key)

    def select_key(self, key):
        super().select_key(key)
        self.select_curve(key)
