import numpy.random as rnd
from scipy import signal
import dionysus as dio
from abc import ABC
import numpy as np

''' ------------ *
 | ABSTRACT TYPE |
 * ------------ '''

class Function(ABC):

    def __init__(self, net):
        self.net, self.S = net, [v for v, w in net.Salpha]

    def induced_vertices(self, f):
        return [max(v, key=lambda j: f[j]) for v in self.S]

    def persist(self, fs):
        S = [dio.Simplex(v, w) for v, w in zip(self.S, fs)]
        F = dio.Filtration(sorted(S, key=lambda s: s.data))
        H = dio.homology_persistence(F)
        D = dio.init_diagrams(H, F)
        return {'filtration' : F,
                'homology' : H,
                'diagram' : D}

    def __call__(self, p):
        pass


''' -------- *
 | BASE TYPE |
 * -------- '''

class ErrorFunction(Function):

    def __init__(self, net, max_error, std=5):
        Function.__init__(self, net)
        self.max_error = max_error
        self.K = self.tkern() # self.gkern(std)

    def gkern(self, std):
        k = signal.gaussian(self.net.net_size, std).reshape(-1, 1)
        return np.outer(k, k)

    def tkern(self):
        k = signal.triang(self.net.net_size).reshape(-1, 1)
        return np.outer(k, k)

    def add_error(self, f, p):
        f, n, m = f.copy(), len(f), int(p * len(f))
        for i in rnd.choice(n, m, replace=False):
            f[i] *= (1 + self.max_error * (rnd.rand() - 0.5))
        return f

    def get_fun(self, K):
        indices = [tuple(map(int, x)) for x in self.net.data]
        return np.array([K[i, j] for i, j in indices])

    def run(self, fclean, p):
        f = self.add_error(fclean, p)
        w = self.induced_vertices(f)
        fs = np.array([f[i] for i in w])
        fc = np.array([fclean[i] for i in w])
        err = sum((fs - fc) ** 2) / len(w)
        return {**self.persist(fs),
                'function' : fs,
                'error' : err}

    def __call__(self, p):
        fclean = self.get_fun(self.K)
        return self.run(fclean, p)


''' --------------- *
 | CONCRETE OBJECTS |
 * --------------- '''
 
class DynamicErrorFunction(ErrorFunction):

    def __init__(self, net, max_error):
        Function.__init__(self, net)
        self.max_error = max_error

    def shift(self, s):
        n, d = self.net.net_size, abs(s)
        l = np.s_[d:] if s < 0 else np.s_[:n]
        return signal.triang(n + d)[l].reshape(-1, 1)

    def tkern(self, t):
        return np.outer(*map(self.shift, t))

    def __call__(self, lpt):
        l, p, t = lpt
        fclean = self.get_fun(self.tkern(t))
        return {'label' : l, **self.run(fclean, p)}
