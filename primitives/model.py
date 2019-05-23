from sklearn.manifold import MDS, TSNE
from util.persist import bneck_mat
import numpy.random as rnd
from scipy import signal
from util import funzip
import dionysus as dio
from tqdm import tqdm
from abc import ABC
import numpy as np

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

class Model(ABC):
    dmat, P = None, None
    EMBEDDINGS = {
        'mds' : (MDS, {
                'dissimilarity' : 'precomputed'
            }),
        'tsne' : (TSNE, {
                'metric' : 'precomputed',
                'method' : 'exact',
                'perplexity' : 15
            })}
    def __init__(self, f):
        self.f = f
    def __call__(self, fp):
        return funzip(map(self.f, tqdm(fp, '[ persistence')))
    def run(self, fp):
        pass
    def init_embedding(self, embedding, **kw):
        pass

class ErrorFunction(Function):
    def __init__(self, net, max_error, std=5):
        Function.__init__(self, net)
        self.max_error = max_error
        self.K = self.gkern(std)
    def gkern(self, std):
        n = self.net.net_size
        k = signal.gaussian(n, std=std).reshape(n, 1)
        return np.outer(k, k)
    def add_error(self, f, p):
        f, n, m = f.copy(), len(f), int(p * len(f))
        for i in rnd.choice(n, m, replace=False):
            f[i] *= (1 + self.max_error * (rnd.rand() - 0.5))
        return f
    def __call__(self, p):
        indices = [tuple(map(int, x)) for x in self.net.data]
        fclean = np.array([self.K[i, j] for i, j in indices])
        f = self.add_error(fclean, p)
        w = self.induced_vertices(f)
        fs = np.array([f[i] for i in w])
        fc = np.array([fclean[i] for i in w])
        err = sum((fs - fc) ** 2)/ len(w)
        return {**self.persist(fs),
                'function' : fs,
                'error' : err}

class StaticModel(Model):
    def __init__(self, f):
        Model.__init__(self, f)
    def run(self, fp):
        for k, v in self(fp).items():
            setattr(self, k, v)
        self.dmat = bneck_mat(self.diagram)
    def init_embedding(self, embedding, **kw):
        fembed, default = self.EMBEDDINGS[embedding]
        f = fembed(**{**kw, **default})
        self.P = f.fit_transform(self.dmat)

class StaticError(StaticModel):
    def __init__(self, net, max_error, std=5):
        f = ErrorFunction(net, max_error, std)
        StaticModel.__init__(self, f)
