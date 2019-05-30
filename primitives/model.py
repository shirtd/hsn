from primitives.function import ErrorFunction, DynamicErrorFunction
from sklearn.manifold import MDS, TSNE
from util.hsn.persist import bneck_mat
from util.tools.fun import funzip
from tqdm import tqdm
from abc import ABC
import numpy as np

''' ------------ *
 | ABSTRACT TYPE |
 * ------------ '''

class Model(ABC):
    EMBEDDINGS = {
        'mds' : (MDS, {
                'dissimilarity' : 'precomputed'
            }),
        'tsne' : (TSNE, {
                'metric' : 'precomputed',
                'method' : 'exact',
                'perplexity' : 15
            })}
    dmat, P = None, None

    def __init__(self, f):
        self.f = f

    def __call__(self, fp):
        return funzip(map(self.f, tqdm(fp, '[ persistence')))

    def run(self, fp):
        pass

    def init_embedding(self, embedding, **kw):
        pass

    def get_dict(self):
        return {'diagram' : [[[(p.birth, p.death) for p in d] for d in dgm] for dgm in self.diagram]}



''' -------- *
 | BASE TYPE |
 * -------- '''

class ErrorModel(Model):

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

    def get_dict(self):
        d = super().get_dict()
        return {**d, 'attributes' : {'error' : self.error,
                                    'function' : self.function,
                                    'dmat' : self.dmat}}


''' --------------- *
 | CONCRETE OBJECTS |
 * --------------- '''

class StaticError(ErrorModel):
    funtion_t = ErrorFunction

    def __init__(self, net, max_error, std=5):
        f = self.funtion_t(net, max_error, std)
        ErrorModel.__init__(self, f)

class DynamicError(ErrorModel):
    funtion_t = DynamicErrorFunction

    def __init__(self, net, max_error):
        f = self.funtion_t(net, max_error)
        ErrorModel.__init__(self, f)

    def get_dict(self):
        d = super().get_dict()
        d['attributes']['label'] = self.label
        return d
