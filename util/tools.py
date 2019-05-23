from collections import defaultdict
from functools import reduce
import numpy as np

def lmap(f, x):
    return list(map(f, x))

def lfilter(f, x):
    return list(filter(f, x))

def lzip(*x):
    return list(zip(x))

def dzip(k, v):
    return dict(zip(k, v))

def amap(f, x):
    return np.array(lmap(f, x))

def afilter(f, x):
    return np.array(lfilter(f, x))

def azip(*x):
    return np.array(lzip(*x))

def dzip(x, y):
    return {k : x[k] + [v] for k, v in y.items()}

def funzip(ld):
    return reduce(dzip, ld, defaultdict(list))

def funzmap(f, ld):
    return funzip(map(lambda p: f(*p), ld))
