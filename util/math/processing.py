import numpy as np

def normalize(x, axis=None):
    x = np.array(x) if isinstance(x, list) else x
    if x.max(axis) != x.min(axis):
        return (x - x.min(axis)) / (x.max(axis) - x.min(axis))
    return x
