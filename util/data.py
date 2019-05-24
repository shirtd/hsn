from numpy.random import normal, randint, rand
from numpy.fft import fft2, ifft2
import numpy as np

def circle(n=20, r=1., uniform=False, noise=0.1):
    t = np.linspace(0, 1, n, False) if uniform else np.random.rand(n)
    e = r * (1 + noise * (2 * np.random.rand(n) - 1)) if noise else r
    return np.array([e*np.cos(2 * np.pi * t),
                    e*np.sin(2*np.pi*t)]).T.astype(np.float32)

def double_circle(n=50, r=(1., 0.7), uniform=False, noise=0.1):
    p1 = circle(int(n * r[0] / sum(r)), r[0], uniform, noise)
    p2 = circle(int(n * r[1] / sum(r)), r[1], uniform, noise)
    return np.vstack([p1 - np.array([r[0] + noise, 0.]),
                    p2 + np.array([r[1] + noise, 0.])])

def torus(n=1000, R=0.7, r=0.25):
    t = 2*np.pi * np.random.rand(2, n)
    x = (R + r * np.cos(t[1])) * np.cos(t[0])
    y = (R + r * np.cos(t[1])) * np.sin(t[0])
    z = r * np.sin(t[1])
    return np.vstack([x, y, z]).T

def grf(alpha, m=1024):
    n = int(np.sqrt(m))
    fin = list(range(0, n // 2 + 1)) + list(range(n // -2 + 1, 0))
    x = np.stack(np.meshgrid(fin, fin), axis=2).reshape(-1, 2)
    f = lambda r: np.sqrt(np.sqrt(r[0] ** 2 + r[1] ** 2) ** alpha) if r.sum() else 0.
    a = abs(ifft2(normal(size=(n, n)) * np.array(list(map(f, x))).reshape(n, n)))
    return (a - a.min()) / (a.max() - a.min())

def grid_idx(n):
    x = range(n) if isinstance(n, int) else n
    return np.stack(np.meshgrid(x, x), axis=2)

def error_probs(n_errors, n_ctl, ctl_noise=0.01):
    return np.concatenate((rand(10), ctl_noise * rand(n_ctl)))

def random_path(n, t):
    return randint(-n, n, size=(t, 2))

def linear_path(n, t):
    return np.tile(np.linspace(-n, n, t, dtype=int), (2, 1)).T
