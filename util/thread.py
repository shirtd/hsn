from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import sys

def pmap(fun, x, *args):
    pool = Pool()
    f = partial(fun, *args)
    try:
        # y = pool.map(f, tqdm(x))
        y = list(pool.map(f, x))
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y

def cmap(funct, x, *args):
    fun = partial(funct, *args)
    f = partial(map, fun)
    pool = Pool()
    try:
        # y = pool.map(f, tqdm(x))
        y = list(pool.map(f, x))
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y
