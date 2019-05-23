import pickle as pkl
import sys, os

def delete_line():
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')

def wait(s=''):
    ret = raw_input(s)
    delete_line()
    return ret

def save_pkl(fname, x):
    with open(fname, 'w') as f:
        pkl.dump(x, f)
    return fname

def load_pkl(fname):
    with open(fname, 'r') as f:
        x = pkl.load(f)
    return x

def save_state(fname, data, **kwargs):
    dout = {'data' : data, 'args' : kwargs}
    print('[ saving %s' % fname)
    save_pkl(fname, dout)
    return dout

def query_save(data, fpath='.', **kwargs):
    ret = raw_input('[ save as: %s/' % fpath)
    if not ret: return None, None
    flist = os.path.split(ret)
    for dir in flist[:-1]:
        fpath = os.path.join(fpath, dir)
        if not os.path.isdir(fpath):
            print(' | creating directory %s' % fpath)
            os.mkdir(fpath)
    fname = os.path.join(fpath, flist[-1])
    return fname, save_state(fname, data, **kwargs)

def load_args(*keys):
    if len(sys.argv) > 1:
        try:
            x = load_pkl(sys.argv[1])
            return [x[k] for k in keys]
        except e:
            print(e)
    return []
