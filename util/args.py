import argparse

TEST = True

N = 16
EXP = -5
NOISE_COEFF = 1.
BOUND = 0.25
DELTA = 0.05

if TEST:
    N = 10
    BOUND = 1.
    NOISE_COEFF = 0

STD = 5
ERROR = 0.5
NERROR = 15
NCTL = 5
TIME = 1

EMBEDDING = 'tsne'

parser = argparse.ArgumentParser(description='HSN Function exploration.')
parser.add_argument('file', nargs='?', default=None, help='pickle file to load')
parser.add_argument('-N', '--net-size', type=int, default=N, help="maximum number of sensors. default: %d" % N)
parser.add_argument('-n', '--noise', type=float, default=NOISE_COEFF, help="noise coefficient c . default: %0.2f "\
                                                                    "(noise := coeff * 10 / sqrt(N))" % NOISE_COEFF)
parser.add_argument('-d', '--delta', type=float, default=DELTA, help='bound step size. default: %0.2f' % DELTA)
parser.add_argument('-b', '--bound', type=float, default=BOUND, help='initial bound: default: %0.2f' % BOUND)
parser.add_argument('-e', '--exponent', type=int, default=EXP, help='gaussian exponent. default: %0.1f' % EXP)
parser.add_argument('-s', '--std', type=float, default=STD, help='function std. deviation. default: %0.1f' % STD)
parser.add_argument('-xe', '--max-error', type=float, default=ERROR, help='maximum allowed error. default: %0.2f' % ERROR)
parser.add_argument('-re', '--random-errors', type=int, default=NERROR, help='number of random errors: default: %d' % NERROR)
parser.add_argument('-ne', '--non-errors', type=int, default=NCTL, help='number of controls: default: %d' % NCTL)
parser.add_argument('-E', '--embedding', type=str, default=EMBEDDING, help='embedding function. default: %s' % EMBEDDING)
parser.add_argument('-t', '--time', type=int, default=1, help='movement time points. default: %s' % TIME)
parser.add_argument('--save', nargs='?', const=None, default=False, help='pickle net (saved to ./data).')

def print_args(args, default=True):
    a = vars(args)
    if not default:
        a = {k : v for k, v in a.items() if v != parser.get_default(k)}
    if len(a):
        l = max(len(k) for k in a.keys())
        print('[ args ]')
        for k, v in a.items():
            print(' | %s%s : %s' % (' ' * (l - len(k)), k, str(v)))
