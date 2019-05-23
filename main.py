#!/usr/bin/env python
from primitives.interact import StaticErrorInteract
from primitives.network import HSN, LoadHSN
from util.args import parser, print_args
import pickle as pkl
import numpy as np

if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args)

    if args.file is None:
        noise = args.noise * 10 / args.net_size
        alpha = (np.sqrt(2) + noise)
        KW = {'net_size' : args.net_size, 'exp' : args.exponent,
                'bound' : args.bound, 'delta' : args.delta,
                'alpha' : alpha, 'beta' : 3 * alpha,
                'dim' : 2, 'noise' : noise}

        net = HSN(**KW)
        if args.save is not False:
            net.save(args.save)

    else:
        print('[ loading %s' % args.file)
        with open(args.file, 'rb') as f:
            KW = pkl.load(f)

        net = LoadHSN(**KW)
        rfp = np.random.rand(args.random_errors)
        fp = list(rfp) + [0] * args.non_errors

        ARGS = (args.max_error, args.std, fp, args.embedding)
        self = StaticErrorInteract(net, *ARGS)
        input('[ press any key to exit ]')
