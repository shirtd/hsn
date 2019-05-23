#!/usr/bin/env python
from primitives.interact import StaticErrorInteract, DynamicErrorInteract
from util.args import parser, print_args
from primitives.hsn import HSN, LoadHSN
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

        if args.time > 1:
            # t = np.random.randint(-net.net_size, net.net_size, size=(len(fp), 5, 2))
            # t = np.random.randint(-net.net_size, net.net_size, size=(args.time, 2))
            t = np.tile(np.linspace(-net.net_size, net.net_size, args.time, dtype=int), (2, 1)).T
            fpt = [(i, x, z) for i, x in enumerate(fp) for z in t]
            ARGS = (args.max_error, fpt, args.embedding)
            finteract = DynamicErrorInteract
        else:
            ARGS = (args.max_error, args.std, fp, args.embedding)
            finteract = StaticErrorInteract

        self = finteract(net, *ARGS)
        input('[ press any key to exit ]')
