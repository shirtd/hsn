#!/usr/bin/env python
from primitives.interact import StaticErrorInteract,\
                                DynamicErrorInteract, load_interact
from util.data import linear_path, error_probs, circular_path
from util.tools.io import load_pkl, save_pkl
from util.args import parser, print_args
from base.hsn import HSN, LoadHSN
import numpy as np
import os

if __name__ == '__main__':

    args = parser.parse_args()
    print_args(args)

    if args.file is None:

        net = HSN(**{'net_size' : args.net_size,
                        'exp' : args.exponent,
                        'bound' : args.bound,
                        'delta' : args.delta,
                        'noise' : args.noise})

        if args.save is not False:
            net.save(args.save)

    elif args.load is not False:

        self = load_interact(args.file, args.load, args.time, args.embedding)
        input('[ press any key to exit ]')

    else:
        net = LoadHSN(args.file)
        fp = error_probs(args.random_errors, args.non_errors)

        if args.time > 1:
            # t = linear_path(net.net_size, args.time)
            t = circular_path(net.net_size, args.time)
            fpt = [(i, x, z) for i, x in enumerate(fp) for z in t]
            ARGS = (args.max_error, fpt, args.embedding)
            finteract = DynamicErrorInteract
        else:
            ARGS = (args.max_error, args.std, fp, args.embedding)
            finteract = StaticErrorInteract

        self = finteract(net, *ARGS)
        self.save(args.file, args.time)
        input('[ press any key to exit ]')
