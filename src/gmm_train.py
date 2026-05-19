#! /usr/bin/env python3

import sys
import argparse
import numpy as np
import yael

parser = argparse.ArgumentParser()
parser.add_argument('--gmm_ncomponents', type = int, required = True)
parser.add_argument('--vocab', nargs = 2, required = True)
parser.add_argument('--seed', type = int, required = False, default = 0)
parser.add_argument('--redo', type = int, required = False, default = 1)
parser.add_argument('--nthreads', type = int, required = False, default = 1)
parser.add_argument('--niter', type = int, required = False, default = 50)
args = parser.parse_args()
cutFrom, cutTo = map(int, args.vocab[0].split('-'))

data = np.loadtxt(sys.stdin, dtype = np.float32, usecols = range(cutFrom, 1 + cutTo))

npoints, nfeatures = data.shape
niter = args.niter
nthreads = args.nthreads
seed = args.seed
redo = args.redo
flags = yael.GMM_FLAGS_W

if ((nthreads < 1) or (nthreads > 24)):
    print('Bad nthreads value: %d' % nthreads)
    raise SystemExit(1)

gmm = yael.gmm_learn(nfeatures, npoints, args.gmm_ncomponents, niter, yael.FloatArray.acquirepointer(yael.numpy_to_fvec(data)), nthreads, seed, redo, flags)

yael.gmm_write(gmm, args.vocab[1])
