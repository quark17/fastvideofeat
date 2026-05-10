#! /usr/bin/env python3

import sys
import argparse
import numpy as np
import joblib as joblib

from skimage.feature import learn_gmm

parser = argparse.ArgumentParser()
parser.add_argument('--gmm_ncomponents', type = int, required = True)
parser.add_argument('--vocab', nargs = 2, required = True)
args = parser.parse_args()
cutFrom, cutTo = map(int, args.vocab[0].split('-'))

data = np.loadtxt(sys.stdin, dtype = np.float32, usecols = range(cutFrom, 1 + cutTo))

k = args.gmm_ncomponents
# Attempt to match 
gm_args = {
    'covariance_type': 'diag',
    'max_iter': 50,
    # Set this to 8 to take the best of 8 seeds!
    'n_init': 1
    # Pass this for reproducible results
    #'random_state': 0
}
gmm = learn_gmm(data, n_modes=k, gm_args=gm_args)

joblib.dump(gmm, args.vocab[1])
