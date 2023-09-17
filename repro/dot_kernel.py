#! /usr/bin/env python3

import os
import sys
from functools import reduce
import numpy as np
import numpy.linalg as linalg

IN = [os.path.join(sys.argv[1], x[:-1] + '.txt') for x in open(sys.argv[2])]

# The number of descriptor lines in the file
n = len(list(filter(lambda x: not x.startswith('#'), open(IN[0]))))
# Initial value for a vector of n elements
ks = [None]*n

def nextNonCommentLine(f):
        res = '#'
        while res.startswith('#'):
                res = f.readline()
        return res

# Open each of the encoded feature files
# Note: this may require increasing the number of allowed open file
#       (see 'ulimit -a')
fs = [open(path) for path in IN]

def fv_norm(fv):
	fv = np.clip(fv, -1000, 1000)
	fv = np.sign(fv) * np.sqrt(np.abs(fv))
	fv /= (1e-4 + linalg.norm(fv))
	return fv

for i in range(len(ks)):
	x = np.vstack(tuple([fv_norm(np.fromstring(nextNonCommentLine(f), dtype = np.float32, sep = '\t')) for f in fs]))
	ks[i] = np.dot(x, x.T)

res = reduce(np.add, ks)
np.savetxt(sys.stdout, res, fmt = '%.6f', delimiter = '\t')
