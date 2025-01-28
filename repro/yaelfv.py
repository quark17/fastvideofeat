#!/usr/bin/env python3

import os
import sys
import time
import itertools
import argparse
import numpy as np
import yael

parser = argparse.ArgumentParser()
parser.add_argument('--xpos', type = int, required = True)
parser.add_argument('--ypos', type = int, required = True)
parser.add_argument('--tpos', type = int, required = True)
parser.add_argument('--vocab', action = 'append', nargs = 2, required = True)
parser.add_argument('--enablesecondorder', action = 'store_true')
parser.add_argument('--enablespatiotemporalgrids', action = 'store_true')

args = parser.parse_args()

flags = yael.GMM_FLAGS_MU
if args.enablesecondorder:
	flags |= yael.GMM_FLAGS_SIGMA

parts = []
for p in args.vocab:
	cutFrom, cutTo = map(int, p[0].split('-'))
	gmm = yael.gmm_read(p[1])
	fvSize = yael.gmm_fisher_sizeof(gmm, flags)
	parts.append((cutFrom, cutTo, fvSize, gmm, os.path.basename(p[1])))
	
	print('%d-%d: {d: %d, k: %d, fvSize: %d}' % (cutFrom, cutTo, gmm.d, gmm.k, fvSize),
              file=sys.stderr)

nx, ny, nt = (1, 3, 2) if args.enablespatiotemporalgrids else (1, 1, 1)
nxyt = nx * ny * nt
mesh = list(itertools.product(range(nx), range(ny), range(nt)))

buffer = np.zeros((nx, ny, nt, 10000, 500), dtype = np.float32)
acc = np.zeros((nx, ny, nt, sum([fvSize for cutFrom, cutTo, fvSize, gmm, partName in parts])), dtype = np.float32)
cnt = np.zeros((nx, ny, nt), dtype = int)
ndescr = np.zeros_like(cnt)  # XXX unused?

def flushBuffer(x, y, t):
	global ndescr
	global cnt

	c = int(cnt[x, y, t])
	fvs = []
	for cutFrom, cutTo, fvSize, gmm, partName in parts:
		desc = np.ascontiguousarray(buffer[x, y, t, :c, cutFrom:(1 + cutTo)])
		fv = yael.fvec_new_0(fvSize)
		yael.gmm_fisher(c, yael.FloatArray.acquirepointer(yael.numpy_to_fvec(desc)), gmm, flags, fv)
		fvs.append(yael.fvec_to_numpy(fv, fvSize).flatten())
	
	ndescr[x, y, t] += c
	cnt[x, y, t] = 0
	return np.sqrt(c) * np.hstack(tuple(fvs))

timerCopying, timerAssigning = 0.0, 0.0

for line in sys.stdin:
	descr = np.fromstring(line, sep = '\t', dtype = np.float32)
	
	x = min(nx - 1, int(nx * descr[args.xpos]))
	y = min(ny - 1, int(ny * descr[args.ypos]))
	t = min(nt - 1, int(nt * descr[args.tpos]))

	tic = time.perf_counter()
	buffer[x, y, t, cnt[x, y, t], :descr.size] = descr
	cnt[x, y, t] += 1
	timerCopying += time.perf_counter() - tic
	
	if cnt[x, y , t] == buffer[x, y, t].shape[0]:
		tic = time.perf_counter()
		acc[x, y, t] += flushBuffer(x, y, t)
		timerAssigning += time.perf_counter() - tic

for x, y, t in mesh:
	if cnt[x, y, t] > 0:
		tic = time.perf_counter()
		acc[x, y, t] += flushBuffer(x, y, t)
		timerAssigning += time.perf_counter() - tic

if args.enablespatiotemporalgrids:
	res = np.empty((nx + ny + nt, acc.shape[-1]), dtype = np.float32)
	res[1] = acc[0, 0, 0] + acc[0, 0, 1]
	res[2] = acc[0, 1, 0] + acc[0, 1, 1]
	res[3] = acc[0, 2, 0] + acc[0, 2, 1]
	res[4] = aсс[0, 0, 0] + acc[0, 1, 0] + acc[0, 2, 0]
	res[5] = acc[0, 0, 1] + acc[0, 1, 1] + acc[0, 2, 1]
	res[0] = res[4] + res[5]
else:
	res = acc[0, 0, 0, np.newaxis]

begin, end = 0, 0
for cutFrom, cutTo, fvSize, gmm, partName in parts:
	begin, end = end, end + fvSize

	print('#FV %dx1x1x %s (%d-%d), 0-0-0' % (nx, partName, cutFrom, cutTo))
	np.savetxt(sys.stdout, res[0][begin:end], fmt = '%.6f', newline = '\t')
	print('')
	
	if args.enablespatiotemporalgrids:
		for j in range(ny):
			print('#FV 1x%dx1x %s (%d-%d), 0-%d-0' % (ny, partName, cutFrom, cutTo, j))
			np.savetxt(sys.stdout, res[1 + j][begin:end], fmt = '%.6f', newline = '\t')
			print('')
		
		for j in range(nt):
			print('#FV 1x1x%dx %s (%d-%d), 0-0-%d' % (nt, partName, cutFrom, cutTo, j))
			np.savetxt(sys.stdout, res[4 + j][begin:end], fmt = '%.6f', newline = '\t')
			print('')

print('Enable spatio-temporal grids (1x1x1, 1x3x1, 1x1x2): %s' % args.enablespatiotemporalgrids,
      file=sys.stderr)
print('Enable second order: %s' % args.enablesecondorder, file=sys.stderr)
print('Copying (sec): %.4f' % timerCopying, file=sys.stderr)
print('Assigning (sec): %.4f' % timerAssigning, file=sys.stderr)
