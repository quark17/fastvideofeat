#! /usr/bin/env python3

import os
import sys
from datetime import datetime
import itertools
from functools import reduce
import numpy as np
import pandas as pd
from sklearn.svm import SVC

if (len(sys.argv) < 3) or (len(sys.argv) > 4):
	print('usage: %s <split_dir> <allclips_file> [--format#]' % sys.argv[0])
	raise SystemExit(1)

RES_DIR = 'results'
EVAL_DIR = sys.argv[1]
allClips = list(map(lambda l: l[:-1], open(sys.argv[2])))
all_k = np.loadtxt(sys.stdin)

if (len(sys.argv) == 4) and (sys.argv[3] == '--format3'):
        # Modified UCF-101 format to be more universal (used for CalTech-256)
        format = 3
elif (len(sys.argv) == 4) and (sys.argv[3] == '--format2'):
        # Support UCF-101 formats
        format = 2
else:
        # Default is HMDB-51 formats
        format = 1

# Based on the format, determine:
#   classLables
#   splits
#
if (format == 3):
	print('Reading format3...')
	classLabels = list(map(lambda l: l.replace('\n',''), open(os.path.join(EVAL_DIR, 'classLabels.txt'))))
	def read_split(SPLIT_IND):
		idx = 1 + SPLIT_IND
		train = []
		for l in open(os.path.join(EVAL_DIR, 'trainlist%02d.txt' % idx)):
			cols = l.split()
			train += [(cols[0], cols[1])]
		test = []
		for l in open(os.path.join(EVAL_DIR, 'testlist%02d.txt' % idx)):
			cols = l.split()
			test += [(cols[0], cols[1])]
		print('Split %d: trainlen %d, testlen %d' % (idx, len(train), len(test)))
		return (train, test)
	splits = list(map(read_split, list(range(3))))
	print()
elif (format == 2):
	print('Reading format2...')
	classLabels = list(map(lambda l: l.split()[-1], open(os.path.join(EVAL_DIR, 'classInd.txt'))))
	def read_split(SPLIT_IND):
		idx = 1 + SPLIT_IND
		train = []
		for l in open(os.path.join(EVAL_DIR, 'trainlist%02d.txt' % idx)):
			cols = (l.split()[0]).split('/')
			train += [(cols[1], cols[0])]
		test = []
		for l in open(os.path.join(EVAL_DIR, 'testlist%02d.txt' % idx)):
			cols = (l.split()[0]).split('/')
			test += [(cols[1], cols[0])]
		print('Split %d: trainlen %d, testlen %d' % (idx, len(train), len(test)))
		return (train, test)
	splits = list(map(read_split, list(range(3))))
	print()
elif (format == 1):
	print('Reading format1...')
	classLabels = sorted(set([f.split('_test_split')[0] for f in os.listdir(EVAL_DIR) if '_test_split' in f]))
	def read_split(SPLIT_IND):
		fixClipName = lambda x: reduce(lambda acc, ch: acc.replace(ch, '_'), '][;()&?!', x)
		train, test = [], []
		for classLabel in classLabels:
			d = dict(list(map(str.split, open(os.path.join(EVAL_DIR, '%s_test_split%d.txt' % (classLabel, 1 + SPLIT_IND))))))
			train += [(fixClipName(k), classLabel) for k, v in d.items() if v == '1']
			test += [(fixClipName(k), classLabel) for k, v in d.items() if v == '2']
		return (train, test)
	splits = list(map(read_split, list(range(3))))
else:
        print('Unrecognized format')
        raise SystemExit(1)

slice_kernel = lambda inds1, inds2: all_k[np.ix_(list(map(allClips.index, inds1)), list(map(allClips.index, inds2)))]
REG_C = 1.0

def svm_train_test(train_k, test_k, ytrain, REG_C):
	model = SVC(kernel = 'precomputed', C = REG_C, max_iter = 10000)
	model.fit(train_k, ytrain)

	flatten = lambda ls: list(itertools.chain(ls))
	train_conf, test_conf = list(map(flatten, map(model.decision_function, [train_k, test_k])))
	return train_conf, test_conf

def one_vs_rest(SPLIT_IND):
	global combined_res_df

	calc_accuracy = lambda chosen, true: sum([int(true[i] == chosen[i]) for i in range(len(chosen))]) / float(len(chosen))
	partition = lambda f, ls: (list(filter(f, ls)), list(itertools.filterfalse(f, ls)))
	train, test = splits[SPLIT_IND]
	xtest, ytest = list(zip(*test))

	confs = []
	for i, classLabel in enumerate(classLabels):
		postrain, negtrain = partition(lambda x: x[1] == classLabel, train)
		xtrain = list(zip(*(postrain + negtrain)))[0]
		ytrain = [1]*len(postrain) + [-1]*len(negtrain)
		train_k, test_k = list(map(lambda x: slice_kernel(x, xtrain), [xtrain, xtest]))
		train_conf, test_conf = svm_train_test(train_k, test_k, ytrain, REG_C)
		confs.append(test_conf)

	ntest = len(test)
	chosen = [max([(confs[k][i], k) for k in range(len(classLabels))])[1] for i in range(ntest)]
	true = [classLabels.index(test[i][1]) for i in range(ntest)]

	res_df = pd.DataFrame(pd.Series.eq(pd.Series(chosen),pd.Series(true))).transpose()
	res_df.columns = pd.DataFrame(test)[0]
	combined_res_df = pd.concat([combined_res_df,res_df], ignore_index=True, sort=False)

	return calc_accuracy(chosen, true)

aps = []
combined_res_df = pd.DataFrame()
for SPLIT_IND in range(len(splits)):
	test_ap = one_vs_rest(SPLIT_IND)
	aps.append(test_ap)
	print('%-15s: %.4f' % ('split_%d' % SPLIT_IND, test_ap))

print('\nmean: %.4f' % np.mean(aps))

if not os.path.exists(RES_DIR):
        os.mkdir(RES_DIR)
res_fname = os.path.join(RES_DIR, 'results-' + str(datetime.now().timestamp()) + '.csv')
combined_res_df.to_csv(res_fname)
