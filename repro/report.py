#! /usr/bin/env python3

import os
import sys
import re
import yaml
import numpy as np

has_feature_logs = False

if has_feature_logs:
	featuresDir = 'logs/features'
	features = [yaml.safe_load(''.join([l for l in open(os.path.join(featuresDir, logFile)) if not l.startswith('[')])) for logFile in sorted(os.listdir(featuresDir))]

fisher_vectorsDir = 'logs/fisher_vectors'
fisher_vectors = [yaml.safe_load(open(os.path.join(fisher_vectorsDir, logFile)).read()) for logFile in sorted(os.listdir(fisher_vectorsDir))]

classification = yaml.safe_load(open('data/classification.txt').read())

if has_feature_logs:
	print('Average frame count: %d' % np.mean([y['Frame count'] for y in features]))
	print('Average frame size: %dx%d' % tuple(list(map(lambda x: np.mean([float(y['Original frame size'].split('x')[x]) for y in features]), range(2)))))
	print('Average descriptor count: %d' % np.mean([y['Calls.ComputeDescriptor'] for y in features]))
	print('')
	print('All fps are reported without taking file reading and writing into account, howevere, video decoding is included.')
	print('')
	print('Features (%s enabled):' % ', '.join([k for k, v in features[0]['Enabled descriptors'].items() if v]))
	print('  Average total fps: %.2f' % np.mean([y['Fps'] for y in features]))

	print('  Average HOG fps: %.2f' % np.mean([y['Frame count'] / float(0.01 + y['Reading (sec)'] + y['Interp (sec)']['HOG'] + y['IntHist (sec)']['HOG'] + y['Desc (sec)']['HOG']) for y in features]))
	print('  Average HOF fps: %.2f' % np.mean([y['Frame count'] / float(0.01 + y['Reading (sec)'] + y['Interp (sec)']['HOFMBH'] + y['IntHist (sec)']['HOF'] + y['Desc (sec)']['HOF']) for y in features]))
	print('  Average MBH fps: %.2f' % np.mean([y['Frame count'] / float(0.01 + y['Reading (sec)'] + y['Interp (sec)']['HOFMBH'] + y['IntHist (sec)']['MBH'] + y['Desc (sec)']['MBH']) for y in features]))
	print('')

# Instead of hardcoding a field like '10-105'
# look for a field that matches the pattern of a numerical range
fv_k = 0
for key in fisher_vectors[0]:
	print("key %s", key)
	if re.match(r'^\d+-\d+$', key):
		fv_k = fisher_vectors[0][key]['k']
		break

print('Fisher vectors (components: %d, s-t grids enabled: %s, knn: %s, second order enabled: %s, FLANN trees: %s, FLANN comparisons: %s):' % (fv_k, fisher_vectors[0]['Enable spatio-temporal grids (1x1x1, 1x3x1, 1x1x2)'], fisher_vectors[0].get('K_nn', 'N/A'), fisher_vectors[0]['Enable second order'], fisher_vectors[0].get('FLANN trees', -1), fisher_vectors[0].get('FLANN checks', -1)))

if has_feature_logs:
	print('  Average total fps: %.2lf' % np.mean([y['Frame count'] / float(z.get('Copying (sec)', 0) + z.get('Flann (sec)', 0) + z.get('Assigning (sec)', 0)) for y, z in zip(features, fisher_vectors)]))
	print('')

print('Classification:')
for k in sorted(set(classification) - set(['mean'])):
	print('  %-15s\t%.4f' % (k, classification[k]))
print('')
print('  mean: %.4f' % classification['mean'])
