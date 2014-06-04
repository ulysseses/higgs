#!/usr/bin/env python
# Usage: ./preproc_train.py DATA_PATH OUTPUT_PATH BINS_PATH
# e.g.:  ./preproc_train.py ../data/training.csv ../working/reduced_train.dat ../working/bins.dat
# DATA_PATH is the path of the input csv file
# OUTPUT_PATH is the path for the output pickle binary file
# BINS_PATH is the path for the bins pickle binary file
# Data will be stored in the following order
# 	Xtr_B
#	Xtr0_B
# 	ytr_B
#	w_B
# 	Xtr_C
#	Xtr0_C
# 	ytr_C
#	w_C
# 	Xtr_AB
# 	Xtr0_AB
# 	ytr_AB
#	w_AB
# 	Xtr_BC
# 	Xtr0_BC
# 	ytr_BC
#	w_BC
# 	Xtr_ABC
# 	Xtr0_ABC
# 	ytr_ABC
#	w_ABC
# 	Xtr_D
# 	Xtr0_D
# 	ytr_D
#	w_D
# Access the pickled file with the following instructions:
# http://stackoverflow.com/questions/20716812/saving-and-loading-multiple-objects-in-python-pickle-file

import time
t0 = time.time()
import sys
import os.path
import cPickle as pickle
from info_measure import discretize
import numpy as np
from types_r import *


''' Load dataset '''
# Load csv into NumPy array
inpath = "../data/training.csv" if len(sys.argv) == 1 else sys.argv[1]
inpath = os.path.abspath(inpath)
dtrain = np.loadtxt(inpath, delimiter=',', skiprows=1,
					converters={32: lambda x: int(x=='s'.encode('utf-8'))})
dtrain[dtrain == -999.0] = np.NaN

# Identify groups of columns
A = list(np.r_[5:8,13,27:30])
B = list(np.r_[24:27])
C = list(np.r_[1])
D = list(np.r_[2:5, 8:13, 14:24,30])
Xtr0 = dtrain[:, D+A+B+C].astype(flt_r)
arr = np.vstack([dtrain[:,col] for col in [D[0], A[0], B[0], C[0]]]).T
w = dtrain[:,31]
ytr = dtrain[:,32].astype(np.int8)

Di = range(0, len(D))
Ai = range(len(D), len(D+A))
Bi = range(len(D+A), len(D+A+B))
Ci = range(len(D+A+B), len(D+A+B+C))

del dtrain
print "loaded from %s" % inpath


''' Mask application and rounding for easier discretization '''
def dynamic_round(X, percentile=0):
	''' Round to (3 - DR[j]) decimals, where DR[j] is the jth column's logarithmic range.
	
		`percentile` controls what % of data away from X.min() and X.max() should bound
		the range of X (e.g. percentile=5 --> range of 5 to 95 percentile region of X)
		
		If there is no range (i.e. DR[j] is NaN), then don't round
		'''
	DR = np.round(np.log10(np.percentile(X, 100-percentile, axis=0) -
						   np.percentile(X, percentile, axis=0))).astype(np.int)
	for j in xrange(X.shape[1]):
		# bit hacks; i.e. if DR[j] is NOT NaN, i.e. skip when given overflow error
		if bool(~np.isfinite(DR[j])) or DR[j] == np.iinfo(np.int).min:
			pass
		else:
			X[:,j] = np.around(X[:,j], decimals=3-DR[j])
	if not X.flags['C_CONTIGUOUS']:
		raise Exception('Say whaa? Not contiguous man!')
	return X

d, a, b, c = [np.isfinite(arr[:,col]) for col in range(4)]
mask_B = d & ~a & b & ~c
mask_C = d & ~a & ~b & c
mask_AB = d & a & b & ~c
mask_BC = d & ~a & b & c
mask_ABC = d & a & b & c
mask_D = d & ~a & ~b & ~c

Xtr0_B = dynamic_round(Xtr0[:, Di+Bi][mask_B], percentile=2)
ytr_B = ytr[mask_B]
w_B = w[mask_B]
print "B mask and pre-rounding done"
Xtr0_C = dynamic_round(Xtr0[:, Di+Ci][mask_C], percentile=2)
ytr_C = ytr[mask_C]
w_C = w[mask_C]
print "C mask and pre-rounding done"
Xtr0_AB = dynamic_round(Xtr0[:, Di+Ai+Bi][mask_AB], percentile=2)
ytr_AB = ytr[mask_AB]
w_AB = w[mask_AB]
print "AB mask and pre-rounding done"
Xtr0_BC = dynamic_round(Xtr0[:, Di+Bi+Ci][mask_BC], percentile=2)
ytr_BC = ytr[mask_BC]
w_BC = w[mask_BC]
print "BC mask and pre-rounding done"
Xtr0_ABC = dynamic_round(Xtr0[:, Di+Ai+Bi+Ci][mask_ABC], percentile=2)
ytr_ABC = ytr[mask_ABC]
w_ABC = w[mask_ABC]
print "ABC mask and pre-rounding done"
Xtr0_D = dynamic_round(Xtr0[:, Di][mask_D], percentile=2)
ytr_D = ytr[mask_D]
w_D = w[mask_D]
print "D mask and pre-rounding done"

del Xtr0


''' MDL Discretization '''
stack, bins_B = [], []
for col in Xtr0_B.T:
	digitized, bins = discretize(col, ytr_B)
	stack.append(digitized)
	bins_B.append(bins)
Xtr_B = np.vstack(stack).T
print "B discretization done"

stack, bins_C = [], []
for col in Xtr0_C.T:
	digitized, bins = discretize(col, ytr_C)
	stack.append(digitized)
	bins_C.append(bins)
Xtr_C = np.vstack(stack).T
print "C discretization done"

stack, bins_AB = [], []
for col in Xtr0_AB.T:
	digitized, bins = discretize(col, ytr_AB)
	stack.append(digitized)
	bins_AB.append(bins)
Xtr_AB = np.vstack(stack).T
print "AB discretization done"

stack, bins_BC = [], []
for col in Xtr0_BC.T:
	digitized, bins = discretize(col, ytr_BC)
	stack.append(digitized)
	bins_BC.append(bins)
Xtr_BC = np.vstack(stack).T
print "BC discretization done"

stack, bins_ABC = [], []
for col in Xtr0_ABC.T:
	digitized, bins = discretize(col, ytr_ABC)
	stack.append(digitized)
	bins_ABC.append(bins)
Xtr_ABC = np.vstack(stack).T
print "ABC discretization done"

stack, bins_D = [], []
for col in Xtr0_D.T:
	digitized, bins = discretize(col, ytr_D)
	stack.append(digitized)
	bins_D.append(bins)
Xtr_D = np.vstack(stack).T
print "D discretization done"

assert np.sum(~np.isfinite(Xtr_B)) == 0
assert np.sum(~np.isfinite(Xtr0_B)) == 0
assert np.sum(~np.isfinite(Xtr_C)) == 0
assert np.sum(~np.isfinite(Xtr0_C)) == 0
assert np.sum(~np.isfinite(Xtr_AB)) == 0
assert np.sum(~np.isfinite(Xtr0_AB)) == 0
assert np.sum(~np.isfinite(Xtr_BC)) == 0
assert np.sum(~np.isfinite(Xtr0_BC)) == 0
assert np.sum(~np.isfinite(Xtr_ABC)) == 0
assert np.sum(~np.isfinite(Xtr0_ABC)) == 0
assert np.sum(~np.isfinite(Xtr_D)) == 0
assert np.sum(~np.isfinite(Xtr0_D)) == 0


''' Save to file '''
outpath = "../working/reduced_train.dat" if len(sys.argv) == 1 else sys.argv[2]
outpath = os.path.abspath(outpath)
with open(outpath, 'wb') as dat:
	pickle.dump(Xtr_B, dat, -1)
	pickle.dump(Xtr0_B, dat, -1)
	pickle.dump(ytr_B, dat, -1)
	pickle.dump(w_B, dat, -1)
	pickle.dump(Xtr_C, dat, -1)
	pickle.dump(Xtr0_C, dat, -1)
	pickle.dump(ytr_C, dat, -1)
	pickle.dump(w_C, dat, -1)
	pickle.dump(Xtr_AB, dat, -1)
	pickle.dump(Xtr0_AB, dat, -1)
	pickle.dump(ytr_AB, dat, -1)
	pickle.dump(w_AB, dat, -1)
	pickle.dump(Xtr_BC, dat, -1)
	pickle.dump(Xtr0_BC, dat, -1)
	pickle.dump(ytr_BC, dat, -1)
	pickle.dump(w_BC, dat, -1)
	pickle.dump(Xtr_ABC, dat, -1)
	pickle.dump(Xtr0_ABC, dat, -1)
	pickle.dump(ytr_ABC, dat, -1)
	pickle.dump(w_ABC, dat, -1)
	pickle.dump(Xtr_D, dat, -1)
	pickle.dump(Xtr0_D, dat, -1)
	pickle.dump(ytr_D, dat, -1)
	pickle.dump(w_D, dat, -1)

print "Data dumped into %s" % outpath

binspath = "../working/bins.dat" if len(sys.argv) == 1 else sys.argv[3]
binspath = os.path.abspath(binspath)
with open(binspath, 'wb') as dat:
	pickle.dump(bins_B, dat, -1)
	pickle.dump(bins_C, dat, -1)
	pickle.dump(bins_AB, dat, -1)
	pickle.dump(bins_BC, dat, -1)
	pickle.dump(bins_ABC, dat, -1)
	pickle.dump(bins_D, dat, -1)
	
print "Bins dumped into %s" % binspath
print 'Please run "python preproc_test.py" now to preprocess the test set'




t1 = time.time()
print "Total time elapsed: %d min %d sec" % ((t1-t0)//60, (t1-t0) - ((t1-t0)//60)*60)
