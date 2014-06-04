#!/usr/bin/env python
# Usage: ./preproc_test.py DATA_PATH OUTPUT_PATH BINS_PATH
# e.g.:  ./preproc_test.py ../data/test.csv ../working/reduced_test.dat ../working/bins.dat
# DATA_PATH is the path of the input csv file
# OUTPUT_PATH is the path for the output pickle binary file
# BINS_PATH is the path of the bins pickle binary file
# Data will be stored in the following order
# 	X_B
#	X0_B
#	events_B
# 	X_C
# 	X0_C
#	events_C
# 	X_AB
# 	X0_AB
#	events_AB
# 	X_BC
# 	X0_BC
#	events_BC
# 	X_ABC
# 	X0_ABC
#	events_ABC
# 	X_D
# 	X0_D
#	events_D
# Access the pickled file with the following instructions:
# http://stackoverflow.com/questions/20716812/saving-and-loading-multiple-objects-in-python-pickle-file

import time
t0 = time.time()
import sys
import os.path
import cPickle as pickle
from itertools import izip
from mdl import discretize
import numpy as np


''' Load dataset '''
# Load bins
binspath = "../working/bins.dat" if len(sys.argv) == 1 else sys.argv[3]
binspath = os.path.abspath(binspath)
with open(binspath, 'rb') as dat:
	bins_B = pickle.load(dat)
	bins_C = pickle.load(dat)
	bins_AB = pickle.load(dat)
	bins_BC = pickle.load(dat)
	bins_ABC = pickle.load(dat)
	bins_D = pickle.load(dat)

# Load csv into NumPy array
inpath = "../data/test.csv" if len(sys.argv) == 1 else sys.argv[1]
inpath = os.path.abspath(inpath)
data_all = np.loadtxt(inpath, delimiter=',', skiprows=1)
data_all[data_all == -999.0] = np.NaN

# Identify groups of columns
A = list(np.r_[5:8,13,27:30])
B = list(np.r_[24:27])
C = list(np.r_[1])
D = list(np.r_[2:5, 8:13, 14:24,30])
X0 = data_all[:, D+A+B+C].astype(np.float32)
events = data_all[:, 0].astype(np.int64)
arr = np.vstack([data_all[:,col] for col in [D[0], A[0], B[0], C[0]]]).T

Di = range(0, len(D))
Ai = range(len(D), len(D+A))
Bi = range(len(D+A), len(D+A+B))
Ci = range(len(D+A+B), len(D+A+B+C))

del data_all
print "loaded from %s" % inpath


''' Mask application '''

d, a, b, c = [np.isfinite(arr[:,col]) for col in range(4)]
mask_B = d & ~a & b & ~c
mask_C = d & ~a & ~b & c
mask_AB = d & a & b & ~c
mask_BC = d & ~a & b & c
mask_ABC = d & a & b & c
mask_D = d & ~a & ~b & ~c

X0_B = X0[:, Di+Bi][mask_B]
events_B = events[mask_B]
print "B mask done"
X0_C = X0[:, Di+Ci][mask_C]
events_C = events[mask_C]
print "C mask done"
X0_AB = X0[:, Di+Ai+Bi][mask_AB]
events_AB = events[mask_AB]
print "AB mask done"
X0_BC = X0[:, Di+Bi+Ci][mask_BC]
events_BC = events[mask_BC]
print "BC mask done"
X0_ABC = X0[:, Di+Ai+Bi+Ci][mask_ABC]
events_ABC = events[mask_ABC]
print "ABC mask done"
X0_D = X0[:, Di][mask_D]
events_D = events[mask_D]
print "D mask done"

del X0


''' MDL Discretization '''
X_B = np.vstack([np.digitize(col, bins, right=False).astype(np.int16) 
	for col,bins in izip(X0_B.T, bins_B)]).T
print "B discretization done"

X_C = np.vstack([np.digitize(col, bins, right=False).astype(np.int16) 
	for col,bins in izip(X0_C.T, bins_C)]).T
print "C discretization done"

X_AB = np.vstack([np.digitize(col, bins, right=False).astype(np.int16) 
	for col,bins in izip(X0_AB.T, bins_AB)]).T
print "AB discretization done"

X_BC = np.vstack([np.digitize(col, bins, right=False).astype(np.int16) 
	for col,bins in izip(X0_BC.T, bins_BC)]).T
print "BC discretization done"

X_ABC = np.vstack([np.digitize(col, bins, right=False).astype(np.int16) 
	for col,bins in izip(X0_ABC.T, bins_ABC)]).T
print "ABC discretization done"

X_D = np.vstack([np.digitize(col, bins, right=False).astype(np.int16) 
	for col,bins in izip(X0_D.T, bins_D)]).T
print "D discretization done"

assert np.sum(~np.isfinite(X_B)) == 0
assert np.sum(~np.isfinite(X0_B)) == 0
assert np.sum(~np.isfinite(X_C)) == 0
assert np.sum(~np.isfinite(X0_C)) == 0
assert np.sum(~np.isfinite(X_AB)) == 0
assert np.sum(~np.isfinite(X0_AB)) == 0
assert np.sum(~np.isfinite(X_BC)) == 0
assert np.sum(~np.isfinite(X0_BC)) == 0
assert np.sum(~np.isfinite(X_ABC)) == 0
assert np.sum(~np.isfinite(X0_ABC)) == 0
assert np.sum(~np.isfinite(X_D)) == 0
assert np.sum(~np.isfinite(X0_D)) == 0


''' Save to file '''
outpath = "../working/reduced_test.dat" if len(sys.argv) == 1 else sys.argv[2]
outpath = os.path.abspath(outpath)
with open(outpath, 'wb') as dat:
	pickle.dump(X_B, dat, -1)
	pickle.dump(X0_B, dat, -1)
	pickle.dump(events_B, dat, -1)
	pickle.dump(X_C, dat, -1)
	pickle.dump(X0_C, dat, -1)
	pickle.dump(events_C, dat, -1)
	pickle.dump(X_AB, dat, -1)
	pickle.dump(X0_AB, dat, -1)
	pickle.dump(events_AB, dat, -1)
	pickle.dump(X_BC, dat, -1)
	pickle.dump(X0_BC, dat, -1)
	pickle.dump(events_BC, dat, -1)
	pickle.dump(X_ABC, dat, -1)
	pickle.dump(X0_ABC, dat, -1)
	pickle.dump(events_ABC, dat, -1)
	pickle.dump(X_D, dat, -1)
	pickle.dump(X0_D, dat, -1)
	pickle.dump(events_D, dat, -1)

print "Data successfully dumped into %s" % outpath




t1 = time.time()
print "Total time elapsed: %d min %d sec" % ((t1-t0)//60, (t1-t0) - ((t1-t0)//60)*60)
