# distutils: language = c++
# cython: boundscheck = False
# cython: wraparound = False
from __future__ import division
from itertools import izip
from operator import itemgetter
from collections import defaultdict, Counter

from types cimport *
import numpy as np
cimport numpy as np
from numpy cimport ndarray as array
from info_measure cimport su_ms as su

cdef flt inconsistency_count(inconsistency_set):
	cdef flt cardinality = 0.0, v
	for v in inconsistency_set.values():
		cardinality += v
	return cardinality - inconsistency_set.most_common(1)[0][1]

def group_inconsistencies(X, array[ints, ndim=1] y):
	inconsistency_dict = defaultdict(Counter)
	cdef ints label
	for instance, label in izip(X, y):
		inconsistency_dict[tuple(instance)][label] += 1
	return inconsistency_dict

cdef flt icr(X, array[ints, ndim=1] y):
	cdef flt ans = 0.0
	for t, inconsistency_set in group_inconsistencies(X, y).iteritems():
		ans += inconsistency_count(inconsistency_set)
	return ans / X.shape[0]

def cc(array[intm, ndim=2] X, array[ints, ndim=1] y, slice_, ind):
	newslice_ = [i for i in slice_ if i != ind]
	return icr(X[:, newslice_], y) - icr(X[:, slice_], y)

def interact(array[intm, ndim=2] X, array[ints, ndim=1] y, threshold):
	N = X.shape[1]
	slice_ = range(X.shape[1])
	su_ind_pairs = []
	for i in xrange(N):
		su_ind_pairs.append((su(X[:,i], y), i))
	su_ind_pairs.sort(key=itemgetter(0), reverse=True)
	if threshold == 0:
		#print "interact keeping all columns"
		return [ind for score, ind in su_ind_pairs]
	for i in xrange(N - 1, -1, -1):
		score, ind = su_ind_pairs[i]
		p = cc(X, y, slice_, ind)
		if p < threshold:
			slice_ = [j for j in slice_ if j != ind]
			#print "deleting ind %2d, p = %.4f" % (ind, p)
			del su_ind_pairs[i]
		else:
			#print "keeping  ind %2d, p = %.4f" % (ind, p)
			su_ind_pairs[i] = ind
	return su_ind_pairs

