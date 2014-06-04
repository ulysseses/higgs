# distutils: language = c++
# cython: wraparound = False
# cython: boundscheck = False
from types cimport *
import numpy as np
cimport numpy as np
from numpy cimport ndarray as array
from info_measure cimport *
from itertools import combinations
import sys  # debugging


cdef flt merit(array[intm, ndim=2] X, array[ints, ndim=1] y, mode='uc'):
	k = X.shape[1]
	#sys.stdout.write('k = %d\n' % k)
	if mode == 'uc':
		func_ms = su_ms
		func_mm = su_mm
	else:
		func_ms = snmdl_ms
		func_mm = snmdl_mm
	avg_rcf = np.mean([func_ms(X[:, col], y) for col in xrange(k)])
	cdef flt n = 0.0, avg_rff = 0.0
	if k != 1:
		# avg_rff = np.mean([func_mm(X[:, col1], X[:, col2]) for col1 in xrange(k)
			# for col2 in xrange(k) if col1 != col2])  # <-- make more efficient
		for col1, col2 in combinations(range(k), 2):
			avg_rff += func_mm(X[:, col1], X[:, col2])
			n += 1
		avg_rff /= n
	else:
		avg_rff = 1.0
	return k*avg_rcf / (k + k*(k - 1)*avg_rff)** 0.5

def cfs(X, y, backward=True, look_ahead=1, mode='uc'):
	counter = look_ahead
	best_score = merit(X, y, mode)
	if backward:
		cols = range(X.shape[1])
		best_cols = cols[:]
		while counter > 0 and len(cols) > 2:
			counter -= 1
			scores = []
			for i in range(len(cols)):
				curr_cols = cols[:]
				curr_cols.pop(i);
				scores.append(merit(X[:, curr_cols], y, mode))
			curr_best_score = max(scores)
			cols.pop(np.argmax(scores));
			if curr_best_score > best_score:
				best_score = curr_best_score
				best_cols = cols[:]
				counter = look_ahead
	else:
		cols = []
		best_cols = cols[:]
		remaining_cols = range(X.shape[1])
		while counter > 0 and len(remaining_cols) > 2:
			counter -= 1
			scores = []
			for i in range(len(remaining_cols)):
				curr_cols = cols[:]
				curr_cols.append(remaining_cols[i])
				scores.append(merit(X[:, curr_cols], y, mode))
			curr_best_score = max(scores)
			c = remaining_cols.pop(np.argmax(scores))
			cols.append(c)
			if curr_best_score > best_score:
				best_score = curr_best_score
				best_cols = cols[:]
				counter = look_ahead
	return best_cols




