# cython: boundscheck = False
# cython: wraparound = False
''' Selective Pre-processing of Imbalanced Data (SPIDER) algorithm 

SPIDER pseudo-code
----------------------------------------------------------------
for x, y in X, labels:
	if classify_knn(x, 3) == y:
		flag[x] = safe
	else:
		flag[x] = noisy

noisies = [x for x in X if flag[x] == noisy]
if weak:
	for x in [x for x in minorX if flag[x] == noisy]:
		amplify x |knn(x, 3, majorX, safe)| times
elif weak and relabel:
	for x in [x for x in minorX if flag[x] == noisy]:
		amplify x |knn(x, 3, majorX, safe)| times
	for x in [x for x in minorX if flag[x] == noisy]:
		for xx in knn(x, 3, major, noisy):
			relabel(xx)
else:  # strong
	for x in [x for x in minorX if flag[x] == safe]:
		amplify x |knn(x, 3, majorX, safe)| times
	for x, y in [x for x in minorX if flag[x] == noisy], labels:
		if classify_knn(x, 5) == y:
			amplify x |knn(x, 3, majorX, safe)| times
		else:
			amplify x |knn(x, 5, majorX, safe)| times
'''

from types cimport *
from types_r import *
import numpy as np
cimport numpy as np
from numpy cimport ndarray as array
from sklearn.neighbors import NearestNeighbors
from libc.math cimport fabs, pow as cpow


def classify_knn(array[flt, ndim=1] x, array[ints, ndim=1] labels, nn, k=3):
	'''
	Returns (dist_weights, nbrs,
			 actual_minor_influence, largest_possible_minor_influence)

	Classify using weighted kNN. `k+1` is passed as the `n_neighbors` parameter
	because we're ignore the `x` in `X` itself.

	`dist_weights` are the weights of the neighbors inversely weighted by distance
	away from `x`. The dot product of `dist_weights` and `labels[nbrs]` gives the
	signal, and the decision function (`classify_knn`) will predict based on
	whether or not this signal (`actual_minor_influence`) is larger than half the
	maximum possible signal (`largest_possible_minor_influence`).

	In simpler words, `actual_minor_influence` is equal to |knn(x, 3, minor)|
	weighted by inverse-distance, while `largest_possible_minor_influence` is
	equal to |knn(x, 3, minor)| also weighted by inverse-distance, but this time x
	is assumed to gauranteed have 3 minor closest neighbors.
	`largest_possible_minor_influence` is really used to just "normalize"
	`actual_minor_influence` for class-decision purposes.

	However, `classify_knn` does not perform the actual classification, b/c the
	SPIDER algorithm will need `actual_minor_influence` to calculate the re-weighting
	factor for noisy, minor examples (hence no normalization after
	dot product).
	'''
	d, n = nn.kneighbors(x, n_neighbors=k+1)
	cdef:
		array[flt, ndim=1] dists = d[0][1:].astype(flt_r)
		array[np.intp_t, ndim=1] nbrs = n[0][1:]
		array[flt, ndim=1] dist_weights = (dists / np.min(dists)) ** -1
		flt actual_minor_influence = np.dot(labels[nbrs], dist_weights)
		flt largest_possible_minor_influence = np.sum(dist_weights)
	return dist_weights, nbrs, actual_minor_influence, largest_possible_minor_influence

cdef flt knn_major_helper(array[ints, ndim=1] labels, array major_noisy_mask,
	array[flt, ndim=1] dist_weights, array[np.intp_t, ndim=1] nbrs, safe=True):
	''' If `safe` is True, `minor_influence` is |knn(x, k, major, safe/noisy)| is returned.
		If else, then relabeling knn(x, 3, major, noisy) is performed '''
	cdef array[np.intp_t, ndim=1] nbrs2  # used only in noisy case
	cdef np.intp_t nbr
	cdef np.intp_t i
	if safe:
		# mask out minor labels and select only safe's
		return <flt> np.dot((labels[nbrs] == 0) & ~major_noisy_mask[nbrs], dist_weights)
	else:
		nbrs2 = nbrs[(dist_weights ** -1) < 3]  # only neighbors close to it,
		nbrs2 = nbrs2[labels[nbrs2] == 0]		   # are major labeled,
		nbrs2 = nbrs2[major_noisy_mask[nbrs2]]	   # and are flagged noisy
		if len(nbrs2) != 0:
			for i in xrange(len(nbrs2)):
				nbr = nbrs2[i]
				labels[nbr] = 1
				major_noisy_mask[nbr] = 0
		return 0.0

def mydist(array[np.float64_t, ndim=1] x, array[np.float64_t, ndim=1] y, np.float64_t p=0.5):
	''' Modified distance function. `p` is set to 0.5 b/c any higher than that,
		and the distance function is less interpretable/useful in higher dimensions. '''
	cdef np.float64_t dist = 0.0
	cdef np.intp_t i
	for i in xrange(len(x)):
		dist += cpow(fabs(x[i] - y[i]), p)
	dist = cpow(dist, 1 / p)
	return dist

def spider(array[flt, ndim=2] X0, array[ints, ndim=1] labels0, array[flt, ndim=1] weights0,
	weak=True, relabel=False, copy_=True, **kwargs):
	'''
	Perform the SPIDER algorithm on dataset X.

	A nearest neighbors "check" is applied in all cases to flag noisy and safe examples
		for both classes.

	For parameters, when `weak` is flagged, minority examples are weighted more heavily
		wherever their nearest neighbors are mostly majority classes.

	When `relabel` is flagged, SPID will re-label a majority example when its
		nearest neighbors are mostly minority classes.

	When `weak` is unflagged (i.e. "strong"), the kNN algorithm is reused, but 'k' is
		now set to 5. For each noisy minority example, if it is correctly categorized,
		then its weight is increased proportional to the number of majority classes in
		its 3 closest neighbors. However, if it is NOT correctly categorized, then its
		weight is increased proportional to the number of majority classes in its 5
		closest neighbors.

	`copy_` controls whether or not to perform SPID inplace or not. I.e. `copy_=False`
		will modify `X`, `labels`, and `weights` inplace.

	Pass in a User-defined `func` as parameter if you want to use a custom metric for
		evaluating distances. This is particularly useful for using p<1 norms for
		high dimensional datasets.
	'''
	cdef:
		np.intp_t i
		array[flt, ndim=2] X
		array[ints, ndim=1] labels
		array[flt, ndim=1] weights

	if copy_:
		if relabel:
			labels = labels0.copy()
		else:
			labels = labels0
		weights = weights0.copy()
	else:
		labels = labels0
		weights = weights0
	X = X0
	# if func:
		# nn = NearestNeighbors(algorithm='auto', metric='pyfunc', func=func)
	# else:
		# nn = NearestNeighbors(algorithm='auto', metric='minkowski', p=p)
	nn = NearestNeighbors(**kwargs)
	nn.fit(X)
	dw_nbrs_cache = [None] * len(labels)

	cdef:
		array noisy = np.zeros(len(labels), dtype=np.bool_)
		array[flt, ndim=1] x
		ints y
		array[flt, ndim=1] dist_weights
		array[np.intp_t, ndim=1] nbrs
		flt score
		flt norm

	for i in xrange(len(labels)):
		x, y = X[i], labels[i]
		dist_weights, nbrs, score, norm = classify_knn(x, labels, nn, k=3)
		if score < norm / 2:
			if y:  # minor & noisy
				noisy[i] = 1
				dw_nbrs_cache[i] = (dist_weights, nbrs)
		else:
			if (1-y):  # major & noisy
				noisy[i] = 1
			else:  # minor & safe
				dw_nbrs_cache[i] = (dist_weights, nbrs)
	# # major_noisy <-- indices of all y in major flagged as noisy
	# # minor_noisy <-- indices of all y in minor flagged as noisy
	cdef array major_noisy_mask = noisy & (labels == 0)
	cdef array minor_noisy_mask = noisy & (labels == 1)
	if weak and relabel:
		for i in np.where(minor_noisy_mask)[0]:
			dist_weights, nbrs = dw_nbrs_cache[i]
			score = knn_major_helper(labels, major_noisy_mask, dist_weights, nbrs, safe=True)
			weights[i] *= 1 + score
		for i in np.where(minor_noisy_mask)[0]:
			# select neighbors not too far away
			dist_weights, nbrs = dw_nbrs_cache[i]
			knn_major_helper(labels, major_noisy_mask, dist_weights, nbrs, safe=False);
	elif weak:
		for i in np.where(minor_noisy_mask)[0]:
			dist_weights, nbrs = dw_nbrs_cache[i]
			score = knn_major_helper(labels, major_noisy_mask, dist_weights, nbrs, safe=True)
			weights[i] *= 1 + score
	else:  # strong
		for i in xrange(len(minor_noisy_mask)):
			if labels[i]:  # ensure minor examples ONLY
				assert labels[i] != 0
				if ~minor_noisy_mask[i]:  # (minor &) safe, reweight
					dist_weights, nbrs = dw_nbrs_cache[i]
				elif minor_noisy_mask[i]:  # minor & noisy, reweight with nn=5
					dist_weights, nbrs, score, norm = classify_knn(x, labels, nn, k=5)
					if score >= norm / 2:  # classified correctly after 5 neighbors
						dist_weights, nbrs = dw_nbrs_cache[i]
				score = knn_major_helper(labels, major_noisy_mask, dist_weights, nbrs, safe=True)
				weights[i] *= 1 + score

	return X, labels, weights  # warning: X is NOT a copy
					










