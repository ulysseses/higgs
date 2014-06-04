# distutils: language = c++
## cython: wraparound = False
## cython: boundscheck = False

from types import *
import numpy as np
cimport numpy as np
from numpy cimport ndarray as array
from libcpp.unordered_set cimport unordered_set as cset
from cython.operator cimport dereference as deref, \
    preincrement as inc, postincrement as inc2
cimport libc.math
import sys  # debugging


########### Mean Functions

cdef inline flt mean_f(array[flt, ndim=1] x, flt c):
	cdef flt ans = 0
	cdef np.intp_t i
	for i in xrange(len(x)):
		if x[i] == c:
			ans += 1
	return ans / len(x)

cdef inline flt mean_m(array[intm, ndim=1] x, intm c):
	cdef flt ans = 0
	cdef np.intp_t i
	for i in xrange(len(x)):
		if x[i] == c:
			ans += 1
	return ans / len(x)

cdef inline flt mean_s(array[ints, ndim=1] y, ints c):
	cdef flt ans = 0
	cdef np.intp_t i
	for i in xrange(len(y)):
		if y[i] == c:
			ans += 1
	return ans / len(y)
	
cdef inline flt joint_mean_ms(array[intm, ndim=1] x, intm c1, array[ints, ndim=1] y, ints c2):
	cdef flt ans = 0
	cdef np.intp_t i
	for i in xrange(len(x)):
		if (x[i] == c1) & (y[i] == c2):
			ans += 1
	return ans / len(y)

cdef inline flt joint_mean_mm(array[intm, ndim=1] x, intm c1, array[intm, ndim=1] y, intm c2):
	cdef flt ans = 0
	cdef np.intp_t i
	for i in xrange(len(x)):
		if (x[i] == c1) & (y[i] == c2):
			ans += 1
	return ans / len(y)

	
########### Univariate Entropy Functions

cdef flt entropy_f(array[flt, ndim=1] x):
	cdef:
		cset[flt] uniques
		np.intp_t i
		flt p, ans = 0.0
	for i in xrange(len(x)):
		uniques.insert(x[i])
	cdef cset[flt].iterator it = uniques.begin()
	while it != uniques.end():
		p = mean_f(x, deref(inc2(it)))
		if p > 0:
			ans += p * libc.math.log2(p)
	return -ans

cdef flt entropy_s(array[ints, ndim=1] y):
	cdef:
		cset[ints] uniques
		np.intp_t i
		flt p, ans = 0.0
	for i in xrange(len(y)):
		uniques.insert(y[i])
	cdef cset[ints].iterator it = uniques.begin()
	while it != uniques.end():
		p = mean_s(y, deref(inc2(it)))
		if p > 0:
			ans += p * libc.math.log2(p)
	return -ans
    
cdef flt entropy_m(array[intm, ndim=1] x):
	cdef:
		cset[intm] uniques
		np.intp_t i
		flt p, ans = 0.0
	for i in xrange(len(x)):
		uniques.insert(x[i])
	cdef cset[intm].iterator it = uniques.begin()
	while it != uniques.end():
		p = mean_m(x, deref(inc2(it)))
		#sys.stdout.write("p: %.4f\n" % p)
		if p > 0:
			ans += p * libc.math.log2(p)
	return -ans


########### Bivariate Entropy Function
# http://orange.biolab.si/blog/2012/06/15/joint-entropy-in-python/

cdef flt entropy2_ms(array[intm, ndim=1] x, array[ints, ndim=1] y):
	cdef:
		cset[intm] xset
		cset[ints] yset
		np.intp_t i
		flt p, ans = 0.0
	for i in xrange(len(x)):
		xset.insert(x[i])
		yset.insert(y[i])
	cdef cset[ints].iterator ity = yset.begin()
	cdef cset[intm].iterator itx
	cdef ints c2
	while ity != yset.end():
		c2 = deref(ity)
		itx = xset.begin()
		while itx != xset.end():
			p = joint_mean_ms(x, deref(inc2(itx)), y, c2)
			if p > 0:
				ans += p * libc.math.log2(p)
		inc(ity)
	return -ans

cdef flt entropy2_mm(array[intm, ndim=1] x, array[intm, ndim=1] y):
	cdef:
		cset[intm] xset
		cset[intm] yset
		np.intp_t i
		flt p, ans = 0.0
	for i in xrange(len(x)):
		xset.insert(x[i])
		yset.insert(y[i])
	cdef cset[intm].iterator ity = yset.begin()
	cdef cset[intm].iterator itx
	cdef intm c2
	while ity != yset.end():
		c2 = deref(ity)
		itx = xset.begin()
		while itx != xset.end():
			p = joint_mean_mm(x, deref(inc2(itx)), y, c2)
			if p > 0:
				ans += p * libc.math.log2(p)
		inc(ity)
	return -ans


########### Symmetric Uncertainty

cdef flt su_ms(array[intm, ndim=1] x, array[ints, ndim=1] y):
	cdef flt H_x_y = entropy_m(x) + entropy_s(y)
	cdef flt H_xy = entropy2_ms(x, y)
	if H_x_y == H_xy == 0:
		return 0.0
	return 2 * (H_x_y - H_xy) / H_x_y

cdef flt su_mm(array[intm, ndim=1] x, array[intm, ndim=1] y):  
	cdef flt H_x_y = entropy_m(x) + entropy_m(y)
	cdef flt H_xy = entropy2_mm(x, y)
	#sys.stdout.write("H_x_y: %s H_xy: %s\n" % (H_x_y, H_xy))
	#sys.stdout.write("H_x_y == H_xy == 0: %s\n" % (H_x_y == H_xy == 0))
	if H_x_y == H_xy == 0:
		return 0.0
	return 2 * (H_x_y - H_xy) / H_x_y


########### Symmetric Normalized Minimum Description Length

cdef flt ln_gamma(np.int64_t n):
	''' use natural log b/c I don't know how to calculate log2(gamma)
		http://stackoverflow.com/questions/3037113/python-calculate-multinomial-probability-density-functions-on-large-dataset
	'''
	if n < 1: return np.inf
	if n < 3: return 0.0
	cdef flt *c_ = [76.18009172947146, -86.50532032941677, \
		24.01409824083091, -1.231739572450155, \
		0.001208650973866179, -0.000005395239384953]
	cdef flt x = n, y = n
	cdef flt tm = x + 5.5
	tm -= (x + 0.5) * libc.math.log(tm)
	cdef flt se = 1.0000000000000190015
	cdef size_t j
	for j in xrange(6):
		y += 1.0
		se += c_[j] / y
	return (-tm + libc.math.log(2.5066282746310005 * se / x))

cdef inline ln_fact(np.int64_t n):
	return ln_gamma(n + 1)

cdef inline flt ln_binomial(np.int64_t top, np.int64_t bottom):
	return ln_fact(top) - ln_fact(bottom) - ln_fact(top - bottom)

cdef flt prior_mdl_s(array[ints, ndim=1] C):
	''' binary class assumption '''
	cdef np.int64_t n = len(C), n1 = 0
	cdef np.intp_t i
	for i in xrange(n):
		if C[i] == 1:
			n1 += 1
	cdef flt first = ln_binomial(n, n1)
	cdef flt second = libc.math.log(n + 1)
	return first + second

cdef flt prior_mdl_m(array[intm, ndim=1] F):
	# assume np.max(F) is the number of unique attribute values
	# and attribute values go from 0..max(F)
	cdef:
		np.int64_t n = np.max(F) + 1, N = len(F)
		np.intp_t i
		flt first = ln_fact(N), second = ln_binomial(N + n - 1, n - 1)
		array[np.int64_t, ndim=1] counts = np.zeros(n, dtype=np.int64)
	for i in xrange(len(F)):
		counts[F[i]] += 1
	for i in xrange(n):
		first -= ln_fact(counts[i])
	return first + second
	
cdef flt post_mdl_ms(array[intm, ndim=1] F, array[ints, ndim=1] C):
	''' binary class assumption '''
	# assume max(F) == number of unique attribute values
	# and attribute values go from 0..max(F)
	cdef:
		np.intp_t j
		np.int64_t k = np.max(F) + 1
		array[np.int64_t, ndim=1] counts1 = np.zeros(k, dtype=np.int64)
		array[np.int64_t, ndim=1] counts2 = np.zeros(k, dtype=np.int64)
	for j in xrange(len(F)):
		counts1[F[j]] += 1
		if F[j] == 1:
			counts2[F[j]] += 1
	cdef flt ans = 0
	for j in xrange(k):
		ans += ln_binomial(counts1[j], counts2[j])
		ans += libc.math.log(counts1[j] + 1)
	return ans

cdef flt post_mdl_sm(array[ints, ndim=1] C, array[intm, ndim=1] F):
	''' binary class assumption '''
	# assume max(F) == number of unique attribute values
	# and attribute values go from 0..max(F)
	cdef np.int64_t k = np.max(F) + 1
	cdef np.intp_t j, i
	cdef flt ans = 0.0
	cdef np.int64_t countsC1 = np.sum(C == 1)
	cdef np.int64_t countsC0 = len(C) - countsC1
	cdef array[np.int64_t, ndim=1] countsF1 = np.zeros(k, dtype=np.int64)
	cdef array[np.int64_t, ndim=1] countsF0 = np.zeros(k, dtype=np.int64)
	for j in xrange(len(F)):
		if C[j] == 1:
			countsF1[F[j]] += 1
		else:
			countsF0[F[j]] += 1
	ans += (ln_fact(countsC1) + ln_fact(countsC0)) * 2
	for j in xrange(k):
		ans -= ln_fact(countsF1[j]) + ln_fact(countsF0[j])
	ans -= ln_binomial(countsC1 + k - 1, k - 1)
	ans -= ln_binomial(countsC0 + k - 1, k - 1)
	return ans

cdef flt post_mdl_mm(array[intm, ndim=1] F1, array[intm, ndim=1] F2):
	cdef np.int64_t k1 = np.max(F1) + 1, k2 = np.max(F2) + 1
	cdef np.intp_t j, i
	cdef flt ans = 0.0
	cdef array[np.int64_t, ndim=1] top = np.zeros(k1, dtype=np.int64)
	cdef array[np.int64_t, ndim=1] bottom
	for j in xrange(len(F1)):
		top[F1[j]] += 1
	for j in xrange(k1):
		ans += ln_fact(top[j])
		bottom = np.zeros(k2, dtype=np.int64)
		for i in xrange(len(F2)):
			if F1[i] == j:
				bottom[F2[i]] += 1
		for i in xrange(k2):
			ans -= ln_fact(bottom[i])
		ans += ln_binomial(top[j] + k2 - 1, k2 - 1)
	return ans

cdef flt snmdl_ms(array[intm, ndim=1] F, array[ints, ndim=1] C):
	cdef flt prior_mdl_f = prior_mdl_m(F)
	cdef flt post_mdl_fc = post_mdl_ms(F, C)
	cdef flt prior_mdl_c = prior_mdl_s(C)
	cdef flt post_mdl_cf = post_mdl_sm(C, F)
	return 0.5 * ((prior_mdl_f - post_mdl_fc)/prior_mdl_f + (prior_mdl_c - post_mdl_cf)/prior_mdl_c)

cdef flt snmdl_mm(array[intm, ndim=1] F1, array[intm, ndim=1] F2):
	cdef flt prior_mdl_f1 = prior_mdl_m(F1)
	cdef flt post_mdl_f1f2 = post_mdl_mm(F1, F2)
	cdef flt prior_mdl_f2 = prior_mdl_m(F2)
	cdef flt post_mdl_f2f1 = post_mdl_mm(F2, F1)
	return 0.5 * ((prior_mdl_f1 - post_mdl_f1f2)/prior_mdl_f1 + (prior_mdl_f2 - post_mdl_f2f1)/prior_mdl_f2)


########### Fayyad Irani Discretization

cdef np.int64_t num_unique_s(array[ints, ndim=1] y):
	cdef cset[ints] uniques
	cdef np.intp_t i
	for i in xrange(len(y)):
		uniques.insert(y[i])
	return uniques.size()

def mdlpc_criterion(array[ints, ndim=1] y, array[ints, ndim=1] y1, array[ints, ndim=1] y2):
	''' gain must be bigger than the MDLPC criterion '''
	cdef flt h, h1, h2
	cdef np.int64_t k, k1, k2, n, n1, n2
	h = entropy_s(y)
	h1, h2 = entropy_s(y1), entropy_s(y2)
	k = num_unique_s(y)
	k1, k2 = num_unique_s(y1), num_unique_s(y2)
	n = len(y)
	n1, n2 = len(y1), len(y2)
	# gain - stuff > 0
	return (1+1.0*k/n)*h - (1.0*(n1-k1)*h1 + 1.0*(n2-k2)*h2 + \
		libc.math.log2(1.0*(n-1)*(3**k-2)))/n > 0

def discretize(array[flt] a, array[ints] y):
	''' Discretize attribute `a` based on the MDL principle
		`discretized_a` is an np.int16 type array '''
	cdef:
		array[flt, ndim=1] acopy
		array[ints, ndim=1] ycopy
		array[ints, ndim=1] y1, y2
		array[flt, ndim=1] unique_sorted_a
		flt j
		#array[np.npy_bool] mask  # booleans not su_mspported in numpy cython
	acopy = a
	ycopy = y
	unique_sorted_a = np.unique(a)
	unique_sorted_a.sort(kind='quicksort')
	bins = []
	for j in unique_sorted_a[1:]:
		mask = acopy < j
		y1, y2 = ycopy[mask], ycopy[~mask]
		if mdlpc_criterion(ycopy, y1, y2):
			bins.append(j)
			ycopy = y2
			acopy = acopy[~mask]
	discretized_a = np.digitize(a, bins, right=False).astype(np.int16)  # or np.int32
	return discretized_a, bins  # bins will have roundoff error,
								# but it won't affect `</<=`, `>/>=`, or `==`
                                
                                
