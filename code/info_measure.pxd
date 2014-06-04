# distutils: language = c++

from types cimport *
from numpy cimport ndarray as array

cdef flt snmdl_ms(array[intm, ndim=1] F, array[ints, ndim=1] C)
cdef flt snmdl_mm(array[intm, ndim=1] F1, array[intm, ndim=1] F2)
cdef flt su_ms(array[intm, ndim=1] x, array[ints, ndim=1] y)
cdef flt su_mm(array[intm, ndim=1] x, array[intm, ndim=1] y)