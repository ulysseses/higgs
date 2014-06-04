# distutils: language = c++
# NumPy dtypes at compiletime
import numpy as np
cimport numpy as np

ctypedef np.float64_t flt  # assume 32-bit
ctypedef np.int8_t ints    # smallest possible integer container
ctypedef np.int16_t intm   # medium integer type