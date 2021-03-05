from cython cimport floating


# BLAS Level 1 ################################################################
cdef floating _dot(int, floating*, int, floating*, int) nogil

cdef void _axpy(int, floating, floating*, int, floating*, int) nogil

cdef floating _nrm2(int, floating*, int) nogil

cdef void _scal(int, floating, floating*, int) nogil