from cython cimport floating

from scipy.linalg.cython_blas cimport sdot, ddot
from scipy.linalg.cython_blas cimport saxpy, daxpy
from scipy.linalg.cython_blas cimport snrm2, dnrm2
from scipy.linalg.cython_blas cimport sscal, dscal


################
# BLAS Level 1 #
################

cdef floating _dot(int n, floating *x, int incx,
                   floating *y, int incy) nogil:
    """x.T.y"""
    if floating is float:
        return sdot(&n, x, &incx, y, &incy)
    else:
        return ddot(&n, x, &incx, y, &incy)


cpdef _dot_memview(floating[::1] x, floating[::1] y):
    return _dot(x.shape[0], &x[0], 1, &y[0], 1)


cdef void _axpy(int n, floating alpha, floating *x, int incx,
                floating *y, int incy) nogil:
    """y := alpha * x + y"""
    if floating is float:
        saxpy(&n, &alpha, x, &incx, y, &incy)
    else:
        daxpy(&n, &alpha, x, &incx, y, &incy)


cpdef _axpy_memview(floating alpha, floating[::1] x, floating[::1] y):
    _axpy(x.shape[0], alpha, &x[0], 1, &y[0], 1)


cdef floating _nrm2(int n, floating *x, int incx) nogil:
    """sqrt(sum((x_i)^2))"""
    if floating is float:
        return snrm2(&n, x, &incx)
    else:
        return dnrm2(&n, x, &incx)


cpdef _nrm2_memview(floating[::1] x):
    return _nrm2(x.shape[0], &x[0], 1)


cdef void _scal(int n, floating alpha, floating *x, int incx) nogil:
    """x := alpha * x"""
    if floating is float:
        sscal(&n, &alpha, x, &incx)
    else:
        dscal(&n, &alpha, x, &incx)


cpdef _scal_memview(floating alpha, floating[::1] x):
    _scal(x.shape[0], alpha, &x[0], 1)
