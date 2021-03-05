"""
Wrapper for liblinear

Author: fabian.pedregosa@inria.fr
"""

import  numpy as np
cimport numpy as np

from cython cimport floating
from scipy.linalg.cython_blas cimport sdot, ddot
from scipy.linalg.cython_blas cimport saxpy, daxpy
from scipy.linalg.cython_blas cimport snrm2, dnrm2
from scipy.linalg.cython_blas cimport sscal, dscal

include "_liblinear.pxi"

np.import_array()


def train_wrap(X,
               np.ndarray[np.float64_t, ndim=1, mode='c'] Y,
               bint is_sparse,
               int solver_type,
               double eps,
               double bias,
               double C,
               np.ndarray[np.float64_t, ndim=1] class_weight,
               int max_iter,
               unsigned random_seed,
               double epsilon,
               np.ndarray[np.float64_t, ndim=1, mode='c'] sample_weight):

    cdef parameter *param
    cdef problem *problem
    cdef model *model
    cdef char_const_ptr error_msg
    cdef int len_w

    if is_sparse:
        problem = csr_set_problem((<np.ndarray>X.data).data,
                                  X.dtype == np.float64,
                                  (<np.ndarray[np.int32_t, ndim=1, mode='c']>X.indices).data,
                                  (<np.ndarray[np.int32_t, ndim=1, mode='c']>X.indptr).data,
                                  (<np.int32_t>X.shape[0]),
                                  (<np.int32_t>X.shape[1]),
                                  (<np.int32_t>X.nnz),
                                  bias,
                                  sample_weight.data,
                                  Y.data)
    else:
        problem = set_problem((<np.ndarray>X).data,
                              X.dtype == np.float64,
                              (<np.int32_t>X.shape[0]),
                              (<np.int32_t>X.shape[1]),
                              (<np.int32_t>np.count_nonzero(X)),
                              bias,
                              sample_weight.data,
                              Y.data)

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.intc)

    param = set_parameter(solver_type,
                          eps,
                          C,
                          class_weight.shape[0],
                          class_weight_label.data,
                          class_weight.data,
                          max_iter,
                          random_seed,
                          epsilon)

    error_msg = check_parameter(problem, param)
    if error_msg:
        free_problem(problem)
        free_parameter(param)
        raise ValueError(error_msg)
    
    cdef BlasFunctions blas_functions
    blas_functions.dot = _dot[double]
    blas_functions.axpy = _axpy[double]
    blas_functions.scal = _scal[double]
    blas_functions.nrm2 = _nrm2[double]

    # early return
    with nogil:
        model = train(problem, param, &blas_functions)

    ### FREE
    free_problem(problem)
    free_parameter(param)

    # get sample coefficients
    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] alpha
    cdef int nr_sample = get_nr_sample(model)
    alpha = np.empty((1, nr_sample), order='F')
    copy_alpha(alpha.data, model, nr_sample)

    # clean up
    free_and_destroy_model(&model)

    return alpha


def set_verbosity_wrap(int verbosity):
    """
    Control verbosity of libsvm library
    """
    set_verbosity(verbosity)


# Private

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
