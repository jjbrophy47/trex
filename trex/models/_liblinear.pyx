"""
Wrapper for liblinear

Adapted from:
https://github.com/scikit-learn/scikit-learn/blob/\
95119c13af77c76e150b753485c662b7c52a41a2/sklearn/svm/_liblinear.pyx
"""

import  numpy as np
cimport numpy as np

from libc.stdio cimport printf

include "_liblinear.pxi"

np.import_array()

def train_wrap(X,
               np.ndarray[np.float64_t, ndim=1, mode='c'] Y,
               int solver_type,
               double eps,
               double bias,
               double C,
               np.ndarray[np.float64_t, ndim=1] class_weight,
               double epsilon):

    cdef parameter *param
    cdef problem *problem
    cdef model *model
    cdef char_const_ptr error_msg

    # initialize problem
    problem = set_problem((<np.ndarray>X).data,
                          X.dtype == np.float64,
                          (<np.int32_t>X.shape[0]),
                          (<np.int32_t>X.shape[1]),
                          (<np.int32_t>np.count_nonzero(X)),
                          bias,
                          Y.data)

    # get class labels
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.intc)

    # initialize parameters
    param = set_parameter(solver_type,
                          eps,
                          C,
                          class_weight.shape[0],
                          class_weight_label.data,
                          class_weight.data,
                          epsilon)

    # error checking
    error_msg = check_parameter(problem, param)
    if error_msg:
        free_problem(problem)
        free_parameter(param)
        raise ValueError(error_msg)

    # early return
    with nogil:
        model = train(problem, param)

    ### FREE
    free_problem(problem)
    free_parameter(param)

    # get alpha coefficients
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
