from common_libs import *
from cublas_functions import *
linalg.init()

def cublas_calculate_transpose_non_batched(h, a_gpu):
    cublas_transpose = get_single_transpose_function(a_gpu)
    m, k = a_gpu.shape
    at_gpu = gpuarray.empty((k, m), a_gpu.dtype)
    k, n = at_gpu.shape
    # Calculate transpose
    transa = transb = 't'
    cublas_transpose(h, transa, transb, m, k, 1.0, a_gpu.gpudata, k, 0.0, a_gpu.gpudata, k, at_gpu.gpudata, m)
    return at_gpu

# Matrix product, there is a batch equivalent for this function too
# Make sure it has 2 dimensions (use reshape in the case is 1d)
def cublas_matrix_product_gemm_non_batched(handle, a_gpu, b_gpu):
    """
    :param handle:
    :param a_gpu: Be carefull to pass X here
    :param b_gpu: Xt should be here
    :return:
    """
    cublas_dot = get_single_dot_function(b_gpu)
    if len(a_gpu.shape)!=2 or len(a_gpu.shape)!=2:
        raise ValueError('Make sure the arrays are 2 dimensional')
    n, l = a_gpu.shape
    k, m = b_gpu.shape
    c_gpu = gpuarray.empty((n, m), b_gpu.dtype)
    lda = max(1, a_gpu.strides[0] // a_gpu.dtype.itemsize)
    ldb = max(1, b_gpu.strides[0] // b_gpu.dtype.itemsize)
    ldc = max(1, c_gpu.strides[0] // c_gpu.dtype.itemsize)
    alpha = np.float32(1.0)
    beta = np.float32(0.0)
    transa = transb = 'n'
    cublas_dot(handle, transb, transa, m, n, k, alpha, b_gpu.gpudata, ldb, a_gpu.gpudata, lda, beta, c_gpu.gpudata, ldc)
    return c_gpu

def cublas_matrix_product_gemm_batched(handle, as_gpu, bs_gpu):
    cublas_dot = get_batched_dot_function(as_gpu)
    if len(a_gpu.shape) != 2 or len(a_gpu.shape) != 2:
        raise ValueError('Make sure the arrays are 2 dimensional')
    # n, z, l
    n, l = as_gpu.shape
    k, m = bs_gpu.shape
    c_gpu = gpuarray.empty((n, m), b_gpu.dtype)
    lda = max(1, a_gpu.strides[0] // a_gpu.dtype.itemsize)
    ldb = max(1, b_gpu.strides[0] // b_gpu.dtype.itemsize)
    ldc = max(1, c_gpu.strides[0] // c_gpu.dtype.itemsize)
    alpha = np.float32(1.0)
    beta = np.float32(0.0)
    transa = transb = 'n'
    cublas_dot(handle, transb, transa, m, n, k, alpha, b_gpu.gpudata, ldb, a_gpu.gpudata, lda, beta, c_gpu.gpudata, ldc)
    return c_gpu


"TODO: Fix this function, like linalg.inv"
def cublas_single_matrix_inversion_non_batched(h, a_gpu, overwrite=False, ipiv_gpu=None):
    (cublas_getrf, bufsize, cublas_getrs) = get_single_inverse_function(a_gpu)
    data_type = a_gpu.dtype
    n = a_gpu.shape[0]
    if ipiv_gpu is None:
        ipiv_gpu = gpuarray.empty((n, 1), np.int32)
    try:
        in_gpu = a_gpu if overwrite else a_gpu.copy()
        Lwork = bufsize(h, n, n, in_gpu.gpudata, n)
        Work = gpuarray.empty(Lwork, data_type)
        devInfo = gpuarray.empty(1, np.int32)
        cublas_getrf(h, n, n, in_gpu.gpudata, n, Work.gpudata, ipiv_gpu.gpudata, devInfo.gpudata)
    except cusolver.CUSOLVER_ERROR as e:
        raise ValueError("Error while generating inverse of the matrix")

    d = devInfo.get()[0]
    if d != 0:
        raise ValueError("Singular matrix or wrong params")
    try:
        b_gpu = linalg.eye(n, data_type)
        cublas_getrs(h, cublas._CUBLAS_OP['n'], n, n,
              in_gpu.gpudata, n, ipiv_gpu.gpudata, b_gpu.gpudata, n,
              devInfo.gpudata)

        # Since CUSOLVER's getrs functions save their output in b_gpu, we
        # need to copy it back to the input matrix if overwrite is requested:
        if overwrite:
            a_gpu.set(b_gpu)
            return a_gpu
        else:
            return b_gpu
    except cusolver.CUSOLVER_ERROR as e:
        raise "Error with cusolver {}".format(e.message)
    return h

def calculate_regression_coeffs_non_batched(handle, x_gpu, y_gpu):
    xt_gpu = cublas_calculate_transpose_non_batched(handle, x_gpu)
    xtx_gpu = cublas_matrix_product_gemm_non_batched(handle, xt_gpu, x_gpu)
    xty_gpu = cublas_matrix_product_gemm_non_batched(handle, xt_gpu, y_gpu)
    # xtx_inv_gpu = cublas_single_matrix_inversion(handle, xtx_gpu)
    xtx_inv_gpu = linalg.inv(xtx_gpu, lib="cusolver")
    b_coefficients = cublas_matrix_product_gemm_non_batched(handle, xtx_inv_gpu, xty_gpu)
    return b_coefficients


def calculate_predictions_from_model_non_batched(handle, x_gpu, b_coefficients_gpu):
    return cublas_matrix_product_gemm_non_batched(handle, x_gpu, b_coefficients_gpu)