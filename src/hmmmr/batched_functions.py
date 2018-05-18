from common_libs import *
from cublas_functions import *

"""

Factoring a matrix with LU factorization

1 Step GETRF 

M =  LU
U = Upper Matrix (Including diagonal)
L = Lower Matrix (Diagonal with 1s)
Returns the pivot vector too and the LU matrix

2 Step 
"""
def massive_inverse_pycuda(handle, As_gpu):
    """
    Uses LU factorization and pycublas batched subroutines to calculate matrix inversions
    :param handle: Cublas handle
    :param As_gpu: Xs matrices in numpy
    :return: {
        'Xinvs_ptr': Pointer to array of inverses
        'Xinvs_gpu': Xinvs array in GPUs
        'return_codes': Array with return codes of the inverse results of each matrix
    }
    """
    # print "Starting pycuda inversion"
    n = As_gpu.shape[1] # Matrices should be nxn, number of cols
    batchSize = As_gpu.shape[0] # Number of matriecs
    # start = default_timer()
    # print "Total time to create cublas handler was {}s".format(default_timer()-start)
    # start = default_timer()
    Xs_ptr = bptrs(As_gpu)  # This is the only one that needs to be passed as the pointes
    info_rf = pycuda.gpuarray.zeros(batchSize, np.int32)  # Will contain the return code for each matrx LU factorization
    Ps_gpu = pycuda.gpuarray.empty((batchSize, n), np.int32)  # Stores the pivots from LU factorization
    Xinvs_gpu = pycuda.gpuarray.empty_like(As_gpu)  # Storage for inverted matrices
    Xinvs_ptr = bptrs(Xinvs_gpu)
    #  print "Total time to copy Xs/Create, Ps, Xinvs_gpu, and info arrrays to cpu was {} s".format(default_timer()-start)
    # start = default_timer()
    (getrf, getri) = get_batched_inverse_functions(As_gpu)
    getrf(handle, n, Xs_ptr.gpudata, n, Ps_gpu.gpudata, info_rf.gpudata, batchSize) # LU Factorization
    # print "Total time for LU factorization {}s. \n" \
    #       "Got this returns codes for LU factorization {}".format(default_timer()-start, info.get())
    # start = default_timer()
    # use factorization to perform inversion
    info_ri = pycuda.gpuarray.zeros(batchSize, np.int32)
    getri(handle, n, Xs_ptr.gpudata, n, Ps_gpu.gpudata, Xinvs_ptr.gpudata,
                               n, info_ri.gpudata, batchSize)
    # print "Total time was getRi: {}s \n " \
    #      "Got this returns codes for getRi batched {}".format(default_timer()-start, info.get())\
    # I will sum return codes in order to know which code results were > 0 for any of the cases
    resumed_info = vector_addition_pycuda(info_ri, info_rf, overwriteAs=False, handle=handle)
    return {
        'Xinvs_ptr': Xinvs_ptr,
        'Xinvs_gpu': Xinvs_gpu,
        'return_codes': resumed_info
    }


def massive_inverse(As, handle=None):
    """
    :param As: A list of square matrices (on cpu, numpy data structure)
    :param handle: Pycuda handle
    :return: Inverses of Xs matrices in a numpy structure
    """
    handle = handle if handle else cublas.cublasCreate()
    As_gpu = gpuarray.to_gpu(As.astype('float32'))
    pycuda_result = massive_inverse_pycuda(handle, As_gpu)
    return pycuda_result['Xinvs_gpu'].get()


def massive_product_row_major(handle, As_gpu, Bs_gpu):
    """
    Batched multiplication for matrices in row major order
    Since CUDA by default assume that matrices are given in column major order we need to trick the multiplication as
    explained here https://www.beechwood.eu/cublas-sgemm-dimensions-mishmash-in-row-major-order-environment/
    :param handle: Cublas handle
    :param As: A matrices
    :param Bs: B matrices
    :return: {
        'Cs_gpu': Array of matrices (Results of A*B)
        'Cs_ptr': Pointers to array of matrices
    }
    """
    assert As_gpu.shape[2] == Bs_gpu.shape[1] # Number of colums on As should be the same as number of rows on B
    assert As_gpu.shape[0] == Bs_gpu.shape[0] # Same number of matrices to multiply
    assert As_gpu.dtype == Bs_gpu.dtype
    multiply_function = get_batched_dot_function(As_gpu)
    transa = 'n'
    transb = 'n'
    As_ptr = bptrs(As_gpu)
    Bs_ptr = bptrs(Bs_gpu)
    m = Bs_gpu.shape[2] # Numbers of rows of As matrices
    n = As_gpu.shape[1] # Numbers of columns of Bs matrices
    k = As_gpu.shape[2] # Number of columns of As and number of rows of Bs
    alpha = np.float32(1.0).astype(As_gpu.dtype)
    lda = max(1, m)
    ldb = max(1, k)
    beta = np.float32(0.0).astype(As_gpu.dtype)
    batchSize = As_gpu.shape[0] # Number of matrices to multiply
    Cs_shape = (batchSize, As_gpu.shape[1], Bs_gpu.shape[2])
    Cs_gpu = gpuarray.zeros(Cs_shape, As_gpu.dtype)
    Cs_ptr = bptrs(Cs_gpu)
    ldc = max(1, m)
    multiply_function(handle, transa, transb, m, n, k, alpha, Bs_ptr.gpudata, lda, As_ptr.gpudata, ldb, beta, Cs_ptr.gpudata, ldc, batchSize)
    return {
        'Cs_gpu': Cs_gpu,
        'Cs_ptr': Cs_ptr
    }


def massive_product_column_major(handle, As_gpu, Bs_gpu):
    assert As_gpu.shape[2] == Bs_gpu.shape[1] # Number of colums on As should be the same as number of rows on B
    assert As_gpu.shape[0] == Bs_gpu.shape[0] # Same number of matrices to multiply
    multiply_function = get_batched_dot_function(As_gpu)
    transa = 'n'
    transb = 'n'
    As_ptr = bptrs(As_gpu)
    Bs_ptr = bptrs(Bs_gpu)
    m = As_gpu.shape[1] # Numbers of rows of As matrices
    n = Bs_gpu.shape[2] # Numbers of columns of Bs matrices
    k = As_gpu.shape[2] # Number of columns of As and number of rows of Bs
    alpha = np.float32(1.0)
    lda = max(1, m)
    ldb = max(1, k)
    beta = np.float32(0.0)
    batchSize = As_gpu.shape[0] # Number of matrices to multiply
    Cs_shape = (batchSize, As_gpu.shape[1], Bs_gpu.shape[2],)
    Cs_gpu = gpuarray.zeros(Cs_shape, As_gpu.dtype)
    Cs_ptr = bptrs(Cs_gpu)
    ldc = max(1, m)
    cublas.cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, As_ptr.gpudata, lda, Bs_ptr.gpudata, ldb, beta, Cs_ptr.gpudata, ldc, batchSize)
    return {
        'Cs_gpu': Cs_gpu,
        'Cs_ptr': Cs_ptr
    }


def _get_massive_product_function(order='r'):
    """
    :param order: r for row major order, c for column major order.
    :return:
    """
    if order=='r':
        return massive_product_row_major
    elif order=='c':
        return massive_product_column_major
    else:
        raise Exception("Not valid order")

def massive_product(As, Bs, order='r', handle=None):
    handle = handle if handle else cublas.cublasCreate()
    cublas_massive_product = _get_massive_product_function(order)
    handle = handle if handle else cublas.cublasCreate()
    As_gpu = gpuarray.to_gpu(As.astype(As.dtype))
    Bs_gpu = gpuarray.to_gpu(Bs.astype(Bs.dtype))
    pycuda_result = cublas_massive_product(handle, As_gpu, Bs_gpu)
    return pycuda_result['Cs_gpu'].get()


def vector_addition_pycuda(As_gpu, Bs_gpu, overwriteAs=False, alpha=np.float32(1.0), handle=None):
    assert As_gpu.size == Bs_gpu.size
    handle = handle if handle else cublas.cublasCreate()
    alpha = alpha.astype(As_gpu.dtype)
    vector_addition = get_vector_addition(As_gpu)
    Bs_gpu = Bs_gpu.copy() if not overwriteAs else Bs_gpu
    vector_addition(handle, As_gpu.size, alpha,  As_gpu.gpudata, 1, Bs_gpu.gpudata, 1)
    return Bs_gpu

def massive_vector_sums(As_gpu, handle=None):
    """
    Uses matrix multiplication to compute the sum of each individual vector
    :param As_gpu: Array of m vectors of n elements (in reality matrices Nx1) matrices
    :param handle: pycuda handle
    :return: Array of matrices (1x1) containing the sum of each vector
    """
    handle = handle if handle else cublas.cublasCreate()
    ones_shape = (As_gpu.shape[0], As_gpu.shape[2], As_gpu.shape[1])
    ones = np.ones(ones_shape, As_gpu.dtype)
    ones_gpu = gpuarray.to_gpu(ones.astype(ones.dtype))
    return massive_product_row_major(handle, ones_gpu, As_gpu)



def vector_sqrt(As_gpu, overwrite=False):
    func = get_sqrt_function(As_gpu)
    if overwrite:
        func(As_gpu, out=As_gpu)
        return As_gpu
    else:
        return func(As_gpu)


# Taken from linalg.multiply
# http://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.linalg.multiply.html
def massive_multiply_one_to_one(x_gpu, y_gpu, handle=None, overWriteAs=False):
    """
    Multiply arguments element-wise.

    Parameters
    ----------
    x_gpu, y_gpu : pycuda.gpuarray.GPUArray
        Input arrays to be multiplied.
    handle : HAndle
    overwrite : bool (default: False)
        If true, return the result in `y_gpu`.
        is false, return the result in a newly allocated array.

    Returns
    -------
    z_gpu : pycuda.gpuarray.GPUArray
        The element-wise product of the input arrays.
    """
    handle = handle if handle else cublas.cublasCreate()
    if x_gpu.shape != y_gpu.shape:
        raise ValueError('input arrays must have the same shape')

    if x_gpu.dtype not in [np.float32, np.float64, np.complex64,
                           np.complex128]:
        raise ValueError('unrecognized type')

    x_ctype = tools.dtype_to_ctype(x_gpu.dtype)
    y_ctype = tools.dtype_to_ctype(y_gpu.dtype)

    if overWriteAs:
        func = ElementwiseKernel("{x_ctype} *x, {y_ctype} *y".format(x_ctype=x_ctype,
                                                                        y_ctype=y_ctype),
                                    "y[i] *= x[i]")
        func(x_gpu, y_gpu)
        return y_gpu
    else:
        result_type = np.result_type(x_gpu.dtype, y_gpu.dtype)
        # workaround for bug #131
        z_gpu = gpuarray.empty(tuple(int(i) for i in x_gpu.shape),
                               result_type)
        func = ElementwiseKernel("{x_ctype} *x, {y_ctype} *y, {z_type} *z".format(x_ctype=x_ctype,
                                                                                  y_ctype=y_ctype,
                                                                                  z_type=tools.dtype_to_ctype(
                                                                                      result_type)),
                                 "z[i] = x[i]*y[i]")
        func(x_gpu, y_gpu, z_gpu)
        return z_gpu


# Builded with Element Wise
def massive_pow_square(x_gpu, overWriteAs=False):
    """
    Calculates vector**2

    Parameters
    ----------
    x_gpu,  pycuda.gpuarray.GPUArray
        Input arrays to be multiplied.
    handle : HAndle
    overwrite : bool (default: False)
        If true, return the result in `x_gpu`.
        is false, return the result in a newly allocated array.

    Returns
    -------
    z_gpu : pycuda.gpuarray.GPUArray
        Result of x_gpu**2
    """
    if x_gpu.dtype not in [np.float32, np.float64, np.complex64,
                           np.complex128]:
        raise ValueError('unrecognized type')

    x_ctype = tools.dtype_to_ctype(x_gpu.dtype)

    if overWriteAs:
        func = ElementwiseKernel("{x_ctype} *x".format(x_ctype=x_ctype),
                                    "x[i] *= x[i]")
        func(x_gpu)
        return x_gpu
    else:
        result_type = np.result_type(x_gpu.dtype)
        # workaround for bug #131
        z_gpu = gpuarray.empty(tuple(int(i) for i in x_gpu.shape),result_type)
        func = ElementwiseKernel("{x_ctype} *x, {z_type} *z".
                                 format(x_ctype=x_ctype, z_type=tools.dtype_to_ctype(result_type)),
                                 "z[i] = x[i]*x[i]")
        func(x_gpu, z_gpu)
        return z_gpu


def scalar_vector_product(x_gpu, scalar, overwrite=False):
    """
    Calculates x_gpu*scalar

    Parameters
    ----------
    x_gpu,  pycuda.gpuarray.GPUArray
        Input arrays to be multiplied.
    scalar = scalar to multiply
    overwrite : bool (default: False)
        If true, return the result in `x_gpu`.
        is false, return the result in a newly allocated array.

    Returns
    -------
    z_gpu : pycuda.gpuarray.GPUArray
        Result of x_gpu**2
    """
    if x_gpu.dtype not in [np.float32, np.float64, np.complex64,
                           np.complex128]:
        raise ValueError('unrecognized type')

    x_ctype = tools.dtype_to_ctype(x_gpu.dtype)

    if overwrite:
        func = ElementwiseKernel("{x_ctype} *x".format(x_ctype=x_ctype),
                                    "x[i] *= {scalar}".format(scalar=str(scalar)))
        func(x_gpu)
        return x_gpu
    else:
        result_type = np.result_type(x_gpu.dtype)
        # workaround for bug #131
        z_gpu = gpuarray.empty(tuple(int(i) for i in x_gpu.shape),result_type)
        func = ElementwiseKernel("{x_ctype} *x, {z_type} *z".
                                 format(x_ctype=x_ctype, z_type=tools.dtype_to_ctype(result_type)),
                                 "z[i] = x[i]* {scalar}".format(scalar=str(scalar)))
        func(x_gpu, z_gpu)
        return z_gpu

def vector_scalar_division(x_gpu, scalar, overwrite=False):
    return scalar_vector_product(x_gpu, 1.0/(scalar+0.0), overwrite=overwrite)