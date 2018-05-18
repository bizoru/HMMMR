from common_libs import *

# NON BATCHED FUNCTIONS
# This should be separated since we can reuse the transpose for other operations
"""
This is not part of BLAS routines but it's included in cuda (geam), also there is a batch equivalent for this function
https://stackoverflow.com/questions/15458552/what-is-the-most-efficient-way-to-transpose-a-matrix-in-cuda
http://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.cublas.cublasSgeam.html#skcuda.cublas.cublasSgeam
"""
def get_single_transpose_function(a_gpu):
    if (a_gpu.dtype == np.complex64):
        func = cublas.cublasCgeam
    elif (a_gpu.dtype == np.float32):
        func = cublas.cublasSgeam
    elif (a_gpu.dtype == np.complex128):
        func = cublas.cublasZgeam
    elif (a_gpu.dtype == np.float64):
        func = cublas.cublasDgeam
    else:
        raise ValueError('unsupported input type')
    return func

"""
BLAS http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html#gaeda3cbd99c8fb834a60a6412878226e1
CUBLAS https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
"""
def get_single_dot_function(a_gpu):
    if (a_gpu.dtype == np.complex64):
        func = cublas.cublasCgemm
    elif (a_gpu.dtype == np.float32):
        func = cublas.cublasSgemm
    elif (a_gpu.dtype == np.complex128):
        func = cublas.cublasZgemm
    elif (a_gpu.dtype == np.float64):
        func = cublas.cublasDgemm
    else:
        raise ValueError('unsupported input type')
    return func


def get_single_inverse_function(a_gpu):
    data_dtype = a_gpu.dtype.type
    if (data_dtype == np.complex64):
        getrf = cusolver.cusolverDnCgetrf
        bufsize = cusolver.cusolverDnCgetrf_bufferSize
        getrs = cusolver.cusolverDnCgetrs
    elif (data_dtype == np.float32):
        getrf = cusolver.cusolverDnSgetrf
        bufsize = cusolver.cusolverDnSgetrf_bufferSize
        getrs = cusolver.cusolverDnSgetrs
    elif (data_dtype == np.complex128):
        getrf = cusolver.cusolverDnZgetrf
        bufsize = cusolver.cusolverDnZgetrf_bufferSize
        getrs = cusolver.cusolverDnZgetrs
    elif (data_dtype == np.float64):
        getrf = cusolver.cusolverDnDgetrf
        bufsize = cusolver.cusolverDnDgetrf_bufferSize
        getrs = cusolver.cusolverDnDgetrs
    # Cula is not available, check:
    return (getrf, bufsize, getrs)


# BATCHED FUNCTIONS
""" 
TO IMPLEMENT

Some docs on this:
Batched Linear Algebra Problems on GPU
Accelerators
    http://trace.tennessee.edu/cgi/viewcontent.cgi?article=5031&context=utk_graddiss

"""
#

def get_batched_dot_function(a_gpu):
    if (a_gpu.dtype == np.complex64):
        func = cublas.cublasCgemmBatched
    elif (a_gpu.dtype == np.float32):
        func = cublas.cublasSgemmBatched
    elif (a_gpu.dtype == np.complex128):
        func = cublas.cublasZgemmBatched
    elif (a_gpu.dtype == np.float64):
        func = cublas.cublasDgemmBatched
    else:
        raise ValueError('unsupported input type')
    return func

def get_batched_transpose_function(a_gpu):
    return None

def get_batched_inverse_functions(a_gpu):
    # https://programtalk.com/python-examples/pycuda.gpuarray.to_gpu/
    if (a_gpu.dtype == np.float32):
        getrf = cublas.cublasSgetrfBatched
        getri = cublas.cublasSgetriBatched
    elif (a_gpu.dtype == np.float64):
        getrf = cublas.cublasDgetrfBatched
        getri = cublas.cublasDgetriBatched
    else:
        raise ValueError('unsupported input type')
    return (getrf, getri)

def get_vector_addition(a_gpu):
    if (a_gpu.dtype == np.complex64):
        func = cublas.cublasCaxpy
    elif (a_gpu.dtype == np.float32 or a_gpu.dtype == np.int32 ):
        func = cublas.cublasSaxpy
    elif (a_gpu.dtype == np.complex128):
        func = cublas.cublasZaxpy
    elif (a_gpu.dtype == np.float64):
        func = cublas.cublasDaxpy
    else:
        raise ValueError('unsupported input type')
    return func


def get_sqrt_function(a_gpu):
        return cumath.sqrt
def bptrs(a):
    """
    Pointer array when input represents a batch of matrices or vectors.

    taken from scikits.cuda tests/test_cublas.py
    """

    return pycuda.gpuarray.arange(a.ptr, a.ptr + a.shape[0] * a.strides[0], a.strides[0],
                                  dtype=cublas.ctypes.c_void_p)
