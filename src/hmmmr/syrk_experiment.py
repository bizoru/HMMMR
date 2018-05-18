from pycuda import gpuarray
from scikits.cuda import cublas

"""
syrk does multiplication by the transpose, why not to use:
1) Does not exist a batched function for this
2) We need to keep the transpose in memory in order to do other operations
3) Fills just the upper or lower triangles
However, some docs on it:
https://stackoverflow.com/questions/34091737/cuda-gemm-transpose-with-numpy
http://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.cublas.cublasSsyrk.html#skcuda.cublas.cublasSsyrk
https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-syrk
Code below from https://www.kinetica.com/docs/_downloads/udf_cublas_py_proc.py
"""
def cublas_matrix_transpose_product_syrk(h,A):
   A_gpu = gpuarray.to_gpu(A)
   C_gpu = gpuarray.zeros((A.shape[0], A.shape[0]), np.float32)
   cublas.cublasSsyrk(h, 'u', 't', A.shape[0], A.shape[1], 1.0, A_gpu.gpudata, A.shape[1], 0.0, C_gpu.gpudata, A.shape[0])
   return C_gpu.get()
