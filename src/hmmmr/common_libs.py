import numpy as np
import sys as sys

try:
    from pycuda import cumath, driver, gpuarray, tools
    from pycuda.elementwise import ElementwiseKernel
    from scikits.cuda import cublas
    import pycuda.autoinit
except Exception as e:
    sys.stderr.write("WARNING: Pycuda or cublas was not found on python path\n")
