from pycuda import cumath, driver, gpuarray, tools
from pycuda.elementwise import ElementwiseKernel
import numpy as np
from scikits.cuda import cublas
from skcuda import linalg
import pycuda.autoinit
