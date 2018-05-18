from pycuda import gpuarray
import numpy as np
from scikits.cuda import cublas


from hmmmr.non_batched_functions import *

"""
Regression explanation
https://onlinecourses.science.psu.edu/stat501/node/382
"""
def calculate_single_regression(handle, XY):
    """
    :param handle: Cublas handle
    :param XY: XY matrix, where the last column are the Y values, the columns [:-2] are predictors values and the columns[-2] is filled of 1s to cover the constant
            XY is in CPU memory, this function transfers to GPU
    :return: Vector Beta of coefficients (gpu pointer)
    """
    Y = XY[:, -1]
    Y = Y.reshape((Y.shape[0], 1))
    X = np.delete(XY, XY.shape[1] - 1, 1)
    x_gpu = gpuarray.to_gpu(X)
    y_gpu = gpuarray.to_gpu(Y)
    b_coefficients_gpu = calculate_regression_coeffs_non_batched(handle, x_gpu, y_gpu)
    y_calculated_gpu = calculate_predictions_from_model_non_batched(handle, x_gpu, b_coefficients_gpu)
    return {'x_gpu': x_gpu, 'y_gpu':y_gpu, 'y_calculated_gpu':y_calculated_gpu, 'b_coeffs_gpu': b_coefficients_gpu}


def test_with_test_data():
    handle = cublas.cublasCreate()
    # 3 vars model + constant
    XY = np.loadtxt(open("TestData/Y=2X1+3X2+4X3+5_valid_predictors.csv", "rb"), delimiter=",", skiprows=1, dtype=np.float32)
    print "3 vars model, B coefficients"
    print calculate_single_regression(handle, XY)

    # 2 vars model
    XY = np.loadtxt(open("TestData/Y=2X1+3X2+5.csv", "rb"), delimiter=",", skiprows=1, dtype=np.float32)
    print "2 vars model, B coefficients"
    print calculate_single_regression(handle, XY)


if __name__ == "__main__":
    test_with_test_data()