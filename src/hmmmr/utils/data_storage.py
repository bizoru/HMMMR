import numpy as np
from copy import copy

def load_x_y_from_csv(filename, delimiter=",", skiprows=1, dtype=np.float32):
    XY = np.loadtxt(open(filename, "rb"), delimiter=delimiter, skiprows=skiprows, dtype=dtype)
    Y = copy(XY[:, -1])
    # Constant is always injected
    XY[:,XY.shape[1]-1] = np.ones(XY.shape[0])
    X = XY

    col_names = []
    with open(filename, 'rb') as f:
        cols_in_file = f.readline().strip().split(',')
        col_names = cols_in_file[:]
        col_names.append(cols_in_file[len(cols_in_file)-1])
        col_names[len(cols_in_file)-1] = "constant"

    return X, Y, np.array(col_names)
