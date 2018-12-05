import numpy as np

def load_x_y_from_csv(filename, delimiter=",", skiprows=1, dtype=np.float32):
    XY = np.loadtxt(open(filename, "rb"), delimiter=delimiter, skiprows=skiprows, dtype=dtype)
    Y = XY[:, -1]
    # Constant is always injected
    XY[:,XY.shape[1]-1] = np.ones(XY.shape[0])
    X = XY
    return X, Y