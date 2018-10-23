from itertools import combinations
from time import time

from hmmmr.batched_functions import *
from hmmmr.common_libs import *
from hmmmr.utils.math import ncr

FLOAT_PRECISSION = np.float64
FLOAT_PRECISSION_SIZE = FLOAT_PRECISSION(1.0).nbytes

def get_column_index_combinations(X, n=3):
    """
    Generates a list of possible predictor combinations, note the last column will be included since it  is the constant var
    :param X: Matrix with predictors data
    :param n: Number of predictors to include
    :return: List of combinations, each combination is of n+1 size since it aggregates the last column
    """
    columns_index = range(X.shape[1])
    combs = combinations(columns_index, n)
    for c in combs:
        yield c


def get_X_matrices_from_combinations(X, index_combinations):
    """

    :param X: Matrix with predictors data. Last column must be Ones
    :param index_combinations: Combinations of columns to test in the model
    :return: Xs = Vector of matrices with combinations of columns for each model
    """
    # Need to iterate over index_combinations to give the right order (shape)
    return np.array([X[:, c] for c in index_combinations], dtype=X.dtype)



def get_Xt_matrices_from_combinations(Xt, index_combinations):
    """
    Same as get_X_matrices_from_combinations but used for the transpost
    :param Xt:
    :param index_combinations:
    :return:
    """
    return np.array([Xt[c,:] for c in index_combinations], dtype=Xt.dtype)


def get_Ys_matrices(Y, n):
    """
    :param Y: Vector with observed data
    :param n: Number of repetitions
    :return: Ys: Y repeated n times
    """
    Y_shape = Y.shape
    Ys_shape = (n, Y.shape[0], 1)
    return np.array([Y]*n, dtype=Y.dtype).reshape(Ys_shape)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def numpy_regression(X, comb, Y):
    X1 = X[:, comb]
    X1t = X1.T
    dot1 = np.dot(X1t, X1)
    invThing = np.linalg.inv(dot1)
    del (dot1)
    dot2 = np.dot(X1t, Y)
    B = np.dot(invThing, dot2)
    del (dot2)
    del (invThing)
    Ysim = np.dot(X1, B)
    metric = rmse(Y, Ysim)
    return {
        'metric': metric,
        'beta_coefficients': B,
        'ys_sim': Ysim
    }


def find_best_models_cpu(file_name='../TestData/Y=2X1+3X2+4X3+5_with_shitty.csv', min_predictors=1, max_predictors=4, handle=None, **kwargs):
    """

    :param file_name: File name containing data, the format is the following
        Columns: Contains data for N-2 predictors, 1 column full of 1s and 1 column with outcome data
            Columns 1 to N-2 contains predictors data
            The N-1 column is always full of 1s (due the constant on the model)
            The N column contains Y data
        Rows:  The first row contains the name of the predictor
            The next rows contains the observations (They need to be real values, no empty/nans are allowed
    :param max_predictors: Max numbers of predictors to test in the regression. Should b N-2 at max
    :return: Ordered array (by RMSE) of tuples containing (predictors_combination, RMSE)
    """
    handle = handle if handle else cublas.cublasCreate()
    XY = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1, dtype=np.float32)
    X = np.delete(XY, XY.shape[1] - 1, 1)
    Y = XY[:, -1]
    combs_rmse = None
    done_regressions = 0
    invalid_regressions = 0
    with open(file_name, 'rb') as f:
        col_names = np.array(f.readline().strip().split(','))
    for n_predictors in range(min_predictors, max_predictors):
        index_combinations = get_column_index_combinations(X, n_predictors) # n predictors - 1 constant
        s_i = ncr(X.shape[1], n_predictors)  # Number of possible combinations
        print "Doing regressions for {} predictors ({} regressions".format(n_predictors, s_i)
        for comb in index_combinations:
            try:
                regression = numpy_regression(X, comb, Y)
                combinations_cols_names = np.array([col_names[x] for x in comb])
                result = np.array(combinations_cols_names, regression['metric'])
                if combs_rmse is None:
                    combs_rmse = np.array(list(result))
                else:
                    combs_rmse = np.concatenate(combs_rmse, result)
                i += 1
            except:
                invalid_regressions += 1
        done_regressions += s_i
    print "{} Regressions has been done, {} invalid".format(done_regressions, invalid_regressions)
    ordered_combs = combs_rmse[combs_rmse[:, 1].argsort()]
    return combs_rmse


"""
start_time = time()
ordered_combs = find_best_models(file_name="/tmp/pronos_ordered_cleaned.csv", max_predictors=10)
print "Using numpy to do regressions took {}".format(time() - start_time)




            regression_results = {}
            regression_results['predictors_combinations'] = np.array(index_combinations[i:end_combination], dtype=np.int32)
            # If the matrix had not inverse then the model is invalid
            invalid_models = np.where(regression_results['inv_results'].get() != 0)[0]
            print "For this batch {} models are invalid".format(len(invalid_models))
            # Cleaning invalid model results
            regression_results['predictors_combinations'] = np.delete(regression_results['predictors_combinations'], invalid_models, 0)
            regression_results['beta_coefficients'] = np.delete(regression_results['beta_coefficients'], invalid_models, 0)
            regression_results['rmse'] = np.delete(regression_results['rmse'], invalid_models, 0)
            regression_results['ys_sim'] = np.delete(regression_results['ys_sim'], invalid_models, 0)
            regression_results['ys_obs'] = np.delete(regression_results['ys_obs'], invalid_models, 0)
            combinations_cols_names = np.array([col_names[x] for x in regression_results['predictors_combinations']])
            if combs_rmse is None:
                combs_rmse = np.array(list(zip(combinations_cols_names, regression_results['rmse'])))
            else:
                combs_rmse = np.concatenate((combs_rmse, np.array(list(zip(combinations_cols_names, regression_results['rmse'])))))
"""
