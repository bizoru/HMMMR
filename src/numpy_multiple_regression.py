from hmmmr.common_libs import *
from hmmmr.config import FLOAT_PRECISSION
from hmmmr.utils.math import ncr, ncr_sum, get_column_index_combinations
from hmmmr.utils.data_storage import load_x_y_from_csv



def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def get_X_Xt_matrix(X, comb):
    X1 = X[:, comb]
    X1t = X1.T
    return X1, X1t

def numpy_regression(X1, X1t, Y):
    dot1 = np.dot(X1t, X1)
    invThing = np.linalg.inv(dot1)
    dot2 = np.dot(X1t, Y)
    B = np.dot(invThing, dot2)
    Ysim = np.dot(X1, B)
    metric = rmse(Y, Ysim)
    return {
        'metric': metric,
        'beta_coefficients': B,
        'ys_sim': Ysim
    }

def find_best_models_cpu(file_name='../TestData/Y=2X1+3X2+4X3+5_with_shitty.csv', min_predictors=1, max_predictors=4,
                         add_constant=True, **kwargs):
    """

    :param file_name: File name containing data, the format is the following
        Columns: Contains data for N-1 predictors, 1 column with outcome data
            Columns 1 to N-1 contains predictors data
            The N column contains Y data
        Rows:  The first row contains the name of the predictor
            The next rows contains the observations (They need to be real values, no empty/nans are allowed
    :param max_predictors: Max numbers of predictors to test in the regression. Should b N-1 at max
    :return: Ordered array (by RMSE) of tuples containing (predictors_combination, RMSE)
    """
    X, Y = load_x_y_from_csv(file_name, delimiter=",", skiprows=1, dtype=FLOAT_PRECISSION)
    combs_rmse = None
    done_regressions = 0
    invalid_regressions = 0
    with open(file_name, 'rb') as f:
        col_names = np.array(f.readline().strip().split(','))
    total_regressions = ncr_sum(X.shape[1]-1, min_predictors, max_predictors+1)
    sys.stdout.write("{} regressions will be performed\n".format(total_regressions))
    combs_rmse = np.ndarray((total_regressions, 2), dtype=np.dtype('O'))
    regression_idx = 0
    for n_predictors in range(min_predictors, max_predictors+1):
        index_combinations = get_column_index_combinations(X, n_predictors, max_batch_size=1, add_constant=add_constant)
        s_i = ncr(X.shape[1]-1, n_predictors)  # Number of possible combinations
        sys.stdout.write("Doing regressions for {} predictors ({}) regressions\n".format(n_predictors, s_i))
        for comb in index_combinations:
            try:
                X1, X1t = get_X_Xt_matrix(X, comb)
                regression = numpy_regression(X1, X1t, Y)
                combinations_cols_names = np.array([col_names[x] for x in comb])
                result = np.array([[combinations_cols_names, regression['metric']]])
                combs_rmse[regression_idx] = result
            except:
                invalid_regressions += 1
            regression_idx += 1
        done_regressions += s_i
    sys.stdout.write("{} Regressions has been done, {} invalid\n".format(done_regressions, invalid_regressions))
    ordered_combs = combs_rmse[combs_rmse[:, 1].argsort()]
    return ordered_combs
