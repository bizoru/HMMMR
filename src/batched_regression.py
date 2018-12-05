from itertools import combinations
from time import time

from hmmmr.batched_functions import *
from hmmmr.common_libs import *
from hmmmr.config import FLOAT_PRECISSION, FLOAT_PRECISSION_SIZE
from hmmmr.utils.math import ncr, get_column_index_combinations
from hmmmr.utils.data_storage import load_x_y_from_csv


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

def _print_memory_usage(text="INFO"):
    mem_info = driver.mem_get_info()
    mem_used = mem_info[1] - mem_info[0]
    b_to_mb = 1024*1024.0
    # sys.stdout.write("{} \n Free memory {}MB, Used Memory {}MB, Total Memory {}MB".format(text, mem_info[0]/b_to_mb, mem_used/b_to_mb, mem_info[1]/b_to_mb))

def rmse_metric(YsObs_gpu, YsSim_gpu, N_data, handle=None):
    handle = handle if handle else cublas.cublasCreate()
    # Evaluation (RMSE)
    # Differences computation (Saxpy alpha=-1)
    diffs_gpu = vector_addition_pycuda(YsObs_gpu, YsSim_gpu, handle=handle, alpha=np.float32(-1))
    # Differences * Differences (same as ^2 )
    diffs2_gpu = massive_pow_square(diffs_gpu)
    # Sums diif^2 for each regression, if N regressions then N sums
    # Calculate the sums
    diifs2_sum_gpu = massive_vector_sums(diffs2_gpu, handle=handle)['Cs_gpu']
    del(diffs2_gpu)
    del(diffs_gpu)
    # diffs2_sum/N
    diifs2_over_n_gpu = vector_scalar_division(diifs2_sum_gpu, N_data)
    del(diifs2_sum_gpu)
    # sqrt
    rmse_gpu = vector_sqrt(diifs2_over_n_gpu)
    return rmse_gpu



def massive_multilineal_regresion(Xs, XTs, Ys, handle=None, only_metric=True):
    """
    Calculates a linear regression using Nvidia computing power!
        n = number of models
        d = number of observations
        c = number of predictors (columns)
    :param Xs: Shape (n, d, c) =>

    :param XTs: Shape (n, c, d)
    :param Ys: (n, d, 1)
    :param handle: Cublas handle (created if not provided)
    :return: Dict with regressions data
        # Note is possible some models at this point are invalid
     {
        'beta_coefficients': Vector of vectors with beta_coefficients for each model
        'ys_obs': Y observed data (repeated n times)
        'ys_sim': Y simulated data for each model
        'rmse': Vector with RMSE for each model
        'inv_results': Vector with the result of inverse operation, if not 0 the model is not valid
     }
    """
    handle = handle if handle else cublas.cublasCreate()

    Xs = np.array(Xs, FLOAT_PRECISSION)
    XTs = np.array(XTs, FLOAT_PRECISSION)
    YsObs = np.array(Ys, FLOAT_PRECISSION)

    Xs_gpu = gpuarray.to_gpu(Xs.astype(Xs.dtype))
    XTs_gpu = gpuarray.to_gpu(XTs.astype(XTs.dtype))
    YsObs_gpu = gpuarray.to_gpu(YsObs.astype(YsObs.dtype))
    _print_memory_usage("Once loaded Xs Xts YsObs")
    N_data = Xs.shape[1]

    # Regression
    XTsXs_gpu = massive_product_row_major(handle, XTs_gpu, Xs_gpu)['Cs_gpu']
    XTsYs_gpu = massive_product_row_major(handle, XTs_gpu, YsObs_gpu)['Cs_gpu']
    _print_memory_usage("Once Calculated XsYs XTsXs")
    del (XTs_gpu)

    # This is where the regression can fail
    inv_results = massive_inverse_pycuda(handle, XTsXs_gpu)
    XTsXsInv_gpu = inv_results['Xinvs_gpu']
    inv_returns = inv_results['return_codes']
    _print_memory_usage("Once Calculated Inverse")
    del (XTsXs_gpu)
    Bs_gpu = massive_product_row_major(handle, XTsXsInv_gpu, XTsYs_gpu)['Cs_gpu']
    del XTsXsInv_gpu
    _print_memory_usage("Once Calculated Bs")
    del (XTsYs_gpu)
    # Simulations
    YsSim_gpu = massive_product_row_major(handle, Xs_gpu, Bs_gpu)['Cs_gpu']
    _print_memory_usage("Once Calculated Ysim")
    del (Xs_gpu)
    _print_memory_usage("Before RMSE")
    rmse_gpu = rmse_metric(YsObs_gpu, YsSim_gpu, N_data, handle)
    _print_memory_usage("After RMSE")

    del(Xs)
    del(XTs)
    del(YsObs)

    results = {'rmse': rmse_gpu.get().flatten(), 'inv_results': inv_returns}

    if not only_metric:
        results.update({'beta_coefficients': Bs_gpu.get(), 'ys_obs': YsObs_gpu.get(),
                       'ys_sim': YsSim_gpu.get()})
    return results


def _get_max_batch_size(cols, n_data):
    """
    Calculates the max batch size for this regression given the available memory on the device
    :param cols: Number of cols (predictors)
    :param n_data: Number of observations
    :return: batchSize
    """
    # Take in account the intermediate data structures used, also the precission

    # 2 times X (X and X transpose) + n_data (Y) + XTsXs_gpu (Square cols x cols) + XTsYs (cols x 1) + Invs ( cols, cols) + n (metric)
    # c = cols
    # n = data
    # 2nc + n + c**2 + c + c**2 + n = 2c**2 + 2nc + 2n
    # 90% of free memory in the card
    estimated_scalars_per_regression = 2*(cols**2) + 2*(cols*n_data) + 2*n_data
    free_memory = driver.mem_get_info()[0]
    required_bytes_per_regression = FLOAT_PRECISSION_SIZE * estimated_scalars_per_regression
    # Closest thousand
    max_batch = (int(free_memory/required_bytes_per_regression)/1000)*1000
    return max_batch


def find_best_models_gpu(file_name='../TestData/Y=2X1+3X2+4X3+5_with_shitty.csv', min_predictors=1, max_predictors=4,
                         metric=None,  window=None, handle=None, max_batch_size=None, add_constant=True, **kwargs):
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
    tt = te = 0 # total time
    handle = handle if handle else cublas.cublasCreate()

    X, Y, col_names = load_x_y_from_csv(file_name, delimiter=",", skiprows=1, dtype=FLOAT_PRECISSION)
    done_regressions = 0
    combs_rmse = None
    for n_predictors in range(min_predictors, max_predictors+1):
        _print_memory_usage("Initial State: ")
        max_batch_size = _get_max_batch_size(n_predictors+1, Y.size)
        index_combinations = get_column_index_combinations(X, n_predictors, max_batch_size=max_batch_size, add_constant=add_constant)
        s_i = ncr(X.shape[1]-1, n_predictors) # Number of possible combinations
        i = 0
        for current_combinations in index_combinations:
            sys.stdout.write("Processing from {} to {} regressions in this batch\n".format(i, i + len(current_combinations)))
            ss = time()
            Xs = get_X_matrices_from_combinations(X, current_combinations)
            XTs = get_Xt_matrices_from_combinations(X.T, current_combinations)
            YsObs = get_Ys_matrices(Y, len(current_combinations))
            te += time() - ss
            ss = time()
            regression_results = massive_multilineal_regresion(Xs, XTs, YsObs, handle=handle)
            tt += time() - ss
            regression_results['predictors_combinations'] = np.array(current_combinations, dtype=np.int32)
            # If the matrix had not inverse then the model is invalid
            invalid_models = np.where(regression_results['inv_results'].get() != 0)[0]
            sys.stdout.write("For this batch {} models are invalid".format(len(invalid_models)))
            # Cleaning invalid model results
            regression_results['predictors_combinations'] = np.delete(regression_results['predictors_combinations'], invalid_models, 0)
            # regression_results['beta_coefficients'] = np.delete(regression_results['beta_coefficients'], invalid_models, 0)
            regression_results['rmse'] = np.delete(regression_results['rmse'], invalid_models, 0)
            # regression_results['ys_sim'] = np.delete(regression_results['ys_sim'], invalid_models, 0)
            # regression_results['ys_obs'] = np.delete(regression_results['ys_obs'], invalid_models, 0)
            combinations_cols_names = np.array([col_names[x] for x in regression_results['predictors_combinations']])
            if combs_rmse is None:
               combs_rmse = np.array(list(zip(combinations_cols_names, regression_results['rmse'])))
            else:
               combs_rmse = np.vstack((combs_rmse, np.array(list(zip(combinations_cols_names, regression_results['rmse'])))))
            i += len(current_combinations)
            done_regressions += len(current_combinations)
    sys.stdout.write("{} Regressions has been done, tt {}, te: {}\n".format(done_regressions, tt, te))
    ordered_combs = combs_rmse[combs_rmse[:, 1].argsort()]
    return ordered_combs

