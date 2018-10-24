import numpy as np
import pyopencl as cl
import sys
import argparse
import os
from time import time

from batched_regression import _print_memory_usage, find_best_models_gpu
from numpy_multiple_regression import find_best_models_cpu

from hmmmr.utils.profiling import do_profile

import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Performs massive multilinear regresions from the combinatorics of predictors from 1 to MaxPredictors). Selects the best model base on some criteria (RMSE) by default")
    parser.add_argument('-i', dest="input_file",
                        help="CSV file with predictors data")
    parser.add_argument('-w', dest="window", default=300,
                        help="Window of data to perform regressions.  Recommended: 300 for hourly data, 50 for daily data")
    parser.add_argument('-mp', dest="max_predictors", help="Max numbers of predictors to combine", default=8)
    parser.add_argument('-np', dest="min_predictors", help="Min numbers of predictors to combine", default=1)
    parser.add_argument('-o', dest="output_file", default=None,
                        help="Output file with combinations and metrics results")
    parser.add_argument('-m', dest="metric", default="rmse",
                        help="Metric to be calculated for the possible model")
    parser.add_argument('-d', dest="device", default="gpu",
                        help="Device to use to perform calculations")
    parser.add_argument('-b', dest="max_batch_size", default=None,
                        help="Size of batch to process")

    args = parser.parse_args()

    input_file = args.input_file
    window = args.window
    max_predictors = args.max_predictors
    min_predictors = args.min_predictors
    max_batch_size = args.max_batch_size
    metric = args.metric
    device = args.device

    output_file = args.ouput_file if args.output_file else "/tmp/{}-w{}-mp{}-{}.csv".format(input_file.split("/")[-1], window, max_predictors, device)

    if any(x is None for x in [input_file, window, max_predictors, output_file, metric]):
        parser.print_help()
        sys.exit(0)

    return input_file, int(window), int(max_predictors), int(min_predictors), metric, output_file, device, max_batch_size


# _print_memory_usage("Initial State: ")
@do_profile(follow=[find_best_models_gpu, find_best_models_cpu])
def perform_regressions():
    start_time = time()
    input_file, window, max_predictors, min_predictors, metric, output_file, device, max_batch_size = parse_arguments()
    if device == "gpu":
        print "Running calculations on GPU"
        ordered_combs = find_best_models_gpu(file_name=input_file, min_predictors=min_predictors, max_predictors=max_predictors, metric=metric,  window=window, max_batch_size=max_batch_size)
        print "Using GPU to do regressions took {}".format(time() - start_time)
    elif device == "cpu":
        ordered_combs = find_best_models_cpu(file_name=input_file, min_predictors=min_predictors, max_predictors=max_predictors, metric=metric,  window=window, max_batch_size=max_batch_size)
    df = pd.DataFrame(ordered_combs)
    df.to_csv(output_file)


perform_regressions()
