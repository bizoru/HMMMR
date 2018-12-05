import operator as op
from itertools import combinations

# Calculate number of possible combinations
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
    return numer//denom


def ncr_sum(n, min_r,max_r):
    return sum([ncr(n, r) for r in range(min_r, max_r) ])


def get_column_index_combinations(X, n, max_batch_size=1, add_constant=True):
    """
    Generates a list of possible predictor combinations, note the last column will be included since it  is the constant var
    :param X: Matrix with predictors data
    :param n: Number of predictors to include
    :return: List of combinations, each combination is of n+1 size since it aggregates the last column
    """
    max_batch_size = int(max_batch_size)
    sys.stdout.write("Generating {} combs for this batch\n".format(max_batch_size))
    columns_index = range(X.shape[1])
    # Exclude the constant from the possible combinations , the constant will be in ALL OR NONE models
    iterator = combinations(len(columns_index) - 1, n)
    current_combs = []
    counter = 0
    for c in iterator:
        combination = list(c) + [columns_index[-1]] if add_constant else list(c)
        if max_batch_size == 1:
            yield combination
        else:
            current_combs.append(combination)
            counter += 1
            if counter % max_batch_size == 0:
                yield current_combs
                current_combs = []

    if current_combs:
         yield current_combs
