import operator as op

# Calculate number of possible combinations
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
    return numer//denom


def ncr_sum(n, min_r,max_r):
    return sum([ncr(n, r) for r in range(min_r, max_r) ])
