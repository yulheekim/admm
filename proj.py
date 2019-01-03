import numpy
def proj(x):
# % x     = input vector
# % bound = 2x1 vector
# %
# % Example: out = proj_bound(x, [1,3]);
# % projects a vector x onto the interval [1,3]
# % by setting x(x>3) = 3, and x(x<1) = 1
# %
# % 2016-07-24 Stanley Chan

    bound = [0,1]
    over_ceiling_idx = x > bound[1]
    x[over_ceiling_idx] = 1
    under_floor_idx = x < bound[0]
    x[under_floor_idx] = 0
    return x
