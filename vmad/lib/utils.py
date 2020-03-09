import numpy as np
from vmad import operator

def finite_diff(param, func, epsilon, args=None, mode='forward'):
    """
    Find the finite differencing of a  given function based off of an input parameter
    Params:
    __________________
    param:   parameter to difference with respect to
    func:    function for finite difference
    epsilon: amount of forward stepping in differencing
    args:    further arguments, if any, for the function

    Returns:
    _________________
    forward differencing solution to function with respect to input parameter

    """
    if mode=='forward':
        k1, k2, k3=1, 0, 1

    elif mode=='backward':
        k1, k2, k3=-1, 0, -1

    elif mode=='central':
        k1, k2, k3=1/2, -1/2, 1

    return k3*(func(param+k1*epsilon) - func(param+k2*epsilon))/epsilon


@operator
class finite_difference:
    ain = {'param': '*'}
    aout = {'diff':'*'}

    def apl(node, param, func, epsilon, mode='central'):
        f = func(param)
        return dict(diff=f)

    def vjp(node, _diff, param, func, epsilon, mode='central'):
        delta = finite_diff(param, lambda x: func(x), epsilon, mode=mode)
        return dict(_param=np.dot(delta, _diff))

    def jvp(node, param_, param, func, epsilon, mode='central'):
        delta = finite_diff(param, lambda x: func(x), epsilon, mode=mode)
        return dict(diff_=np.dot(delta, param_))


