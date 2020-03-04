import numpy as np
from vmad import operator

def forward_difference(param, func, epsilon, args=None):
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
    if args is None:
        return (func(param+epsilon) - func(param))/epsilon
    else:
        return (func(param+epsilon, *args) - func(param, *args))/epsilon



@operator
class finite_difference:
    ain = {'param': '*'}
    aout = {'diff':'*'}

    def apl(node, param, func, epsilon, args=None):
        delta = forward_difference(param, func, epsilon, args)
        return dict(diff = delta)

    def vjp(node, _diff, param, func, epsilon, args=None):
        delta = forward_difference(param, func, epsilon, args)
        return dict(_param = np.dot(delta, _diff))

    def jvp(node, param_, param, func, epsilon, args=None):
        delta = forward_difference(param, func, epsilon, args)
        return dict(diff_ = np.dot(delta, param_))
