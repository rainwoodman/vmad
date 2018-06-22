"""
    This module provides utils for debugging a model.

    These functions are triggered during model execution.
"""

from vmad.core.operator import operator

@operator
class eval:
    ain  = 'x'
    aout = 'y'

    def apl(self, x, function):
        """ Evaluates a function on the symbol x at run time.

            Parameters
            ----------
            function : callable or string.
                if a string is provided for the evaluation, x is the variable.
                if a function is provided, it shall be function(x)

        """
        if hasattr(function, '__call__'):
            return dict(y=function(x))
        else:
            from builtins import eval
            return dict(y=eval(function, dict(x=x)))

    def vjp(self): return 0
    def jvp(self): return 0

@operator
class watchpoint:
    """ printing a variable or a list of variables.

        >>> with Builder() as m:
        >>>    x = m.input('x')
        >>>    watchpoint(x, lambda x: print(x.shape))
        >>>    watchpoint(x, lambda x: print(x.size))
        >>>    watchpoint((x, x), lambda (x1, x2): print(x1.size))
        >>>    m.output(x)
    """
    ain  = {'x': '*'}
    aout = {}

    def apl(self, x, monitor=print):
        monitor(x)
        return dict(y = x)

    def vjp(self): return 0
    def jvp(self): return

@operator
class assert_isinstance:
    """ assert a variable is of a certain type """
    ain = 'obj'
    aout = []
    def apl(self, obj, class_or_tuple):
        if not isinstance(obj, class_or_tuple):
            raise TypeError('Expecting an instance of %s, got %s', repr(class_or_tuple), repr(type(obj)))

    def vjp(self): return 0
    def jvp(self): return 0

@operator
class assert_true:
    """ assert if a function applied to a variable is true.

        Examples:

        >>> with Builder() as m:
        >>>    x = m.input('x')
        >>>    assert_true(x, lambda x: isinstance(x, int)))
        >>>    m.output(x)
    """

    ain = 'x'
    aout = []
    def apl(self, x, func):
        if not func(x):
            raise AssertionError('Assertion failed on %s(%s).' % (func, x))

    def vjp(self): return 0
    def jvp(self): return 0
