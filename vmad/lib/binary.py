from vmad import operator
from vmad.core.symbol import Literal, ZeroLiteral
from vmad.core.symbol import Symbol
import numpy

class binary_ufunc:
    ain = 'x1', 'x2'
    aout = 'y'

    @staticmethod
    def f(x1, x2): return x1

    @staticmethod
    def fprime1(x1, x2): return 1

    @staticmethod
    def fprime2(x1, x2): return 0

    def apl(node, x1, x2):
        return node.operator.prototype.f(x1, x2)

    def vjp(node, x1, x2, _y):
        return dict(
                _x1 = node.operator.prototype.fprime1(x1, x2) * _y,
                _x2 = node.operator.prototype.fprime2(x1, x2) * _y
                )

    def jvp(node, x1, x2, x1_, x2_):
        return node.operator.prototype.fprime1(x1, x2) * x1_ \
            +  node.operator.prototype.fprime2(x1, x2) * x2_

@operator
class fmax(binary_ufunc):
    @staticmethod
    def f(x1, x2):
        return numpy.fmax(x1, x2)

    @staticmethod
    def fprime1(x1, x2):
        return x1 >= x2

    @staticmethod
    def fprime2(x1, x2):
        return ~(x1 >= x2)

maximum = fmax

@operator
class fmin(binary_ufunc):
    @staticmethod
    def f(x1, x2):
        return numpy.fmin(x1, x2)

    @staticmethod
    def fprime1(x1, x2):
        return x1 < x2

    @staticmethod
    def fprime2(x1, x2):
        return ~(x1 <= x2)

minimum = fmax

@operator
class add(binary_ufunc):
    @staticmethod
    def f(x1, x2):
        return numpy.add(x1, x2)

    @staticmethod
    def fprime1(x1, x2):
        return 1

    @staticmethod
    def fprime2(x1, x2):
        return 1

