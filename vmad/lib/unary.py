from vmad import operator
from vmad.core.symbol import Literal, ZeroLiteral
from vmad.core.symbol import Symbol
import numpy

class unary_ufunc:
    ain = 'x'
    aout = 'y'

    @staticmethod
    def f(x): return x

    @staticmethod
    def fprime(x): return 1

    def apl(node, x):
        return node.operator.prototype.f(x)

    def vjp(node, x, _y):
        return node.operator.prototype.fprime(x) * _y

    def jvp(node, x, x_):
        return node.operator.prototype.fprime(x) * x_

@operator
class sinc(unary_ufunc):
    @staticmethod
    def f(x): return numpy.sinc(x)
    @staticmethod
    def fprime(x):
        # FIXME: expand this properly:
        y = 1 / x * (
          numpy.cos(numpy.pi * x)
        - numpy.sin(numpy.pi * x) / (numpy.pi * x))
        return numpy.where(x == 0, 0, y)

@operator
class sin(unary_ufunc):
    @staticmethod
    def f(x): return numpy.sin(x)
    @staticmethod
    def fprime(x): return numpy.cos(x)

@operator
class cos(unary_ufunc):
    @staticmethod
    def f(x): return numpy.cos(x)
    @staticmethod
    def fprime(x): return -numpy.sin(x)

@operator
class tan(unary_ufunc):
    @staticmethod
    def f(x): return numpy.tan(x)
    # FIXME: hard code a faster version which saves tan.
    @staticmethod
    def fprime(x): return 1 + numpy.tan(x) ** 2

@operator
class arcsin(unary_ufunc):
    @staticmethod
    def f(x): return numpy.arcsin(x)
    @staticmethod
    def fprime(x): return 1 / (1 - x **2) **0.5

@operator
class arccos(unary_ufunc):
    @staticmethod
    def f(x): return numpy.arccos(x)
    @staticmethod
    def fprime(x): return -1 / (1 - x **2) **0.5

@operator
class arctan(unary_ufunc):
    @staticmethod
    def f(x): return numpy.arctan(x)
    @staticmethod
    def fprime(x): return 1 / (1 + x **2)

@operator
class sinh(unary_ufunc):
    @staticmethod
    def f(x): return numpy.sinh(x)
    @staticmethod
    def fprime(x): return numpy.cosh(x)

@operator
class cosh(unary_ufunc):
    @staticmethod
    def f(x): return numpy.cosh(x)
    @staticmethod
    def fprime(x): return numpy.sinh(x)


@operator
class log(unary_ufunc):
    @staticmethod
    def f(x): return numpy.log(x)
    @staticmethod
    def fprime(x): return 1.0 / x

@operator
class log10(unary_ufunc):
    @staticmethod
    def f(x): return numpy.log10(x)
    @staticmethod
    def fprime(x, log10=1.0 / numpy.log(10)): return 1.0 / x * log10

@operator
class exp(unary_ufunc):
    @staticmethod
    def f(x): return numpy.exp(x)
    @staticmethod
    def fprime(x): return numpy.exp(x)

@operator
class absolute(unary_ufunc):
    @staticmethod
    def f(x): return numpy.absolute(x)
    @staticmethod
    def fprime(x): return 1.0 * (x > 0) + -1.0 * (x < 0)

fabs = absolute
