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

