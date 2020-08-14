from vmad import Builder
from vmad.lib import linalg
from vmad.lib import unary
from vmad.testing import BaseVectorTest

import numpy
from pprint import pprint

class UnaryUfuncVectorTest(BaseVectorTest):
    ufunc = None

    def y(self, x):
        return type(self).ufunc.prototype.f(x)

    def model(self, x):
        return type(self).ufunc(x)

class Test_sinc(UnaryUfuncVectorTest):
    ufunc = unary.sinc
    x = numpy.arange(-10, 10)


class Test_sin(UnaryUfuncVectorTest):
    ufunc = unary.sin
    x = numpy.arange(-10, 10)

class Test_cos(UnaryUfuncVectorTest):
    ufunc = unary.cos
    x = numpy.arange(-10, 10)

class Test_tan(UnaryUfuncVectorTest):
    ufunc = unary.tan
    x = numpy.linspace(-1, 1, 10)

class Test_arcsin(UnaryUfuncVectorTest):
    ufunc = unary.arcsin
    x = numpy.linspace(-0.99, 0.99, 8)

class Test_arccos(UnaryUfuncVectorTest):
    ufunc = unary.arccos
    x = numpy.linspace(-0.99, 0.99, 8)

class Test_arctan(UnaryUfuncVectorTest):
    ufunc = unary.arctan
    x = numpy.arange(-10, 10)

class Test_log(UnaryUfuncVectorTest):
    ufunc = unary.log
    x = numpy.arange(10) + 1

class Test_log10(UnaryUfuncVectorTest):
    ufunc = unary.log10
    x = numpy.arange(10) + 1

class Test_exp(UnaryUfuncVectorTest):
    ufunc = unary.exp
    x = numpy.linspace(-0.99, 0.99, 8)

class Test_absolute(UnaryUfuncVectorTest):
    ufunc = unary.absolute
    x = numpy.linspace(-0.99, 0.99, 8)

class Test_sinh(UnaryUfuncVectorTest):
    ufunc = unary.sinh
    x = numpy.linspace(-0.99, 0.99, 8)

class Test_cosh(UnaryUfuncVectorTest):
    ufunc = unary.cosh
    x = numpy.linspace(-0.99, 0.99, 8)
