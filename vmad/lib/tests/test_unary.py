from __future__ import print_function
from pprint import pprint
from vmad.lib import linalg
import numpy

from vmad.lib import binary_ufunc

from vmad import Builder
from vmad.testing import BaseVectorTest

class UnaryUfuncVectorTest(BaseVectorTest):
    ufunc = None

    def y(self, x):
        return type(self).ufunc.prototype.f(x)

    def model(self, x):
        return type(self).ufunc(x)

class Test_sin(UnaryUfuncVectorTest):
    ufunc = binary_ufunc.sin
    x = numpy.arange(10)

class Test_cos(UnaryUfuncVectorTest):
    ufunc = binary_ufunc.cos
    x = numpy.arange(10)

class Test_log(UnaryUfuncVectorTest):
    ufunc = binary_ufunc.log
    x = numpy.arange(10) + 1

class Test_log10(UnaryUfuncVectorTest):
    ufunc = binary_ufunc.log10
    x = numpy.arange(10) + 1

class Test_exp(UnaryUfuncVectorTest):
    ufunc = binary_ufunc.exp
    x = numpy.arange(10) + 1

class Test_absolute(UnaryUfuncVectorTest):
    ufunc = binary_ufunc.absolute
    x = numpy.arange(10) - 3
