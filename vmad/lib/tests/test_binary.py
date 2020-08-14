from vmad import Builder
from vmad.lib import binary
from vmad.lib import linalg
from vmad.testing import BaseVectorTest

import numpy
from pprint import pprint

class BinaryUfuncVectorTest(BaseVectorTest):
    ufunc = None

    def y(self, x):
        return type(self).ufunc.prototype.f(x[0], x[1])

    def model(self, x):
        x1 = linalg.take(x, 0, axis=0)
        x2 = linalg.take(x, 1, axis=0)
        return type(self).ufunc(x1, x2)


class Test_fmax(BinaryUfuncVectorTest):
    ufunc = binary.fmax
    x = numpy.arange(10).reshape(2, 5)

class Test_fmin(BinaryUfuncVectorTest):
    ufunc = binary.fmin
    x = numpy.arange(10).reshape(2, 5)

class Test_add(BinaryUfuncVectorTest):
    ufunc = binary.add
    x = numpy.arange(10).reshape(2, 5)

