from __future__ import print_function
from pprint import pprint
from vmad.lib import linalg
import numpy

from vmad import Builder
from vmad.testing import BaseScalarTest

class LinalgScalarTest(BaseScalarTest):
    to_scalar = staticmethod(linalg.to_scalar)

class Test_to_scalar(LinalgScalarTest):

    x = numpy.arange(10)
    y = sum(x ** 2)
    x_ = numpy.eye(10)

    def model(self, x):
        return x

class Test_pack_complex(LinalgScalarTest):
    x = numpy.arange(10) # will pack to complex of x + x * 1j
    y = sum(x ** 2) * 1.25
    x_ = numpy.eye(10)

    def model(self, x):
        c = linalg.pack_complex(x, linalg.mul(x, 0.5))
        return c

class Test_unpack_complex(LinalgScalarTest):
    x = numpy.arange(10) # will pack to complex of x + x * 1j
    y = sum(x ** 2) * 4
    x_ = numpy.eye(10)

    def model(self, x):
        c = linalg.pack_complex(x, x)
        r, i = linalg.unpack_complex(c)
        return linalg.add(r, i)

class Test_reshape(LinalgScalarTest):
    x = numpy.arange(10) # will pack to complex of x + x * 1j
    y = sum(x ** 2)
    x_ = numpy.eye(10)

    def model(self, x):
        c = linalg.reshape(x, (5, 2))
        return c

class Test_einsum(LinalgScalarTest):
    x = numpy.arange(10)
    y = sum(x ** 2) ** 2
    x_ = numpy.eye(10)

    def model(self, x):
        # testing uncontracted dimensions
        a = linalg.reshape(x, (10, 1, 1))
        b = linalg.reshape(x, (1, 10))
        return linalg.einsum("abc, ca->b", [a, b])

class Test_einsum2(LinalgScalarTest):
    x = numpy.arange(10)
    y = sum(x ** 2) ** 2
    x_ = numpy.eye(10)

    def model(self, x):
        # testing contraction
        a = linalg.reshape(x, (10,))
        b = linalg.reshape(x, (10))
        return linalg.einsum("i, i->", [a, b])

class Test_mul1(LinalgScalarTest):
    x = numpy.arange(10)
    y = sum(x ** 2)
    x_ = numpy.eye(10)

    def model(self, x):
        return linalg.mul(x, 1.0)

class Test_mul2(LinalgScalarTest):
    x = numpy.arange(10) + 1
    y = sum(x**2)
    x_ = numpy.eye(10)
    eps = 1e-3
    def model(self, x):
        y1 = linalg.pow(x, 0.3)
        y2 = linalg.pow(x, 0.7)
        return linalg.mul(y1, y2)

class Test_mul3(LinalgScalarTest):
    x = numpy.arange(10) 
    y = sum(x**2)
    x_ = numpy.eye(10)

    def model(self, x):
        return linalg.mul(1, x)

class Test_sum(LinalgScalarTest):
    x = numpy.arange(10)
    y = sum(x.reshape(5, 2).sum(axis=0) ** 2)
    x_ = numpy.eye(10)

    def model(self, x):
        x = linalg.reshape(x, (5, 2))

        return linalg.sum(x, axis=0)

class Test_pow(LinalgScalarTest):
    x = numpy.arange(10)
    y = sum(x ** 2)
    x_ = numpy.eye(10)

    def model(self, x):
        return linalg.pow(x, 1.0)

class Test_abs(LinalgScalarTest):
    x = numpy.arange(10) - 4.5 # avoid 0 because numerial is bad.
    y = sum((abs(x) + x) ** 2)
    x_ = numpy.eye(10)

    def model(self, x):
        return linalg.add(linalg.abs(x), x)

class Test_log(LinalgScalarTest):
    logx = 1 + numpy.arange(10)
    x = numpy.exp(logx)
    y = sum(logx ** 2)
    x_ = numpy.eye(10)

    def model(self, x):
        return linalg.log(x)

class Test_add(LinalgScalarTest):
    x = numpy.arange(10)
    y = sum((x + 5.0)** 2)
    x_ = numpy.eye(10)

    def model(self, x):
        return linalg.add(x, 5.0)

class Test_copy(LinalgScalarTest):
    x = numpy.arange(10)
    y = sum(x ** 2)
    x_ = numpy.eye(10)

    def model(self, x):
        return linalg.copy(x)

class Test_stack(LinalgScalarTest):
    x = numpy.arange(10)
    y = sum(x ** 2) * 2
    x_ = numpy.eye(10)

    def model(self, x):
        return linalg.stack([x, x], axis=0)

class Test_take(LinalgScalarTest):

    x = numpy.arange(10)
    y = 2 ** 2
    x_ = numpy.eye(10)

    def model(self, x):
        return linalg.take(x, 2, axis=0)

class Test_sumat(LinalgScalarTest):
    x = numpy.arange(10)
    at = [0, 1, 3]
    y = numpy.sum(numpy.add.reduceat(x.reshape(5, 2), at, axis=0) ** 2)
    x_ = numpy.eye(10)

    def model(self, x):
        x = linalg.reshape(x, (5, 2))

        return linalg.sumat(x, at=self.at, axis=0)

