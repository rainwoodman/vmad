from __future__ import print_function
from pprint import pprint
from vmad.lib import linalg
import numpy

from vmad import Builder
from vmad.testing import BaseVectorTest

class Test_pack_complex(BaseVectorTest):
    x = numpy.arange(10) # will pack to complex of x + x * 1j
    y = numpy.sum(2 * x ** 2)

    def model(self, x):
        # FIXME: pack_complex only works with to_scalar
        # but it doesn't work with mul and conj
        # there is a missing negative sign. one of these is wrong.
        c = linalg.pack_complex(x, x)
        c = linalg.to_scalar(c)
        return c

class Test_conj_mul(BaseVectorTest):
    x = numpy.arange(10) # will pack to complex of x + x * 1j
    y = x ** 2

    def model(self, x):
        # FIXME: pack_complex only works with to_scalar
        # but it doesn't work with mul and conj
        # there is a missing negative sign. one of these is wrong.
        x = x * 1j
        c = linalg.conj(x)
        return x * c

class Test_unpack_complex(BaseVectorTest):
    x = numpy.arange(10) # will pack to complex of x + x * 1j
    y = x * 2

    def model(self, x):
        c = linalg.pack_complex(x, x)
        r, i = linalg.unpack_complex(c)
        return linalg.add(r, i)

class Test_reshape(BaseVectorTest):
    x = numpy.arange(10) 
    y = x.reshape(5, 2)

    def model(self, x):
        c = linalg.reshape(x, (5, 2))
        return c

class Test_einsum(BaseVectorTest):
    x = numpy.arange(10)
    y = numpy.sum(x ** 2).reshape(1)

    def model(self, x):
        # testing uncontracted dimensions
        a = linalg.reshape(x, (10, 1, 1))
        b = linalg.reshape(x, (1, 10))
        return linalg.einsum("abc, ca->b", [a, b])

class Test_einsum2(BaseVectorTest):
    x = numpy.arange(10)
    y = numpy.sum(x ** 2)

    def model(self, x):
        # testing contraction
        a = linalg.reshape(x, (10,))
        b = linalg.reshape(x, (10))
        return linalg.einsum("i, i->", [a, b])

class Test_mul1(BaseVectorTest):
    x = numpy.arange(10)
    y = x

    def model(self, x):
        return linalg.mul(x, 1.0)

class Test_mul2(BaseVectorTest):
    x = numpy.arange(10) + 1
    y = x
    def model(self, x):
        y1 = linalg.pow(x, 0.3)
        y2 = linalg.pow(x, 0.7)
        return linalg.mul(y1, y2)

class Test_mul3(BaseVectorTest):
    x = numpy.arange(10) 
    y = x

    def model(self, x):
        return linalg.mul(1, x)

class Test_sum(BaseVectorTest):
    x = numpy.arange(10)
    y = x.reshape(5, 2).sum(axis=0)

    def model(self, x):
        x = linalg.reshape(x, (5, 2))

        return linalg.sum(x, axis=0)

class Test_pow(BaseVectorTest):
    x = numpy.arange(10)
    y = x ** 2.0

    def model(self, x):
        return linalg.pow(x, 2.0)

class Test_abs(BaseVectorTest):
    x = numpy.arange(10) - 4.5 # avoid 0 because numerial is bad.
    y = (abs(x) + x)

    def model(self, x):
        return linalg.add(linalg.abs(x), x)

class Test_log(BaseVectorTest):
    logx = 1 + numpy.arange(10)
    x = numpy.exp(logx)
    y = logx
    epsilon = 1e-3 # enlarge epsilon for loss of precision in log.
    def model(self, x):
        return linalg.log(x)

class Test_add(BaseVectorTest):
    x = numpy.arange(10)
    y = (x + 5.0)

    def model(self, x):
        return linalg.add(x, 5.0)

class Test_copy(BaseVectorTest):
    x = numpy.arange(10)
    y = x

    def model(self, x):
        return linalg.copy(x)

class Test_stack(BaseVectorTest):
    x = numpy.arange(10)
    y = numpy.stack([x, x])

    def model(self, x):
        return linalg.stack([x, x], axis=0)

class Test_take(BaseVectorTest):

    x = numpy.arange(10)
    y = numpy.array(4)

    def model(self, x):
        return linalg.sum(linalg.take(x, [2, 2], axis=0), axis=0)

from vmad.core.stdlib import watchpoint
class Test_take_vector(BaseVectorTest):

    x = numpy.arange(10)
    y = numpy.array(4)
    def model(self, x):
        a = linalg.take(x, [[2, 2], [2, 2]], axis=0)
        b = linalg.take(a, 0, axis=0)
        c = linalg.take(b, 0, axis=0)
        #d = linalg.take(b, 0, axis=0)
        #watchpoint(b, lambda b:print('b=', b))
        #watchpoint(c, lambda c:print('c=', c))
    #    watchpoint(d, lambda d:print('d=', d))
        return c + c

class Test_concatenate(BaseVectorTest):

    x = numpy.arange(10).reshape(5, 2)
    y = numpy.concatenate([x, x], axis=1)

    def model(self, x):
        #a = linalg.concatenate([x, x], axis=0)
        b = linalg.concatenate([x, x], axis=1)
        return b

class Test_transpose(BaseVectorTest):

    x = numpy.arange(10).reshape(5, 2)
    y = numpy.transpose(x, (1, 0))

    def model(self, x):
        #a = linalg.concatenate([x, x], axis=0)
        b = linalg.transpose(x, (1, 0))
        return b

class Test_sumat(BaseVectorTest):
    x = numpy.arange(10).reshape(5, 2)
    at = [0, 1, 3]
    y = numpy.add.reduceat(x, at, axis=0)

    def model(self, x):
        return linalg.sumat(x, at=self.at, axis=0)


def test_take_chained():
    from vmad import autooperator
    from numpy.testing import assert_array_equal

    @autooperator('x->y')
    def func(x):
        b = 0
        for i in range(3):
            b = b + linalg.take(x, i, axis=0)
        return b

    (y,), (_x, ) = (func.build().compute_with_vjp(init=dict(x=numpy.array([3, 4, 5])), v=dict(_y=1.0)))
    assert y == 12
    assert_array_equal(_x, (1, 1, 1))
