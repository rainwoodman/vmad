from vmad import operator
from vmad.core.symbol import Literal, ZeroLiteral
from vmad.core.symbol import Symbol

import numpy

# a few commonly used operators are expected to be
# compatible with the python ones.
from vmad.core.stdlib import mul, add, abs, pow

@operator
class matmul:
    ain = {'x':'*'}
    aout = {'C':'*'}

    def apl(self, x, A):
        C=numpy.einsum('ik,ijk->ij',x,A) 
        return dict(C=C)

    def vjp(self, _C,A):
        _x=numpy.einsum('ijk,ij->ik',A,_C)
        return dict(_x=_x)

    def jvp(self, x_,A):
        C_=numpy.einsum('ik,ijk->ij',x_,A)
        return dict(C_=C_)

from numpy.core.einsumfunc import _parse_einsum_input

def _join_einsum_sub(sub_op, sub_y):
    return ','.join(sub_op) + '->' + sub_y

@operator
class einsum:
    ain = 'x'
    aout = 'y'
    def apl(self, subscripts, x):
        operands = []
        operands.append(subscripts)
        operands.extend(x)
        # we use the internal einsum function to parse subscripts to a normal form.
        sub_op, sub_y, operands = _parse_einsum_input(operands)
        sub_op = sub_op.split(',')
        return dict(y=numpy.einsum(_join_einsum_sub(sub_op, sub_y), *x), sub_op=sub_op, sub_y=sub_y, x=x)

    def vjp(self, x, _y, sub_op, sub_y):
        _x = []
        for i, arg in enumerate(x):
            # Jacobian is removing the item from the chain. 
            # vjp is replace the item with _y.
            a = list(sub_op)[:]
            x1 = list(x)[:]
            new_sub_y = a[i]
            a[i] = sub_y
            x1[i] = _y
            _x.append(numpy.einsum(_join_einsum_sub(a, new_sub_y), *x1))

        return _x

    def jvp(self, x, x_, sub_op, sub_y):
        y_ = []
        for i, arg in enumerate(x):
            # Jacobian is removing the item from the chain. 
            # jvp is replace the item with x_
            x1 = list(x)[:]
            x1[i] = x_[i]
            y_.append(numpy.einsum(_join_einsum_sub(sub_op, sub_y), *x1))

        # sum of all branches
        return numpy.sum(y_, axis=0)

@operator
class unpack_complex:
    ain = {'x' : '*'}
    aout = [('real', '*'), ('imag', '*')]

    def apl(self, x):
        return dict(real=x.real, imag=x.imag)

    def vjp(self, _real, _imag):
        return dict(_x = _real + _imag * 1j)

    def jvp(self, x_):
        return dict(real_ = x_.real, imag_ = x_.imag)

@operator
class pack_complex:
    ain = [('real', '*'), ('imag', '*')]
    aout = {'y' : '*'}

    def apl(self, real, imag):
        return dict(y = real + imag * 1j)

    def vjp(self, _y):
        return dict(_real = _y.real, _imag = _y.imag)

    def jvp(self, real_, imag_):
        return dict(y_ = real_ + imag_ * 1j)

@operator
class to_scalar:
    ain  = {'x': 'ndarray'}
    aout = {'y': '*'}

    def apl(self, x):
        return dict(y = (x * numpy.conj(x)).sum())

    def vjp(self, _y, x):
        return dict(_x = 2 * numpy.conj(_y) * x)

    def jvp(self, x_, x):
        return dict(y_ = (x_ * numpy.conj(x) + numpy.conj(x_) * x).sum())

@operator
class log:
    ain = {'x' : '*',
          }
    aout = {'y' : '*'}

    def apl(self, x):
        return dict(y=numpy.log(x))

    def vjp(self, _y, x):
        return dict(_x = _y * 1. / x)

    def jvp(self, x_, x):
        return dict(y_ = x_ * 1. / x)

@operator
class copy:
    ain = {'x' : 'ndarray'}
    aout = {'y' : 'ndarray'}

    def apl(self, x):
        return dict(y = numpy.copy(x))

    def vjp(self, _y):
        return dict(_x = numpy.copy(_y))

    def jvp(self, x_):
        return dict(y_ = numpy.copy(x_))

@operator
class stack:
    ain = {'x' : 'ndarray',}
    aout = {'y' : 'ndarray'}

    def apl(self, x, axis):
        return dict(y=numpy.stack(x, axis=axis))

    def vjp(self, _y, axis):
        return dict(_x=[numpy.take(_y, i, axis=axis)
                for i in range(numpy.shape(_y)[axis])])

    def jvp(self, x_, axis):
        return dict(y_=numpy.stack(x_, axis=axis))

@operator
class take:
    ain = {'x' : 'ndarray',}
    aout = {'y' : 'ndarray'}

    def apl(self, x, i, axis):
        if axis is None:
            raise AssertionError('Assertion error. axis keyword in linalg.take cannot be None.')
        return dict(y=numpy.take(x, i, axis=axis))

    def rcd(self, x, i, axis, y):
        return dict(xshape = numpy.shape(x), i=i, axis=axis)

    def vjp(self, _y, i, axis, xshape):
        _x = numpy.zeros(xshape)
        _x = numpy.swapaxes(_x, 0, axis)
        _x[i] = _y
        _x = numpy.swapaxes(_x, 0, axis)
        return dict(_x=_x)

    def jvp(self, x_, i, axis):
        return dict(y_=numpy.take(x_, i, axis=axis))

@operator
class reshape:
    ain  = {'x' : '*'}
    aout = {'y': '*'}

    def apl(self, x, shape):
        return dict(y = numpy.reshape(x, shape))

    def rcd(self, x, shape, y):
        return dict(xshape = numpy.shape(x), shape=shape)

    def vjp(self, _y, xshape):
        return dict(_x=_y.reshape(xshape))

    def jvp(self, x_, shape):
        return dict(y_=x_.reshape(shape))

@operator
class sumat:
    ain  = {'x' : '*'}
    aout = {'y': '*'}

    def apl(self, x, at, axis=0):
        if not (numpy.diff(at) >= 0).all():
            raise ValueError('at must be monotonically increasing')

        y = numpy.add.reduceat(x, at, axis=axis, dtype='f8')
        return y

    def rcd(self, x, y, at, axis=0):
        return dict(xshape = numpy.shape(x), at=at, axis=axis)

    def vjp(self, _y, xshape, at, axis):
        _x = numpy.ones(xshape)
        N = numpy.diff(numpy.concatenate([at, [xshape[axis]]], axis=0))
        return numpy.repeat(_y, N, axis=axis)

    def jvp(self, x_, at, axis):
        return numpy.add.reduceat(x_, at, axis=axis, dtype='f8')

@operator
class sum:
    ain  = {'x' : '*'}
    aout = {'y': '*'}

    def apl(self, x, axis=None):
        return dict(y = numpy.sum(x, axis=axis, dtype='f8'))

    def rcd(self, x, y, axis=None):
        return dict(xshape = numpy.shape(x), axis=axis)

    def vjp(self, _y, xshape, axis):
        _x = numpy.ones(xshape)

        if axis is not None:
            # prepend to the correct axis
            _yshape = list(numpy.shape(_y))
            _yshape.insert(axis, 1)
            _y = _y.reshape(_yshape)

        _x *= _y
        return dict(_x = _x)

    def jvp(self, x_, xshape, axis):
        return numpy.sum(x_, axis=axis, dtype='f8')

