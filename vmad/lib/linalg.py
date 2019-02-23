from vmad import operator

import numpy

# a few commonly used operators are expected to be
# compatible with the python ones.
from vmad.core.stdlib import mul, add, abs, pow

# import all functions defined in unary module
from vmad.lib.unary import *

from numpy.core.einsumfunc import _parse_einsum_input

def _join_einsum_sub(sub_op, sub_y):
    return ','.join(sub_op) + '->' + sub_y

@operator
class einsum:
    ain = 'x'
    aout = 'y'
    def apl(node, subscripts, x):
        operands = []
        operands.append(subscripts)
        operands.extend(x)
        # we use the internal einsum function to parse subscripts to a normal form.
        sub_op, sub_y, operands = _parse_einsum_input(operands)
        sub_op = sub_op.split(',')
        return dict(y=numpy.einsum(_join_einsum_sub(sub_op, sub_y), *x), sub_op=sub_op, sub_y=sub_y, x=x)

    def vjp(node, x, _y, sub_op, sub_y):
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

    def jvp(node, x, x_, sub_op, sub_y):
        y_ = []
        for i, arg in enumerate(x):
            # Jacobian is removing the item from the chain. 
            # jvp is replace the item with x_
            x1 = list(x)[:]
            x1[i] = x_[i]
            y_.append(numpy.einsum(_join_einsum_sub(sub_op, sub_y), *x1))

        # sum of all branches
        return numpy.sum(y_, axis=0)

def matmul(x, A):
    return einsum('ik,ijk->ij', x, A)

@operator
class unpack_complex:
    ain = {'x' : '*'}
    aout = [('real', '*'), ('imag', '*')]

    def apl(node, x):
        return dict(real=x.real, imag=x.imag)

    def vjp(node, _real, _imag):
        return dict(_x = _real + _imag * 1j)

@operator
class pack_complex:
    ain = [('real', '*'), ('imag', '*')]
    aout = {'y' : '*'}

    def apl(node, real, imag):
        return dict(y = real + imag * 1j)

    def vjp(node, _y):
        return dict(_real = _y.real, _imag = _y.imag)

@operator
class conj:
    ain = 'x'
    aout = 'y'

    def apl(node, x):
        return numpy.conj(x)

    def vjp(node, _y):
        return numpy.conj(_y)

@operator
class to_scalar:
    ain  = {'x': 'ndarray'}
    aout = {'y': '*'}

    def apl(node, x):
        return dict(y = (x * numpy.conj(x)).sum())

    def vjp(node, _y, x):
        return dict(_x = 2 * numpy.conj(_y) * x)

    def jvp(node, x_, x):
        return dict(y_ = (x_ * numpy.conj(x) + numpy.conj(x_) * x).sum())

@operator
class copy:
    ain = {'x' : 'ndarray'}
    aout = {'y' : 'ndarray'}

    def apl(node, x):
        return dict(y = numpy.copy(x))

    def vjp(node, _y):
        return dict(_x = numpy.copy(_y))

@operator
class stack:
    ain = {'x' : 'ndarray',}
    aout = {'y' : 'ndarray'}

    def apl(node, x, axis):
        return dict(y=numpy.stack(x, axis=axis))

    def vjp(node, _y, axis):
        return dict(_x=[numpy.take(_y, i, axis=axis)
                for i in range(numpy.shape(_y)[axis])])

@operator
class take:
    ain = {'x' : 'ndarray',}
    aout = {'y' : 'ndarray'}

    def apl(node, x, i, axis):
        if axis is None:
            raise AssertionError('Assertion error. axis keyword in linalg.take cannot be None.')
        i = numpy.array(i, dtype='intp')
        y = numpy.take(x, i, axis=axis)
        yshape = numpy.shape(y)
        xshape = numpy.shape(x)
        return dict(y=y, yshape=yshape, xshape=xshape, i=i, axis=axis)

    def vjp(node, _y, i, axis, xshape, yshape):
        # shape of y is x.shape [..., i
        _x = numpy.zeros(xshape)
        _y = numpy.broadcast_to(_y, yshape)

        indices = [slice(None)] * len(xshape)
        indices[axis] = i

        # one element can show up several times, add all gradients.
        numpy.add.at(_x, tuple(indices), _y)
        return dict(_x=_x)

@operator
class concatenate:
    ain = {'x' : '*',}
    aout = {'y' : 'ndarray'}

    def apl(node, x, axis):
        if axis is None:
            raise AssertionError('Assertion error. axis keyword in linalg.take cannot be None.')
        return dict(y=numpy.concatenate(x, axis=axis))

    def rcd(node, x, axis, y):
        return dict(xshapes=[numpy.shape(x1) for x1 in x], axis=axis)

    def vjp(node, _y, axis, xshapes):
        # chopping of _y along the axis will do
        _y = numpy.swapaxes(_y, 0, axis)
        offset = 0
        _x = []

        for xshape in xshapes:
            _x1 = _y[offset:offset+xshape[axis]]
            _x.append(numpy.swapaxes(_x1, 0, axis))
            offset = offset + xshape[axis]

        return dict(_x=_x)

@operator
class transpose:
    ain = {'x' : 'ndarray',}
    aout = {'y' : 'ndarray'}

    def apl(node, x, axes):
        return dict(y=numpy.transpose(x, axes))

    def rcd(node, x, axes, y):
        return dict(axes=axes)

    def vjp(node, _y, axes):
        inverse = [0] * len(axes)
        for i, a in enumerate(axes):
            inverse[a] = i
        _x = _y.transpose(inverse)
        return dict(_x=_x)

@operator
class reshape:
    ain  = {'x' : '*'}
    aout = {'y': '*'}

    def apl(node, x, shape):
        return dict(y = numpy.reshape(x, shape))

    def rcd(node, x, shape, y):
        return dict(xshape = numpy.shape(x), shape=shape)

    def vjp(node, _y, xshape):
        return dict(_x=_y.reshape(xshape))

@operator
class sumat:
    ain  = {'x' : '*'}
    aout = {'y': '*'}

    def apl(node, x, at, axis=0):
        if not (numpy.diff(at) >= 0).all():
            raise ValueError('at must be monotonically increasing')

        y = numpy.add.reduceat(x, at, axis=axis, dtype='f8')
        return y

    def rcd(node, x, y, at, axis=0):
        return dict(xshape = numpy.shape(x), at=at, axis=axis)

    def vjp(node, _y, xshape, at, axis):
        _x = numpy.ones(xshape)
        N = numpy.diff(numpy.concatenate([at, [xshape[axis]]], axis=0))
        return numpy.repeat(_y, N, axis=axis)

@operator
class sum:
    ain  = {'x' : '*'}
    aout = {'y': '*'}

    def apl(node, x, axis=None):
        return dict(y = numpy.sum(x, axis=axis, dtype='f8'))

    def rcd(node, x, y, axis=None):
        return dict(xshape = numpy.shape(x), axis=axis)

    def vjp(node, _y, xshape, axis):
        _x = numpy.ones(xshape)

        if axis is not None:
            # prepend to the correct axis
            _yshape = list(numpy.shape(_y))
            _yshape.insert(axis, 1)
            _y = _y.reshape(_yshape)

        _x *= _y
        return dict(_x = _x)

