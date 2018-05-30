from vmad import operator
from vmad.core.symbol import Literal, ZeroLiteral
import numpy

@operator
class mul:
    ain = {'x1' : '*',
           'x2' : '*',
          }
    aout = {'y' : '*'}

    def apl(self, x1, x2):
        return dict(y = x1 * x2)

    def rcd(self, x1, x2, y):
        model = self.find_model()
        # the other value is not needed, 0 should work.
        if isinstance(self.varin['x1'].symbol, Literal):
            x2 = 0
        if isinstance(self.varin['x2'].symbol, Literal):
            x1 = 0
        return dict(x1=x1, x2=x2)

    def vjp(self, _y, x1, x2):
        return dict(_x1 = _y * x2,
                    _x2 = _y * x1)

    def jvp(self, x1_, x2_, x1, x2):
        return dict(y_ = x1_* x2 + x1 * x2_)

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
class add:
    ain  = {'x1': '*',
            'x2': '*',
           }
    aout = {'y': '*'}

    def apl(self, x1, x2):
        return dict(y = x1 + x2)

    def vjp(self, _y):
        return dict(_x1 = _y, _x2 = _y)

    def jvp(self, x1_, x2_):
        return dict(y_ = x1_ + x2_)

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
class abs:
    ain = {'x' : '*',
          }
    aout = {'y' : '*'}

    def apl(self, x):
        return dict(y=numpy.abs(x))

    def vjp(self, _y, x):
        return dict(_x = _y * numpy.sign(x))

    def jvp(self, x_, x):
        return dict(y_ = x_ * numpy.sign(x))

@operator
class pow:
    ain = {'x' : '*',
          }
    aout = {'y' : '*'}

    def apl(self, x, n):
        return dict(y=x ** n)

    def vjp(self, _y, x, n):
        fac = x ** (n - 1) if n != 1 else 1
        return dict(_x = n * _y * fac)

    def jvp(self, x_, x, n):
        fac = x ** (n - 1) if n != 1 else 1
        return dict(y_ = n * x_ * fac)

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

        y = numpy.add.reduceat(x, at, axis=axis)
        return y

    def rcd(self, x, y, at, axis=0):
        return dict(xshape = numpy.shape(x), at=at, axis=axis)

    def vjp(self, _y, xshape, at, axis):
        _x = numpy.ones(xshape)
        N = numpy.diff(numpy.concatenate([at, [xshape[axis]]], axis=0))
        return numpy.repeat(_y, N, axis=axis)

    def jvp(self, x_, at, axis):
        return numpy.add.reduceat(x_, at, axis=axis)

@operator
class sum:
    ain  = {'x' : '*'}
    aout = {'y': '*'}

    def apl(self, x, axis=None):
        return dict(y = numpy.sum(x, axis=axis))

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
        return numpy.sum(x_, axis=axis)

