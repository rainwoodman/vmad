from vmad.core.operator import operator

class unary:
    ain = 'x'
    aout = 'y'

class binary:
    ain = 'x1', 'x2'
    aout = 'y'


@operator
class neg(unary):
    def apl(self, x):
        return dict(y = -x1)

    def vjp(self, _y):
        return dict(_x = -_y)

    def jvp(self, x_):
        return dict(y_ = -x_)

@operator
class pos(unary):
    def apl(self, x):
        return dict(y = +x1)

    def vjp(self, _y):
        return dict(_x = +_y)

    def jvp(self, x_):
        return dict(y_ = +x_)

@operator
class abs(unary):
    def apl(self, x):
        return dict(y = abs(x))

    def vjp(self, _y, x):
        return dict(_x = _y * (x > 0) + -_y * (x < 0))

    def jvp(self, x_, x):
        return dict(y_ = x_ * (x > 0) + -x_ * (x < 0))

@operator
class add(binary):
    def apl(self, x1, x2):
        return dict(y = x1 + x2)

    def vjp(self, _y):
        return dict(_x1 = _y, _x2 = _y)

    def jvp(self, x1_, x2_):
        return dict(y_ = x1_ + x2_)

@operator
class sub(binary):
    def apl(self, x1, x2):
        return dict(y = x1 - x2)

    def vjp(self, _y):
        return dict(_x1 = _y, _x2 = -_y)

    def jvp(self, x1_, x2_):
        return dict(y_ = x1_ + x2_)

@operator
class mul(binary):
    def apl(self, x1, x2):
        return dict(y = x1 * x2)

    def rcd(self, x1, x2, y):
        from vmad.core.symbol import Literal
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
class div(binary):
    def apl(self, x1, x2):
        x2inv = 1 / x2
        return dict(y = x1 * x2inv, x2inv=x2inv)

    def rcd(self, x1, x2inv, y, x2):
        from vmad.core.symbol import Literal
        # the other value is not needed, 0 should work.
        if isinstance(self.varin['x1'].symbol, Literal):
            x2inv = 0
        else:
            x2inv = 1 / x2
        if isinstance(self.varin['x2'].symbol, Literal):
            x1 = 0
        return dict(x1=x1, x2inv=x2inv)

    def vjp(self, _y, x1, x2inv):
        return dict(_x1 = _y * x2,
                    _x2 = -_y * x1 * (x2inv * x2inv))

    def jvp(self, x1_, x2_, x1, x2inv):
        return dict(y_ = x1_ * x2inv - x1 * (x2inv * x2inv) * x2_)

# only supports unary pow -- now gradient against n yet.
@operator
class pow(unary):
    def apl(self, x, n):
        return dict(y=x ** n)

    def vjp(self, _y, x, n):
        fac = x ** (n - 1) if n != 1 else 1
        return dict(_x = n * _y * fac)

    def jvp(self, x_, x, n):
        fac = x ** (n - 1) if n != 1 else 1
        return dict(y_ = n * x_ * fac)
