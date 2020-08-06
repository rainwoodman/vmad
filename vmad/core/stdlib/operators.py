from vmad.core.operator import operator

class unary:
    ain = 'x'
    aout = 'y'

class binary:
    ain = 'x1', 'x2'
    aout = 'y'


@operator
class neg(unary):
    def apl(node, x):
        return dict(y = -x)

    def vjp(node, _y):
        return dict(_x = -_y)

    def jvp(node, x_):
        return dict(y_ = -x_)

@operator
class pos(unary):
    def apl(node, x):
        return dict(y = +x)

    def vjp(node, _y):
        return dict(_x = +_y)

    def jvp(node, x_):
        return dict(y_ = +x_)

@operator
class abs(unary):
    def apl(node, x):
        # function name was replaced
        from builtins import abs
        return dict(y = abs(x))

    def vjp(node, _y, x):
        return dict(_x = _y * (x > 0) + -_y * (x < 0))

    def jvp(node, x_, x):
        return dict(y_ = x_ * (x > 0) + -x_ * (x < 0))

@operator
class add(binary):
    def apl(node, x1, x2):
        return dict(y = x1 + x2)

    def vjp(node, _y):
        return dict(_x1 = _y, _x2 = _y)

    def jvp(node, x1_, x2_):
        return dict(y_ = x1_ + x2_)

@operator
class sub(binary):
    def apl(node, x1, x2):
        return dict(y = x1 - x2)

    def vjp(node, _y):
        return dict(_x1 = _y, _x2 = -_y)

    def jvp(node, x1_, x2_):
        return dict(y_ = x1_ - x2_)

@operator
class mul(binary):
    def apl(node, x1, x2):
        return dict(y = x1 * x2)

    def rcd(node, x1, x2, y):
        # the other value is not needed, 0 should work.
        if node.is_literal('x1'):
            x2 = 0
        if node.is_literal('x2'):
            x1 = 0
        return dict(x1=x1, x2=x2)

    def vjp(node, _y, x1, x2):
        return dict(_x1 = _y * x2,
                    _x2 = _y * x1)

    def jvp(node, x1_, x2_, x1, x2):
        return dict(y_ = x1_* x2 + x1 * x2_)

@operator
class div(binary):
    def apl(node, x1, x2):
        x2inv = 1 / x2
        return dict(y = x1 * x2inv, x2inv=x2inv)

    def rcd(node, x1, x2inv, y, x2):
        from vmad.core.symbol import Literal
        # the other value is not needed, 0 should work.
        if node.is_literal('x1'):
            x1fac = 0
        else:
            x1fac = 1 / x2
        if node.is_literal('x2'):
            x2fac = 0
        else:
            x2fac = x1 * (1 / x2) * (1 / x2)
        return dict(x1fac=x1fac, x2fac=x2fac)

    def vjp(node, _y, x1fac, x2fac):
        return dict(_x1 = _y * x1fac,
                    _x2 = -_y * x2fac)

    def jvp(node, x1_, x2_, x1fac, x2fac):
        return dict(y_ = x1_ * x1fac - x2fac * x2_)

@operator
class mod(binary):
    def apl(node, x1, x2):
        return dict(y = x1 % x2, n = x1 // x2)

    def vjp(node, _y, x1, x2, n):
        return dict(_x1 = _y,
                    _x2 = (- n) * _y)

    def jvp(node, x1_, x2_, x1, x2, n):
        return dict(y_ = x1_ - n * x2_)

@operator
class pow(binary):
    def apl(node, x1, x2):
        if not node.is_literal('x2'):
            from numpy import log
            logx1 = log(x1)
        else:
            # no need for logx2, as _x2 will be zeros.
            logx1 = 0

        return dict(y=x1 ** x2, logx1=logx1)

    def vjp(node, _y, x1, x2, logx1):
        fac = x1 ** (x2 - 1) if x2 != 1 else 1
        return dict(_x1 = x2 * _y * fac,
                    _x2 = _y * x1**x2 * logx1)

    def jvp(node, x1_, x2_, x1, x2, logx1):
        fac = x1 ** (x2 - 1) if x2 != 1 else 1
        return dict(y_ = x2 * x1_ * fac + x2_ * x1**x2 * logx1)


