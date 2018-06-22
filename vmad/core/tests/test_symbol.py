from vmad.core.symbol import Symbol
from vmad.core.operator import operator
from vmad import Builder

def test_symbol_eval():
    with Builder() as m:
        a = m.input('a')
        m.output(
                b=a.eval(lambda a: len(a)),
                c=a.eval("len(x)"),
                )

    m.compute(['b', 'c'], init=dict(a=[1, 2, 3]))
    m.compute_with_vjp(init=dict(a=[1, 2, 3]), v=dict(_b=1.0, _c=1.0))
    m.compute_with_jvp(['b', 'c'], init=dict(a=[1, 2, 3]), v=dict(a_=1.0))

def test_symbol_add():
    with Builder() as m:
        a, b = m.input('a', 'b')
        m.output(
                c = (a + b) * b / b ** 2
                )

    c = m.compute('c', init=dict(a=2, b=3))
    assert round(c, 4) == round((2 + 3) * 3 / 3 ** 2, 4)

def test_subclass_symbol():

    class MySymbol(Symbol):
        @operator
        class __add__:
            ain = 'x', 'y'
            aout = 'z'
            def apl(self, x, y):
                return dict(z=x + y)
            def vjp(self, _z):
                return dict(_x=_z, _y=_z)
            def jvp(self, x_, y_):
                return dict(z_ = x_+ y_)

    with Builder() as m:
        a, b = m.input(MySymbol('a'), MySymbol('b'))
        c = a + b
        m.output(c=c)

    m.compute('c', init=dict(a=1, b=2))
    m.compute_with_vjp(init=dict(a=1, b=2), v=dict(_c=1.0))
    m.compute_with_jvp(['c'], init=dict(a=1, b=2), v=dict(a_=1.0, b_=1.0))

