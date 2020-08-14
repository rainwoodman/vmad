from vmad.core.model import Builder
from vmad.core.model import Model
from vmad.lib.linalg import add
from vmad.lib.linalg import mul

from pprint import pprint
import pytest

def test_model_build():
    from vmad import autooperator, operator
    from vmad.core.symbol import Literal

    @autooperator('Sl->Xh')
    def upsample2(Sl, Ql):  #pm has the same resolution as Ql

        layout = add(Ql, 1)
        print(id(Model.get_model(layout)), len(Model.get_model(layout)))
        Ql1 = add(Ql, layout)
        print(id(Model.get_model(layout)), id(Model.get_model(Ql1)), len(Model.get_model(layout)))
        Sl1 = add(Sl, layout)
        print(id(Model.get_model(layout)), id(Model.get_model(Ql1)), len(Model.get_model(layout)))

        dis_d = add(Ql1, Sl1)
        print(id(Model.get_model(layout)), id(Model.get_model(Ql1)), len(Model.get_model(layout)))

        return dis_d

    Sl = 1
    model2 = upsample2.build(Ql=Sl)

    Xh = model2.compute('Xh', init=dict(Sl=Sl))
    (y, ), [vjp] = model2.compute_with_vjp(init=dict(Sl=Sl), v=dict(_Xh=1))

def test_model_partial():
    with Builder() as m:
        a = m.input('a')
        t1 = add(x1=a, x2=a)
        m.output(c=t1)

    init = dict(a=3)

    c, tape = m.compute(init=init, vout='c', return_tape=True)
    assert c == 6

    vjp = tape.get_vjp()
    init = dict(_c=1.0)
    _a = vjp.compute(init=init, vout='_a', monitor=print)
    assert _a == 2.0

    jvp = tape.get_jvp()
    init = dict(a_=1.0)
    c_ = jvp.compute(init=init, vout='c_', monitor=print)
    assert c_ == 2.0

def test_model_unused():
    with Builder() as m:
        a, b = m.input('a', 'b')
        # use a twice with a dependency
        # triggers problem with last_ref in autodiff;
        # because this line is not executed by the context;
        # last_ref is not True for the last ref on the tape.
        d = (a + b) + a
        m.output(c=1.0)

    init = dict(a=3, b=4)
    c, tape = m.compute(init=init, vout='c', return_tape=True)
    assert c == 1.0

    vjp = tape.get_vjp()
    init = dict(_c=1.0)
    _a, _b = vjp.compute(init=init, vout=['_a', '_b'], monitor=print)
    assert _a == 0
    assert _b == 0

    jvp = tape.get_jvp()
    init = dict(a_=1.0, b_=1.0)
    c_ = jvp.compute(init=init, vout='c_', monitor=print)
    assert c_ == 0

def test_model_partial_out():
    with Builder() as m:
        a = m.input('a')
        t1 = add(x1=a, x2=a)
        m.output(c=t1)
        m.output(a=a)

    init = dict(a=3)

    (a, c), tape = m.compute(init=init, vout=['a', 'c'], return_tape=True)
    assert c == 6
    assert a == 3

    vjp = tape.get_vjp()

    # test two outputs individually
    init = dict(_c=1.0, _a=0.0)
    _a = vjp.compute(init=init, vout='_a', monitor=print)
    assert _a == 2.0

    init = dict(_c=0.0, _a=1.0)
    _a = vjp.compute(init=init, vout='_a', monitor=print)
    assert _a == 1.0

    jvp = tape.get_jvp()
    init = dict(a_=1.0)
    a_, c_ = jvp.compute(init=init, vout=['a_', 'c_'], monitor=print)
    assert c_ == 2.0
    assert a_ == 1.0

def test_model_many_rewrites():
    # this is a nasty model with many variable rewrites.
    n = 2
    with Builder() as m:
        x = m.input('x')
        for i in range(2):
            x = add(x1=x, x2=x)

        m.output(y=x)

    init = dict(x=1.0)
    y, tape = m.compute(init=init, vout='y', return_tape=True)
    assert y == 4.0

    vjp = tape.get_vjp()
    init = dict(_y = 1.0)
    _x = vjp.compute(init=init, vout='_x', monitor=print)
    assert _x == 4.0

def test_model_compute_with_vjp():
    with Builder() as m:
        a = m.input('a')
        t1 = add(x1=a, x2=a)
        m.output(b=t1)

    init = [('a', 1)]
    [b], [_a] = m.compute_with_vjp(init=init, v=[('_b', 1.0)])
    assert b == 2.0
    assert _a == 2.0


def test_model_compute_with_jvp():
    with Builder() as m:
        a = m.input('a')
        t1 = add(x1=a, x2=a)
        m.output(b=t1)

    init = [('a', 1)]
    b, b_ = m.compute_with_jvp(vout='b', init=init, v=[('a_', 1.0)])
    assert b == 2.0
    assert b_ == 2.0

def test_model_compute_with_gnDp():
    with Builder() as m:
        a = m.input('a')
        t1 = mul(x1=a, x2=a)
        m.output(b=t1)

    init = [('a', 1)]
    b, [_a_] = m.compute_with_gnDp(
                            vout='b',
                            init=init,
                            v=[('a_', 1.0)],
                            )
    assert b == 1.0
    assert _a_ == 4.0

def test_model_attr():
    import numpy
    with Builder() as m:
        a, b = m.input('a', 'b')
        d = add(x1=b, x2=1)
        t1 = add(x1=a, x2=b.eval(lambda b: b.size))
        m.output(c=t1)

    init = dict(a=2, b=numpy.array([2,]))

    c, tape = m.compute(init=init, vout='c', return_tape=True)
    assert c == 3

    vjp = tape.get_vjp()
    init = dict(_c=1.0)
    _a = vjp.compute(init=init, vout='_a', monitor=print)
    assert _a == 1.0

    jvp = tape.get_jvp()
    init = dict(a_=1.0)
    c_ = jvp.compute(init=init, vout='c_', monitor=print)
    assert c_ == 1.0

