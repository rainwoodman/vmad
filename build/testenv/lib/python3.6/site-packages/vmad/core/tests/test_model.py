from __future__ import print_function
from pprint import pprint
from vmad.core.operator import add
from vmad.core.model import Builder
import pytest

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

