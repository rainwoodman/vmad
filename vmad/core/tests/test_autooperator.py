from __future__ import print_function

from vmad.core.model import Builder
from vmad.core.error import BadArgument
from vmad.core.autooperator import autooperator
from vmad.lib.linalg import add
import pytest

@autooperator
class example:
    ain  = {'x': '*'}
    aout = {'y': '*'}

    # must take both extra parameters and input parameters
    def main(self, x, n):
        for i in range(n):
            x = add(x1=x, x2=x)
        return dict(y=x)

@autooperator('x->y')
def example_func(x, n):
    for i in range(n):
        x = add(x1=x, x2=x)
    return dict(y=x)

def test_autooperator_annotations():
    @autooperator
    def example_func(x :'*', n) -> 'y':
        return dict(y=x)
    assert 'x' in example_func.ain
    assert 'n' not in example_func.ain
    assert 'y' in example_func.aout

    y = example_func.build(n=2).compute(vout='y', init=dict(x=1))

def test_autooperator_explicit():
    @autooperator('x', '->', 'y')
    def example_func(x, n):
        return dict(y=x)
    assert 'x' in example_func.ain
    assert 'n' not in example_func.ain
    assert 'y' in example_func.aout

    y = example_func.build(n=2).compute(vout='y', init=dict(x=1))

def test_autooperator_explicit_short():
    @autooperator('x->y')
    def example_func(x, n):
        return dict(y=x)

    assert 'x' in example_func.ain
    assert 'n' not in example_func.ain
    assert 'y' in example_func.aout

    y = example_func.build(n=2).compute(vout='y', init=dict(x=1))

def test_autooperator_explicit_short_space():
    @autooperator(' x ->y')
    def example_func(x, n):
        return dict(y=x)

    assert 'x' in example_func.ain
    assert 'n' not in example_func.ain
    assert 'y' in example_func.aout

    y = example_func.build(n=2).compute(vout='y', init=dict(x=1))

def test_autooperator_as_member():
    class MyType(object):
        def __init__(self):
            self.n = 3

        @autooperator('x->y')
        def example_func(self, x):
            return dict(y=x * self.n)

    obj = MyType()
    with Builder() as m:
        a = m.input('a')
        b = obj.example_func(a)
        m.output(b=b)

    y = m.compute(vout='b', init=dict(a=1))
    assert y == 3

    y = obj.example_func.build().compute(vout='y', init=dict(x=1.))
    assert y == 3

def test_model_nested():

    with Builder() as m:
        a = m.input('a')
        b = example(a, 2)
        c = example.forgetful(a, 2)
        m.output(b=b, c=c)

    init = dict(a = 1.0)
    (b, c), tape = m.compute(init=init, vout=['b', 'c'], monitor=print, return_tape=True)

    assert b == 4.0
    assert c == 4.0

    vjp = tape.get_vjp()
    init = dict(_b = 1.0, _c=1.0)
    _a = vjp.compute(init=init, vout='_a', monitor=print)
    assert _a == 8.0

    jvp = tape.get_jvp()
    init = dict(a_ = 1.0)
    b_, c_ = jvp.compute(init=init, vout=['b_', 'c_'], monitor=print)
    assert b_ == 4.0
    assert c_ == 4.0

def test_autooperator_additional_hyper():
    @autooperator
    def example_func(x :'*') -> 'y':
        return x

    assert 'x' in example_func.ain
    assert 'y' in example_func.aout

    @autooperator
    def main1(x : '*', n) -> 'y':
        return example_func(x, n=n)

    with pytest.raises(BadArgument):
        y = main1.build(n=2).compute(vout='y', init=dict(x=1))

def test_model_nested_build():

    m = example.build(n=2)
    init = dict(x = 1.0)
    y, tape = m.compute(init=init, vout='y', monitor=print, return_tape=True)
    assert y == 4.0

def test_autooperator_bind():

    op1 = example.bind(n=2)
    m = example.build(n=2)

    assert 'n' in op1.hyperargs
    assert 'n' not in example.hyperargs

    with Builder() as m:
        a = m.input('a')
        b = example(a, 2)
        c = op1(a)
        m.output(b=b, c=c)

    init = dict(a = 1.0)
    (b,c), tape = m.compute(init=init, vout=['b', 'c'], monitor=print, return_tape=True)

    assert b == c
    assert b == 4.0

    assert example.__bound_model__ is None
    assert op1.__bound_model__ is not None

    op1.build()

def test_autooperator_compute():

    y = example.build(n=2).compute(vout='y', init=dict(x=1))

def test_autooperator_forgetful_compute():

    y = example.forgetful.build(n=2).compute(vout='y', init=dict(x=1))

def test_autooperator_precompute():

    op1 = example.precompute(n=2, x=1)
    m = example.build(n=2)

    with Builder() as m:
        a = m.input('a')
        b = example(a, 2)
        c = op1(a, n=2)
        m.output(b=b, c=c)

    init = dict(a = 1.0)
    (b,c), tape = m.compute(init=init, vout=['b', 'c'], monitor=print, return_tape=True)

    assert b == c
    assert b == 4.0

    assert example.__bound_tape__ is None
    assert op1.__bound_tape__ is not None

    op1.build()

def test_autooperator_precompute2():

    op1 = example_func.precompute(n=2, x=1)
    m = example_func.build(n=2)

    with Builder() as m:
        a = m.input('a')
        b = example_func(a, 2)
        c = op1(a, n=2)
        m.output(b=b, c=c)

    init = dict(a = 1.0)
    (b,c), tape = m.compute(init=init, vout=['b', 'c'], monitor=print, return_tape=True)

    assert b == c
    assert b == 4.0

    assert example_func.__bound_tape__ is None
    assert op1.__bound_tape__ is not None

    op1.build()

