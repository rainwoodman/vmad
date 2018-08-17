from __future__ import print_function

from vmad.core.model import Builder
from vmad.core.autooperator import autooperator, autograd
from vmad.lib.linalg import add

@autooperator
class example:
    ain  = {'x': '*'}
    aout = {'y': '*'}

    # must take both extra parameters and input parameters
    def main(self, x, n):
        for i in range(n):
            x = add(x1=x, x2=x)
        return dict(y=x)

def test_autograd_annotations():
    @autograd
    def example_func(x :'*', n) -> 'y':
        return dict(y=x)
    assert 'x' in example_func.ain
    assert 'n' not in example_func.ain
    assert 'y' in example_func.aout

def test_autograd_explicit():
    @autograd('x', '->', 'y')
    def example_func(x, n):
        return dict(y=x)
    assert 'x' in example_func.ain
    assert 'n' not in example_func.ain
    assert 'y' in example_func.aout

def test_model_nested():

    with Builder() as m:
        a = m.input('a')
        b = example(a, 2)
        m.output(b=b)

    init = dict(a = 1.0)
    b, tape = m.compute(init=init, vout='b', monitor=print, return_tape=True)

    assert b == 4.0

    vjp = tape.get_vjp()
    init = dict(_b = 1.0)
    _a = vjp.compute(init=init, vout='_a', monitor=print)
    assert _a == 4.0

    jvp = tape.get_jvp()
    init = dict(a_ = 1.0)
    b_ = jvp.compute(init=init, vout='b_', monitor=print)
    assert b_ == 4.0

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

    assert not hasattr(example, '__bound_model__')
    assert hasattr(op1, '__bound_model__')

    op1.build()

def test_autooperator_compute():

    y = example.build(n=2).compute(vout='y', init=dict(x=1))

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

    assert not hasattr(example, '__bound_tape__')
    assert hasattr(op1, '__bound_tape__')

    op1.build()

