from __future__ import print_function

from vmad.core.model import Builder

def test_model_nested():


    from vmad.core.autooperator import example

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

    from vmad.core.autooperator import example

    m = example.build(n=2)
    init = dict(x = 1.0)
    y, tape = m.compute(init=init, vout='y', monitor=print, return_tape=True)
    assert y == 4.0

def test_autooperator_bind():

    from vmad.core.autooperator import example

    op1 = example.bind(n=2)
    m = example.build(n=2)

    assert 'n' in op1.hyperargs

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

def test_autooperator_precompute():

    from vmad.core.autooperator import example

    op1 = example.precompute(n=2, x=1)
    m = example.build(n=2)

    with Builder() as m:
        a = m.input('a')
        b = example(a, 2)
        c = op1(a)
        m.output(b=b, c=c)

    init = dict(a = 1.0)
    (b,c), tape = m.compute(init=init, vout=['b', 'c'], monitor=print, return_tape=True)

    assert b == c
    assert b == 4.0

    assert not hasattr(example, '__bound_tape__')
    assert hasattr(op1, '__bound_tape__')

    op1.build()

