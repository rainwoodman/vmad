from vmad.core.model import Builder
from vmad.lib import linalg

from pprint import pprint
import pytest

def test_operator_watchpoint():
    from vmad.core.stdlib import watchpoint
    foo = [0]
    def monitor(x):
        foo[0] = 1

    with Builder() as m:
        a = m.input('a')
        b = linalg.add(a, a)
        watchpoint(a, monitor=monitor)
        m.output(c=b)

    init = [('a', 1)]
#    for node in m:
#        print('m', node)
    c, tape = m.compute(init=init, vout='c', return_tape=True)
#    vjp = tape.get_vjp()
#    for node in vjp:
#        print('vjp', node)

    [c], [_a] = m.compute_with_vjp(init=init, v=[('_c', 1.0)])
    assert foo[0] == 1
    assert c == 2
    assert _a == 2.0

    foo[0] = 0
    c, c_ = m.compute_with_jvp(vout='c', init=init, v=[('a_', 1.0)])
    assert foo[0] == 1
    assert c == 2
    assert c_ == 2.0


def test_operator_assert_isinstance():
    from vmad.core.stdlib import assert_isinstance

    with Builder() as m:
        a = m.input('a')
        assert_isinstance(a, int)
        m.output(c=a)

    init = [('a', 1)]
    c, tape = m.compute(init=init, return_tape=True, vout='c')

    [c], [_a] = m.compute_with_vjp(init=init, v=[('_c', 1.0)])
    assert c == 1
    assert _a == 1.0

    c, c_ = m.compute_with_jvp(vout='c', init=init, v=[('a_', 1.0)])
    assert c == 1
    assert c_ == 1.0

    with pytest.raises(TypeError):
        c = m.compute(vout='c', init=dict(a=1.09))

def test_operator_assert_true():
    from vmad.core.stdlib import assert_true

    with Builder() as m:
        a = m.input('a')
        assert_true(a, lambda x: isinstance(x, int))
        m.output(c=a)

    with pytest.raises(AssertionError):
        c = m.compute(vout='c', init=dict(a=1.09))

    c = m.compute(vout='c', init=dict(a=1))

def test_div_error():
    from vmad import autooperator
    @autooperator('a->c')
    def test(a, b):
        c= b/a
        return c
    model = test.build(b=3)
    (y, ), (vjp, ) = model.compute_with_vjp(init=dict(a=2), v=dict(_c=1.0))
    assert y == 1.5
    assert vjp == -0.75
