from __future__ import print_function

from pprint import pprint
from vmad.core.model import Builder
import pytest

def test_operator_watchpoint():
    from vmad.lib.utils import watchpoint
    foo = [0]
    def monitor(x):
        foo[0] = 1

    with Builder() as m:
        a = m.input('a')
        watchpoint(a, monitor=monitor)
        m.output(c=a)
    init = [('a', 1)]

    [c], [_a] = m.compute_with_vjp(init=init, v=[('_c', 1.0)])
    assert foo[0] == 1
    assert c == 1
    assert _a == 1.0

    foo[0] = 0
    c, c_ = m.compute_with_jvp(vout='c', init=init, v=[('a_', 1.0)])
    assert foo[0] == 1
    assert c == 1
    assert c_ == 1.0

def test_operator_assert_isinstance():
    from vmad.lib.utils import assert_isinstance

    with Builder() as m:
        a = m.input('a')
        assert_isinstance(a, int)
        m.output(c=a)
    init = [('a', 1)]

    [c], [_a] = m.compute_with_vjp(init=init, v=[('_c', 1.0)])
    assert c == 1
    assert _a == 1.0

    c, c_ = m.compute_with_jvp(vout='c', init=init, v=[('a_', 1.0)])
    assert c == 1
    assert c_ == 1.0

    with pytest.raises(TypeError):
        c = m.compute(vout='c', init=dict(a=1.09))
