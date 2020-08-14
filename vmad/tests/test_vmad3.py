from vmad import autooperator
from vmad import Builder
from vmad import operator
from vmad.lib.linalg import add

from pprint import pprint

def test_vmad3_functional():
    """ this test demonstrates building a model directly """
    with Builder() as m:
        a, b = m.input('a', 'b')

        t1 = add(a, a)
        t2 = add(b, 0)
        c = add(t1, t2)

        m.output(c=c)

    print("----- model -----")
    pprint(m)
    pprint(m[:])

    print("----- compute -----")
    init = dict(a=3, b=4)

    c = m.compute(init=init, vout='c')
    print(init, c)

    print("----- tape -----")
    init = dict(a=3, b=4)
    c, tape = m.compute(init=init, vout='c', return_tape=True)
    print(init, c)
    pprint(tape)

    print("----- vjp -----")
    vjp = tape.get_vjp()
    pprint(vjp)
    pprint(vjp[:])

    init = dict(_c=1.0)
    _a, _b = vjp.compute(init=init, vout=['_a', '_b'], monitor=print)
    print('_a, _b = ', _a, _b)

    print("----- jvp -----")
    jvp = tape.get_jvp()
    pprint(jvp)
    pprint(jvp[:])

    init = dict(a_=1.0, b_=1.0)
    c_, = jvp.compute(init=init, vout=['c_'], monitor=print)
    print('c_ = ', c_)

def test_autooperator():
    """ this test demonstrates using autooperator
        to define an operator based on a model, whose
        vjp and jvp are obtained via autodiff.
    """
    @autooperator
    class mymodel:
        ain = {'a' : '*',
               'b' : '*'}
        aout = {'c' : '*'}

        def main(model, a, b, n):
            for i in range(n):
                a = add(a, a)

            t2 = add(b, 0)
            return dict(c=add(a, t2))

    init = dict(a=3, b=4)

    m3 = mymodel.build(n=3)
    print("----- model 2-----")
    pprint(m3)
    c = m3.compute(init=init, vout='c')
    assert c == 3 * 2 ** 3 + 4

    m2 = mymodel.build(n=2)
    print("----- model 3-----")
    pprint(m2)
    c = m2.compute(init=init, vout='c')
    assert c == 3 * 2 ** 2 + 4

    # complicated model is longer
    assert len(m3[:]) > len(m2[:])

