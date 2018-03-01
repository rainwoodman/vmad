
def example():
    graph = []

    graph.append(add(x1='a', x2='b', y='c'))

    c, tape = compute(graph, vout='c', init=dict(a=10, b=20), return_tape=True)

from vmad.core.registry import Registry

def test_registry():
    reg = Registry()

    reg.register("add***", dict(x="*", y="*", z='*'), name='add')
    reg.register("add**", dict(x="*", y="*"), name='add')
    reg.register("addii", dict(x=int, y=int), name='add')
    reg.register("addi*", dict(x=int, y="*"), name='add')
    reg.register("add*i", dict(x="*", y=int), name='add')


    assert reg.match("add", dict(x=1, y=2., z="12")) == 'add***'
    assert reg.match("add", dict(x=1, y=2.)) == 'addi*'
    assert reg.match("add", dict(x=1, y=1 )) == 'addii'
    assert reg.match("add", dict(x="s", y=1 )) == 'add*i'
    assert reg.match("add", dict(x="s", y="s" )) == 'add**'





