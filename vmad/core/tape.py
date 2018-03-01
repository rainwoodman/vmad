class Record(object):
    """ A record on the tape. 

        A record contains the node and the resolved arg symbols to the node.

    """
    def __init__(self, node, impl_kwargs):
        self.node = node
        self.impl_kwargs = impl_kwargs

    def __repr__(self):
        return '%s / %s' % (self.node, self.impl_kwargs)

class Tape(list):
    def __init__(self, model):
        self.model = model

    def append(self, node, impl_kwargs):
        list.append(self, Record(node, impl_kwargs))

    def get_vjp(self):
        # to avoid cicurlar reference; this is not a strong dependency
        from .autodiff import vjpmodel
        return vjpmodel(self)

    def get_jvp(self):
        # to avoid cicurlar reference; this is not a strong dependency
        from .autodiff import jvpmodel
        return jvpmodel(self)

    def compute_jvjp(self, vout, aout, init):
        jvp = self.get_jvp()
        aout_ = [a + '_' for a in aout]
        t = jvp.compute(aout_, init)
        vjp = self.get_vjp()
        p = vjp.compute(vout, init=dict([('_' + a, t1) for a, t1 in zip(aout, t)]))
        return p

