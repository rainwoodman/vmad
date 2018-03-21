from .operator import _make_primitive, Operator, unbound
from .model import Builder
from .context import Context
from .error import ModelError

def autooperator(kls):
    """ Create an operator with automated differentiation.

        ain : input arguments
        aout : output arguments

        main : function(model, ...) building the model;
                the arguments are the input symbols
                returns the dict of output symbols,
                shall match the output arguments

        see the example below in this file.
    """

    from collections import OrderedDict

    impl = unbound(kls.main)

    # use the argnames of main function
    argnames = impl.__code__.co_varnames[1:impl.__code__.co_argcount]
    argnames_vjp = list(argnames)
    argnames_jvp = list(argnames)

    argnames_vjp.append('##tape')
    argnames_jvp.append('##tape')

    kls.ain = OrderedDict(kls.ain)
    kls.aout = OrderedDict(kls.aout)

    # add v arguments
    for argname in kls.aout:
        argnames_vjp.append('_' + argname)
    for argname in kls.ain:
        argnames_jvp.append(argname + '_')

    def _build(kwargs):

        model_args = {}
        # copy extra args of the main function
        for argname in argnames:
            if argname not in kls.ain:
                model_args[argname] = kwargs[argname]

        with Builder() as m:
            # add input args as variables
            for argname in kls.ain:
                model_args[argname] = m.input(argname)
            r = impl(m, **model_args)
            # assert outputs are generated
            for argname in kls.aout:
                if argname not in r:
                    raise ModelError("output arg '%s' is not produced by the model" % argname)
            m.output(**r)
        return m

    def apl(self, **kwargs):
        m = _build(kwargs)
        init = [(a, kwargs[a]) for a in self.ain.keys()]
        vout = list(self.aout.keys())
        y, tape = m.compute(vout, init=init, return_dict=True, return_tape=True)
        y['##tape'] = tape
        return y

    def rcd(self, **kwargs):
        return kwargs

    def vjp(self, **kwargs):
        tape = kwargs['##tape']
        m = _build(kwargs)

        init = [(a.name, kwargs[a.name]) for a in m._vin]
        vout = [var.name for var in m._vout]
        v =    [(a, kwargs[a]) for a in self.ain.keys() if a.startswith('_')]
        y, vjp = m.compute_with_vjp(init=init, v=v, return_dict=True)
        return vjp

    def jvp(self, **kwargs):
        tape = kwargs['##tape']
        m = _build(kwargs)
        init = [(a.name, kwargs[a.name]) for a in m._vin]
        vout = [var.name for var in m._vout]
        v =    [(a, kwargs[a]) for a in self.ain.keys() if a .endswith('_')]
        y, jvp = m.compute_with_jvp(vout, init=kwargs, v=v, return_dict=True)
        return jvp

    kls._apl = _make_primitive(kls, 'apl', apl, argnames=argnames, record_impl=rcd)
    kls._vjp = _make_primitive(kls, 'vjp', vjp, argnames=argnames_vjp)
    kls._jvp = _make_primitive(kls, 'jvp', jvp, argnames=argnames_jvp)

    # FIXME: add docstring / argnames
    # shall be the list of extra args
    def build(**kwargs):
        for argname in kwargs:
            if argname in kls.ain:
                raise ModelError("argname %s is an input, shall not be used to produce a model" % argname)

        return _build(kwargs)

    return type(kls.__name__, (Operator, kls, kls._apl), {'build' : staticmethod(build)})

@autooperator
class example:
    ain  = {'x': '*'}
    aout = {'y': '*'}

    # must take both extra parameters and input parameters
    def main(self, x, n):
        from .operator import add

        for i in range(n):
            x = add(x1=x, x2=x)
        return dict(y=x)

