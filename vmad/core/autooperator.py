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
        if hasattr(self, '__bound_model__'):
            m = self.__bound_model__
            init = [(a, kwargs[a]) for a in self.ain.keys()]
        else:
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

        v =    [(a, kwargs[a]) for a in self.ain.keys() if a.startswith('_')]

        vjp = tape.get_vjp()
        vjpvout = tape.get_vjp_vout()
        vjpout = vjp.compute(vjpvout, init=v)
        return dict(zip(vjpvout, vjpout))

    def jvp(self, **kwargs):
        tape = kwargs['##tape']

        v =    [(a, kwargs[a]) for a in self.ain.keys() if a .endswith('_')]

        jvp = tape.get_jvp()
        jvpvout = tape.get_jvp_vout()
        jvpout = jvp.compute(jvpvout, init=v)
        return dict(zip(jvpvout, jvpout))

    kls._apl = _make_primitive(kls, 'apl', apl, argnames=argnames, record_impl=rcd)
    kls._vjp = _make_primitive(kls, 'vjp', vjp, argnames=argnames_vjp)
    kls._jvp = _make_primitive(kls, 'jvp', jvp, argnames=argnames_jvp)

    # FIXME: add docstring / argnames
    # shall be the list of extra args
    def build(**kwargs):
        """ Create a computing graph model for the operator with the given hyper parameters """
        for argname in kwargs:
            if argname in kls.ain:
                raise ModelError("argname %s is an input, shall not be used to produce a model" % argname)

        return _build(kwargs)

    def bind(**kwargs):
        """ Create a bound autooperator where the hyperparameter are already given.

            Instantiating the returned operator no longer requires the hyperparameters.
        """
        for argname in kwargs:
            if argname in kls.ain:
                raise ModelError("argname %s is an input, shall not be used to produce a model" % argname)

        m = _build(kwargs)
        def build(): return m
        return type(kls.__name__, (Operator, kls, kls._apl), {'__bound_model__' : m, 'build': staticmethod(build)})

    return type(kls.__name__, (Operator, kls, kls._apl), {'build' : staticmethod(build), 'bind' : staticmethod(bind)})

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
