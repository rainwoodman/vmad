from .operator import _make_primitive, Operator, unbound, _to_ordereddict
from .model import Builder
from .context import Context
from .error import ModelError

class AutoOperator(Operator):

    def __init__(self, prototype, argnames):
        Operator.__init__(self, prototype)

        impl = unbound(prototype.main)

        # use the argnames of main function
        if argnames is None:
            argnames = impl.__code__.co_varnames[1:impl.__code__.co_argcount]

        self.argnames = argnames
        self.hyperargs = {}

    @property
    def apl(self):
        def apl(node, **kwargs):
            y = _compute(node.operator, kwargs)
            return y

        def rcd(node, **kwargs):
            return kwargs

        return _make_primitive(self, 'apl', apl, argnames=self.argnames, record_impl=rcd)

    @property
    def vjp(self):
        # add v arguments
        argnames_vjp = list(self.argnames)
        argnames_vjp.append('##tape')
        for argname in self.aout:
            argnames_vjp.append('_' + argname)

        def vjp(node, **kwargs):
            tape = kwargs['##tape']

            v =    [(a, kwargs[a]) for a in node.ain.keys() if a.startswith('_')]

            vjp = tape.get_vjp()
            vjpvout = tape.get_vjp_vout()
            vjpout = vjp.compute(vjpvout, init=v)
            return dict(zip(vjpvout, vjpout))

        return _make_primitive(self, 'vjp', vjp, argnames=argnames_vjp)

    @property
    def jvp(self):
        argnames_jvp = list(self.argnames)
        argnames_jvp.append('##tape')

        for argname in self.ain:
            argnames_jvp.append(argname + '_')

        def jvp(node, **kwargs):
            tape = kwargs['##tape']

            v =    [(a, kwargs[a]) for a in node.ain.keys() if a .endswith('_')]

            jvp = tape.get_jvp()
            jvpvout = tape.get_jvp_vout()
            jvpout = jvp.compute(jvpvout, init=v)
            return dict(zip(jvpvout, jvpout))

        return _make_primitive(self, 'jvp', jvp, argnames=argnames_jvp)

    # FIXME: add docstring / argnames
    # shall be the list of extra args
    def build(obj, **kwargs):
        """ Create a computing graph model for the operator with the given hyper parameters """
        for argname in kwargs:
            if argname in obj.ain:
                raise ModelError("argname %s is an input, shall not be used to produce a model" % argname)

        return _build(obj, kwargs)

    def precompute(obj, **kwargs):
        """ Create a bound autooperator where the hyperparameter and parameters are both given; the model
            is compuated, and the tape is recorded.

            Instantiating the returned operator will be an exact replay of the tape, regardless of the parameters
        """
        y = _compute(obj, kwargs)
        m = _build(obj, kwargs)
        tape = y['##tape']

        # create a new operator, because we need new primitives that points to this operator.
        obj = autooperator(obj.prototype)
        obj.__bound_tape__ = tape
        obj.__bound_model__ = m

        return obj

    def bind(obj, **hyperargs):
        """ Create a bound autooperator where the hyperparameter are already given.

            Instantiating the returned operator no longer requires the hyperparameters.
        """
        for argname in hyperargs:
            if argname in obj.ain:
                raise ModelError("argname %s is an input, shall not be used to produce a model" % argname)

        m = _build(obj, hyperargs)

        # remove those args already defined
        argnames = tuple([
                argname for argname in obj.apl.argnames if argname not in hyperargs
                ])

        # create a new operator, because we need new primitives that points to this operator.
        obj = autooperator(obj.prototype, argnames=argnames)
        obj.__bound_model__ = m
        obj.hyperargs = hyperargs
        obj.argnames = argnames

        return obj

def autooperator(kls, argnames=None):
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

    obj = AutoOperator(kls, argnames)

    return obj

def _build(obj, kwargs):
    if hasattr(obj, '__bound_model__'):
        m = obj.__bound_model__
        return m

    impl = unbound(obj.prototype.main)

    model_args = {}
    # copy extra args of the main function
    for argname in obj.apl.argnames:
        if argname not in obj.ain:
            model_args[argname] = kwargs[argname]

    with Builder() as m:
        # add input args as variables
        for argname in obj.ain:
            model_args[argname] = m.input(argname)
        r = impl(m, **model_args)
        # assert outputs are generated
        for argname in obj.aout:
            if argname not in r:
                raise ModelError("output arg '%s' is not produced by the model" % argname)
        m.output(**r)
    return m

def _compute(obj, kwargs):
    if hasattr(obj, '__bound_tape__'):
        tape = obj.__bound_tape__
        y = {}
        y.update(tape.out)
    else :
        m = _build(obj, kwargs)
        init = [(a, kwargs[a]) for a in obj.ain.keys()]

        vout = list(obj.aout.keys())
        y, tape = m.compute(vout, init=init, return_dict=True, return_tape=True)
    y['##tape'] = tape
    return y



