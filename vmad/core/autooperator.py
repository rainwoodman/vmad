from .error import ModelError
from .model import Builder
from .operator import BaseOperator
from .operator import _make_primitive
from .operator import unbound

import inspect

class AutoOperator(BaseOperator):
    """ Base class to support operators with on demand tape and prerecorded tape """

    def __init__(self, prototype, argnames):
        BaseOperator.__init__(self, prototype)

        impl = unbound(prototype.main)

        # use the argnames of main function
        if argnames is None:
            argnames = impl.__code__.co_varnames[1:impl.__code__.co_argcount]

        self.argnames = argnames
        self.hyperargs = {}
        self.__bound_tape__ = None
        self.__bound_model__ = None
        self.__tapename__ = '$$tape$$'

    @property
    def forgetful(self):
        obj = AutoOperator(self.prototype, self.argnames)
        obj.__bound_tape__ = self.__bound_tape__
        obj.__bound_model__ = self.__bound_model__
        obj.__tapename__ = None

        return obj

    @property
    def apl(self):
        return self.get_apl(tapename=self.__tapename__)

    def get_apl(self, tapename):
        """ get an apl primitive, if tapename is not None, the primitive returns the tape as well. """
        def apl(node, **kwargs):
            # node.operator shall be the same as self.
            y = node.operator._compute(kwargs, tapename=tapename)
            return y

        def rcd(node, **kwargs):
            return kwargs

        outnames = list(self.aout.keys())

        return _make_primitive(self, 'apl', apl, argnames=self.argnames, outnames=outnames, record_impl=rcd)

    @property
    def vjp(self):
        return self.get_vjp(tapename=self.__tapename__)

    def get_vjp(self, tapename):
        # add v arguments
        argnames_vjp = list(self.argnames)
        if tapename is not None:
            argnames_vjp.append(tapename)

        for argname in self.aout:
            argnames_vjp.append('_' + argname)

        def vjp(node, **kwargs):
            if tapename is not None:
                tape = kwargs.pop(tapename)
            else:
                # run the apl operator to obtain the tape
                y = node.operator._compute(kwargs, '$$tape$$')
                tape = y['$$tape$$']

            v =    [(a, kwargs[a]) for a in node.primitive.ain.keys() if a.startswith('_')]

            vjp = tape.get_vjp()
            vjpvout = tape.get_vjp_vout()
            vjpout = vjp.compute(vjpvout, init=v)
            return dict(zip(vjpvout, vjpout))

        return _make_primitive(self, 'vjp', vjp, argnames=argnames_vjp)

    @property
    def jvp(self):
        return self.get_jvp(tapename=self.__tapename__)

    def get_jvp(self, tapename):
        argnames_jvp = list(self.argnames)
        if tapename is not None:
            argnames_jvp.append(tapename)

        for argname in self.ain:
            argnames_jvp.append(argname + '_')

        def jvp(node, **kwargs):
            if tapename is not None:
                tape = kwargs[tapename]
            else:
                # run the apl operator to obtain the tape
                y = node.operator._compute(kwargs, '$$tape$$')
                tape = y['$$tape$$']

            v =    [(a, kwargs[a]) for a in node.primitive.ain.keys() if a .endswith('_')]

            jvp = tape.get_jvp()
            jvpvout = tape.get_jvp_vout()
            jvpout = jvp.compute(jvpvout, init=v)
            return dict(zip(jvpvout, jvpout))

        return _make_primitive(self, 'jvp', jvp, argnames=argnames_jvp)

    def _compute(self, kwargs, tapename):
        if self.__bound_tape__ is not None:
            tape = self.__bound_tape__
            y = {}
            y.update(tape.out)
        else :
            m = self._build(kwargs)
            init = [(a, kwargs[a]) for a in self.ain.keys()]

            vout = list(self.aout.keys())
            if tapename is not None:
                y, tape = m.compute(vout, init=init, return_dict=True, return_tape=True)
            else:
                y = m.compute(vout, init=init, return_dict=True, return_tape=False)

        if tapename is not None:
            y[tapename] = tape

        return y

    def _build(self, kwargs):
        if self.__bound_model__ is not None:
            m = self.__bound_model__
            return m

        impl = unbound(self.prototype.main)

        model_args = {}
        # copy extra args of the main function
        for argname in self.apl.argnames:
            if argname not in self.ain:
                model_args[argname] = kwargs[argname]

        with Builder() as m:
            # add input args as variables
            for argname in self.ain:
                model_args[argname] = m.input(argname)
            r = impl(m, **model_args)
            # assert outputs are generated
            for argname in self.aout:
                if argname not in r:
                    raise ModelError("output arg '%s' is not produced by the model" % argname)
            m.output(**r)
        return m

    # FIXME: add docstring / argnames
    # shall be the list of extra args
    def build(__instance__1234__, **kwargs):
        """ Create a computing graph model for the operator with the given hyper parameters """
        self = __instance__1234__
        for argname in kwargs:
            if argname in self.ain:
                raise ModelError("argname %s is an input, shall not be used to produce a model" % argname)

        return self._build(kwargs)


    def precompute(self, **kwargs):
        """ Create a bound autooperator where the hyperparameter and parameters are both given; the model
            is compuated, and the tape is recorded.

            Instantiating the returned operator will be an exact replay of the tape, regardless of the parameters
        """
        y = self._compute(kwargs, '$$tape$$')
        m = self._build(kwargs)
        tape = y['$$tape$$']

        # create a new operator, because we need new primitives that points to this operator.
        obj = AutoOperator(self.prototype, argnames=self.argnames)
        obj.__bound_tape__ = tape
        obj.__bound_model__ = m

        return obj

    def bind(self, **hyperargs):
        """ Create a bound autooperator where the hyperparameter are already given.

            Instantiating the returned operator no longer requires the hyperparameters.
        """
        for argname in hyperargs:
            if argname in self.ain:
                raise ModelError("argname %s is an input, shall not be used to produce a model" % argname)

        m = self._build(hyperargs)

        # remove those args already defined
        argnames = tuple([
                argname for argname in self.apl.argnames if argname not in hyperargs
                ])

        # create a new operator, because we need new primitives that points to this operator.
        obj = AutoOperator(self.prototype, argnames=argnames)
        obj.__bound_model__ = m
        obj.hyperargs = hyperargs

        return obj

def _autograd(func, ain, aout):

    def main(__unused_argument_3333__, *args, **kwargs):
        kwargs = kwargs.copy()
        # merge the positional arguments
        kwargs.update(dict(zip(ain, args)))

        # always call with the kwargs syntax, such that
        r = func(**kwargs)
        # normalize the output to a dict.
        if isinstance(r, tuple):
            r = dict(zip(aout, r))
        elif not isinstance(r, dict) and len(aout) == 1:
            r = dict(zip(aout, [r]))

        return r

    argnames = func.__code__.co_varnames[:func.__code__.co_argcount]
    prototype = type(func.__name__, (),
            dict(
                ain=ain,
                aout=aout,
                main=main,
                )
            )
    return AutoOperator(prototype, argnames=argnames)

def _parse_annotations(func):
    annotations = getattr(func, '__annotations__', {})
    if len(annotations) == 0:
        raise ValueError("Must provide ain, aout or function annotations. see docstring of autograd")
    ain = [
        (key, value)
        for key, value in annotations.items() if key != 'return'
    ]
    aout = annotations['return']

    return ain, aout, func

def _parse_autograd_spec(args):
    if len(args) == 1:
        if hasattr(args[0], '__call__'):
            return _parse_annotations(args[0])

        spec = args[0]
        split = spec.index('->')
        left = spec[:split]
        right = spec[split+2:]

        ain = [a.strip() for a in left.split(',')]
        aout = [a.strip() for a in right.split(',')]
    else:
        split = args.index('->')
        ain = args[:split]
        aout = args[split + 1:]
    return ain, aout, None

def autooperator(*args):
    """ autooperator creates a operator from a prototype class
        or a function that uses vmad to build a model.

        The function or the main method must only use vmad operators
        on the input autooperator symbols.

        1. Use Python 3 annotation syntax to declare
        the input and output arguments.

        .. code ::

            @autooperator
            def function(x : '*', y : '*', n) -> 'z':
                ...
                return z

        2. Explicitly provide the lis of input and output
           arguments.

        .. code ::

            @autooperator('x', 'y', '->', 'z')
            def function(x, y, n):
                ...
                return z

        or 

        .. code ::

            @autooperator('x, y->z')
            def function(x, y, n):
                ...
                return z

        or
        .. code ::

            @autooperator('x, y->z')
            def function(n, **kwargs):
                ...
                return z

        3. Use a class
         Create an operator with automated differentiation.

        ain : input arguments
        aout : output arguments

        main : function(model, ...) building the model;
                the arguments are the input symbols
                returns the dict of output symbols,
                shall match the output arguments

        see the example below in this file.

        .. code ::
            @autooperator
            class function:
                ain = 'x', 'y'
                aout = 'z'
                def main(model, x, y, n):
                    ...
                    return dict(z = ....)

    """
    if inspect.isclass(args[0]):
        return AutoOperator(args[0], argnames=None)

    ain, aout, func = _parse_autograd_spec(args)
    def wrapped(func):
        return _autograd(func, ain, aout)

    if func is None:
        return wrapped
    else:
        return wrapped(func)

