from __future__ import print_function
"""
    Routines to define an operator

    use @operator decorator on a class to define an operator.

    Example: see the source code of :class:`add`

    use @nested.modeloperator to define a nested operator, where
    you only need to define a model and the ain, aout

"""
class EmptyPrimitive:
    argnames = []

def lazyproperty(fn):
    """
        Decorator that makes a property lazy-evaluated.
        This is used to defer the creation of the primitives of
        an operator, such that they do not create an import time
        circular dependency.
    """
    attr_name = '___' + fn.__name__ + '___'

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property

class Operator(object):
    def __init__(self, prototype):
        self.prototype = prototype
        self.ain = _to_ordereddict(prototype.ain)
        self.aout = _to_ordereddict(prototype.aout)


    def __call__(self, *args, **kwargs):
        return self.apl(*args, **kwargs)

    def __get__(self, instance, owner):
        if instance is not None:
            return InstanceOperator(self, instance)
        else:
            return self

    @lazyproperty
    def apl(self):
        if hasattr(self.prototype, 'rcd'):
            record_impl = unbound(self.prototype.rcd)
        else:
            record_impl = record_copy_autodiff

        return _make_primitive(self, 'apl', unbound(self.prototype.apl),
                record_impl=record_impl)

    @lazyproperty
    def vjp(self):
        if hasattr(self.prototype, 'vjp'):
            return _make_primitive(self, 'vjp', unbound(self.prototype.vjp))
        else:
            return EmptyPrimitive

    @lazyproperty
    def jvp(self):
        if hasattr(self.prototype, 'jvp'):
            return _make_primitive(self, 'jvp', unbound(self.prototype.jvp))
        else:
            return EmptyPrimitive

class InstanceOperator(Operator):
    def __init__(self, base, instance):
        self.base = base
        self.instance = instance

    def __call__(self, *args, **kwargs):
        return self.base(self.instance, *args, **kwargs)

    def __getattr__(self, attrname):
        return getattr(self.base, attrname)

def _to_ordereddict(ain):
    from collections import OrderedDict

    # convert ain to an ordered dict,
    # supported syntax:
    # [('a', '*'), ('b', '*')]
    # ['a', ('b', '*')]
    # ['a', 'b']
    # or a dictionary

    if isinstance(ain, (list, tuple)):
        ain = [
            a if isinstance(a, tuple) else (a, '*') for a in ain
        ]
    elif not isinstance(ain, (dict, OrderedDict)):
        # like a scalar, assuming it is string?
        ain = [ (ain, '*') ]

    return OrderedDict(ain)

def record_copy_all(node, **kwargs):
    """ A default rcd implementation that copies all kwargs to the tape.

        the impl is used for the vjp and vjp primitives.

        the impl is can be used for the apl primitives if no rcd is given;
        but we use 'record_copy_autodiff' to save a much smaller subset
        of variables.
    """
    return kwargs

def record_copy_autodiff(node, **kwargs):
    """ A default rcd implementation that copies `useful` kwargs to the tape
        for the autodiff.

        the impl is used for the apl primitives if no rcd is given;
    """
    jvp = node.find_primitive_type('jvp')
    vjp = node.find_primitive_type('vjp')
    record = {}
    for argname, value in kwargs.items():
        if argname in jvp.argnames or argname in vjp.argnames:
            record[argname] = value

    return record

def unbound(method):
    if hasattr(method, 'im_func'):
        # python 2.7 has this unbound method thing
        return method.im_func
    # python 3, cool
    return method

def operator(kls):
    """ Decorator to declare an operator object from an operator class.

        The decorator is similar to a constructor. It produces a
        new object of type Operator, and with attributes apl, jvp and vjp as
        primitives.

        An operator class must define `ain, aout` and apl, vjp, jvp functions.

        ain : dict(name => type_pattern) describes the input arguments of
              the operator
        aout : dict(name => type_pattern) describes the output arguments of
              the operator

        Currently the type_pattern is not used; the plan is to add multi-dispatch
        if it is proven to be useful.

        apl : function(self, ...) the application of the operator;
              it shall return a dictionary
              of the evaluated values (exactly the same number of aout);
              except when there is only one output argument, then
              the result can be returned directly.

              all input arguments are resolved to python objects;
              it can have extra arguments in addition to ain.
              self is the node object that is used in the model

        rcd : function(self, ...) recording the arguments for
              invoking jvp and vjp. It shall return a dict.
              the only items included in the dict are available
              to vjp and vjp; if not defined, all arguments to apl are recorded.

        jvp : function(self, ...) the jacobian vector product. The convention
              is to use '_' + argname as the name of vectors. used for back-prop.
              self is the node object that is used in the model

        vjp : function(self, ...) the vector jacobian product. The convention
              is to use argname + '_' as the name of vectors. used for foward-prop.
              self is the node object that is used in the model

    """

    opr = Operator(kls)

    return opr

def zerobypass(impl):
    def zerobypassimpl(self, **kwargs):
        ain = self.ain
        aout = self.aout
        if all(kwargs[argname] is 0 for argname in ain):
            d = {}
            for argname in aout:
                d[argname] = 0
            return d
        return impl(self, **kwargs)
    return zerobypassimpl

def _make_primitive(opr, func, impl, argnames=None, record_impl=record_copy_all):
    """ create primitives for the operator.

        This is used to define a primitive based on the unbound method
        defined in the operator class.

    """
    from .primitive import Primitive

    assert func in ('apl', 'vjp', 'jvp')

    aout = {}
    ain = {}
    if argnames is None:
        argnames = impl.__code__.co_varnames[1:impl.__code__.co_argcount]

    if func == 'apl':
        ain = opr.ain
        aout = opr.aout
    elif func == 'vjp' : # in and out are prefixed.
        for arg in opr.ain:
            aout['_' + arg] = opr.ain[arg]

        for arg in opr.aout:
            # skip unused input vector args
            if '_' + arg not in argnames: continue
            ain['_' + arg] = opr.aout[arg]
        impl = zerobypass(impl)

    elif func == 'jvp' : # in and out are prefixed.
        for arg in opr.ain:
            # skip unused input vector args
            if arg + '_' not in argnames: continue
            ain[arg + '_'] = opr.ain[arg]

        for arg in opr.aout:
            aout[arg + '_'] = opr.aout[arg]
        impl = zerobypass(impl)

    primitive = Primitive(func, opr)

    primitive.impl     = impl
    primitive.record_impl     = record_impl
    primitive.func     = func
    primitive.ain      = ain
    primitive.aout     = aout
    primitive.argnames = argnames

    return primitive

