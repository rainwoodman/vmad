from __future__ import print_function
"""
    Routines to define an operator

    use @operator decorator on a class to define an operator.

    Example: see the source code of :class:`add`

    use @nested.modeloperator to define a nested operator, where
    you only need to define a model and the ain, aout

"""

class Operator(object): pass

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

def find_primitive_type(node, func):
    # we will only do this on the apl primitives
    # because otherwise this is undefined
    # the algebra of autodiff in vmad3 is explicitly not closed!
    assert isinstance(node, type(node).operator._apl)

    assert func in ['vjp', 'jvp', 'apl']

    if func == 'jvp': return node.operator._jvp
    if func == 'vjp': return node.operator._vjp
    if func == 'apl': return node.operator._apl

def record_copy_all(self, **kwargs):
    """ A default rcd implementation that copies all kwargs to the tape.

        the impl is used for the vjp and vjp primitives.

        the impl is can be used for the apl primitives if no rcd is given;
        but we use 'record_copy_autodiff' to save a much smaller subset
        of variables.
    """
    return kwargs

def record_copy_autodiff(self, **kwargs):
    """ A default rcd implementation that copies `useful` kwargs to the tape
        for the autodiff.

        the impl is used for the apl primitives if no rcd is given;
    """
    jvp = find_primitive_type(self, 'jvp')
    vjp = find_primitive_type(self, 'vjp')
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
    """ Decorator to declare an operator. 

        The decorator is similar to a meta-class. It produces a
        new class with Operator as a baseclass, and apl, jvp and vjp are
        converted to primitives.

        An operator must define `ain, aout` and apl, vjp, jvp functions.

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

    if hasattr(kls, 'rcd'):
        record_impl = unbound(kls.rcd)
    else:
        record_impl = record_copy_autodiff

    kls.ain = _to_ordereddict(kls.ain)
    kls.aout = _to_ordereddict(kls.aout)

    kls._apl = _make_primitive(kls, 'apl', unbound(kls.apl),
        record_impl=record_impl)

    if hasattr(kls, 'vjp'):
        kls._vjp = _make_primitive(kls, 'vjp', unbound(kls.vjp))
    if hasattr(kls, 'jvp'):
        kls._jvp = _make_primitive(kls, 'jvp', unbound(kls.jvp))

    return type(kls.__name__, (Operator, kls, kls._apl), {})

def zerobypass(impl):
    def zerobypassimpl(self, **kwargs):
        ain = type(self).ain
        aout = type(self).aout
        if all(kwargs[argname] is 0 for argname in ain):
            d = {}
            for argname in aout:
                d[argname] = 0
            return d
        return impl(self, **kwargs)
    return zerobypassimpl

def _make_primitive(operator, func, impl, argnames=None, record_impl=record_copy_all):
    """ create primitives for the operator.

        This is used to define a primitive based on the unbound method
        defined in the operator class.

    """
    from .primitive import Primitive
    from .symbol import Symbol

    assert func in ('apl', 'vjp', 'jvp')

    kls = operator


    aout = {}
    ain = {}
    if argnames is None:
        argnames = impl.__code__.co_varnames[1:impl.__code__.co_argcount]

    if func == 'apl':
        ain = kls.ain
        aout = kls.aout
    elif func == 'vjp' : # in and out are prefixed.
        for arg in kls.ain:
            aout['_' + arg] = kls.ain[arg]

        for arg in kls.aout:
            ain['_' + arg] = kls.aout[arg]
        impl = zerobypass(impl)

    elif func == 'jvp' : # in and out are prefixed.
        for arg in kls.ain:
            ain[arg + '_'] = kls.ain[arg]

        for arg in kls.aout:
            aout[arg + '_'] = kls.aout[arg]
        impl = zerobypass(impl)

    members =  dict(
                impl     = impl,
                record_impl     = record_impl,
                func     = func,
                ain      = ain,
                aout     = aout,
                argnames = argnames,
                operator = operator,
                )

    bases = (Primitive,)

    primitive = type(operator.__name__ + '-' + func,
            bases,
            members
            )
    return primitive

# special operator used for partial gradient summation
@operator
class add:
    ain  = {'x1': '*',
            'x2': '*',
           }
    aout = {'y': '*'}

    def apl(self, x1, x2):
        return dict(y = x1 + x2)

    def vjp(self, _y):
        return dict(_x1 = _y, _x2 = _y)

    def jvp(self, x1_, x2_):
        return dict(y_ = x1_ + x2_)

# special operator for marking an output
@operator
class terminal:
    ain  = {'x': '*'}
    aout = {'y': '*'}

    def apl(self, x):
        return dict(y=x)
    def vjp(self, _y):
        return dict(_x=_y)
    def jvp(self, x_):
        return dict(y_=x_)


