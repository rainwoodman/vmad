import weakref

from .error import InferError, UnpackError, OverwritePrecaution, MissingArgument, BrokenPrimitive, BadArgument
from .symbol import Symbol, Literal, List

def make_symbol(model, obj):
    """ Make a symbol out of an input Python object.

    """
    if isinstance(obj, Primitive):
        # unpack single output primitives
        # allows a = primitive(...)
        # also see __iter__ for explict unpacking (a, b = primitive(...))
        if len(obj.varout) > 1:
            raise UnpackError("More than one output variable, need to unpack them")
        obj = next(iter(obj.varout.values()))

        # FIXME: Alternatively, make Primitive a subclass of Symbol.
        # But then some Primitives shall be subclasses of List.
        # the class hierarchy is a bit hairy and sounds like we will
        # in the end always need to deal with single output and multi output
        # primitives specially anyways. Shall give it an attempt.

    if isinstance(obj, (list, tuple)):
        obj = List(model, [make_symbol(model, i) for i in obj])

    # still not a Symbol? Must be some raw python object
    # intended to be used as a Literal
    if not isinstance(obj, Symbol):
        obj = Literal(model, obj)

    return obj


class Primitive(object):
    """ Primitives are building blocks of models.

        Instantiation of a primitive creates a node on a model.

        This is the base class for all operators. Primitive classes are generated
        and attached to the operators classes via the `operator` decorator.

    """

    def __init__(self, *args, **kwargs):
        """ Creating a node and append it to the model's execution graph

            The model is inferred from the input arguments
        """
        kls = type(self)

        _check_primitive_class(kls)

        self.varin = {}
        self.varout = {}
        self.hyper_args = {}

        kwargs = _parse_args(kls, args, kwargs)

        model = _find_model(kls, kwargs)

        self._name = model.unique_name(kls.__name__)

        for argname in kls.ain:
            var = make_symbol(model, kwargs[argname])

            # checking symbol references
            #print(self._name, var.name, id(var), id(model.get(var.name)))

            ref = var.add_reference(self)
            self.varin[argname] = ref

        for argname in kls.aout:
            if not argname in kwargs:
                # if a name is not supplied, generate a name
                varname = self.name + '-' + argname
                var = model.define(varname)
            else:
                var = kwargs[argname]
                var = make_symbol(model, kwargs[argname])

                # already given a symbol, overwrite it
                # but this doesn't work for gradients / tape
                # so we die here
                _check_var_references(var)

                # make a new symbol of the same name
                # var = model.define(var.name)

            self.varout[argname] = var

        # record all `hyper` arguments that do not go into derivatives.
        for k, v in kwargs.items():
            if k not in kls.ain and k not in kls.aout:
                self.hyper_args[k] = v

        # append self to the model.
        model.append(self)

    def __iter__(self):
        """ for unpacking the varout during model construction, e.g.

            .. code::

                a, b, c = split_three_way(x)

        """

        for argname in type(self).aout:
            yield self.varout[argname]

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return "%s(%s=>%s)" % (self._name, self.varin, self.varout)

    def call(self, **kwargs):
        """ call the implementation function of the primitive;

            invoked by the Context

            kwargs: the arguments that goes into the impl function

            Returns: dict, result for each varout.
        """
        r = type(self).impl(self, **kwargs)

        # allow returning without using a dict
        # if there is only a single output argument
        if not isinstance(r, dict) and len(self.varout) == 1:
            argname = next(iter(self.varout.keys()))
            r = {argname:r}

        return r

    def record(self, kwargs):
        """ generate the kwargs that goes into the tape;
            default is to record the entire kwargs.

            Sometimes we do not need the entire kwargs; e.g.
            for linear operators we only need enough information to create
            the output array of the back-prop gradient
            but we don't need the actual parameters.

            invoked by the Context.

            kwargs: the arguments that goes into the impl function

            Returns: dict that goes into the tape
        """
        return type(self).record_impl(self, **kwargs)


def _infer_model(var):
    if isinstance(var, Primitive):
        var = next(iter(var.varout.values()))

    if isinstance(var, Symbol):
        model = var.model
        return model

    if isinstance(var, (list, tuple)):
        for v in var:
            model = _infer_model(v)
            if model is not None:
                return model

    return None

def _check_primitive_class(kls):
    # assert the primitive is properly defined.
    for attr in ['ain', 'aout', 'impl', 'func', 'argnames', 'operator', 'record_impl']:
        if not hasattr(kls, attr):
            raise BrokenPrimitive("primitive class attribute '%s' is not defined" % attr)

def _check_var_references(var):
    if isinstance(var, List):
        for v in var.value:
            _check_var_references(v)
        return

    if len(var.references) != 0:
        raise OverwritePrecaution("Overwritting used symbols is not supported. Because it breaks vjp.")

def _parse_args(kls, args, kwargs):
    """ map arguments give as args and kwargs to argnames.
    """
    kwargs = kwargs.copy() # will modify

    # first attempt to map args into kwargs
    if len(args) > len(kls.argnames):
        raise BadArgument("Argument list longer than total number of args")

    for argname, arg in zip(kls.argnames, args):
        if argname in kwargs:
            raise BadArgument("argument %s already provided as keyword" % argname)

        kwargs[argname] = arg

    return kwargs

def _find_model(kls, kwargs):
    model = None

    for argname in kls.ain:
        if not argname in kwargs: raise MissingArgument("input argument '%s' not provided" % argname)

        var = kwargs[argname]
        model = _infer_model(var)
        if model is not None:
            return model

    for argname in kls.aout:
        if argname not in kwargs: continue
        var = kwargs[argname]

        model = _infer_model(var)
        if model is not None:
            return model

    raise InferError("Cannot infer model from variables -- try to mark at least one literal argument explicitly as Literal")
