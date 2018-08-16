import inspect

from .error import UnpackError, OverwritePrecaution, MissingArgument, BrokenPrimitive, BadArgument
from .node import Node

from .symbol import Symbol, assymbol, BaseSymbol
from .model import Model

class Primitive:
    """ Primitives are building blocks of models.

        Calling a primitive creates a node on a model.

        A primitive object for an operator is generated
        and attached to the operators object via the `operator` decorator.

    """

    def __init__(self, func, opr, ain, aout, argnames, impl, record_impl):
        self.name = opr.prototype.__name__ + '-' + func
        self.func = func
        self.operator = opr
        self.ain = ain
        self.aout = aout
        self.argnames = argnames

        self.impl = impl
        self.record_impl = record_impl

        # a few others are created in make_primitive

    def __eq__(self, other):
        """ If two primitives are the same, they must be for the same operator
            and the same function type. """
        if self.operator is not other.operator: return False
        if self.func != other.func: return False
        return True

    # When there are multiple return values, or if the return variable(s)
    # are fed on the calling arguments, it behaves like a list,
    # and must be unpacked. A strange error (during resolving) is raised if not
    # unpacked.

    def __call__(self, *args, **kwargs):
        """ Creating a node and append it to the model's execution graph

            The model is inferred from the input arguments

            kwargs['__stacklevel__'] is the adjustment for stacklevel
            -1 is the caller. -2 is the caller of the caller
        """

        # remember the frame info
        stacklevel = kwargs.pop('__stacklevel__', -2)
        previous_frame = inspect.currentframe()
        while stacklevel < 0:
            previous_frame = previous_frame.f_back
            stacklevel = stacklevel + 1

        _frameinfo = inspect.getframeinfo(previous_frame)

        node = Node(self, _frameinfo)

        kwargs = self._parse_args(args, kwargs)

        # gather models
        models = self._walk_models(kwargs)
        # consolidate
        model = Model.consolidate(models)
        # replace models mentioned in kwargs
        self._walk_models(kwargs, reset=model)

        basename = model.unique_name(self.name)

        for argname in self.ain:
            var = assymbol(kwargs[argname])

            # checking symbol references
            #print(basename, var.name, id(var), id(model.get(var.name)))

            ref = var._add_reference(node)
            node._varin[argname] = ref

        for argname in self.aout:
            if not argname in kwargs:
                # if a name is not supplied, generate a name
                varname = basename + '-' + argname
                var = Symbol(varname, model=model)
            else:
                var = kwargs[argname]
                var = assymbol(kwargs[argname])

                # already given a symbol, overwrite it
                # but this doesn't work for gradients / tape
                # so we die here
                _check_var_references(var)

            node._varout[argname] = var

        # record all `hyper` arguments that do not go into derivatives.
        for argname in self.argnames:
            if argname in self.ain or argname in self.aout: continue
            # if it is not provided (e.g. default value defined in the prototype is used)
            if argname not in kwargs: continue

            v = kwargs[argname]
            if isinstance(v, BaseSymbol):
                raise BadArgument("argument %s is declared as a hyper argument, but a symbol '%s' is passed in."
                    % (argname, v))

            node.hyper_args[argname] = v


        # append node to the model.
        model.append(node)

        # return the output symbols
        vout = [node.varout[argname] for argname in self.aout]
        if len(self.aout) == 1:
            return vout[0]
        else:
            return vout

    def _parse_args(self, args, kwargs):
        """ map arguments give as args and kwargs to argnames.
        """

        kwargs = kwargs.copy() # will modify

        # first attempt to map args into kwargs
        if len(args) > len(self.argnames):
            raise BadArgument("Argument list longer than total number of args")

        for argname, arg in zip(self.argnames, args):
            if argname in kwargs:
                raise BadArgument("argument %s already provided as keyword" % argname)

            kwargs[argname] = arg

        return kwargs

    def _walk_models(self, kwargs, reset=None):
        models = set()

        for argname in self.ain:
            if not argname in kwargs: raise MissingArgument("input argument '%s' not provided" % argname)

            var = kwargs[argname]
            models = models.union(_infer_models(var, reset=reset))

        for argname in self.aout:
            if argname not in kwargs: continue
            var = kwargs[argname]

            models = models.union(_infer_models(var, reset=reset))

        return models



def _check_var_references(var):
    from .symbol import List
    if isinstance(var, List):
        for v in var:
            _check_var_references(v)
        return

    if var._has_reference():
        raise OverwritePrecaution("Overwritting used symbols is not supported. Because it breaks vjp.")

def _infer_models(var, reset=None):

    if isinstance(var, Symbol):
        if reset is not None:
            reset.anchor(var)

        model = var._model
        if model is not None:
            return set([model])
        else:
            return set()

    if isinstance(var, (list, tuple)):
        models = set()
        for v in var:
            models = models.union(_infer_models(v, reset=reset))

        return models

    return set()

