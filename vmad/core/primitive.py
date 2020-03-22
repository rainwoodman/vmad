import inspect
from collections import OrderedDict

from .error import UnpackError, OverwritePrecaution, MissingArgument, BrokenPrimitive, BadArgument
from .node import Node

from .symbol import Symbol, assymbol, BaseSymbol, Literal, List
from .model import Model

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

class Primitive:
    """ Primitives are building blocks of models.

        Calling a primitive creates a node on a model.

        A primitive object for an operator is generated
        and attached to the operators object via the `operator` decorator.

    """

    def __init__(self, func, opr, ain, aout, argnames, outnames, impl, record_impl):
        self.name = opr.prototype.__name__ + '-' + func
        self.func = func
        self.operator = opr
        self.ain = ain
        self.aout = aout
        self.argnames = argnames
        self.outnames = outnames

        self.impl = impl
        self._default_kwargs = get_default_args(self.impl)
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

    def create_node(self, kwargs, kwout=None, stacklevel=-2):
        """ Creating a node and append it to the model's execution graph

            The model is inferred from the input arguments

            stacklevel is the adjustment for stacklevel
            -1 is the caller. -2 is the caller of the caller

            kwargs are the input arguments; values are symbols.
            kwout are the output arguments, if an output argument is not
            given, a new symbol will be created.
        """
        if kwout is None: # generate output arguments.
            kwout = {}
        # remember the frame info
        previous_frame = inspect.currentframe()

        while stacklevel < 0:
            previous_frame = previous_frame.f_back
            stacklevel = stacklevel + 1

        _frameinfo = inspect.getframeinfo(previous_frame)

        node = Node(self, _frameinfo)

        # FIXME: this is tricky.
        # we guarentee prepending all necessary sub-models
        # necessary for this node before this
        # node; therefore this shall always work as long
        # as we always build a model this way.
        # the algorithm here does not explicitly say this.

        # gather models
        models = self._walk_models(kwargs, kwout)
        # consolidate
        model = Model.consolidate(models)
        basename = model.unique_name(self.name)

        for argname in self.argnames:
            if argname in self.ain:
                var = assymbol(kwargs[argname])
            else:
                # if it is not provided (e.g. default value defined in the prototype is used)
                if argname not in kwargs:
                    continue
                # not a autodiff argument, must be a Literal
                # but we still allow passing in a symbol
                # just ignore its gradient propagation.
                var = assymbol(kwargs[argname])

            # checking symbol references
            #print(basename, var.name, id(var), id(model.get(var.name)))

            ref = var._add_reference(node)
            node._varin[argname] = ref

        for argname in self.outnames:
            if not argname in kwout:
                # if a name is not supplied, generate a name
                varname = basename + '-' + argname
                var = Symbol(varname, model=model)
            else:
                var = assymbol(kwout[argname])

                # already given a symbol, overwrite it
                # but this doesn't work for gradients / tape
                # so we die here
                _check_var_references(var)

            node._varout[argname] = var

        # append node to the model.
        model.append(node)

        # return the output symbols
        vout = [node.varout[argname] for argname in self.outnames]

        return OrderedDict([(argname, node.varout[argname]) for argname in self.outnames])

    def _parse_args(self, args, kwargs):
        """ map arguments give as args and kwargs to argnames. Used in __call__

        """

        kwout = {}
        # first attempt to map args into kwargs
        if len(args) > len(self.argnames):
            raise BadArgument("Argument list longer than total number of args")

        for argname, arg in zip(self.argnames, args):
            if argname in kwargs:
                raise BadArgument("argument %s already provided as keyword" % argname)
            kwargs[argname] = arg

        for argname, arg in self._default_kwargs.items():
            if argname not in kwargs:
                kwargs[argname] = arg

        for argname in list(kwargs.keys()):
            if argname in self.outnames:
                if argname not in self.argnames:
                    import warnings
                    warnings.warn("Supplying keyword argument to an operator for an output. Prefer to use the verbose .create_node API instead of this", DeprecationWarning, stacklevel=3)
                    kwout[argname] = kwargs.pop(argname)

            elif argname not in self.argnames:
                raise BadArgument("Argument %s is passed in , but not declared as an argument" % argname)

        for argname in self.argnames:
            if argname not in kwargs:
                raise MissingArgument("Argument %s is missing." % argname)

        return kwargs, kwout


    def _walk_models(self, kwargs, kwout):
        models = list()

        q = []
        for argname in self.argnames:
            var = kwargs[argname]
            q.append(var)

        for argname in self.outnames:
            # will automatically generate kwout, and they will be properly anchored
            if argname not in kwout: continue
            var = kwout[argname]
            q.append(var)

        for var in q:
            m1 = _infer_models(var)
            _join_models(models, m1)
        return models



def _check_var_references(var):
    from .symbol import List
    if isinstance(var, List):
        for v in var:
            _check_var_references(v)
        return

    if var._has_reference():
        raise OverwritePrecaution("Overwritting used symbols is not supported. Because it breaks vjp.")

def _join_models(models, m1):
    for m in m1:
        if m not in models:
            models.append(m)

def _infer_models(var):

    if isinstance(var, Symbol):
        model = var._model
        if model is not None:
            return [model]
        else:
            return []

    if isinstance(var, (List, list, tuple)):
        models = []
        for v in var:
            m1 = _infer_models(v)
            _join_models(models, m1)

        return models

    return []

