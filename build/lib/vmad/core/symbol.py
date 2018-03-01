from .error import ResolveError
import weakref

class Symbol(object):
    """ A symbol for building models.

        A symbol is named.

        A symbol can be resolved to a python object
        given a context in respect to its name.

        A symbol may be referenced by multiple

        operators if they take the symbol as an input.

        A symbol is bound to a model; this is useful
        for the operators to infer the model during model construction.
    """
    def __init__(self, model, name=None):
        from .model import Model

        if name is not None:
            assert isinstance(name, str)
        assert isinstance(model, Model)

        self._model = weakref.ref(model)
        self.name = name

        # a list of nodes that makes use of the symbol
        self.references = []

    def add_reference(self, node):
        return Ref(self, node)

    @property
    def vjp_name(self):
        return '_' + self.name

    @property
    def jvp_name(self):
        return self.name + '_'

    @property
    def model(self):
        return self._model()

    def __repr__(self):
        return "[%s:]" % self.name

    def resolve(self, context):
        if self.name not in context:
            raise ResolveError("Symbol %s does not exist in the context" % self.name)
        return context[self.name]

    def store(self, context, value):
        context[self.name] = value

    def has_reference(self):
        return len(self.references) > 0

class List(Symbol):
    def __init__(self, model, value):
        Symbol.__init__(self, model, None)
        self.value = value

    def __repr__(self):
        return "L%s" % (str(self.value))

    def resolve(self, context):
        return [v.resolve(context) for v in self.value]

    def store(self, context, value):
        for var, v in zip(self.value, value):
            var.store(context, v)

    def add_reference(self, node):
        return ListRef(self, node)

    def has_reference(self):
        return any(v.has_reference() for v in self.value)

class Ref(object):
    def __init__(self, symbol, node):
        self.symbol = symbol
        self.node = node
        symbol.references.append(weakref.ref(self))
        self.ref_id = len(symbol.references)

    def __repr__(self):
        return "&[%s:%d]" % (self.symbol.name, self.ref_id)

    def get_symbol_names(self):
        return set([self.symbol.name])

class ListRef(object):
    def __init__(self, symbol, node):
        self.symbol = symbol
        # make sure the last reference shows up first.
        self.value = list(reversed([Ref(v, node) for v in reversed(symbol.value)]))

    def __repr__(self):
        return "& %s" % (str(self.value))

    def get_symbol_names(self):
        r = set()
        for ref in self.value:
            r = r.union(ref.get_symbol_names())
        return r

class Literal(Symbol):
    """ A literal is a special symbol that does not resolve with a context.

        Literals do not participate in gradient propagation.
    """
    def __init__(self, model, value):
        Symbol.__init__(self, model, None)
        self.value = value

    def __repr__(self):
        return "%s" % (str(self.value))

    def resolve(self, context):
        return self.value

class ZeroLiteral(Literal):
    """ A ZeroLiteral is specially used to mark zeros in gradient propagation

    """
    def __init__(self, model):
        Symbol.__init__(self, model, None)

    def __repr__(self):
        return "[ZERO]"

    def resolve(self, context):
        return 0

