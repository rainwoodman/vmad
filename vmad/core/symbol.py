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
        self._parent = None
        self.name = name

        # a list of nodes that makes use of the symbol
        self.references = []

    def __getattr__(self, attrname):
        if attrname.startswith('_'):
            raise AttributeError
        return AttrSymbol(self.model, self, attrname)

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
        l = set([])
        symbol = self.symbol
        # recusrively get the name of the parents
        # to handle AttrSymbol and CallSymbol
        while hasattr(symbol, '_parent'):
            if symbol.name is not None:
                l = l.union([symbol.name])
            symbol = symbol._parent
        return l

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

class AttrSymbol(Literal):
    """ Represents accessing a member attribute of a symbol.

        The attribute is always treated as a literal. The suitable context
        is saying getting the size of an array.
    """
    def __init__(self, model, parent, attrname):
        Literal.__init__(self, model, None)
        self._parent = parent
        self._attrname = attrname

    def resolve(self, context):
        return getattr(self._parent.resolve(context), self._attrname)

    def __call__(self, *args, **kwargs):
        return CallSymbol(self.model, self, args, kwargs)

    def __getitem__(self, index):
        return GetItemSymbol(self.model, self, index)

    def __repr__(self):
        return "%s.%s" % (str(self._parent), self._attrname)

class CallSymbol(Literal):
    """ Represents calling a member attribute of a symbol.
    """
    def __init__(self, model, parent, args, kwargs):
        Literal.__init__(self, model, None)
        self._parent = parent
        self._args = args
        self._kwargs = kwargs

    def resolve(self, context):
        return self._parent.resolve(context)(*self._args, **self._kwargs)

class GetItemSymbol(Literal):
    """ Represents getting an item of a symbol.
    """
    def __init__(self, model, parent, index):
        Literal.__init__(self, model, None)
        self._parent = parent
        self._index = index

    def resolve(self, context):
        return self._parent.resolve(context)[index]

class ZeroLiteral(Literal):
    """ A ZeroLiteral is specially used to mark zeros in gradient propagation

    """
    def __init__(self, model):
        Symbol.__init__(self, model, None)

    def __repr__(self):
        return "[ZERO]"

    def resolve(self, context):
        return 0

