from .error import ResolveError
import weakref

class BaseSymbol(object):
    """ A symbol for building models.

        A symbol is named.

        A symbol can be resolved to a python object
        given a context in respect to its name.

        A symbol may be referenced by multiple

        operators if they take the symbol as an input.

        A symbol is bound to a model; this is useful
        for the operators to infer the model during model construction.

        Only named symbols are registered for autodiff.
    """
    def __init__(self, model):
        from .model import Model

        assert isinstance(model, Model)

        self._model = weakref.ref(model)

        self._parent = None

        # a list of nodes that makes use of the symbol
        self._references = []

    @property
    def model(self):
        return self._model()

    def resolve(self, context):
        raise NotImplementedError

    def store(self, context, value):
        """ Storing the value of the variable to the context """
        # Not doing anything to the context by default
        pass

    def add_reference(self, node):
        self._references.append(weakref.ref(self))
        ref_id = len(self._references)
        return Ref(self, node, ref_id)

    def has_reference(self):
        return len(self._references) > 0


class Ref(object):
    """
        A reference is the relation between a symbol and a node.
    """
    def __init__(self, symbol, node, ref_id):
        self.symbol = symbol
        # the node variable is currently not used anywhere
    #    self.node = node
        self.ref_id = ref_id

    def __repr__(self):
        return "&[%s:%d]" % (self.symbol, self.ref_id)

    def is_last_ref(self):
        return self.ref_id == len(self.symbol._references)

    def get_symbol_names(self):
        """ Returns a list of symbol names that are referenced by
            this object, recursively
        """
        from .symbol import Symbol
        l = set([])
        symbol = self.symbol
        # recusrively get the name of the parents
        # to handle AttrSymbol and CallSymbol
        while hasattr(symbol, '_parent'):
            if isinstance(symbol, Symbol):
                l = l.union([symbol.name])
            symbol = symbol._parent
        return l

class Symbol(BaseSymbol):
    def __init__(self, model, name):
        assert isinstance(name, str)

        BaseSymbol.__init__(self, model)

        self.name = name

        model.register(self)

    def __getattr__(self, attrname):
        if attrname.startswith('_'):
            raise AttributeError
        return AttrSymbol(self.model, self, attrname)

    @property
    def vjp_name(self):
        return '_' + self.name

    @property
    def jvp_name(self):
        return self.name + '_'

    def resolve(self, context):
        if self.name not in context:
            raise ResolveError("Symbol %s does not exist in the context" % self.name)
        return context[self.name]

    def store(self, context, value):
        context[self.name] = value

class List(BaseSymbol):
    def __init__(self, model, value):
        BaseSymbol.__init__(self, model)
        self._items = value

    def __repr__(self):
        return "L%s" % (str(self._items))

    def resolve(self, context):
        return [v.resolve(context) for v in self]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        for v in self._items:
            yield v

    def __reversed__(self):
        for v in reversed(self._items):
            yield v

    def store(self, context, value):
        for var, v in zip(self, value):
            var.store(context, v)

    def add_reference(self, node):
        self._references.append(weakref.ref(self))
        ref_id = len(self._references)
        return ListRef(self, node, ref_id)

    def has_reference(self):
        return (
            BaseSymbol.has_reference(self) or
            any(v.has_reference() for v in self)
        )

class ListRef(Ref):
    def __init__(self, symbol, node, ref_id):
        Ref.__init__(self, symbol, node, ref_id)
        self.symbol = symbol
        # make sure the last reference shows up first.
        self.items = list(reversed([v.add_reference(node) for v in reversed(symbol)]))

    def __repr__(self):
        return "& %s" % (str(self.items))

    def __iter__(self):
        return iter(self.items)

    def get_symbol_names(self):
        r = set()
        for ref in self.items:
            r = r.union(ref.get_symbol_names())
        return r

class Literal(BaseSymbol):
    """ A literal is a special symbol that does not resolve with a context.

        Literals do not participate in gradient propagation.
    """
    def __init__(self, model, value):
        BaseSymbol.__init__(self, model)
        self.value = value

    def __repr__(self):
        return "L(%s)" % (str(self.value))

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
        Literal.__init__(self, model, None)

    def __repr__(self):
        return "[ZERO]"

    def resolve(self, context):
        return 0

