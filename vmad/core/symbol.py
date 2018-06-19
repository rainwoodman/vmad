from .error import ResolveError
import weakref
from .refcount import RefCounted
from .refcount import Ref
from .refcount import ListRef

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

    @property
    def model(self):
        return self._model()

    def resolve(self, context):
        raise NotImplementedError

    def store(self, context, value):
        """ Storing the value of the variable to the context """
        # Not doing anything to the context by default
        pass

class Symbol(BaseSymbol, RefCounted):
    def __init__(self, model, name):
        assert isinstance(name, str)

        BaseSymbol.__init__(self, model)
        RefCounted.__init__(self)

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

class List(BaseSymbol, RefCounted):
    def __init__(self, model, value):
        BaseSymbol.__init__(self, model)
        RefCounted.__init__(self)

        self.value = value

    def __repr__(self):
        return "L%s" % (str(self.value))

    def resolve(self, context):
        return [v.resolve(context) for v in self.value]

    def __len__(self):
        return len(self.value)

    def __getitem__(self, i):
        return self.value[i]

    def __iter__(self):
        for v in self.value:
            yield v

    def store(self, context, value):
        for var, v in zip(self.value, value):
            var.store(context, v)

    def add_reference(self, node):
        return ListRef(self, node)

    def has_reference(self):
        return any(v.has_reference() for v in self.value)

class Literal(BaseSymbol, RefCounted):
    """ A literal is a special symbol that does not resolve with a context.

        Literals do not participate in gradient propagation.
    """
    def __init__(self, model, value):
        BaseSymbol.__init__(self, model)
        RefCounted.__init__(self)
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

