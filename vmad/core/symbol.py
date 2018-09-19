from .error import ResolveError
import weakref

from . import stdlib

class BaseSymbol(object):
    """ A symbol for building models.

        A symbol is named.

        A symbol can be resolved to a python object
        given a context in respect to its name.

        A symbol may be referenced by multiple

        operators if they take the symbol as an input.

        Only named symbols are registered for autodiff.
    """
    def __init__(self):
        self._parent = None

        # a list of nodes that makes use of the symbol
        self._references = []

    def _resolve(self, context):
        raise NotImplementedError

    def _store(self, context, value):
        """ Storing the value of the variable to the context """
        # Not doing anything to the context by default
        pass

    def _add_reference(self, node):
        self._references.append(weakref.ref(self))
        ref_id = len(self._references)
        return Ref(self, node, ref_id)

    def _has_reference(self):
        return len(self._references) > 0

    def eval(self, function):
        """ Evaluates an expression or a function on the symbol. see vmad.core.stdlib.eval. """
        return stdlib.eval(self, function)

    # workaround numpy's operator overrides, which
    # is higher priority than rxxx methodds (radd) defined here.
    # this will be deprecated and we will need to add an
    # __array_ufunc__ to dispatch this in the future.
    __array_priority__ = 15.0

    def __add__(self, other): return stdlib.add(self, other, __stacklevel__=-3)
    def __radd__(self, other): return stdlib.add(other, self, __stacklevel__=-3)
    def __sub__(self, other): return stdlib.sub(self, other, __stacklevel__=-3)
    def __rsub__(self, other): return stdlib.sub(other, self, __stacklevel__=-3)
    def __mul__(self, other): return stdlib.mul(self, other, __stacklevel__=-3)
    def __rmul__(self, other): return stdlib.mul(other, self, __stacklevel__=-3)
    def __truediv__(self, other): return stdlib.div(self, other, __stacklevel__=-3)
    def __rtruediv__(self, other): return stdlib.div(other, self, __stacklevel__=-3)
    def __pow__(self, other, modulo=None):
        if modulo is not None:
            raise ValueError("pow with modulo is not supported")
        return stdlib.pow(self, other, __stacklevel__=-3)
    def __rpow__(self, other):
        raise
        return stdlib.pow(other, self, __stacklevel__=-3)

    def __floordiv__(self, other): raise TypeError("floor div is not supported in autodiff")
    def __rfloordiv__(self, other): raise TypeError("floor div is not supported in autodiff")

    def __neg__(self): return stdlib.neg(self, __stacklevel__=-3)
    def __pos__(self): return stdlib.pos(self, __stacklevel__=-3)
    def __abs__(self): return stdlib.abs(self, __stacklevel__=-3)

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
        return "&[%s:%d]" % (repr(self.symbol), self.ref_id)

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
                l = l.union([symbol._name])
            symbol = symbol._parent
        return l

class Symbol(BaseSymbol):
    """ A symbol remembers the model only for eaiser model construction.

        A symbol is bound to a model; this is useful
        for the operators to infer the model during model construction.
    """
    def __init__(self, name, model=None):
        assert isinstance(name, str)

        BaseSymbol.__init__(self)

        self._name = name


        self._model_ref = None

        if model is not None:
            # anchor it now
            model.anchor(self)

    @property
    def _model(self):
        from .model import ModelInTransient
        if self._model_ref is None:
            return None
        if isinstance(self._model_ref, ModelInTransient):
            return self._model_ref
        else:
            return self._model_ref()

    @property
    def _anchored(self):
        return self._model_ref is not None

    @property
    def _transient(self):
        from .model import ModelInTransient
        return isinstance(self._model_ref, ModelInTransient)

    def __repr__(self):
        return self._name

    @property
    def _vjp_name(self):
        return '_' + self._name

    @property
    def _jvp_name(self):
        return self._name + '_'

    def _resolve(self, context):
        if self._name not in context:
            raise ResolveError("Symbol %s does not exist in the context" % self._name)
        return context[self._name]

    def _store(self, context, value):
        context[self._name] = value

class List(BaseSymbol):
    def __init__(self, value):
        BaseSymbol.__init__(self)
        self._items = value

    def __repr__(self):
        return "L%s" % (str(self._items))

    def _resolve(self, context):
        return [v._resolve(context) for v in self]

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

    def _store(self, context, value):
        for var, v in zip(self, value):
            var._store(context, v)

    def _add_reference(self, node):
        self._references.append(weakref.ref(self))
        ref_id = len(self._references)
        return ListRef(self, node, ref_id)

    def _has_reference(self):
        return (
            BaseSymbol._has_reference(self) or
            any(v._has_reference() for v in self)
        )

class ListRef(Ref):
    def __init__(self, symbol, node, ref_id):
        Ref.__init__(self, symbol, node, ref_id)
        self.symbol = symbol
        # make sure the last reference shows up first.
        self.items = list(reversed([v._add_reference(node) for v in reversed(symbol)]))

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
    def __init__(self, value):
        BaseSymbol.__init__(self)
        self._value = value

    def __repr__(self):
        return "L(%s)" % (str(self._value))

    def _resolve(self, context):
        return self._value

    def __getattr__(self, attrname):
        if attrname.startswith('_'):
            raise AttributeError
        return AttrSymbol(self, attrname)

class AttrSymbol(Literal):
    """ Represents accessing a member attribute of a symbol.

        The attribute is always treated as a literal. The suitable context
        is saying getting the size of an array.
    """
    def __init__(self, parent, attrname):
        Literal.__init__(self, None)
        self._parent = parent
        self._attrname = attrname

    def _resolve(self, context):
        return getattr(self._parent._resolve(context), self._attrname)

    def __repr__(self):
        return "%s.%s" % (str(self._parent), self._attrname)

class CallSymbol(Literal):
    """ Represents calling a member attribute of a symbol.
    """
    def __init__(self, parent, args, kwargs):
        Literal.__init__(self, None)
        self._parent = parent
        self._args = args
        self._kwargs = kwargs

    def _resolve(self, context):
        return self._parent._resolve(context)(*self._args, **self._kwargs)

class GetItemSymbol(Literal):
    """ Represents getting an item of a symbol.
    """
    def __init__(self, parent, index):
        Literal.__init__(self, None)
        self._parent = parent
        self._index = index

    def _resolve(self, context):
        return self._parent._resolve(context)[index]

class ZeroLiteral(Literal):
    """ A ZeroLiteral is specially used to mark zeros in gradient propagation

    """
    def __init__(self):
        Literal.__init__(self, None)

    def __repr__(self):
        return "[ZERO]"

    def _resolve(self, context):
        return 0

def assymbol(obj):
    """ Make a symbol out of an input Python object.

    """
    # cast list or tuple to a list object:
    if isinstance(obj, (list, tuple)):
        obj = List([assymbol(i) for i in obj])

    # not a Symbol? Must be some raw python object
    # intended to be used as a Literal
    if not isinstance(obj, BaseSymbol):
        obj = Literal(obj)

    return obj
