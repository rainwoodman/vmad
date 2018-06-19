import weakref

class RefCounted:
    def __init__(self):
        # a list of nodes that makes use of the symbol
        self._references = []

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
        return "&[%s:%d]" % (self.symbol.name, self.ref_id)

    def is_last_ref(self):
        return self.ref_id == len(self.symbol._references)

    def get_symbol_names(self):
        """ Returns a list of symbol names that are referenced by
            this object, recursively
        """
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
        self.value = list(reversed([v.add_reference(node) for v in reversed(symbol.value)]))

    def __repr__(self):
        return "& %s" % (str(self.value))

    def get_symbol_names(self):
        r = set()
        for ref in self.value:
            r = r.union(ref.get_symbol_names())
        return r

