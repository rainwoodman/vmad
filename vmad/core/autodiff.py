from .symbol import ZeroLiteral, Literal, Symbol, ListRef, List
from .stdlib import terminal, add

class SymbolCollection(dict):
    """ A dictionary to look up collected symbols generated
        during the autodiff
    """
    def __init__(self, model, tape):
        self.model = model
        self.zero = ZeroLiteral()
        self._max_ref_id = {}

        def _visit_ref(ref):
            if isinstance(ref, ListRef):
                for r1 in ref:
                    _visit_ref(r1)
            elif isinstance(ref.symbol, Symbol):
                self._max_ref_id[ref.symbol._name] = max(
                    self._max_ref_id.get(ref.symbol._name, 0), ref.ref_id
                    )

        for record in tape:
            node = record.node
            for ref in node.varin.values():
                _visit_ref(ref)


    def add(self, var):
        self[var._name] = var
        return var

    def add_vjp(self, var, ref_id=None):
        if ref_id is None:
            var_p = Symbol(var._vjp_name, model=self.model)
        # there can be many versions of vjp with distinct values.
        else:
            # first occurance, must have the max ref_id.
            var_p = Symbol(var._vjp_name + '#%d' % ref_id, model=self.model)
        return self.add(var_p)

    def add_jvp(self, var):
        return self.add(Symbol(var._jvp_name, model=self.model))

    def is_last_ref(self, var, ref_id):
        if var._name not in self._max_ref_id or ref_id == self._max_ref_id[var._name]:
            return True

        return False

    def get_vjp(self, var, ref_id=None):
        if ref_id is None:
            return self[var._vjp_name]
        else:
            return self[var._vjp_name + '#%d' % ref_id]

    def get_jvp(self, var):
        return self[var._jvp_name]

    def check(self, model):
        for var in self.values():
            assert var._model is model

def prepare_opr_kwargs(record, model):
    """ generate a first guess of kwargs based on the record.

    """
    p = record.node
    impl_kwargs = record.impl_kwargs

    kwargs = {}
    kwargs.update(impl_kwargs)
    return kwargs

def create_output_vjp(ref, symbols):

    # make lists for lists
    if isinstance(ref, ListRef):
        return List([
                    create_output_vjp(r, symbols)
                    for r in ref]
                )

    var = ref.symbol

    # bypass literal arguments
    if isinstance(var, Literal):
        return None

    # ref id is used to distinguish the first
    # time seeing a variable; when we will
    # create an unversioned gradient symbol.
    if symbols.is_last_ref(var, ref.ref_id):
        return symbols.add_vjp(var)
    else:
        return symbols.add_vjp(var, ref.ref_id)

def connect_output_vjp(ref, symbols):

    # make lists for lists
    if isinstance(ref, ListRef):
        for r in ref:
            connect_output_vjp(r, symbols)
        return

    var = ref.symbol

    # bypass literal arguments
    if isinstance(var, Literal):
        return

    # accummulate the partials
    if not symbols.is_last_ref(var, ref.ref_id):
        var_f = symbols.get_vjp(var)
        var_p = symbols.get_vjp(var, ref.ref_id)
        # create a new symbol for the result, with the same name
        # because we intent to overwrite it.
        var_f2 = symbols.add_vjp(var)

        add(x1=var_f, x2=var_p, y=var_f2)

def create_output_jvp(var, symbols):
    if isinstance(var, List):
        return [create_output_jvp(v, symbols) for v in var]

    if isinstance(var, Literal):
        raise RuntimError("This shall not happen, vjp is from an output which can never be a literal")

    return symbols.add_jvp(var)

def create_input_jvp(var, symbols):
    if isinstance(var, List):
        return [create_input_jvp(v, symbols) for v in var]

    if isinstance(var, Literal):
        return symbols.zero

    return symbols[var._jvp_name]

def create_input_vjp(var, symbols):
    if isinstance(var, List):
        return [create_input_vjp(v, symbols) for v in var]

    if isinstance(var, Literal):
        raise RuntimError("This shall not happen, vjp is from an output which can never be a literal")

    if not var._vjp_name in symbols:
        return symbols.zero
    return symbols[var._vjp_name]

def vjpmodel(tape):
    """ generate a vector jacobian product model based on a tape """
    model = type(tape.model)()
    symbols = SymbolCollection(model, tape)

    for var in tape.model._vout:
        model.input(symbols.add_vjp(var))

    for i, record in enumerate(tape[::-1]):
        p = record.node

        vjp_of_p = p.find_primitive_type(func='vjp')

        kwargs = prepare_opr_kwargs(record, model)

        # initialize 'v'
        for argname, var in p.varout.items():
            kwargs['_' + argname] = create_input_vjp(var, symbols)

        # create output vjps
        for argname, ref in p.varin.items():
            var_p = create_output_vjp(ref, symbols)

            if var_p is not None:
                kwargs['_' + argname] = var_p

        node = vjp_of_p(**kwargs)

        # combine partial derivatives.
        for argname, ref in p.varin.items():
            connect_output_vjp(ref, symbols)

    # mark outputs
    for var in tape.model._vin:
        if not var._vjp_name in symbols:
            varout = symbols.zero
        else:
            varout = symbols[var._vjp_name]
        model.output(**{var._vjp_name : varout})

    return model

def jvpmodel(tape):
    """ generate a jacobian vector product model based on a tape """
    model = type(tape.model)()
    symbols = SymbolCollection(model, tape)

    for var in tape.model._vin:
        model.input(symbols.add_jvp(var))

    for i, record in enumerate(tape):
        p = record.node

        jvp_of_p = p.find_primitive_type(func='jvp')

        kwargs = prepare_opr_kwargs(record, model)

        # initialize 'v'
        for argname, ref in p.varin.items():
            jvp_var = create_input_jvp(ref.symbol, symbols)
            kwargs[argname + '_'] = jvp_var

        # create output symbols
        for argname, var in p.varout.items():
            jvp_var = create_output_jvp(var, symbols)
            kwargs[argname + '_'] = jvp_var

        jvp_of_p(**kwargs)

    # mark outputs
    for var in tape.model._vout:
        if not var._jvp_name in symbols:
            varout = symbols.zero
        else:
            varout = symbols[var._jvp_name]
        model.output(**{var._jvp_name : varout})

    symbols.check(model)
    return model
