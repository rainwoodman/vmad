from .symbol import ZeroLiteral, Literal, Symbol, ListRef, List
from .model import Model
from .operator import terminal, add
from .operator import find_primitive_type

def prepare_opr_kwargs(record, model):
    """ generate a first guess of kwargs based on the record.

    """
    p = record.node
    impl_kwargs = record.impl_kwargs

    kwargs = {}
    return impl_kwargs

def create_output_vjp(ref, model):

    # make lists for lists
    if isinstance(ref, ListRef):
        return List([
                    create_output_vjp(r, model)
                    for r in ref]
                )

    var = ref.symbol

    # bypass literal arguments
    if isinstance(var, Literal):
        return None

    if ref.is_last_ref():
        # largest reference_id, must be the
        # first time seeing the partial derivative
        # define the symbol for the full derivative
        var_p = Symbol(var._vjp_name, model=model)
    else:
        var_p = Symbol(var._vjp_name + '#%d' % ref.ref_id, model=model)

    return var_p

def connect_output_vjp(ref, model):

    # make lists for lists
    if isinstance(ref, ListRef):
        for r in ref:
            connect_output_vjp(r, model)
        return

    var = ref.symbol

    # bypass literal arguments
    if isinstance(var, Literal):
            return

    # accummulate the partials
    if not ref.is_last_ref():
        var_f = model.get(var._vjp_name)
        var_p = model.get(var._vjp_name + '#%d' % ref.ref_id)
        # create a new symbol for the result, with the same name
        # because we intent to overwrite it.
        var_f2 = Symbol(var._vjp_name, model=model)

        add(x1=var_f, x2=var_p, y=var_f2)

def create_output_jvp(var, model):
    if isinstance(var, List):
        return [create_output_jvp(v, model) for v in var]

    if isinstance(var, Literal):
        raise RuntimError("This shall not happen, vjp is from an output which can never be a literal")

    return Symbol(var._jvp_name, model=model)

def create_input_jvp(var, model):
    if isinstance(var, List):
        return [create_input_jvp(v, model) for v in var]

    if isinstance(var, Literal):
        return ZeroLiteral()

    return model.get(var._jvp_name)

def create_input_vjp(var, model):
    if isinstance(var, List):
        return [create_input_vjp(v, model) for v in var]

    if isinstance(var, Literal):
        raise RuntimError("This shall not happen, vjp is from an output which can never be a literal")

    if not model.has(var._vjp_name):
        # the variable is not declared on the model
        # FIXME: this can either be a bug or the variable is unused.
        return ZeroLiteral()

    return model.get(var._vjp_name)

def vjpmodel(tape):
    """ generate a vector jacobian product model based on a tape """
    model = Model()
    for var in tape.model._vout:
        model.input(var._vjp_name)

    for i, record in enumerate(tape[::-1]):
        p = record.node

        vjp_of_p = find_primitive_type(p, func='vjp')

        kwargs = prepare_opr_kwargs(record, model)

        # initialize 'v'
        for argname, var in p.varout.items():
            kwargs['_' + argname] = create_input_vjp(var, model)

        # create output vjps
        for argname, ref in p.varin.items():
            var_p = create_output_vjp(ref, model)

            if var_p is not None:
                kwargs['_' + argname] = var_p

        node = vjp_of_p(**kwargs)

        # combine partial derivatives.
        for argname, ref in p.varin.items():
            connect_output_vjp(ref, model)

    # mark outputs
    for var in tape.model._vin:
        if not model.has(var._vjp_name):
            varout = ZeroLiteral()
        else:
            varout = model.get(var._vjp_name)
        model.output(**{var._vjp_name : varout})

    return model

def jvpmodel(tape):
    """ generate a jacobian vector product model based on a tape """
    model = Model()
    for var in tape.model._vin:
        model.input(var._jvp_name)

    for i, record in enumerate(tape):
        p = record.node

        jvp_of_p = find_primitive_type(p, func='jvp')

        kwargs = prepare_opr_kwargs(record, model)

        # initialize 'v'
        for argname, ref in p.varin.items():
            jvp_var = create_input_jvp(ref.symbol, model)
            kwargs[argname + '_'] = jvp_var

        # create output symbols
        for argname, var in p.varout.items():
            jvp_var = create_output_jvp(var, model)
            kwargs[argname + '_'] = jvp_var

        jvp_of_p(**kwargs)

    # mark outputs
    for var in tape.model._vout:
        if not model.has(var._jvp_name):
            varout = ZeroLiteral()
        else:
            varout = model.get(var._jvp_name)
        model.output(**{var._jvp_name : varout})

    return model
