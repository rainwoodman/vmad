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
        return List(model, [
                    create_output_vjp(r, model)
                    for r in ref.value]
                )

    var = ref.symbol

    # bypass literal arguments
    if isinstance(var, Literal):
        return None

    if ref.ref_id == len(var.references):
        # largest reference_id, must be the
        # first time seeing the partial derivative
        # define the symbol for the full derivative
        var_p = model.define(var.vjp_name)
    else:
        var_p = model.define(var.vjp_name + '#%d' % ref.ref_id)

    return var_p

def connect_output_vjp(ref, model):

    # make lists for lists
    if isinstance(ref, ListRef):
        for r in ref.value:
            connect_output_vjp(r, model)
        return

    var = ref.symbol

    # bypass literal arguments
    if isinstance(var, Literal):
            return

    # accummulate the partials
    if ref.ref_id != len(var.references):
        var_f = model.get(var.vjp_name)
        var_p = model.get(var.vjp_name + '#%d' % ref.ref_id)
        # create a new symbol for the result, with the same name
        # because we intent to overwrite it.
        var_f2 = model.define(var.vjp_name)

        add(x1=var_f, x2=var_p, y=var_f2)

def create_output_jvp(var, model):
    if isinstance(var, List):
        return [create_output_jvp(v, model) for v in var.value]

    if isinstance(var, Literal):
        raise RuntimError("This shall not happen, vjp is from an output which can never be a literal")

    return model.define(var.jvp_name)

def create_input_jvp(var, model):
    if isinstance(var, List):
        return [create_input_jvp(v, model) for v in var.value]

    if isinstance(var, Literal):
        return ZeroLiteral(model)

    return model.get(var.jvp_name)

def create_input_vjp(var, model):
    if isinstance(var, List):
        return [create_input_vjp(v, model) for v in var.value]

    if isinstance(var, Literal):
        raise RuntimError("This shall not happen, vjp is from an output which can never be a literal")

    if not model.has(var.vjp_name):
        # the variable is not declared on the model
        # FIXME: this can either be a bug or the variable is unused.
        return ZeroLiteral(model)

    return model.get(var.vjp_name)

def vjpmodel(tape):
    """ generate a vector jacobian product model based on a tape """
    model = Model()
    for var in tape.model._vout:
        model.input(var.vjp_name)

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
        if not model.has(var.vjp_name):
            varout = ZeroLiteral(model)
        else:
            varout = model.get(var.vjp_name)
        model.output(**{var.vjp_name : varout})

    return model

def jvpmodel(tape):
    """ generate a jacobian vector product model based on a tape """
    model = Model()
    for var in tape.model._vin:
        model.input(var.jvp_name)

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
        if not model.has(var.jvp_name):
            varout = ZeroLiteral(model)
        else:
            varout = model.get(var.jvp_name)
        model.output(**{var.jvp_name : varout})

    return model
