from .symbol import Symbol, Literal
from .error import DuplicatedOutput, InferError
from .context import Context
from .tape import Tape

from .stdlib import terminal

from collections import OrderedDict
import uuid
import weakref

class Model(list):
    def __init__(self):
        self._vin = []
        self._vout = []

        self._name = uuid.uuid4().hex

    def __hash__(self):
        return hash(self._name)

    def extend(self, other):
        """ concatenate the model with another model.
        """

        # _syms are merged as we reanchor the _model of
        # symbols.

        self._vin.extend(other._vin)
        self._vout.extend(other._vout)

        list.extend(self, other)

    def input(self, *args):
        """ Declare multiple input variables.

            Parameters
            ----------
            args : list of input variables.
                    can be either instances of Symbol or
                    a string, in which case a Symbol is
                    created.

            Returns
            -------
            variables : list or a single object if len(args) == 1
                Symbols, marked as inputs and anchored to the model.

        """
        # FIXME: assert the Symbol hasn't been used
        v = []
        for name in args:
            if isinstance(name, Symbol):
                r = name
            else:
                r = Symbol(name)

            # anchor it to self
            self.anchor(r)
            v.append(r)

        self._vin.extend(v)

        if len(args) == 1:
            return v[0]

        return v

    def anchor(self, symbol):
        if symbol._anchored:
            if not symbol._transient:
                if symbol._model is not self:
                    raise ModelError("cannot change the model after the symbol is no longer in transient.")

        if isinstance(self, ModelInTransient):
            # use strong reference to keep these Transient models alive.
            # they will be replaced by the Model in the end as we put symbols
            # together via primitives.
            symbol._model_ref = self
        else:
            symbol._model_ref = weakref.ref(self)

    def output(self, **kwargs):
        for varname, oldvar in kwargs.items():
            for var in self._vout:
                if var._name == varname:
                    raise DuplicatedOutput("Variable %s is already marked as an output" % varname)

            var = Symbol(varname, model=self)
            terminal(x=oldvar, y=var)
            self._vout.append(var)

    def compile(self): pass

    def unique_name(self, str):
        return '%s-%s' % (str, uuid.uuid4().hex)

    def __repr__(self):
        return "Model: %s => %s" % (self._vin, self._vout)

    def compute(self, vout, init, return_tape=False, return_dict=False, monitor=None):
        """
            compute a model with the initial values

            init : dictionary
        """
        if isinstance(vout, str):
            single_return = True
            vout = [vout]
        else:
            single_return = False

        init = OrderedDict(init)

        tape = Tape(self, init)
        ctx = Context(**init)

        out = ctx.compute(self, vout=vout,
                        tape=tape,
                        monitor=monitor)

        tape.finalize(OrderedDict(zip(vout, out)))

        if return_dict:
            out = OrderedDict(zip(vout, out))
        else:
            if single_return:
                out = out[0]

        if return_tape:
            out = out, tape

        return out

    def compute_with_vjp(self, init, v, return_dict=False, monitor=None):
        """
            compute a model with the vjp

            Parameters
            ----------
            init : dict, or list of tuples
                initial values of the module. List will
                be convert to an ordereddict
            v    : dict, or list of tuples
                initial vectors dotted to the jacobian. List will
                be convert to an ordereddict

            The vout of the model computation is inferred
            from variable names of v.

            The vout of the vjp computation is inferred from
            variable names of init.

            Returns
            -------
            out : a list of computed values of the model;
                ordered by the keys of the dictionary.

            vjpout : a list of computed values of the vjp
                ordered by the keys of the dictionary.

        """

        init = OrderedDict(init)
        v = OrderedDict(v)

        # assert all of the vectors start with '_', such that
        # we can properly infer the vout of the model
        assert all(varname.startswith('_') for varname in v.keys())

        vout = [varname[1:] for varname in v.keys()]
        out, tape = self.compute(vout, init=init, return_tape=True)
        vjpvout = tape.get_vjp_vout()

        vjp = tape.get_vjp()
        vjpout = vjp.compute(vjpvout, init=v)

        if return_dict:
            out = OrderedDict(zip(vout, out))
            vjpout = OrderedDict(zip(vjpvout, vjpout))

        return out, vjpout

    def compute_with_jvp(self, vout, init, v, return_dict=False, monitor=None):
        """
            compute a model with the jvp

            Parameters
            ----------
            vout : list or string
                output variable to compute.
            init : dict, or list of tuples
                initial values of the module.
            v    : dict, or list of tuples
                initial vectors dotted by the jacobian.

            The vout of the vjp computation is inferred from
            variable names of vout.

            Returns
            -------
            out : a list of computed values of the model.
                if vout is a string, out is delisted (a single python object)

            jvpout : a list of computed values of the jvp
                ordered by the keys of the dictionary.
                if vout is a string, out is delisted (a single python object)

        """
        init = OrderedDict(init)
        v = OrderedDict(v)

        # assert all of the vectors ends with '_'.
        assert all(varname.endswith('_') for varname in v.keys())

        out, tape = self.compute(vout, init=init, return_tape=True)

        jvpvout = tape.get_jvp_vout()

        jvp = tape.get_jvp()
        jvpout = jvp.compute(jvpvout, init=v)

        if return_dict:
            out = OrderedDict(zip(vout, out))
            jvpout = OrderedDict(zip(jvpvout, jvpout))
        else:
            if not isinstance(vout, (tuple, list)):
                jvpout = jvpout[0]

        return out, jvpout

    def compute_with_gnDp(self, vout, init, v, return_dict=False, monitor=None):
        """
            compute a model with the gauss-newton approximated
            hessian vector product (D dot v = JTJ dot v)

            Parameters
            ----------
            vout : list or string
                output variable to compute.
            init : dict, or list of tuples
                initial values of the model.
            v    : dict, or list of tuples
                initial vectors dotted by the jacobian; shall
                match the variables in init, but with '_' appended.

            The vout of the vjp computation is inferred from
            variable names of vout.

            Returns
            -------
            out : a list of computed values of the model.
                if vout is a string, out is delisted (a single python object)

            gnhpout: a list of computed values of the gnhp
                ordered by the keys of the dictionary init.

        """
        init = OrderedDict(init)
        v = OrderedDict(v)

        # assert all of the vectors ends with '_'.
        assert all(varname.endswith('_') for varname in v.keys())

        out, tape = self.compute(vout, init=init, return_tape=True)

        jvp = tape.get_jvp()
        jvpvout = tape.get_jvp_vout()
        jvpout = jvp.compute(jvpvout, init=v)

        vjp = tape.get_vjp()
        vjpvout = tape.get_vjp_vout()
        # connecting the output of jvp as input of vjp
        vjpvin = ['_' + i[:-1] for i in jvpvout]
        gnhpout = vjp.compute(vjpvout, init=zip(vjpvin, jvpout))

        if return_dict:
            out = OrderedDict(zip(vout, out))
            gnhpout = OrderedDict(zip(vjpvout, gnhpout))

        return out, gnhpout

    @staticmethod
    def consolidate(models):
        """ Consolidate several models into a single one.

            The non-transient model is preferred.

            After this call all but the returned model
            shall be considered invalid.
        """
        model = None
        # find the non-transient model
        for i in models:
            if isinstance(i, ModelInTransient): continue
            if i is None: continue
            if model is None: # first time seeing a non transient
                model = i
            else:
                raise InferError("Multiple non transient models are found")

        if model is None:
            if len(models) == 0:
                model = ModelInTransient()
            else:
                # get the first model
                model = next(iter(models))

        for i in models:
            if i is model: continue # skip 'the' model
            model.extend(i)

        return model

class ModelInTransient(Model):
    """ A model that is in transient, eventually will be added to a 'real' model.

        This is necessary as we start building a model with symbols that haven't been
        associated to the desired model.

        They will be first added to a transient model,
        then when the true model is seen they will be added to the model.
    """
    pass

class Builder(Model):
    """ A context manager to signify the process of buildig a model.

    """
    def __enter__(self): return self
    def __exit__(self, *args): self.compile()
