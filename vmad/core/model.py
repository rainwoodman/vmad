from .symbol import Symbol, Literal
from .operator import terminal
from .error import DuplicatedOutput
from .context import Context
from .tape import Tape
from collections import OrderedDict
import uuid

class Model(list):
    def __init__(self):
        self._vin = []
        self._vout = []

        # symbols that are registered for autodiff
        self._syms = {}
        self._name = uuid.uuid4().hex

    def __hash__(self):
        return hash(self._name)

    def extend(self, other):
        """ concatenate the model with another model.
        """

        self._vin.extend(other._vin)
        self._vout.extend(other._vout)
        self._syms.update(other._syms)

        print('extending', list(self), list(other))
        list.extend(self, other)

    def register(self, r):
        assert r._name is not None

        self._syms[r._name] = r
        return r

    def get(self, varname):
        return self._syms[varname]

    def has(self, varname):
        return varname in self._syms

    def input(self, *args):
        r = [Symbol(self, a) for a in args]
        self._vin.extend(r)
        if len(args) == 1:
            r = r[0]
        return r

    def output(self, **kwargs):
        for varname, oldvar in kwargs.items():
            for var in self._vout:
                if var._name == varname:
                    raise DuplicatedOutput("Variable %s is already marked as an output" % varname)

            var = Symbol(self, varname)
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
