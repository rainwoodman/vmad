from .error import UnexpectedOutput, makeExecutionError, ModelError
from .stdlib import terminal

_raise_internal_errors = True

def set_raise_internal_errors(flag):
    """ If raise_internal_errors is set to True, then the errors in
        node execution are directly raised.

        If False, we will produce a wrapped messsage,
        which contains the line number where the
        node is declared. It is easier for debugging and in Python 3+
        the underlying error message is also printed.
    """
    global _raise_internal_errors
    _raise_internal_errors = flag


class Context(dict):
    """ A context is a collection of python objects referred by symbol names;

        This is where the execution of a model occurs.

        Context is the internal API. Use the compute method of a model instead.
    """
    def __init__(self, **init):
        self.update(init)

    def remove_unused(self, nodes):
        """ remove objects not used by nodes"""
        used = set()
        for p in nodes:
            for argname, ref in p.varin.items():
                used = used.union(ref.get_symbol_names())

        toremove = []
        for key in self:
            if not key in used:
                toremove.append(key)

        for key in toremove:
            self.pop(key)

    @classmethod
    def result_used(self, node):
        # FIXME: this doesn't remove all of the unused
        # may need to fix this in 'compile' or 'optimize'.
        if node.primitive == terminal.apl:
            return True

        for argname, var in node.varout.items():
            if var._has_reference(): return True

        if len(node.varout) == 0:
            # special operators without any output always gets run.
            return True
        return False

    def compute(self, model, vout, tape, monitor=None):
        """
            compute a model in the current context (self)
        """
        _voutnames = set([var._name for var in model._vout])

        for varname in vout:
            if varname not in _voutnames:
                raise UnexpectedOutput("Requested vout %s is not defined by the model as an output; available ones are %s" % (varname, _voutnames))

        r = {}
        for i, node in enumerate(model):

            if self.result_used(node):
                self.execute(node, tape)

            if node.primitive == terminal.apl:
                for argname, var in node.varout.items():
                    r[var._name] = self[var._name]

            self.remove_unused(model[i+1:])

            if monitor is not None:
                monitor(node, node.varin, node.varout, self)

        r = [r[varname] for varname in vout]

        return r

    def execute(self, node, tape):
        """ execute a node on the context, recording the
            arguments of the impl to the tape for replay, debuggin,
            or autodiff.
        """

        resolved = {}
        for argname, ref in node.varin.items():
            var = ref.symbol
            resolved[argname] = var._resolve(self)

        kwargs = {}
        kwargs.update(resolved)

        # add the hyper arguments used by the impl
        for argname, value in node.hyper_args.items():
            kwargs[argname] = value

        if _raise_internal_errors:
            r = node.call(kwargs)
        else:
            try:
                r = node.call(kwargs)
            except Exception as e:
                raise makeExecutionError(
                    "Error computing node : %s" % (node), e)

        tape.append(node, node.record(kwargs, r))

        for argname, var in node.varout.items():
            var._store(self, r[argname])

