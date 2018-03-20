from .operator import terminal
from .error import UnexpectedOutput, ExecutionError, ModelError

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

    def result_used(self, node):
        # FIXME: this doesn't remove all of the unused
        # may need to fix this in 'compile' or 'optimize'.
        if isinstance(node, terminal._apl):
            return True
        for argname, var in node.varout.items():
            if var.has_reference(): return True
        return False

    def compute(self, model, vout, tape, monitor=None):
        """
            compute a model in the current context (self)
        """

        _voutnames = set([var.name for var in model._vout])

        for varname in vout:
            if varname not in _voutnames:
                raise UnexpectedOutput("Requested vout %s is not defined by the model as an output; available ones are %s" % (varname, _voutnames))

        r = {}
        for i, node in enumerate(model):

            if self.result_used(node):
                #try:
                    self.execute(node, tape)
                #except ModelError:
                #    raise
                #except Exception as e:
                #    raise ExecutionError("Error computing node : %s. model = %s" % (node, model), e)

            if isinstance(node, terminal._apl):
                for argname, var in node.varout.items():
                    r[var.name] = self[var.name]

            self.remove_unused(model[i+1:])

            if monitor is not None:
                monitor(node, self)

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
            resolved[argname] = var.resolve(self)

        kwargs = {}
        kwargs.update(resolved)

        # add the hyper arguments used by the impl
        for argname, value in node.hyper_args.items():
            if argname in node.argnames:
                kwargs[argname] = value

        tape.append(node, node.record(kwargs))

        r = node.call(**kwargs)

        for argname, var in node.varout.items():
            var.store(self, r[argname])

