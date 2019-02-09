class Node:
    """ A node on the computing graph.

        The node is the first argument to apl(node, ....) and
        jvp / vjp functions.

        node[argname] gives the input symbol
    """
    def __init__(self, primitive, _frameinfo):
        self.primitive = primitive
        self.operator = primitive.operator
        self._frameinfo = _frameinfo

        # add a few aliases for accessing primitive attributes
        # 
        self.name = primitive.name

        self._varin = {} # references
        self._varout = {}

    def __getitem__(self, key):
        """ getting input variables as symbols """
        # varin are references.
        return self._varin[key].symbol

    @property
    def varin(self):
        return self._varin

    @property
    def varout(self):
        return self._varout

    def __repr__(self):
        #return "%s(%s=>%s) at %s:%d" % (type(self).__name__, self.varin, self._varout, self._frameinfo[0], self._frameinfo[1])
        return "%s @ %s : %s " % (self.name, self._frameinfo[0], self._frameinfo[1])

    def call(self, kwargs):
        """ call the implementation function of the primitive;

            invoked by the Context

            kwargs: the arguments that goes into the impl function

            Returns: dict, result for each varout.
        """
        from .symbol import BaseSymbol
        for key, value in kwargs.items():
            assert not isinstance(value, BaseSymbol)

        r = self.primitive.impl(self, **kwargs)

        # allow returning without using a dict
        # if there is only a single output argument
        if not isinstance(r, dict):
            if len(self.varout) == 1:
                argname = next(iter(self.varout.keys()))
                r = {argname:r}
            if len(self.varout) == 0:
                if r is not None:
                    raise ValueError("Return value of the primitive is not None, while no output arguments are defined")
                r = {}

        for key, value in r.items():
            assert not isinstance(value, BaseSymbol)

        return r

    def record(self, kwargs, r):
        """ generate the kwargs that goes into the tape;
            default is to record the entire kwargs.

            Sometimes we do not need the entire kwargs; e.g.
            for linear operators we only need enough information to create
            the output array of the back-prop gradient
            but we don't need the actual parameters.

            invoked by the Context.

            kwargs: the arguments that goes into the impl function
            r : the result of the calculation operator apl, dict from argname to value
                see above.

            Returns: dict that goes into the tape, will be available in vjp and jpv
        """
        # merge the two dictionaries, prioritizing kwargs (inputs).
        d = {}
        d.update(r)
        d.update(kwargs)
        from .symbol import BaseSymbol
        for key, value in d.items():
            assert not isinstance(value, BaseSymbol)
        return self.primitive.record_impl(self, **d)

    def find_primitive_type(node, func):
        # we will only do this on the apl primitives
        # because otherwise this is undefined
        # the algebra of autodiff in vmad3 is explicitly not closed!
        assert node.primitive == node.operator.apl

        assert func in ['vjp', 'jvp', 'apl']

        if func == 'jvp': return node.operator.jvp
        if func == 'vjp': return node.operator.vjp
        if func == 'apl': return node.operator.apl

