class Node:
    def __init__(self, primitive, _frameinfo):
        self.primitive = primitive
        self.operator = primitive.operator
        self._frameinfo = _frameinfo

        # add a few aliases for accessing primitive attributes.
        self.name = primitive.name
        self.ain = primitive.ain
        self.aout = primitive.aout

        self._varin = {}
        self._varout = {}

        # FIXME: why is this useful at all?
        self.hyper_args = {}

    @property
    def varin(self):
        return self._varin

    @property
    def varout(self):
        return self._varout

    def __repr__(self):
        #return "%s(%s=>%s) at %s:%d" % (type(self).__name__, self.varin, self._varout, self._frameinfo[0], self._frameinfo[1])
        return "%s" % self.name

    def call(self, **kwargs):
        """ call the implementation function of the primitive;

            invoked by the Context

            kwargs: the arguments that goes into the impl function

            Returns: dict, result for each varout.
        """
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
        return self.primitive.record_impl(self, **d)

