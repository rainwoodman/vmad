"""
Vocaburary
----------

universal : a variable whose value is replicated on all ranks.
distributed : a variable whose value is partitioned on all ranks.
root-only   : a variable whose value is only stored on the root rank.

collective : an operation that is performed by all ranks.

"""

from vmad import operator, autooperator
import numpy

@operator
class allreduce:
    ain = [('x', '*')]
    aout = [('y', '*')]

    def apl(node, x, comm):
        return comm.allreduce(x)

    def vjp(node, _y, comm):
        return dict(_x = _y)

    def jvp(node, x_, comm):
        return dict(y_ = comm.allreduce(x_))

@operator
class allbcast:
    """
    The invert operator of allreduce.

    Converting an universal variable 'x' into a distributed variable 'x'.

    The input variable is assumed to be identical
    on all ranks. Therefore in the forward pass the allbcast operator
    is an nop operation.

    Unlike broadcast which converts a master variable into a universal
    variable.

    """

    ain = [('x', '*')]
    aout = [('y', '*')]

    def apl(node, x, comm):
        # value should already be identical on all ranks.
        comm.barrier()
        return x

    def vjp(node, _y, comm):
        return dict(_x = comm.allreduce(_y))

    def jvp(node, x_):
        return dict(y_ = x_)


