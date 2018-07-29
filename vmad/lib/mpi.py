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


