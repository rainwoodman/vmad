from __future__ import print_function
from pprint import pprint
from vmad.lib import linalg, mpi

import numpy

from vmad.testing import BaseScalarTest
from mpi4py import MPI

class Test_allreduce(BaseScalarTest):
    to_scalar = staticmethod(linalg.to_scalar)

    comm = MPI.COMM_WORLD

    x = comm.rank + 1.0
    y = comm.allreduce(x) ** 2

    x_ = numpy.eye(1)

    # self.x is distributed, thus allreduce along the rank axis.
    def inner(self, a, b):
        return self.comm.allreduce(numpy.sum(a * b))

    def model(self, x):
        return mpi.allreduce(x, self.comm)


class Test_allbcast(BaseScalarTest):
    to_scalar = staticmethod(lambda x: x)
    comm = MPI.COMM_WORLD
    x = 2.0
    y = comm.allreduce(x * (comm.rank + 1))

    x_ = numpy.eye(1)

    # self.x is universal, thus no special allreduce here.
    def inner(self, a, b):
        return numpy.sum(a*b)

    def model(self, x):
        x = mpi.allbcast(x, self.comm)
        x = x * (self.comm.rank + 1)
        return mpi.allreduce(x, comm=self.comm)
