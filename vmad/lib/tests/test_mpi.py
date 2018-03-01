from __future__ import print_function
from pprint import pprint
from vmad.lib import linalg, mpi

import numpy

from vmad.testing import BaseScalarTest
from mpi4py import MPI

class Test_allreduce(BaseScalarTest):
    to_scalar = linalg.to_scalar

    comm = MPI.COMM_WORLD

    x = comm.rank + 1.0
    y = comm.allreduce(x) ** 2

    x_ = numpy.eye(1)

    def inner(self, x, y):
        return self.comm.allreduce(numpy.sum(x * y))

    def model(self, x):
        return mpi.allreduce(x, self.comm)
