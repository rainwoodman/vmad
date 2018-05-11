from vmad import Builder, autooperator
from vmad.lib import mpi, linalg

from pmesh.pm import ParticleMesh
import numpy

@autooperator
class MPIChiSquareOperator:
    ain = [('x', '*')]
    aout = [('y', '*')]

    def main(self, x, comm):
        chi2 = linalg.sum(linalg.mul(x, x))
        chi2 = mpi.allreduce(chi2, comm)
        return dict(y = chi2)

from abopt.abopt2 import Problem as BaseProblem, VectorSpace

class MPIChiSquareProblem(BaseProblem):
    """ Defines a chisquare problem, which is

        .. math::

            y = \sum_i [R_i(F(s))]^2

        F : forward_operator

        [R_i] : residuals

        comm : The problem is defined on a MPI communicator -- must be consistent
        with forward_operator and residuals. (usually just pm.comm or MPI.COMM.WORLD)

    """
    def __init__(self, comm, forward_operator, residuals):
        self.residuals = residuals
        self.forward_operator = forward_operator
        self.comm = comm

        with Builder() as m:
            x = m.input('x')
            y = 0
            fx = forward_operator(x)
            # fixme: need a way to directly include a subgraphs
            # rather than building it again.
            for operator in self.residuals:
                r = operator(*fx)
                chi2 = MPIChiSquareOperator(r, comm)
                y = linalg.add(y, chi2)
            m.output(y=y)

        def objective(x):
            return m.compute(vout='y', init=dict(x=x))

        def gradient(x):
            y, [vjp] = m.compute_with_vjp(init=dict(x=x), v=dict(_y=1.0))
            return vjp

        def hessian_vector_product(x, v):
            Dv = 0

            replay = forward_operator.precompute(x=x)

            for operator in self.residuals:
                with Builder() as m:
                    xx = m.input('x')
                    fx = replay(xx)
                    r = operator(*fx)
                    m.output(y=r)

                y, [Dv1] = m.compute_with_gnDp(vout='y',
                            init=dict(x=x),
                            v=dict(x_=v))
                Dv = Dv + Dv1
            # H is 2 JtJ, see wikipedia on Gauss Newton.
            return Dv * 2

        def addmul(a, b, c, p=1):
            """ a + b * c ** p, follow the type of b """
            if p is not 1: c = c ** p
            c = b * c
            if a is not 0: c = c + a
            return c

        def dot(a, b):
            """ einsum('i,i->', a, b) """
            return self.comm.allreduce(numpy.sum(a * b))

        vs = VectorSpace(addmul=addmul, dot=dot)
        BaseProblem.__init__(self,
                        vs = vs,
                        objective=objective,
                        gradient=gradient,
                        hessian_vector_product=hessian_vector_product)
