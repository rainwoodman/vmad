from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg

from pmesh.pm import ParticleMesh
import numpy

@autooperator
class FastPMOperator:
    ain = [('x', '*')]
    aout = [('wn', '*'), ('s', '*'), ('fs', '*')]

    def main(self, x, q, stages, cosmology, powerspectrum, pm):
        wnk = fastpm.as_complex_field(x, pm)

        wn = fastpm.c2r(wnk)
        rholnk = fastpm.induce_correlation(wnk, powerspectrum, pm)

        if len(stages) == 0:
            rho = fastpm.c2r(rholnk)
            rho = linalg.add(rho, 1.0)
        else:
            dx, p, f = fastpm.nbody(rholnk, q, stages, cosmology, pm)
            x = linalg.add(q, dx)
            layout = fastpm.decompose(x, pm)
            rho = fastpm.paint(x, mass=1, layout=layout, pm=pm)

        return dict(fs=rho, s=rholnk, wn=wn)

@autooperator
class NLResidualOperator:
    ain = [('wn', '*'), ('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, wn, s, fs, d, sigma):
        r = linalg.add(fs, d * -1)
        r = linalg.mul(r, sigma ** -1)

        return dict(y = r)

    @classmethod
    def evaluate(self, wn, s, fs, d, sigma):
        return None

@autooperator
class SmoothedNLResidualOperator:
    ain = [('wn', '*'), ('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, wn, s, fs, d, sigma, scale):
        r = linalg.add(fs, d * -1)
        def tf(k):
            k2 = sum(ki ** 2 for ki in k)
            return numpy.exp(- 0.5 * k2 * scale ** 2)
        c = fastpm.r2c(r)
        c = fastpm.apply_transfer(c, tf)
        r = fastpm.c2r(c)
        r = linalg.mul(r, sigma ** -1)
        return dict(y = r)

    @classmethod
    def evaluate(self, wn, s, fs, d, sigma, scale):
        return None

@autooperator
class LNResidualOperator:
    ain = [('wn', '*'), ('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, wn, s, fs):
        r = linalg.add(wn, 0)
        fac = linalg.pow(wn.Nmesh.prod(), -0.5)
        r = linalg.mul(r, fac)
        return dict(y = r)

    @classmethod
    def evaluate(self, wn, s, fs):
        return None

@autooperator
class ChiSquareOperator:
    ain = [('x', '*')]
    aout = [('y', '*')]

    def main(self, x, comm):
        chi2 = linalg.sum(linalg.mul(x, x))
        chi2 = mpi.allreduce(chi2, comm)
        return dict(y = chi2)

from abopt.abopt2 import Problem as BaseProblem, VectorSpace

class ChiSquareProblem(BaseProblem):
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
            wn, s, fs = forward_operator(x)
            # fixme: need a way to directly include a subgraphs
            # rather than building it again.
            for operator in self.residuals:
                r = operator(wn, s, fs)
                chi2 = ChiSquareOperator(r, comm)
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
                    wn, s, fs = replay(xx)
                    r = operator(wn, s, fs)
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
            return self.comm.allreduce((a * b).sum())

        vs = VectorSpace(addmul=addmul, dot=dot)
        BaseProblem.__init__(self,
                        vs = vs,
                        objective=objective,
                        gradient=gradient,
                        hessian_vector_product=hessian_vector_product)

    def save(self, filename, state):
        with Builder() as m:
            x = m.input('x')
            wn, s, fs = self.forward_operator(x)
            m.output(wn=wn, s=s, fs=fs)

        wn, s, fs = m.compute(['wn', 's', 'fs'], init=dict(x=state['x']))

        from nbodykit.lab import FieldMesh

        wn = FieldMesh(wn)
        s = FieldMesh(s)
        fs = FieldMesh(fs)

        s.attrs['y'] = state['y']
        s.attrs['nit'] = state['nit']
        s.attrs['gev'] = state['gev']
        s.attrs['fev'] = state['fev']
        s.attrs['hev'] = state['hev']

        wn.save(filename, dataset='wn', mode='real')
        s.save(filename, dataset='s', mode='real', )
        fs.save(filename, dataset='fs', mode='real')

    def evaluate(self, x):
        from nbodykit.lab import FFTPower, FieldMesh

        with Builder() as m:
            xx = m.input('x')
            wn, s, fs = self.forward_operator(xx)
            m.output(wn=wn, s=s, fs=fs)

        wn, s, fs = m.compute(['wn', 's', 'fs'], init=dict(x=x))

        d = []

        for operator in self.residuals:
            d.append(operator.evaluate(wn, s, fs, **operator.hyperargs))

        return d
