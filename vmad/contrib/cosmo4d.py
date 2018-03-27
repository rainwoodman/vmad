from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg

from pmesh.pm import ParticleMesh

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

@autooperator
class LNResidualOperator:
    ain = [('wn', '*'), ('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, wn, s, fs):
        r = linalg.add(wn, 0)
        fac = linalg.pow(wn.Nmesh.prod(), -0.5)
        r = linalg.mul(r, fac)
        return dict(y = r)

@autooperator
class ChiSquareOperator:
    ain = [('x', '*')]
    aout = [('y', '*')]

    def main(self, x, comm):
        chi2 = linalg.sum(linalg.mul(x, x))
        chi2 = mpi.allreduce(chi2, comm)
        return dict(y = chi2)

from abopt.abopt2 import real_vector_space, Problem as BaseProblem, VectorSpace

class ChiSquareProblem(BaseProblem):

    def __init__(self, comm, forward_operator, residuals):
        """
        """
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
            print('obj', (x**2).sum())
            return m.compute(vout='y', init=dict(x=x))

        def gradient(x):
            print('grad', (x**2).sum())
            y, [vjp] = m.compute_with_vjp(init=dict(x=x), v=dict(_y=1.0))
            return vjp

        def hessian_vector_product(x, v):
            print('hvp', (x**2).sum())
            Dv = 0

            replay = forward_operator.precompute(x=x)

            for operator in self.residuals:
                with Builder() as m:
                    x = m.input('x')
                    wn, s, fs = replay(x)
                    r = operator(wn, s, fs)
                    m.output(y=r)

                y, [Dv1] = m.compute_with_gnDp(vout='y',
                            init=dict(x=x),
                            v=dict(x_=v))
                Dv = Dv + Dv1
            # H is 2 JtJ, see wikipedia on Gauss Newton.
            return Dv * 2

        vs = real_vector_space

        BaseProblem.__init__(self,
                        vs = vs,
                        objective=objective,
                        gradient=gradient,
                        hessian_vector_product=hessian_vector_product)

