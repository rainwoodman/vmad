from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg
from abopt.trustregion import TrustRegionCG

import numpy

from abopt.lbfgs import LBFGS

from abopt.abopt2 import real_vector_space, Problem, VectorSpace

from nbodykit.cosmology import Planck15, LinearPower

@autooperator
class nl3d:
    ain = [('x', '*')]
    aout = [('rho', '*')]

    def main(self, x, q, stages, cosmology, powerspectrum, pm):
        wnk = fastpm.as_complex_field(x, pm)

        rhok = fastpm.induce_correlation(wnk, powerspectrum, pm)

        if len(stages) == 0:
            rho = fastpm.c2r(rhok)
            rho = linalg.add(rho, 1)
        else:
            dx, p, f = fastpm.nbody(rhok, q, stages, cosmology, pm)
            x = linalg.add(q, dx)
            layout = fastpm.decompose(x, pm)
            rho = fastpm.paint(x, mass=1, layout=layout, pm=pm)

        return dict(rho = rho)

@autooperator
class ResidualOperator:
    ain = [('x', '*')]
    aout = [('y', '*')]
    def main(self, x, d, powerspectrum, pm):
        rho = nl3d(x, q, [1.0], Planck15, powerspectrum, pm)
        r = linalg.add(rho, d * -1)

        return dict(y = r)

@autooperator
class PriorOperator:
    ain = [('x', '*')]
    aout = [('y', '*')]
    def main(self, x, d, powerspectrum, pm):
        return dict(y = x)

@autooperator
class ObjectiveOperator:
    ain = [('x', '*')]
    aout = [('y', '*'), ('chi2', '*'), ('prior', '*')]

    def main(self, x, d, powerspectrum, pm):
        r = ResidualOperator(x, d, powerspectrum, pm)

        chi2 = linalg.sum(linalg.mul(r, r))
        chi2 = mpi.allreduce(chi2, pm.comm)

        p = PriorOperator(x, d, powerspectrum, pm)
        prior = linalg.sum(linalg.mul(p, p))
        prior = mpi.allreduce(prior, pm.comm)

        y = linalg.add(chi2, prior)

        return dict(y = y, prior=prior, chi2=chi2)


class MyProblem(Problem):

    def __init__(self, d, powerspectrum, pm):
        self.full_graph = ObjectiveOperator.build(d=d, powerspectrum=powerspectrum, pm=pm)
        self.subgraphs = [
                    ResidualOperator.build(d=d, powerspectrum=powerspectrum, pm=pm),
                    PriorOperator.build(d=d, powerspectrum=powerspectrum, pm=pm),
                    ]

        def objective(x):
            print('obj', (x**2).sum())
            return self.full_graph.compute(vout='y', init=dict(x=x))

        def gradient(x):
            print('grad', (x**2).sum())
            y, [vjp] = self.full_graph.compute_with_vjp(init=dict(x=x), v=dict(_y=1.0, _chi2=0, _prior=0))
            return vjp

        def hessian_vector_product(x, v):
            print('hvp', (x**2).sum())
            Dv = 0
            for graph in self.subgraphs:
                y, [Dv1] = graph.compute_with_gnDp(vout='y', init=dict(x=x), v=dict(x_=v))
                Dv = Dv + Dv1
            # H is 2 JtJ, see wikipedia on Gauss Newton.
            return Dv * 2

        vs = real_vector_space

        Problem.__init__(self,
                        vs = vs,
                        objective=objective,
                        gradient=gradient,
                        hessian_vector_product=hessian_vector_product)

pm = fastpm.ParticleMesh([128, 128, 128], BoxSize=400.)
powerspectrum = LinearPower(Planck15, 0)

q = pm.generate_uniform_particle_grid()
wn = pm.generate_whitenoise(555, unitary=True)
x = wn[...]

x = numpy.stack([x.real, x.imag], -1)

forward_model = nl3d.build(q=q, stages=[1.0], cosmology=Planck15, powerspectrum=powerspectrum, pm=pm)
d = forward_model.compute('rho', init=dict(x=x))

problem = MyProblem(d=d, powerspectrum=powerspectrum, pm=pm)
problem.maxradius = 100
problem.initradus = 1
problem.cg_rtol = 0.1
problem.cg_maxiter= 10
trcg = TrustRegionCG()
trcg.cg_monitor = print

lbfgs = LBFGS(maxiter=30)
print(problem.f(x))
print(problem.g(x))
print(problem.hessian_vector_product(x, x))
"""
x1 = lbfgs.minimize(problem, x * 0.001, monitor=print)
x1 = trcg.minimize(problem, x1['x'], monitor=print)
"""
x1 = trcg.minimize(problem, x * 0.001, monitor=print)
