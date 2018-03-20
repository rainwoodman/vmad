from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg

import numpy
#from abopt.lbfgs import LBFGS

from nbodykit.cosmology import Planck15, LinearPower

pm = fastpm.ParticleMesh([32, 32, 32], BoxSize=128.)
powerspectrum = LinearPower(Planck15, 0)

q = pm.generate_uniform_particle_grid()

@autooperator
class nl3d:
    ain = [('x', '*')]
    aout = [('rho', '*')]

    def main(self, x, q, stages, cosmology, powerspectrum, pm):
        wnk = fastpm.as_complex_field(x, pm)

        rhok = fastpm.induce_correlation(wnk, powerspectrum, pm)

        dx, p, f = fastpm.nbody(rhok, q, stages, cosmology, pm)
        x = linalg.add(q, dx)
        layout = fastpm.decompose(x, pm)
        rho = fastpm.paint(x, mass=1, layout=layout, pm=pm)
        return dict(rho = rho)

@autooperator
class F:
    ain = [('x', '*')]
    aout = [('y', '*')]
    def main(self, x, d, powerspectrum, pm):
        rho = nl3d(x, q, [0.1, 0.6, 1.0], Planck15, powerspectrum, pm)
        r = linalg.add(rho, d * -1)

        return dict(y = r)

@autooperator
class P:
    ain = [('x', '*')]
    aout = [('y', '*')]
    def main(self, x, d, powerspectrum, pm):
        return dict(y = x)

@autooperator
class objective:
    ain = [('x', '*')]
    aout = [('y', '*'), ('chi2', '*'), ('prior', '*')]
    def main(self, x, d, powerspectrum, pm):
        r = F(x, d, powerspectrum, pm)

        chi2 = linalg.sum(linalg.mul(r, r))
        chi2 = mpi.allreduce(chi2, pm.comm)

        p = P(x, d, powerspectrum, pm)
        prior = linalg.sum(linalg.mul(p, p))
        prior = mpi.allreduce(prior, pm.comm)

        y = linalg.add(chi2, prior)

        return dict(y = y, prior=prior, chi2=chi2)


wn = pm.generate_whitenoise(555, unitary=True)
x = wn[...]

x = numpy.stack([x.real, x.imag], -1)

model = nl3d.build(q=q, stages=[0.1, 0.6, 1.0], cosmology=Planck15, powerspectrum=powerspectrum, pm=pm)
d = model.compute('rho', init=dict(x=x))

model = objective.build(d=d, powerspectrum=powerspectrum, pm=pm)
y, chi2, prior = model.compute(['y', 'chi2', 'prior'], init=dict(x=x))
print(y, chi2, prior)

Fmodel = F.build(d=d, powerspectrum=powerspectrum, pm=pm)
Pmodel = P.build(d=d, powerspectrum=powerspectrum, pm=pm)

f = Fmodel.compute('y', init=dict(x=x))
p = Pmodel.compute('y', init=dict(x=x))

def gn_hessian_dot(Fmodel, Pmodel, x, v):
    # gauss newton hessian dot is
    f, Ftape = Fmodel.compute('y', init=dict(x=x), return_tape=True)
    p, Ptape = Pmodel.compute('y', init=dict(x=x), return_tape=True)
    jvjp1 = Ftape.compute_jvjp('_x', ['y'], init=dict(x_ = v))
    jvjp2 = Ptape.compute_jvjp('_x', ['y'], init=dict(x_ = v))

    return jvjp1 + jvjp2

#print(f, p)
print(hessian_dot(Fmodel, Pmodel, x, x))
