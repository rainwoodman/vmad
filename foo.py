from vmad import Builder
from fastpm.force.lpt import lpt1, lpt2source

from vmad.lib import fastpm
import numpy

from nbodykit.cosmology import Planck15, LinearPower

pm = fastpm.ParticleMesh([32, 32, 32], BoxSize=128.)
powerspectrum = LinearPower(Planck15, 0)

q = pm.generate_uniform_particle_grid()

with Builder() as model:
    x = model.input('x')

    wnk = fastpm.as_complex_field(x, pm)

    rhok = fastpm.induce_correlation(wnk, powerspectrum, pm)
    dx1, dx2 = fastpm.lpt(rhok, q, pm)
    dx, p, f = fastpm.nbody(rhok, q, [0.1, 0.6, 1.0], Planck15, pm)
    dx0, p0, f0 = fastpm.nbody(rhok, q, [0.1], Planck15, pm)
    model.output(dx1=dx1, dx2=dx2, dx=dx, p=p, dx0=dx0, p0=p0, f0=f0, f=f)

wn = pm.generate_whitenoise(555, unitary=True)
x = wn[...]

x = numpy.stack([x.real, x.imag], -1)

from fastpm.core import Solver, leapfrog

solver = Solver(pm, Planck15, B=1)
linear = solver.linear(wn, powerspectrum)

dx1_f = lpt1(linear, q)
dx2_f = lpt1(lpt2source(linear), q)
lpt = solver.lpt(linear, Q=q, a=0.1)

print('comparing lpt order by order')

dx1, dx2 = model.compute(['dx1', 'dx2'], init=dict(x=x))

print('model', dx1.std(axis=0), dx2.std(axis=0))
print('fastpm', dx1_f.std(axis=0), dx2_f.std(axis=0))
print('model', dx1[0], dx2[0])
print('fastpm', dx1_f[0], dx2_f[0])


print('comparing lpt dx and p ')
dx0, p0 = model.compute(['dx0', 'p0'], init=dict(x=x))
print('model', dx0.std(axis=0), p0.std(axis=0))
print('fastpm', lpt.S.std(axis=0), lpt.P.std(axis=0))

print('comparing single step')
lpt = solver.lpt(linear, Q=q, a=0.1)
state = solver.nbody(lpt, leapfrog([0.1]))
dx0, p0, f0 = model.compute(['dx0', 'p0', 'f0'], init=dict(x=x))
print('model', dx0.std(axis=0), p0.std(axis=0), f0.std(axis=0))
print('fastpm', state.S.std(axis=0), state.P.std(axis=0), state.F.std(axis=0))
print('model', dx0[0], p0[0], f0[0])
print('fastpm', state.S[0], state.P[0], state.F[0])

print('comparing multi step')
dx, p, f = model.compute(['dx', 'p', 'f'], init=dict(x=x))

lpt = solver.lpt(linear, Q=q, a=0.1)
state = solver.nbody(lpt, leapfrog([0.1, 0.6, 1.0]))
print('model', dx.std(axis=0), p.std(axis=0), f.std(axis=0))
print('fastpm', state.S.std(axis=0), state.P.std(axis=0), state.F.std(axis=0))

print('model', dx[0], p[0], f[0])
print('fastpm', state.S[0], state.P[0], state.F[0])
