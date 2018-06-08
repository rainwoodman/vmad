from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg
from vmad.contrib import cosmo4d
from abopt.trustregion import TrustRegionCG
from abopt.lbfgs import LBFGS

import numpy

from nbodykit.cosmology import Planck15, LinearPower

pm = cosmo4d.ParticleMesh([4, 4, 4], BoxSize=300.)

def print(*args, **kwargs):
    comm = pm.comm
    from builtins import print
    if comm.rank == 0:
        print(*args, **kwargs)


powerspectrum = LinearPower(Planck15, 0)
noise_powerspectrum = lambda k: 1.

n = cosmo4d.convolve(noise_powerspectrum, pm.generate_whitenoise(333, unitary=True, type='real'))
N = cosmo4d.convolve(noise_powerspectrum, pm.create(value=1, type='real')) ** 2 * pm.Nmesh.prod()
s_truth = cosmo4d.convolve(powerspectrum, pm.generate_whitenoise(555, unitary=True))
S = cosmo4d.convolve(powerspectrum, pm.create(type='complex', value=1)) ** 2
S[S == 0] = 1.0

print(numpy.var(n), N.cmean(), N[0, 0, 0], noise_powerspectrum(0) / (pm.BoxSize / pm.Nmesh).prod())
print(S[0, 0, 1])


ForwardModelHyperParameters = dict(
            q = pm.generate_uniform_particle_grid(),
            stages=[0.4, 0.6, 1.0],
            cosmology=Planck15,
            powerspectrum=powerspectrum,
            pm=pm)

ForwardOperator = cosmo4d.FastPMOperator.bind(**ForwardModelHyperParameters)

fs, s = ForwardOperator.build().compute(('fs', 's'), init=dict(x=s_truth))

def save_truth(filename, fs, s, n):
    from nbodykit.lab import FieldMesh

    print('<fs>, <s>', fs.cmean(), s.c2r().cmean())

    d = FieldMesh(fs + n)
    fs = FieldMesh(fs)
    s = FieldMesh(s)

    s.save(filename, dataset='s', mode='real')
    fs.save(filename, dataset='fs', mode='real')
    d.save(filename, dataset='d', mode='real')

save_truth('/tmp/bar-truth', fs=fs, s=s, n=n)

problem = cosmo4d.ChiSquareProblem(pm.comm,
        ForwardOperator,
        [
            cosmo4d.PriorOperator.bind(invS=S ** -1),
            cosmo4d.NLResidualOperator.bind(d=fs + n, invN=N ** -1),
        ]
        )

trcg = TrustRegionCG(
    maxradius = 10000,
    minradius = 1e-4,
    initradus = 1,
    atol = 0.1,
    cg_rtol = 0.1,
    cg_maxiter= 10,
)

trcg.cg_monitor = print

def monitor(state):
    #problem.save('/tmp/bar-%04d' % state['nit'], state)
    print(state)

lbfgs = LBFGS(maxiter=30)
print('objective(truth) =', problem.f(s_truth), 'expecting', pm.Nmesh.prod() * len(problem.residuals))
print('objective(truth) =', problem.f(s_truth * 0.001), 'expecting', pm.Nmesh.prod() * len(problem.residuals))

print('gradient = ', problem.g(s_truth))
#print('hessian vector product = ', problem.hessian_vector_product(x, x))
#print('hessian vector product = ', problem.hessian_vector_product(x, x).shape)

"""
x1 = lbfgs.minimize(problem, x * 0.001, monitor=print)
x1 = trcg.minimize(problem, x1['x'], monitor=print)
"""
#x1 = lbfgs.minimize(problem, s_truth * 0.001, monitor=monitor)

x1 = trcg.minimize(problem, s_truth * 0.001, monitor=monitor)

