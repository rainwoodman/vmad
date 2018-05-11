from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg
from vmad.contrib import cosmo4d

from abopt.trustregion import TrustRegionCG

import numpy

from abopt.lbfgs import LBFGS

from abopt.abopt2 import real_vector_space, Problem, VectorSpace

from nbodykit.cosmology import Planck15, LinearPower

oldprint = print

pm = cosmo4d.ParticleMesh([4, 4, 4], BoxSize=400.)

def print(*args):
    if pm.comm.rank == 0:
        oldprint(*args)

powerspectrum = LinearPower(Planck15, 0)
noise_powerspectrum = lambda k: 100.

n = pm.generate_whitenoise(333, unitary=True).apply(
        lambda k, v: v * (noise_powerspectrum(k) / pm.BoxSize.prod()) ** 0.5).c2r()

noise_variance = noise_powerspectrum(0.) / (pm.BoxSize / pm.Nmesh).prod()

wn = pm.generate_whitenoise(555, unitary=True)

x = wn[...]
x = numpy.stack([x.real, x.imag], -1)

ForwardModelHyperParameters = dict(
            q = pm.generate_uniform_particle_grid(),
            stages=[0.4, 0.6, 1.0],
            cosmology=Planck15,
            powerspectrum=powerspectrum,
            pm=pm)

ForwardOperator = cosmo4d.FastPMOperator.bind(**ForwardModelHyperParameters)

fs, wn, s = ForwardOperator.build().compute(('fs', 'wn', 's'), init=dict(x=x))

def save_truth(filename, fs, wn, s, n):
    from nbodykit.lab import FieldMesh


    print('<fs>, <wn>, <s>', fs.cmean(), wn.cmean(), s.c2r().cmean())

    d = FieldMesh(fs + n)
    wn = FieldMesh(wn)
    fs = FieldMesh(fs)
    s = FieldMesh(s)

    wn.save(filename, dataset='wn', mode='real')
    s.save(filename, dataset='s', mode='real')
    fs.save(filename, dataset='fs', mode='real')
    d.save(filename, dataset='d', mode='real')

save_truth('/tmp/bar-truth', fs=fs, wn=wn, s=s, n=n)

problem = cosmo4d.ChiSquareProblem(pm.comm,
        ForwardOperator,
        [
            cosmo4d.PriorOperator,
#            cosmo4d.LNResidualOperator.bind(d=wn, invvar=0),
            cosmo4d.NLResidualOperator.bind(d=fs + n, invvar=noise_variance ** -1),
        ]
        )

problem.maxradius = 100
problem.initradus = 1
problem.atol = 0.1
problem.cg_rtol = 0.1
problem.cg_maxiter= 10
trcg = TrustRegionCG()
trcg.cg_monitor = print

def monitor(state):
    #problem.save('/tmp/bar-%04d' % state['nit'], state)
    print(state)

lbfgs = LBFGS(maxiter=30)
print('objective(truth) =', problem.f(x), 'expecting', pm.Nmesh.prod() * len(problem.residuals))

#print('gradient = ', problem.g(x))
#print('hessian vector product = ', problem.hessian_vector_product(x, x))
#print('hessian vector product = ', problem.hessian_vector_product(x, x).shape)

"""
x1 = lbfgs.minimize(problem, x * 0.001, monitor=print)
x1 = trcg.minimize(problem, x1['x'], monitor=print)
"""
#x1 = lbfgs.minimize(problem, x * 0.001, monitor=print)

x1 = trcg.minimize(problem, x * 0.001, monitor=monitor)

