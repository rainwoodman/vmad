from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg
from vmad.contrib import cosmo4d
from abopt.trustregion import TrustRegionCG
from abopt.lbfgs import LBFGS

import numpy

from nbodykit.cosmology import Planck15, LinearPower

pm = cosmo4d.ParticleMesh([32, 32, 32], BoxSize=16.)

def print(*args, **kwargs):
    comm = pm.comm
    from builtins import print
    if comm.rank == 0:
        print(*args, **kwargs)

def pprint(*args, **kwargs):
    comm = pm.comm
    from pprint import pprint
    if comm.rank == 0:
        pprint(*args, **kwargs)



Pss = LinearPower(Planck15, 0)
Pnn = lambda k: 1.0

ForwardModelHyperParameters = dict(
            q = pm.generate_uniform_particle_grid(),
            stages=[1.0],
            cosmology=Planck15,
            pm=pm)

ForwardOperator = cosmo4d.FastPMOperator.bind(**ForwardModelHyperParameters)

def ProblemFactory(pm, ForwardOperator, S, N, d):
    problem = cosmo4d.ChiSquareProblem(pm.comm,
            ForwardOperator,
            [
                cosmo4d.PriorOperator.bind(invS=S ** -1),
                cosmo4d.NLResidualOperator.bind(d=d, invN=N ** -1),
    #            cosmo4d.SmoothedNLResidualOperator.bind(d=fs + n, invN=N ** -1, scale=4.0),
            ]
            )
    return problem

sim_t = cosmo4d.SynthData.create(ForwardOperator, 333, Pss, Pnn)

pprint(sim_t.attrs)

sim_t.save('/tmp/bar-truth')

sim_t = cosmo4d.SynthData.load('/tmp/bar-truth', pm.comm)

problem = ProblemFactory(pm, ForwardOperator, sim_t.S, sim_t.N, sim_t.d)

trcg = TrustRegionCG(
    maxradius = 100000,
    minradius = 1e-2,
    initradus = 1,
    atol = 0.1,
    cg_rtol = 0.1,
    cg_maxiter= 10,
)

trcg.cg_monitor = print

def monitor(state):
    #problem.save('/tmp/bar-%04d' % state['nit'], state)
    z = state.g
    print(abs(z.c2r().r2c() - z)[...].max() / z.cnorm())
    print(state)

print('objective(truth) =', problem.f(sim_t.s), 'expecting', pm.Nmesh.prod() * len(problem.residuals))
print('objective(0) =', problem.f(sim_t.s * 0.001))

#print('gradient = ', problem.g(t * 0.001))
#print('hessian vector product = ', problem.hessian_vector_product(x, x))
#print('hessian vector product = ', problem.hessian_vector_product(x, x).shape)

lbfgs = LBFGS(maxiter=3)
problem.set_preconditioner('complex')
s1 = sim_t.s * 0.001
#state = lbfgs.minimize(problem, t* 0.001, monitor=monitor)
#s1 = state['x']
#problem.set_preconditioner('real')
state = trcg.minimize(problem, s1, monitor=monitor)

