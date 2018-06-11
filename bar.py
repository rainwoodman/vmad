from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg
from vmad.contrib import cosmo4d
from abopt.trustregion import TrustRegionCG
from abopt.lbfgs import LBFGS

import numpy

from nbodykit.cosmology import Planck15, LinearPower

pm = cosmo4d.ParticleMesh([8, 8, 8], BoxSize=16.)

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

def monitor(state):
    #problem.save('/tmp/bar-%04d' % state['nit'], state)
    z = state.g
    print(abs(z.c2r().r2c() - z)[...].max() / z.cnorm())
    print(state)


sim_t = cosmo4d.SynthData.create(ForwardOperator, 333, Pss, Pnn)

for i in range(3):
    sim_b = cosmo4d.SynthData.create(ForwardOperator, 333, Pss, Pnn)

pprint(sim_t.attrs)

sim_t.save('/tmp/bar-truth')

sim_t = cosmo4d.SynthData.load('/tmp/bar-truth', pm.comm)

trcg = TrustRegionCG(
    maxradius = 100000,
    minradius = 1e-2,
    initradus = 1,
    atol = 0.1,
    cg_rtol = 0.1,
    cg_maxiter= 10,
)

trcg.cg_monitor = print

class MAPInversion(cosmo4d.MAPInversion):
    def problem_factory(self, S, N, d, smoothing):
        """ Create a problem object for a given set of args.

            Parameters
            ----------
            S : ComplexField
                prior power
            N : RealField
                noise variance

            d : RealField
                data

            smoothing : float
                subsampling fraction (length scale of a gaussian smoothing)
        """
        pm = self.ForwardOperator.hyperargs['pm']

        problem = cosmo4d.ChiSquareProblem(pm.comm,
                self.ForwardOperator,
                [
                    cosmo4d.PriorOperator.bind(invS=S ** -1),
                    cosmo4d.SmoothedNLResidualOperator.bind(d=d, invN=N ** -1, scale=smoothing),
                ]
                )

        return problem

    def schedule(self, S, N):
        self.schedule_problem('S', S)
        self.schedule_problem('N', N)
        self.schedule_optimizer('maxiter', [1, 2, 3, 4])
        self.schedule_problem('smoothing', [2, 1, 0, 0])

mapinv = MAPInversion(trcg, ForwardOperator)

# checking the problem
problem = mapinv.problem_factory(S=sim_t.S, N=sim_t.N, d=sim_t.d, smoothing=0)

print('objective(truth) =', problem.f(sim_t.s), 'expecting', pm.Nmesh.prod() * len(problem.residuals))
print('objective(0) =', problem.f(sim_t.s * 0.001))


mapinv.schedule(S=sim_t.S, N=sim_t.N)
shat_t = mapinv.apply(sim_t.d, epochs=[0, 1, 2, 3],
            monitor_epoch=print,
            monitor_progress=monitor)

sims = [None] * 8
shats = [None] * 8
for i in range(8):
    sims[i] = cosmo4d.SynthData.create(ForwardOperator, i, Pss, Pnn)
    shats[i] = mapinv.apply(sims[i].d, epochs=[0, 1, 2, 3],
                monitor_epoch=print,
                monitor_progress=monitor)


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

