from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg
from vmad.contrib import cosmo4d
from abopt.trustregion import TrustRegionCG
from abopt.lbfgs import LBFGS

import numpy

from nbodykit.cosmology import Planck15, LinearPower

pm = cosmo4d.ParticleMesh([32, 32, 32], BoxSize=32.)

def print(*args, **kwargs):
    comm = pm.comm
    from builtins import print
    if comm.rank == 0:
        print(*args, **kwargs)



Pss = LinearPower(Planck15, 0)
Pnn = lambda k: 1.0

ForwardModelHyperParameters = dict(
            q = pm.generate_uniform_particle_grid(),
            stages=[1.0],
            cosmology=Planck15,
            pm=pm)

ForwardOperator = cosmo4d.FastPMOperator.bind(**ForwardModelHyperParameters)

class SynData:
    def __init__(self, S, s, N, n, fs, d):
        self.s = s
        self.fs = fs
        self.d = d
        self.n = n
        self.S = S
        self.N = N

    @classmethod
    def create(kls, ForwardOperator, seed, Pss, Pnn):
        pm = ForwardOperator.hyperargs['pm']

        rng = numpy.random.RandomState(seed)
        noiseseed = rng.randint(0xffffff)
        signalseed = rng.randint(0xffffff)

        powerspectrum = lambda k: Pss(k)
        noise_powerspectrum = lambda k: Pnn(k)

        # n is a RealField
        # N is covariance, diagonal thus just a RealField
        n = cosmo4d.convolve(noise_powerspectrum, pm.generate_whitenoise(noiseseed, unitary=True, type='real'))
        N = cosmo4d.convolve(noise_powerspectrum, pm.create(value=1, type='real')) ** 2 * pm.Nmesh.prod()

        # t is a ComplexField
        # S is covariance, diagonal thus just a ComplexField
        t = cosmo4d.convolve(powerspectrum, pm.generate_whitenoise(signalseed, unitary=True, type='complex'))
        S = cosmo4d.convolve(powerspectrum, pm.create(type='complex', value=1)) ** 2
        S[S == 0] = 1.0

        print(numpy.var(n), N.cmean(), N[0, 0, 0], noise_powerspectrum(0) / (pm.BoxSize / pm.Nmesh).prod())
        print(S[0, 0, 1])

        fs, s = ForwardOperator.build().compute(('fs', 's'), init=dict(x=t))

        return kls(S=S, s=s, fs=fs, d=fs+n, N=N, n=n)

    def save(self, filename):
        from nbodykit.lab import FieldMesh

        print('<fs>, <s>', self.fs.cmean(), self.s.c2r().cmean())

        d = FieldMesh(self.d)
        fs = FieldMesh(self.fs)
        s = FieldMesh(self.s)
        n = FieldMesh(self.n)
        N = FieldMesh(self.N)
        S = FieldMesh(self.S)

        s.save(filename, dataset='s', mode='real')
        S.save(filename, dataset='S', mode='real')
        n.save(filename, dataset='n', mode='real')
        N.save(filename, dataset='N', mode='real')
        fs.save(filename, dataset='fs', mode='real')
        d.save(filename, dataset='d', mode='real')


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

sim_t = SynData.create(ForwardOperator, 333, Pss, Pnn)

sim_t.save('/tmp/bar-truth')

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

