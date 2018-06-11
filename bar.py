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



Pss = LinearPower(Planck15, 0)
Pnn = lambda k: 1.0

ForwardModelHyperParameters = dict(
            q = pm.generate_uniform_particle_grid(),
            stages=[1.0],
            cosmology=Planck15,
            pm=pm)

ForwardOperator = cosmo4d.FastPMOperator.bind(**ForwardModelHyperParameters)

class SynthData:
    """ This object represents a synthetic data.

        A synthetic data is draw with a random seed
        from 
    """
    def __init__(self, S, s, N, n, fs, d, attrs={}):
        self.s = s
        self.fs = fs
        self.d = d
        self.n = n
        self.S = S
        self.N = N

        self.attrs = {}
        self.attrs.update(attrs)

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

        attrs = {}
        attrs['noiseseed'] = noiseseed
        attrs['signalseed'] = signalseed
        attrs['noisevariance'] = (n ** 2).cmean()
        attrs['fsvariance'] = ((fs - 1)**2).cmean()

        print(attrs)
        return kls(S=S, s=s, fs=fs, d=fs+n, N=N, n=n, attrs=attrs)

    @classmethod
    def load(kls, filename, comm):
        from nbodykit.lab import BigFileMesh
        from bigfile import File

        d = BigFileMesh(filename, dataset='d', comm=comm).compute(mode='real')
        fs = BigFileMesh(filename, dataset='fs', comm=comm).compute(mode='real')
        s = BigFileMesh(filename, dataset='s', comm=comm).compute(mode='complex')
        n = BigFileMesh(filename, dataset='n', comm=comm).compute(mode='real')
        N = BigFileMesh(filename, dataset='N', comm=comm).compute(mode='real')
        S = BigFileMesh(filename, dataset='S', comm=comm).compute(mode='complex')

        attrs = {}
        with File(filename) as bf:
            with bf['Header'] as bb:
                for key in bb.attrs:
                    attrs[key] = bb.attrs[key]

        print(attrs)
        return kls(S=S, s=s, fs=fs, d=fs+n, N=N, n=n, attrs=attrs)

    def save(self, filename):
        from nbodykit.lab import FieldMesh
        from bigfile import File
        print('<fs>, <s>', self.fs.cmean(), self.s.c2r().cmean())

        d = FieldMesh(self.d)
        fs = FieldMesh(self.fs)
        s = FieldMesh(self.s)
        n = FieldMesh(self.n)
        N = FieldMesh(self.N)
        S = FieldMesh(self.S)

        s.save(filename, dataset='s', mode='complex')
        S.save(filename, dataset='S', mode='complex')
        n.save(filename, dataset='n', mode='real')
        N.save(filename, dataset='N', mode='real')
        fs.save(filename, dataset='fs', mode='real')
        d.save(filename, dataset='d', mode='real')

        with File(filename) as bf:
            with bf.create("Header") as bb:
                for key in self.attrs:
                    bb.attrs[key] = self.attrs[key]

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

sim_t = SynthData.create(ForwardOperator, 333, Pss, Pnn)

sim_t.save('/tmp/bar-truth')

sim_t = SynthData.load('/tmp/bar-truth', pm.comm)

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

