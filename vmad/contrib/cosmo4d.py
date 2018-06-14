from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg
from pmesh.pm import ParticleMesh, Field
from .chisquare import MPIChiSquareProblem, MPIVectorSpace, MAPInversion
from abopt.abopt2 import Preconditioner

import numpy

@autooperator
class FastPMOperator:
    ain = [('x', '*')]
    aout = [('s', '*'), ('fs', '*')]

    def main(self, x, q, stages, cosmology, pm):
        rholnk = x

        if len(stages) == 0:
            rho = fastpm.c2r(rholnk)
            rho = linalg.add(rho, 1.0)
        else:
            dx, p, f = fastpm.nbody(rholnk, q, stages, cosmology, pm)
            x = linalg.add(q, dx)
            layout = fastpm.decompose(x, pm)
            rho = fastpm.paint(x, mass=1, layout=layout, pm=pm)

        return dict(fs=rho, s=rholnk)

@autooperator
class NLResidualOperator:
    ain = [('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, s, fs, d, invN):
        r = linalg.add(fs, d * -1)
        r = linalg.mul(r, invN ** 0.5)

        return dict(y = r)

@autooperator
class SmoothedNLResidualOperator:
    ain = [('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, s, fs, d, invN, scale):
        r = linalg.add(fs, d * -1)
        def tf(k):
            k2 = sum(ki ** 2 for ki in k)
            return numpy.exp(- 0.5 * k2 * scale ** 2)
        c = fastpm.r2c(r)
        c = fastpm.apply_transfer(c, tf)
        r = fastpm.c2r(c)
        r = linalg.mul(r, invN ** 0.5)
        return dict(y = r)

@autooperator
class LNResidualOperator:
    ain = [('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, s, fs, d, invN):
        """ t is the truth, used only in evaluation. """
        r = linalg.add(s, d * -1)
        fac = linalg.pow(wn.Nmesh.prod(), -0.5)
        fac = linalg.mul(fac, invN ** 0.5)
        r = linalg.mul(r, fac)
        return dict(y = r)

@autooperator
class PriorOperator:
    ain = [('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, s, fs, invS):
        # when |s|^2 and invS are the same, this is supposed
        # to return 1.0
        fac = linalg.pow(s.pm.Nmesh.prod(), -0.5)
        fac = linalg.mul(fac, invS ** 0.5)
        s_over_Shalf = linalg.mul(s, fac)
        r = fastpm.c2r(s_over_Shalf)
        return dict(y = r)

class FastPMVectorSpace(MPIVectorSpace):
    def dot(self, a, b):
        """ einsum('i,i->', a, b) """
        if isinstance(a, Field):
            return a.cdot(b).real
        else:
            return self.comm.allreduce(numpy.sum(a * b))

class ChiSquareProblem(MPIChiSquareProblem):
    ComplexOptimizer = Precondition(Pvp=lambda x, direction: x, vPp=lambda x, direction: x)

    RealOptimizer = Preconditioner(
        Pvp=lambda x, direction:
            x.c2r() / x.Nmesh.prod() if direction > 0 else x.c2r(),
        vPp=lambda x, direction:
            x.r2c() if direction > 0 else x.r2c() * x.Nmesh.prod()
    )

    def __init__(self, comm, forward_operator, residuals):
        vs = FastPMVectorSpace(comm)
        MPIChiSquareProblem.__init__(self, comm, forward_operator, residuals, vs)

    def save(self, filename, state):
        with Builder() as m:
            s = m.input('s')
            s, fs = self.forward_operator(s)
            m.output(s=s, fs=fs)

        s, fs = m.compute(['s', 'fs'], init=dict(s=state['x']))

        from nbodykit.lab import FieldMesh

        s = FieldMesh(s)
        fs = FieldMesh(fs)

        s.attrs['y'] = state['y']
        s.attrs['nit'] = state['nit']
        s.attrs['gev'] = state['gev']
        s.attrs['fev'] = state['fev']
        s.attrs['hev'] = state['hev']

        s.save(filename, dataset='s', mode='real', )
        fs.save(filename, dataset='fs', mode='real')

def convolve(power_spectrum, field):
    """ Convolve a field with a power spectrum.
        Preserve the type of the output field.
    """
    c = field.cast(type='complex')
    c = c.apply(lambda k, v: (power_spectrum(k.normp() ** 0.5) / c.pm.BoxSize.prod()) ** 0.5 * v)
    return c.cast(type=type(field))

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
    def create(kls, ForwardOperator, seed, Pss, Pnn, unitary=False):
        pm = ForwardOperator.hyperargs['pm']

        rng = numpy.random.RandomState(seed)
        noiseseed = rng.randint(0xffffff)
        signalseed = rng.randint(0xffffff)

        powerspectrum = lambda k: Pss(k)
        noise_powerspectrum = lambda k: Pnn(k)

        # n is a RealField
        # N is noise-covariance, diagonal thus just a RealField.
        n = convolve(noise_powerspectrum, pm.generate_whitenoise(noiseseed, unitary=unitary, type='real'))
        # N is the same as the variance of n, due to the multiplication of Nmesh.
        N = convolve(noise_powerspectrum, pm.create(value=1, type='real')) ** 2 * pm.Nmesh.prod()

        # t is a ComplexField
        # S is covariance, diagonal thus just a ComplexField
        t = convolve(powerspectrum, pm.generate_whitenoise(signalseed, unitary=unitary, type='complex'))
        # S the power of S ** 0.5 is the same as power spectrum
        S = convolve(powerspectrum, pm.create(type='complex', value=1)) ** 2
        S[S == 0] = 1.0

        fs, s = ForwardOperator.build().compute(('fs', 's'), init=dict(x=t))

        attrs = {}
        attrs['noiseseed'] = noiseseed
        attrs['signalseed'] = signalseed
        attrs['noisevariance'] = (n ** 2).cmean()
        attrs['expected_noisevariance'] = noise_powerspectrum(0) / (pm.BoxSize / pm.Nmesh).prod()
        attrs['fsvariance'] = ((fs - 1)**2).cmean()

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

        return kls(S=S, s=s, fs=fs, d=fs+n, N=N, n=n, attrs=attrs)

    @classmethod
    def save_extra(kls, filename, dataset, field):
        """ some extra data. """
        from nbodykit.lab import FieldMesh
        s = FieldMesh(field)
        s.save(filename, dataset=dataset, mode='complex')

    @classmethod
    def load_extra(kls, filename, dataset, comm):
        from nbodykit.lab import BigFileMesh
        s = BigFileMesh(filename, dataset=dataset, comm=comm).compute(mode='complex')
        return s

    def save(self, filename):
        from nbodykit.lab import FieldMesh
        from bigfile import File

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

