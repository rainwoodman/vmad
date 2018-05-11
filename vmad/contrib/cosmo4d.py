from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg
from pmesh.pm import ParticleMesh
from .chisquare import MPIChiSquareProblem
import numpy

@autooperator
class FastPMOperator:
    ain = [('x', '*')]
    aout = [('wn', '*'), ('s', '*'), ('fs', '*')]

    def main(self, x, q, stages, cosmology, powerspectrum, pm):
        wnk = fastpm.as_complex_field(x, pm)

        wn = fastpm.c2r(wnk)
        rholnk = fastpm.induce_correlation(wnk, powerspectrum, pm)

        if len(stages) == 0:
            rho = fastpm.c2r(rholnk)
            rho = linalg.add(rho, 1.0)
        else:
            dx, p, f = fastpm.nbody(rholnk, q, stages, cosmology, pm)
            x = linalg.add(q, dx)
            layout = fastpm.decompose(x, pm)
            rho = fastpm.paint(x, mass=1, layout=layout, pm=pm)

        return dict(fs=rho, s=rholnk, wn=wn)

@autooperator
class NLResidualOperator:
    ain = [('wn', '*'), ('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, wn, s, fs, d, invvar):
        r = linalg.add(fs, d * -1)
        r = linalg.mul(r, invvar ** 0.5)

        return dict(y = r)

    @classmethod
    def evaluate(self, wn, s, fs, d, sigma):
        from nbodykit.lab import FFTPower
        r_x = FFTPower(d, second=fs, mode='1d') 
        r_fs = FFTPower(fs, mode='1d')  
        r_d = FFTPower(d, mode='1d')

        return dict(r_fs=r_fs, r_d=r_d, r_x=r_x)

@autooperator
class SmoothedNLResidualOperator:
    ain = [('wn', '*'), ('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, wn, s, fs, d, invvar, scale):
        r = linalg.add(fs, d * -1)
        def tf(k):
            k2 = sum(ki ** 2 for ki in k)
            return numpy.exp(- 0.5 * k2 * scale ** 2)
        c = fastpm.r2c(r)
        c = fastpm.apply_transfer(c, tf)
        r = fastpm.c2r(c)
        r = linalg.mul(r, invvar ** 0.5)
        return dict(y = r)

@autooperator
class LNResidualOperator:
    ain = [('wn', '*'), ('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, wn, s, fs, d, invvar):
        """ t is the truth, used only in evaluation. """
        r = linalg.add(wn, d * -1)
        fac = linalg.pow(wn.Nmesh.prod(), -0.5)
        fac = linalg.mul(fac, invvar ** 0.5)
        r = linalg.mul(r, fac)
        return dict(y = r)

    @classmethod
    def evaluate(self, wn, s, fs, t):
        from nbodykit.lab import FFTPower
        r_x = FFTPower(s, second=t, mode='1d')
        r_s = FFTPower(s, mode='1d')
        r_t = FFTPower(t, mode='1d')

        return dict(r_s=r_s, r_t=r_t, r_x=r_x)

@autooperator
class PriorOperator:
    ain = [('wn', '*'), ('s', '*'), ('fs', '*')]
    aout = [('y', '*')]
    def main(self, wn, s, fs):
        r = linalg.add(wn, 0)
        fac = linalg.pow(wn.Nmesh.prod(), -0.5)
        r = linalg.mul(r, fac)
        return dict(y = r)

    @classmethod
    def evaluate(self, wn, s, fs):
        return None

class ChiSquareProblem(MPIChiSquareProblem):
    def save(self, filename, state):
        with Builder() as m:
            x = m.input('x')
            wn, s, fs = self.forward_operator(x)
            m.output(wn=wn, s=s, fs=fs)

        wn, s, fs = m.compute(['wn', 's', 'fs'], init=dict(x=state['x']))

        from nbodykit.lab import FieldMesh

        wn = FieldMesh(wn)
        s = FieldMesh(s)
        fs = FieldMesh(fs)

        s.attrs['y'] = state['y']
        s.attrs['nit'] = state['nit']
        s.attrs['gev'] = state['gev']
        s.attrs['fev'] = state['fev']
        s.attrs['hev'] = state['hev']

        wn.save(filename, dataset='wn', mode='real')
        s.save(filename, dataset='s', mode='real', )
        fs.save(filename, dataset='fs', mode='real')

