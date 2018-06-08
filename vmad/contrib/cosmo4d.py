from vmad import Builder, autooperator
from vmad.lib import fastpm, mpi, linalg
from pmesh.pm import ParticleMesh, Field
from .chisquare import MPIChiSquareProblem, MPIVectorSpace
import numpy

@autooperator
class FastPMOperator:
    ain = [('x', '*')]
    aout = [('s', '*'), ('fs', '*')]

    def main(self, x, q, stages, cosmology, powerspectrum, pm):
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

