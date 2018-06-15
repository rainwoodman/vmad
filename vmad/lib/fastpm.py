from vmad import operator, autooperator
from vmad.core.model import Literal
from pmesh.pm import ParticleMesh
from vmad.lib import linalg
import numpy

@operator
class to_scalar:
    ain = {'x' : 'RealField'}
    aout = {'y' : '*'}

    def apl(self, x):
        return dict(y = x.cnorm())

    def vjp(self, _y, x):
        return dict(_x= x * (2 * _y))

    def jvp(self, x_, x):
        return dict(y_ = x.cdot(x_) * 2)

# FIXME: this is probably not correct.
"""
@operator
class to_scalar_co:
    ain = {'x' : 'ndarray'}
    aout = {'y' : '*'}

    def apl(self, x, comm):
        return dict(y = comm.allreduce((x * numpy.conj(x)).sum()))

    def vjp(self, _y, x, comm):
        return dict(_x = 2 * numpy.conj(_y) * x)

    def jvp(self, x_, x, comm):
        return dict(y_ = comm.allreduce((x_ * numpy.conj(x) + numpy.conj(x_) * x).sum()))
"""

@operator
class as_complex_field:
    ain = {'x' : '*'}
    aout = {'y' : 'ComplexField'}

    def apl(self, x, pm):
        y = pm.create(mode='complex')
        y.real[...] = x[..., 0]
        y.imag[...] = x[..., 1]
        return dict(y=y)

    def vjp(self, _y, pm):
        _x = numpy.stack([_y.real, _y.imag], axis=-1)
        return dict(_x=_x)

    def jvp(self, x_, pm):
        y_ = pm.create(mode='complex')
        y_.real[...] = x_[..., 0]
        y_.imag[...] = x_[..., 1]
        return dict(y_=y_)

@operator
class r2c:
    ain = {'x' : 'RealField'}
    aout = {'y' : 'ComplexField'}

    def apl(self, x):
        return dict(y=x.r2c(), pm=x.pm)
    def vjp(self, _y, pm):
        _y = pm.create(mode='complex', value=_y)
        return dict(_x=_y.r2c_vjp())
    def jvp(self, x_, pm):
        x_ = pm.create(mode='real', value=x_)
        return dict(y_=x_.r2c())

@operator
class c2r:
    ain = {'x' : 'ComplexField'}
    aout = {'y' : 'RealField'}

    def apl(self, x):
        return dict(y=x.c2r(), pm=x.pm)
    def vjp(self, _y, pm):
        _y = pm.create(mode='real', value=_y)
        return dict(_x=_y.c2r_vjp())
    def jvp(self, x_, pm):
        x_ = pm.create(mode='complex', value=x_)
        return dict(y_=x_.c2r())

@operator
class apply_transfer:
    ain = {'x' : 'ComplexField'}
    aout = {'y' : 'ComplexField'}

    def apl(self, x, tf):
        filter = lambda k, v: v * tf(k)
        return dict(y=x.apply(filter))

    def vjp(self, _y, tf):
        filter = lambda k, v: v * numpy.conj(tf(k))
        return dict(_x=_y.apply(filter))

    def jvp(self, x_, tf):
        filter = lambda k, v: v * tf(k)
        return dict(y_=x_.apply(filter))

@operator
class paint:
    aout = {'mesh' : 'RealField'}
    ain =  {'x': 'ndarray',
           'layout': 'Layout',
           'mass' : 'ndarray'
           }

    def apl(self, x, mass, layout, pm):
        mesh = pm.paint(x, mass=mass, layout=layout, hold=False)
        return dict(mesh=mesh)

    def vjp(self, _mesh, x, mass, layout, pm):
        N = pm.comm.allreduce(layout.oldlength)
        _mesh = pm.create(mode='real', value=_mesh)
        _x, _mass = pm.paint_vjp(_mesh, x, layout=layout, mass=mass)
        return dict(
            _layout = 0,
            _x=_x,
            _mass=_mass)

    def jvp(self, x_, x, layout, mass, layout_, mass_, pm):
        if x_ is 0: x_ = None
        if mass_ is 0: mass_ = None # force cast it to a scalar 0, so make it None
        mesh_ = pm.paint_jvp(x, v_mass=mass_, mass=mass, v_pos=x_, layout=layout)

        return dict(mesh_=mesh_)

@operator
class readout:
    aout = {'value' : 'ndarray'}

    ain = {'x': 'ndarray',
        'mesh': 'RealField',
      'layout' : 'Layout'}

    def apl(self, mesh, x, layout, resampler=None):
        N = mesh.pm.comm.allreduce(len(x))
        value = mesh.readout(x, layout=layout, resampler=resampler)
        return dict(value=value, pm=mesh.pm)

    def vjp(self, _value, x, layout, mesh, pm, resampler=None):
        _mesh, _x = mesh.readout_vjp(x, _value, layout=layout, resampler=resampler)
        return dict(_mesh=_mesh, _x=_x, _layout=0)

    def jvp(self, x_, mesh_, x, layout, layout_, mesh, pm, resampler=None):
        if mesh_ is 0: mesh_ = None
        if x_ is 0: x_ = None
        mesh = pm.create(mode='real', value=mesh)
        value_ = mesh.readout_jvp(x, v_self=mesh_, v_pos=x_, layout=layout, resampler=resampler)
        return dict(value_=value_)

@operator
class decompose:
    aout={'layout' : 'Layout'}
    ain={'x': 'ndarray'}

    def apl(self, x, pm):
        return dict(layout=pm.decompose(x))

    def vjp(engine, _layout):
        return dict(_x=0)

    def jvp(engine, x_):
        return dict(layout_=0)

@operator
class gather:
    aout='y'
    ain ='x', 'layout'

    def apl(self, x, layout):
        return dict(y=layout.gather(x))

    def vjp(engine, _y, layout):
        _x = layout.exchange(_y)
        return dict(_layout=0, _x=_x)

    def jvp(engine, layout, layout_, x_,):
        return dict(y_=layout.gather(x_))

@operator
class exchange:
    aout='y'
    ain ='x', 'layout'

    def apl(self, x, layout):
        return dict(y=layout.exchange(x))

    def vjp(engine, _y, layout):
        _x = layout.gather(_y)
        return dict(_layout=0, _x=_x)

    def jvp(engine, layout, layout_, x_,):
        return dict(y_=layout.exchange(x_))

def fourier_space_neg_gradient(dir, pm):
    nyquist =  numpy.pi / pm.BoxSize[dir] * (pm.Nmesh[dir] - 0.01)
    def filter(k):
        mask = abs(k[dir]) < nyquist
        return - 1j * k[dir] * mask
    return filter

def fourier_space_laplace(k):
    k2 = k.normp(2)
    bad = k2 == 0
    k2[bad] = 1
    k2 = - 1 / k2
    k2[bad] = 0
    return k2

@autooperator
class lpt1:
    ain = [
            ('rhok',  'ComplexField'),
          ]
    aout = [
            ('dx1', '*'),
         #, ('dx2', '*'),
           ]

    def main(self, rhok, q, pm):
        p = apply_transfer(rhok, fourier_space_laplace)
        q = Literal(self, q)

        layout = decompose(q, pm)

        r1 = []
        for d in range(pm.ndim):
            dx1_c = apply_transfer(p, fourier_space_neg_gradient(d, pm))
            dx1_r = c2r(dx1_c)
            dx1 = readout(dx1_r, q, layout)
            r1.append(dx1)

        dx1 = linalg.stack(r1, axis=-1)

        return dict(dx1 = dx1)

@autooperator
class lpt2src:
    ain = [
            ('rhok',  'ComplexField'),
          ]
    aout = [
            ('rho_lpt2', 'RealField'),
           ]

    def main(self, rhok, pm):
        if pm.ndim != 3:
            raise ValueError("LPT 2 only works in 3d")

        D1 = [1, 2, 0]
        D2 = [2, 0, 1]

        potk = apply_transfer(rhok, fourier_space_laplace)

        Pii = []
        for d in range(pm.ndim):
            t = apply_transfer(potk, fourier_space_neg_gradient(d, pm))
            Pii1 = apply_transfer(t, fourier_space_neg_gradient(d, pm))
            Pii1 = c2r(Pii1)
            Pii.append(Pii1)

        source = None
        for d in range(pm.ndim):
            source1 = linalg.mul(Pii[D1[d]], Pii[D2[d]])
            if source is None:
                source = source1
            else:
                source = linalg.add(source, source1)

        for d in range(pm.ndim):
            t = apply_transfer(potk, fourier_space_neg_gradient(D1[d], pm))
            Pij1 = apply_transfer(t, fourier_space_neg_gradient(D2[d], pm))
            Pij1 = c2r(Pij1)
            neg = linalg.mul(Pij1, -1)
            source1 = linalg.mul(Pij1, neg)
            source = linalg.add(source, source1)

        source = linalg.mul(source, 3.0/7 )

        return dict(rho_lpt2=source)

@autooperator
class induce_correlation:
    ain = [
            ('wnk',  'ComplexField'),
          ]
    aout = [
            ('c', 'ComplexField'),
           ]

    def main(self, wnk, powerspectrum, pm):
        def tf(k):
            k = sum(ki ** 2 for ki in k) ** 0.5
            return (powerspectrum(k) / pm.BoxSize.prod()) ** 0.5

        c = apply_transfer(wnk, tf)
        return dict(c = c)

@autooperator
class lpt:
    ain = [
            ('rhok',  'RealField'),
          ]

    aout = [
            ('dx1', '*'),
            ('dx2', '*'),
           ]

    def main(self, rhok, q, pm):

        dx1 = lpt1(rhok, q, pm)
        source2 = lpt2src(rhok, pm)
        rhok2 = r2c(source2)
        dx2 = lpt1(rhok2, q, pm)

        return dict(dx1=dx1, dx2=dx2)


@autooperator
class gravity:
    ain = [ ('dx', '*'),
          ]
    aout = [ ('f', '*')]

    def main(self, dx, q, pm):
        from vmad.lib.utils import watchpoint
        x = linalg.add(dx, q)
        layout = decompose(x, pm)

        xl = exchange(x, layout)
        rho = paint(xl, 1.0, None, pm)

        # convert to 1 + delta
        fac = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(q))

        rho = linalg.mul(rho, fac)
        rhok = r2c(rho)

        p = apply_transfer(rhok, fourier_space_laplace)

        r1 = []
        for d in range(pm.ndim):
            dx1_c = apply_transfer(p, fourier_space_neg_gradient(d, pm))
            dx1_r = c2r(dx1_c)
            dx1l = readout(dx1_r, xl, None)
            dx1 = gather(dx1l, layout)
            r1.append(dx1)

        f = linalg.stack(r1, axis=-1)
        return dict(f=f)

def KickFactor(pt, ai, ac, af):
    return 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)

def DriftFactor(pt, ai, ac, af):
    return 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)

@autooperator
class leapfrog:
    ain = [ ('dx_i', '*'), ('p_i', '*') ]
    aout = [ ('dx', '*'), ('p', '*'), ('f', '*') ]

    def main(self, dx_i, p_i, q, stages, pt, pm):

        Om0 = pt.Om(1.0)

        dx = dx_i
        p = p_i
        f = gravity(dx, q, pm)

        for ai, af in zip(stages[:-1], stages[1:]):
            ac = (ai * af) ** 0.5

            # kick
            dp = linalg.mul(f, KickFactor(pt, ai, ai, ac) * 1.5 * Om0)
            p = linalg.add(p, dp)

            # drift
            ddx = linalg.mul(p, DriftFactor(pt, ai, ac, af))
            dx = linalg.add(dx, ddx)

            # force
            f = gravity(dx, q, pm)

            # kick
            dp = linalg.mul(f, KickFactor(pt, ac, af, af) * 1.5 * Om0)
            p = linalg.add(p, dp)

        f = linalg.mul(f, 1.5 * Om0)
        return dict(dx=dx, p=p, f=f)

@autooperator
class nbody:
    ain = [
            ('rhok',  'RealField'),
          ]

    aout = [
            ('dx', '*'), ('p', '*'), ('f', '*')
           ]

    def main(self, rhok, q, stages, cosmology, pm):
        from fastpm.background import PerturbationGrowth

        dx1, dx2 = lpt(rhok, q, pm)

        stages = numpy.array(stages)
        mid = (stages[1:] * stages[:-1]) ** 0.5
        support = numpy.concatenate([mid, stages])
        support.sort()
        pt = PerturbationGrowth(cosmology, a=support)
        a = stages[0]

        E = pt.E(a)
        D1 = pt.D1(a)
        D2 = pt.D2(a)
        f1 = pt.f1(a)
        f2 = pt.f2(a)

        dx1 = linalg.mul(dx1, D1)
        dx2 = linalg.mul(dx2, D2)

        p1 = linalg.mul(dx1, a ** 2 * f1 * E)
        p2 = linalg.mul(dx2, a ** 2 * f2 * E)

        p = linalg.add(p1, p2)
        dx = linalg.add(dx1, dx2)

        dx, p, f = leapfrog(dx, p, q, stages, pt, pm)

        return dict(dx=dx, p=p, f=f)

@operator
class cdot:
    ain = {'x1' : 'ComplexField', 'x2' : 'Complexfield'}
    aout = {'y' : '*'}

    # only keep the real part, assuming two fields are hermitian.
    def apl(self, x1, x2):
        return dict(y=x1.cdot(x2).real)

    def vjp(self, x1, x2, _y):
        _x1 = x2.cdot_vjp(_y)
        _x2 = x1.cdot_vjp(_y)
        return dict(_x1=_x1, _x2=_x2)

    def jvp(self, x1_, x2_, x1, x2):
        return dict(y_=x1.cdot(x2_).real + x2.cdot(x1_).real)

