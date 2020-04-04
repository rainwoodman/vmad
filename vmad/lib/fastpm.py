from vmad import operator, autooperator
from pmesh.pm import ParticleMesh
from vmad.lib import linalg
import numpy

@operator
class to_scalar:
    ain = {'x' : 'RealField'}
    aout = {'y' : '*'}

    def apl(node, x):
        return dict(y = x.cnorm())

    def vjp(node, _y, x):
        return dict(_x= x * (2 * _y))

    def jvp(node, x_, x):
        return dict(y_ = x.cdot(x_) * 2)

@operator
class as_complex_field:
    ain = {'x' : '*'}
    aout = {'y' : 'ComplexField'}

    def apl(node, x, pm):
        y = pm.create(type='complex')
        y.real[...] = x[..., 0]
        y.imag[...] = x[..., 1]
        return dict(y=y)

    def vjp(node, _y, pm):
        _x = numpy.stack([_y.real, _y.imag], axis=-1)
        return dict(_x=_x)

    def jvp(node, x_, pm):
        y_ = pm.create(type='complex')
        y_.real[...] = x_[..., 0]
        y_.imag[...] = x_[..., 1]
        return dict(y_=y_)

@operator
class r2c:
    ain = {'x' : 'RealField'}
    aout = {'y' : 'ComplexField'}

    def apl(node, x):
        return dict(y=x.r2c(), pm=x.pm)
    def vjp(node, _y, pm):
        _y = pm.create(type='complex', value=_y)
        return dict(_x=_y.r2c_vjp())
    def jvp(node, x_, pm):
        x_ = pm.create(type='real', value=x_)
        return dict(y_=x_.r2c())

@operator
class c2r:
    ain = {'x' : 'ComplexField'}
    aout = {'y' : 'RealField'}

    def apl(node, x):
        return dict(y=x.c2r(), pm=x.pm)
    def vjp(node, _y, pm):
        _y = pm.create(type='real', value=_y)
        return dict(_x=_y.c2r_vjp())
    def jvp(node, x_, pm):
        x_ = pm.create(type='complex', value=x_)
        return dict(y_=x_.c2r())

@operator
class apply_transfer:
    ain = {'x' : 'ComplexField'}
    aout = {'y' : 'ComplexField'}

    def apl(node, x, tf, kind='wavenumber'):
        filter = lambda k, v: v * tf(k)
        return dict(y=x.apply(filter, kind=kind))

    def vjp(node, _y, tf, kind):
        filter = lambda k, v: v * numpy.conj(tf(k))
        return dict(_x=_y.apply(filter, kind=kind))

    def jvp(node, x_, tf, kind):
        filter = lambda k, v: v * tf(k)
        return dict(y_=x_.apply(filter, kind=kind))

def _take_default(array, ind, conj, default):
    ind1 = ind.clip(0, len(array) - 1)
    r = array[ind1]
    r[conj] = numpy.conj(r)[conj]
    # out of bound values, set to default
    numpy.putmask(r, ind != ind1, default)
    return r

@operator
class apply_digitized:
    """
    Apply a digitized transfer function, represented by
    `tf` and a digitizer function, which converts the mode label
    to indices in `tf`.

    if ind < 0 or ind >= len(tf), the mode is skipped.

    y = x * sum Pi lambda_i

    where lambda is the transfer function per binned k, Pi is
        projection operator for binned k.

    Parameters
    ----------
    tf : array_like
        values of digitized transfer function

    digitizer : func([k0, k1, k2, ...]) -> ind, conj
        decided which bin the mode falls in, and if the mode needs
        to be conjugated.
        the mode labels depends on the value of kind.

    kind : string, 'wavenumber', 'circular', 'index'.

    Notes
    -----
    vjp:
        _lambda_i = <x * Pi , _y>
        this is the binned cross power between x and _y.
        for harmonic functions this is always real.
        _x = _y * sum Pi lambda_i

    jvp:
        y_ = x sum Pi lambda_i_ + x_ sum Pi lambda_i

    """
    ain = 'x', 'tf'
    aout = 'y'

    @staticmethod
    def isotropic_wavenumber(kedges):
        def digitizer(k):
            k2 = k.normp(2)
            ind = numpy.digitize(k2 ** 0.5, kedges) - 1
            return ind, numpy.zeros_like(ind, dtype='?')
        return digitizer

    def apl(node, x, tf, digitizer, kind='wavenumber', mode='amplitude'):
        assert mode in ('amplitude', 'phase')

        # build the projection operators, stored as ind, where value of ind is the index in tf.
        ind = numpy.zeros(x.value.shape, dtype='intp')
        conj = numpy.zeros(x.value.shape, dtype='?')

        x.apply(lambda k, v: digitizer(k)[0], kind=kind, out=ind)
        x.apply(lambda k, v: digitizer(k)[1], kind=kind, out=conj)

        # if ind is out of bound from tf, then we return 1
        # because the transfer function is constant 1 on those bins.
        if mode == 'amplitude':
            y = x * _take_default(tf, ind, conj, 1)
            eit = None
        elif mode == 'phase':
            eit = numpy.exp(1j * tf)
            y = x * _take_default(eit, ind, conj, 1)
        return dict(y=y, ind=ind, conj=conj, eit=eit)

    def vjp(node, _y, x, tf, ind, conj, eit, mode):
        ind1 = ind.clip(0, len(tf) - 1)
        mask = ind1 != ind

        if mode == 'amplitude':
            # compute the cross power between _y and x
            weights = _y * numpy.conj(x)
            # count how many modes each value in the field contributes
            weights = weights.apply(x._expand_hermitian, kind='index', out=Ellipsis)
            # clear out of bound modes
            numpy.putmask(weights.value, mask, 0)
            _tf_real = numpy.bincount(ind1.flat, weights=weights.real.flat, minlength=len(tf))
    #        _tf_imag = numpy.bincount(ind1.flat, weights=weights.imag.flat, minlength=len(tf))
            _tf = _tf_real # assuming harmonic functions
            _x = _y * _take_default(tf, ind, conj, 1)

        elif mode == 'phase':
            y = x * _take_default(eit, ind, conj, 0)

            # compute the cross power between _y and x
            weights = _y * numpy.conj(1j * y)
            # count how many modes each value in the field contributes
            weights = weights.apply(x._expand_hermitian, kind='index', out=Ellipsis)
            # clear out of bound modes
            numpy.putmask(weights.value, mask, 0)

            _tf = numpy.bincount(ind1.flat, weights=weights.real.flat, minlength=len(tf))
            _x = _y * _take_default(eit.conj(), ind, conj, 1)

        # gather from other ranks
        _tf = x.pm.comm.allreduce(_tf)

        return dict(_tf = _tf, _x = _x)

    def jvp(node, tf_, x_, x, tf, ind, conj, eit, mode):
        y_ = 0
        if mode == 'amplitude':
            if tf_ is not 0:
                y_ = y_ + x * _take_default(tf_, ind, conj, 0)
            if x_ is not 0:
                y_ = y_ + x_ * _take_default(tf, ind, conj, 1)
        elif mode == 'phase':
            if tf_ is not 0:
                y_ = y_ + x * _take_default(eit * 1j * tf_, ind, conj, 0)
            if x_ is not 0:
                y_ = y_ + x_ * _take_default(eit, ind, conj, 1)

        return y_

@operator
class paint:
    aout = {'mesh' : 'RealField'}
    ain =  {'x': 'ndarray',
           'layout': 'Layout',
           'mass' : 'ndarray'
           }

    def apl(node, x, mass, layout, pm):
        mesh = pm.paint(x, mass=mass, layout=layout, hold=False)
        return dict(mesh=mesh)

    def vjp(node, _mesh, x, mass, layout, pm):
        _mesh = pm.create(type='real', value=_mesh)
        _x, _mass = pm.paint_vjp(_mesh, x, layout=layout, mass=mass)
        return dict(
            _layout = 0,
            _x=_x,
            _mass=_mass)

    def jvp(node, x_, x, layout, mass, layout_, mass_, pm):
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

    def apl(node, mesh, x, layout, resampler=None):
        N = mesh.pm.comm.allreduce(len(x))
        value = mesh.readout(x, layout=layout, resampler=resampler)
        return dict(value=value, pm=mesh.pm)

    def vjp(node, _value, x, layout, mesh, pm, resampler=None):
        _mesh, _x = mesh.readout_vjp(x, _value, layout=layout, resampler=resampler)
        return dict(_mesh=_mesh, _x=_x, _layout=0)

    def jvp(node, x_, mesh_, x, layout, layout_, mesh, pm, resampler=None):
        if mesh_ is 0: mesh_ = None
        if x_ is 0: x_ = None
        mesh = pm.create(type='real', value=mesh)
        value_ = mesh.readout_jvp(x, v_self=mesh_, v_pos=x_, layout=layout, resampler=resampler)
        return dict(value_=value_)

@operator
class decompose:
    aout={'layout' : 'Layout'}
    ain={'x': 'ndarray'}

    def apl(node, x, pm):
        return dict(layout=pm.decompose(x))

    def vjp(engine, _layout):
        return dict(_x=0)

    def jvp(engine, x_):
        return dict(layout_=0)

@operator
class gather:
    aout='y'
    ain ='x', 'layout'

    def apl(node, x, layout):
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

    def apl(node, x, layout):
        return dict(y=layout.exchange(x))

    def vjp(engine, _y, layout):
        _x = layout.gather(_y)
        return dict(_layout=0, _x=_x)

    def jvp(engine, layout, layout_, x_,):
        return dict(y_=layout.exchange(x_))

def fourier_space_neg_gradient(dir, pm, order):
    if order == 0:
        nyquist =  numpy.pi / pm.BoxSize[dir] * (pm.Nmesh[dir] - 0.01)
        def filter(k):
            mask = abs(k[dir]) < nyquist
            return - 1j * k[dir] * mask
    elif order == 1:
        def filter(k):
            cellsize = (pm.BoxSize[dir] / pm.Nmesh[dir])
            w = k[dir] * cellsize

            a = 1 / (6.0 * cellsize) * (8 * numpy.sin(w) - numpy.sin(2 * w))
            # a is already zero at the nyquist to ensure field is real
            return (-1j * a)
    else:
        raise ValueError("only order 1 and 2 are supproted")
    return filter

def fourier_space_laplace(k):
    k2 = k.normp(2)
    bad = k2 == 0
    k2[bad] = 1
    k2 = - 1 / k2
    k2[bad] = 0
    return k2

@autooperator('rhok,q->dx1')
def lpt1(rhok, q, pm):
    p = apply_transfer(rhok, fourier_space_laplace)

    layout = decompose(q, pm)

    r1 = []
    for d in range(pm.ndim):
        dx1_c = apply_transfer(p, fourier_space_neg_gradient(d, pm, order=1))
        dx1_r = c2r(dx1_c)
        dx1 = readout(dx1_r, q, layout)
        r1.append(dx1)

    dx1 = linalg.stack(r1, axis=-1)

    return dict(dx1 = dx1)

@autooperator('rhok->rho_lpt2')
def lpt2src(rhok, pm):
    if pm.ndim != 3:
        raise ValueError("LPT 2 only works in 3d")

    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    potk = apply_transfer(rhok, fourier_space_laplace)

    Pii = []
    for d in range(pm.ndim):
        t = apply_transfer(potk, fourier_space_neg_gradient(d, pm, order=1))
        Pii1 = apply_transfer(t, fourier_space_neg_gradient(d, pm, order=1))
        Pii1 = c2r(Pii1)
        Pii.append(Pii1)

    source = None
    for d in range(pm.ndim):
        source1 = Pii[D1[d]] * Pii[D2[d]]
        if source is None:
            source = source1
        else:
            source = source + source1

    for d in range(pm.ndim):
        t = apply_transfer(potk, fourier_space_neg_gradient(D1[d], pm, order=1))
        Pij1 = apply_transfer(t, fourier_space_neg_gradient(D2[d], pm, order=1))
        Pij1 = c2r(Pij1)
        source1 = - Pij1 * Pij1
        source = source + source1

    source = (3.0 / 7) * source

    return dict(rho_lpt2=source)

@autooperator('wnk->c')
def induce_correlation(wnk, powerspectrum, pm):
    def tf(k):
        k = sum(ki ** 2 for ki in k) ** 0.5
        return (powerspectrum(k) / pm.BoxSize.prod()) ** 0.5

    c = apply_transfer(wnk, tf)
    return dict(c = c)

@autooperator('rhok->dx1,dx2')
def lpt(rhok, q, pm):

    dx1 = lpt1(rhok, q, pm)
    source2 = lpt2src(rhok, pm)
    rhok2 = r2c(source2)
    dx2 = lpt1(rhok2, q, pm)

    return dict(dx1=dx1, dx2=dx2)


@autooperator('dx->f,potk')
def gravity(dx, q, pm):
    from vmad.core.stdlib import watchpoint
    x = q + dx
    #def w(q): print('q', q)
    #watchpoint(x, w)
    #watchpoint(x, lambda x: print('x', q))
    #watchpoint(dx, lambda dx: print('dx', dx))
    layout = decompose(x, pm)

    xl = exchange(x, layout)
    rho = paint(xl, 1.0, None, pm)

    # convert to 1 + delta
    fac = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(q))

    rho = rho * fac
    rhok = r2c(rho)

    p = apply_transfer(rhok, fourier_space_laplace)

    r1 = []
    for d in range(pm.ndim):
        dx1_c = apply_transfer(p, fourier_space_neg_gradient(d, pm, order=1))
        dx1_r = c2r(dx1_c)
        dx1l = readout(dx1_r, xl, None)
        dx1 = gather(dx1l, layout)
        r1.append(dx1)

    f = linalg.stack(r1, axis=-1)
    return dict(f=f, potk=p)

def KickFactor(Om0, support, FactoryCache, ai, ac, af):
    pt = FactoryCache.get_cosmology(Om0, a=support)
    return 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)

def DriftFactor(Om0, support, FactoryCache, ai, ac, af):
    pt        = FactoryCache.get_cosmology(Om0, a=support)
    return 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)


class CosmologyFactory():

    from fastpm.background import MatterDominated

    def __init__(self):
        self.cosmo_cache = dict()

    def get_cosmology(self, Om0, a):
        cosmo_id = hash((Om0))
        if cosmo_id in self.cosmo_cache:
            return self.cosmo_cache[cosmo_id]
        pt = CosmologyFactory.MatterDominated(Om0,a=a)
        self.cosmo_cache[cosmo_id] = pt
        return pt



@autooperator(' dx_i,p_i, Om0->dx,p,f')
def leapfrog(dx_i, p_i, Om0, q, stages, pm):

    stages = numpy.array(stages)
    mid = (stages[1:] * stages[:-1]) ** 0.5
    support = numpy.concatenate([mid, stages])
    support.sort()
    FactoryCache = CosmologyFactory()
    dx = dx_i
    p = p_i
    f, potk = gravity(dx, q, pm)

    for ai, af in zip(stages[:-1], stages[1:]):
        ac = (ai * af) ** 0.5

        # kick
        dp = f * (KickFactor(Om0, support, FactoryCache, ai, ai, ac) * 1.5 * Om0)
        p = p + dp

        # drift
        ddx = p * DriftFactor(Om0, support, FactoryCache, ai, ac, af)
        dx = dx + ddx

        # force
        f, potk = gravity(dx, q, pm)

        # kick
        dp = f * (KickFactor(Om0, support, FactoryCache, ac, af, af) * 1.5 * Om0)
        p = p + dp

    f = f * (1.5 * Om0)
    return dict(dx=dx, p=p, f=f)

@autooperator('rhok->dx,p')
def firststep(rhok, q, a, pt, pm):

    dx1, dx2 = lpt(rhok, q, pm)

    E = pt.E(a)
    D1 = pt.D1(a)
    D2 = pt.D2(a)
    f1 = pt.f1(a)
    f2 = pt.f2(a)

    dx1 = dx1 * D1
    dx2 = dx2 * D2

    p1 = dx1 * (a ** 2 * f1 * E)
    p2 = dx2 * (a ** 2 * f2 * E)

    p = p1 + p2
    dx = dx1 + dx2
    return dict(dx=dx, p=p)

@autooperator('rhok, Om0->dx,p,f')
def nbody(rhok,Om0, q, stages, cosmology, pm):
    from fastpm.background import MatterDominated

    stages = numpy.array(stages)
    mid = (stages[1:] * stages[:-1]) ** 0.5
    support = numpy.concatenate([mid, stages])
    support.sort()
    pt = CosmologyFactory().get_cosmology(Om0, support)

    dx, p = firststep(rhok, q, stages[0], pt, pm)

    dx, p, f = leapfrog( dx, p, Om0, q, stages, pm)

    return dict(dx=dx, p=p, f=f)

@operator
class cdot:
    ain = {'x1' : 'ComplexField', 'x2' : 'Complexfield'}
    aout = {'y' : '*'}

    # only keep the real part, assuming two fields are hermitian.
    def apl(node, x1, x2):
        return dict(y=x1.cdot(x2).real)

    def vjp(node, x1, x2, _y):
        _x1 = x2.cdot_vjp(_y)
        _x2 = x1.cdot_vjp(_y)
        return dict(_x1=_x1, _x2=_x2)

    def jvp(node, x1_, x2_, x1, x2):
        return dict(y_=x1.cdot(x2_).real + x2.cdot(x1_).real)

class FastPMSimulation:
    def __init__(self, stages, cosmology, pm, B=1, q=None):
        from fastpm.background import MatterDominated

        if q is None:
            q = pm.generate_uniform_particle_grid()

        stages = numpy.array(stages)
        mid = (stages[1:] * stages[:-1]) ** 0.5
        self.support = numpy.concatenate([mid, stages])
        self.support.sort()
        self.pt = MatterDominated(cosmology.Omega0, a=self.support)#MatterDominated(cosmology.Om0, a=self.support)
        self.stages = stages
        self.pm = pm
        self.fpm = ParticleMesh(Nmesh=pm.Nmesh * B,
                        BoxSize=pm.BoxSize,
                        dtype=pm.dtype,
                        comm=pm.comm,
                        resampler=pm.resampler)
        self.q = q

    def KickFactor(self, ai, ac, af):
        return 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)

    def DriftFactor(self, ai, ac, af):
        return 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)

    @autooperator('rhok->dx, p')
    def firststep(self, rhok):
        q = self.q
        pm = self.pm
        pt = self.pt
        a = self.stages[0]

        dx1, dx2 = lpt(rhok, q, pm)

        E = pt.E(a)
        D1 = pt.D1(a)
        D2 = pt.D2(a)
        f1 = pt.f1(a)
        f2 = pt.f2(a)

        dx1 = dx1 * D1
        dx2 = dx2 * D2

        p1 = dx1 * (a ** 2 * f1 * E)
        p2 = dx2 * (a ** 2 * f2 * E)

        p = p1 + p2
        dx = dx1 + dx2
        return dict(dx=dx, p=p)

    @autooperator('dx->f,potk')
    def gravity(self, dx):
        q = self.q
        # use Force PM resolution including B factor.
        pm = self.fpm

        x = q + dx

        layout = decompose(x, pm)

        xl = exchange(x, layout)
        rho = paint(xl, 1.0, None, pm)

        # convert to 1 + delta
        fac = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(q))

        rho = rho * fac
        rhok = r2c(rho)

        p = apply_transfer(rhok, fourier_space_laplace)

        r1 = []
        for d in range(pm.ndim):
            dx1_c = apply_transfer(p, fourier_space_neg_gradient(d, pm, order=1))
            dx1_r = c2r(dx1_c)
            dx1l = readout(dx1_r, xl, None)
            dx1 = gather(dx1l, layout)
            r1.append(dx1)

        f = linalg.stack(r1, axis=-1)
        return dict(f=f, potk=p)

    @autooperator('Om0, rhok->dx,p,f')
    def run(self, Om0, rhok):

        dx, p = self.firststep(rhok)

        pt = self.pt
        stages = self.stages
        q = self.q

        f, potk = self.gravity(dx)

        for ai, af in zip(stages[:-1], stages[1:]):
            ac = (ai * af) ** 0.5

            # kick
            dp = f * (self.KickFactor(ai, ai, ac) * 1.5 * Om0)
            p = p + dp

            # drift
            ddx = p * self.DriftFactor(ai, ac, af)
            dx = dx + ddx

            # force
            f, potk = self.gravity(dx)

            # kick
            dp = f * (self.KickFactor(ac, af, af) * 1.5 * Om0)
            p = p + dp

        f = f * (1.5 * Om0)
        return dict(dx=dx, p=p, f=f)


