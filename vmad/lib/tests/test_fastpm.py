from __future__ import print_function
from pprint import pprint
from vmad.lib import fastpm, linalg
import numpy
from vmad import autooperator

from vmad.testing import BaseScalarTest
from mpi4py import MPI
from numpy.testing import assert_allclose
from pmesh.pm import RealField, ComplexField

def create_bases(x):
    bases = numpy.eye(x.size).reshape([-1] + list(x.shape))
    if isinstance(x, RealField):
        pm = x.pm
        # FIXME: remove this after pmesh 0.1.36
        def create_field(pm, data):
            real = pm.create(type='real')
            real[...] = data
            return real
        return [create_field(pm, i) for i in bases]
    else:
        return [i for i in bases]

class Test_r2c_c2r(BaseScalarTest):
    to_scalar = staticmethod(fastpm.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    x = pm.generate_whitenoise(seed=300, unitary=True, type='real')
    y = x.cnorm()
    x_ = create_bases(x)

    def model(self, x):
        c = fastpm.r2c(x)
        r = fastpm.c2r(c)
        return r

#    def teardown(self):
#        print(self.y_)

def transfer(k):
    k2 = sum(ki ** 2 for ki in k)
    mask = k2 == 0
    k2[mask] = 1
    r = 1 / k2
    r[mask] = 0
    return r

class Test_r2c_transfer_c2r(BaseScalarTest):
    to_scalar = staticmethod(fastpm.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    x = pm.generate_whitenoise(seed=300, unitary=True, type='real')
    y = x.r2c().apply(lambda k, v: transfer(k) * v).c2r().cnorm()
    x_ = create_bases(x)

    def model(self, x):
        c = fastpm.r2c(x)
        c = fastpm.apply_transfer(c, tf=transfer)
        r = fastpm.c2r(c)
        return r

#    def teardown(self):
#        print(self.y_)

class Test_apply_digitized_x(BaseScalarTest):
    to_scalar = staticmethod(fastpm.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4], BoxSize=4.0, comm=MPI.COMM_SELF)
    kedges = numpy.linspace(0, 2 * numpy.pi / 4.0 * 3, 8)
    tf = numpy.arange(len(kedges) - 1) + 1
    x = pm.generate_whitenoise(seed=300, unitary=True, type='real')
    x_ = create_bases(x)
    y = NotImplemented

    def model(self, x):
        c = fastpm.r2c(x)
        c = fastpm.apply_digitized(c, tf=self.tf, kedges=self.kedges)
        r = fastpm.c2r(c)
        return r

class Test_apply_digitized_tf(BaseScalarTest):
    to_scalar = staticmethod(fastpm.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=4.0, comm=MPI.COMM_SELF)
    kedges = numpy.linspace(0, 2 * numpy.pi / 4.0 * 3, 8)
    y = NotImplemented
    x = numpy.arange(len(kedges) - 1)
    x_ = create_bases(x)
    atol = 1e-9
    def model(self, x):
        c0 = self.pm.generate_whitenoise(seed=300, unitary=True, type='complex', mean=1.0)
        c = fastpm.apply_digitized(c0, tf=x, kedges=self.kedges)
        r = fastpm.c2r(c)
        return r

class Test_cdot(BaseScalarTest):
    to_scalar = staticmethod(linalg.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    x = pm.generate_whitenoise(seed=300, unitary=True, type='real')
    x1 = pm.generate_whitenoise(seed=301, unitary=True, type='real')
    y = (x.cdot(x1)) ** 2 * pm.Nmesh.prod() ** -2.
    x_ = create_bases(x)

    def model(self, x):
        x1 = fastpm.r2c(self.x1)
        x = fastpm.r2c(x)
        return fastpm.cdot(x, x1)


class Test_paint_x(BaseScalarTest):
    to_scalar = staticmethod(fastpm.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    mesh = pm.generate_whitenoise(seed=300, unitary=True, type='real')

    x = pm.generate_uniform_particle_grid(shift=0.1)
    y = (1 + mesh).cnorm()
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
        y = fastpm.paint(x, layout=None, mass=1.0, pm=self.pm)
        y = linalg.add(y, self.mesh) # biasing a bit to get non-zero derivatives.
        return y

class Test_decompose_paint_x(BaseScalarTest):
    to_scalar = staticmethod(fastpm.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    mesh = pm.generate_whitenoise(seed=300, unitary=True, type='real')

    x = pm.generate_uniform_particle_grid(shift=0.1)
    y = (1 + mesh).cnorm()
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
        layout = fastpm.decompose(x, pm=self.pm)
        y = fastpm.paint(x, layout=layout, mass=1.0, pm=self.pm)
        y = linalg.add(y, self.mesh) # biasing a bit to get non-zero derivatives.
        return y

class Test_decompose_exchange(BaseScalarTest):
    to_scalar = staticmethod(fastpm.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    mesh = pm.generate_whitenoise(seed=300, unitary=True, type='real')

    x = pm.generate_uniform_particle_grid(shift=0.1)
    y = (1 + mesh).cnorm()
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
        layout = fastpm.decompose(x, pm=self.pm)
        x1 = fastpm.exchange(x, layout)
        y1 = fastpm.paint(x1, mass=1.0, layout=None, pm=self.pm)
        y = linalg.add(y1, self.mesh) # biasing a bit to get non-zero derivatives.
        return y

class Test_paint_mass(BaseScalarTest):
    to_scalar = staticmethod(fastpm.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    mesh = pm.generate_whitenoise(seed=300, unitary=True, type='real')

    pos = pm.generate_uniform_particle_grid(shift=0.1)
    x = numpy.ones(len(pos))
    y = x.sum()
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
        y = fastpm.paint(self.pos, layout=None, mass=x, pm=self.pm)
        return y

class Test_readout_x(BaseScalarTest):
    to_scalar = staticmethod(linalg.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    mesh = pm.generate_whitenoise(seed=300, unitary=True, type='real')

    x = pm.generate_uniform_particle_grid(shift=0.5)
    y = (mesh.readout(x) ** 2).sum()
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
        y = fastpm.readout(self.mesh, x, layout=None)
        return y

class Test_readout_mesh(BaseScalarTest):
    to_scalar = staticmethod(linalg.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    x = pm.generate_whitenoise(seed=300, unitary=True, type='real')

    pos = pm.generate_uniform_particle_grid(shift=0.5)
    y = (x.readout(pos) ** 2).sum()
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
        y = fastpm.readout(x, self.pos, layout=None)
        return y

class Test_lpt1(BaseScalarTest):
    to_scalar = staticmethod(linalg.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    x = pm.generate_whitenoise(seed=300, unitary=True, type='real')

    pos = pm.generate_uniform_particle_grid(shift=0.5)
    y = NotImplemented
    x_ = create_bases(x)

    epsilon = 1e-4

    def model(self, x):
        dx1 = fastpm.lpt1(fastpm.r2c(x), q=self.pos, pm=self.pm)
        return dx1

class Test_lpt2src(BaseScalarTest):
    to_scalar = staticmethod(fastpm.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    x = pm.generate_whitenoise(seed=300, unitary=True, type='real')

    y = NotImplemented
    x_ = create_bases(x)

    epsilon = 1e-4
    rtol = 1e-6
    def model(self, x):
        return fastpm.lpt2src(fastpm.r2c(x), pm=self.pm)

class Test_lpt(BaseScalarTest):
    to_scalar = staticmethod(linalg.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    x = pm.generate_whitenoise(seed=300, unitary=True, type='real')

    pos = pm.generate_uniform_particle_grid(shift=0.5)
    y = NotImplemented
    x_ = create_bases(x)

    epsilon = 1e-4

    def model(self, x):
        dx1, dx2 = fastpm.lpt(fastpm.r2c(x), q=self.pos, pm=self.pm)
        return linalg.add(dx1, dx2)

class Test_nbody(BaseScalarTest):
    to_scalar = staticmethod(linalg.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    x = pm.generate_whitenoise(seed=300, unitary=True, type='real')

    pos = pm.generate_uniform_particle_grid(shift=0.5)
    y = NotImplemented
    x_ = create_bases(x)

    epsilon = 1e-4
    def model(self, x):
        from nbodykit.cosmology import Planck15
        dx, p, f = fastpm.nbody(fastpm.r2c(x), q=self.pos, stages=[0.1, 0.5, 1.0], pm=self.pm, cosmology=Planck15)
        return linalg.stack([dx, p, f], axis=-1)

class Test_fastpm(BaseScalarTest):
    to_scalar = staticmethod(linalg.to_scalar)

    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    x = pm.generate_whitenoise(seed=300, unitary=True, type='real')

    pos = pm.generate_uniform_particle_grid(shift=0.5)
    y = NotImplemented
    x_ = create_bases(x)

    epsilon = 1e-4
    def model(self, x):
        from nbodykit.cosmology import Planck15
        sim = fastpm.FastPMSimulation(pm=self.pm, cosmology=Planck15, stages=[0.1, 0.5, 1.0], q=self.pos)

        dx, p, f = sim.run(fastpm.r2c(x))
        return linalg.stack([dx, p, f], axis=-1)


def test_add_scalar():
    from vmad import autooperator
    from vmad.core.stdlib import eval
    from numpy.testing import assert_array_equal
    pm = fastpm.ParticleMesh(Nmesh=[4, 4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

    x = pm.generate_whitenoise(seed=300, unitary=True, type='real')
    x[...] = 1.0

    @autooperator('x, n->y')
    def func(x, n):
        a = linalg.take(n, 0, axis=0)
        a = linalg.broadcast_to(a, eval(x, lambda x : x.shape))
        return x + a

    n = numpy.array([3, 4])
    (y, ), (_x, _n) = (func.build().compute_with_vjp(init=dict(x=x, n=n), v=dict(_y=x)))
    assert_array_equal(y, 4)
    assert_array_equal(_x, 1)
    assert_array_equal(_n, (64, 0))
