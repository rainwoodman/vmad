vmad stands for virtual machine automated differentation framework.

The word virtual machine here does not mean much (think vm in LLVM), other
than emphasizing that the computation graph is always traversed with
an explicit linear order -- the order the nodes are triggerred. This means
we do not even try to do dependency based scheduling. The only optimization
is to skip irrelevant nodes during the calculation.

This is the third iteration of the design; it has become too big to be
kept as an embed file in abopt. Two previous iterations
can be found in the optimization package abopt (gradually being deprecated):

    https://github.com/bccp/abopt


The core algorithm is stable. vmad uses tape-based automated differentation.
A good reference on tape based auto-diff is in PhD thesis of Maclaurin,
who wrote autograd: https://dougalmaclaurin.com/phd-thesis.pdf .

Each operator consists of three types of primitives:

- apl : the application of operator; forward propagation
- jvp : jacobian vector product; forward jacobian propagation
- vjp : vector jacobian product; backward jacobian propagation
- rcd : declaring what to be recorded on the tape. (unstable)

Unlike autograd, the operators in vmad do not form a closed group.
As a result there is no support for higher order differentiation (e.g. Hessian),
but it makes the syntax for declaring operators leaner.
Note that Hessian of a Chi-square problem can be approximated as a product of
jvp and vjp. (See wikipedia on Gauss-Newton optimization). In many cases this
is sufficiently good. [reference needed]

Complex numbers are treated as individual degrees of freedom. This is
mathematically inferior, but it is more intuitive and avoids a hermitian conjugate
between gradients and parameter update.

The interface is still experimental; hence documentation is sparse. There are three
main concepts, ``model``, ``operator``, and ``autooperator``.

Automated differentation can be performed on a ``model``, which is a collection
of operators, connected by symbols. The forward pass on the model will substitute
symbols with values, and resulting a tape that records these values for backward and
forward propagation of the jacobians, which is what automated differentiation does.

``autooperator`` is basically a ``model``, but built on demand and provides the
jacobian operators via automated differentiation. Currently the tape of the
autooperator node is recorded on the full tape, we will add option to recompute
the tape (trading computation and memory).

A small library of operators for linear algebra (backed by numpy),
particle-mesh simulation(backed by pmesh), and MPI (mpi4py) are in `vmad.lib`.
The focus has been on CPU and MPI, mostly because we are close to a
super computing facility (NERSC) that provide plenty of CPU and MPI. There are
already PyTorch or Tensorflow for GPU oriented work (though not so suitable for MPI).

A higher level Chi-square inversion problems are built on with abopt,
and can be found in `contrib` directory. This mostly implements the procedures
described in http://adsabs.harvard.edu/abs/2017JCAP...12..009S .

Here is an example operator, ``add``,

.. code::

    from vmad import operator

    @operator
    class add:
        ain = 'x1', 'x2'
        aout = 'y'

        def apl(ctx, x1, x2):
            return dict(y=x1 + x2)

        def vjp(ctx, _y):
            return dict(_x1=_y, _x2=_y)

        def jvp(ctx, x1_, x2_):
            return dict(y_=x1_+x2_)


and and example autooperator

.. code::

    from vmad import operator
    from vmad.lib import linalg

    @autooperator
    class myoperator:
        ain = 'x1', 'x2', 'x3'
        aout = 'y'

        # (x1 + x2) * |x3|^2

        def main(ctx, x1, x2, x3):
            x12 = x1 + x2
            x3sq = linalg.to_scalar(x3)
            return dict(y = x12 * x3sq)


