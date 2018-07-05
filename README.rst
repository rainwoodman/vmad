vmad stands for virtual machine automated differentation framework.

The word virtual machine here does not mean much (think of llvm), other
than the computation graph is always traversed with an explicit linear
order -- the order the nodes are triggerred.

This is the third iteration of the design; two previous interfaces
can be found in the optimization package abopt:

    https://github.com/bccp/abopt

this iteration has become too big to be kept as an embed file in abopt.

The kernel algorithm is stable. We use tape based automated differentation.
Each operator consists of three types of primitives:

  - apl (the application of operator)
  - jvp (jacobian vector product)
  - vjp (vector jacobian product)

The operators do not form a closed group. There is no support for
hessian products; though hessian of a Chi-square problem can be approximated
as a product of jvp and vjp. (See wikipedia on Gauss-Newton optimization).

Automated differentation can be performed on a ``model``,
or through an ``autooperator``, which is an automated differentiable operator.

The interface is still experimental; hence documentation is sparse.

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


