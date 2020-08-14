from vmad import Builder, autooperator
from vmad.lib import mpi, linalg

import numpy
from pmesh.pm import ParticleMesh

@autooperator
class MPIChiSquareOperator:
    ain = [('x', '*')]
    aout = [('y', '*')]

    def main(self, x, comm):
        chi2 = linalg.sum(x * x)
        chi2 = mpi.allreduce(chi2, comm)
        return dict(y = chi2)

from abopt.abopt2 import Problem as BaseProblem, VectorSpace

class MPIVectorSpace(VectorSpace):
    def __init__(self, comm):
        self.comm = comm

    def addmul(self, a, b, c, p=1):
        """ a + b * c ** p, follow the type of b """
        if p is not 1: c = c ** p
        c = b * c
        if a is not 0: c = c + a
        return c

    def dot(self, a, b):
        """ einsum('i,i->', a, b) """
        return self.comm.allreduce(numpy.sum(a * b))

class MPIChiSquareProblem(BaseProblem):
    """ Defines a chisquare problem, which is

        .. math::

            y = \sum_i [R_i(F(s))]^2

        F : forward_operator

        [R_i] : residuals

        comm : The problem is defined on a MPI communicator -- must be consistent
        with forward_operator and residuals. (usually just pm.comm or MPI.COMM.WORLD)

    """
    def __init__(self, comm, forward_operator, residuals, vectorspace=None):
        self.residuals = residuals
        self.forward_operator = forward_operator
        self.comm = comm

        if vectorspace is None:
            vectorspace = MPIVectorSpace(comm)

        with Builder() as m:
            x = m.input('x')
            y = 0
            fx = forward_operator(x)
            # fixme: need a way to directly include a subgraphs
            # rather than building it again.
            for operator in self.residuals:
                r = operator(*fx)
                chi2 = MPIChiSquareOperator(r, comm)
                y = linalg.add(y, chi2)
            m.output(y=y)

        def objective(x):
            return m.compute(vout='y', init=dict(x=x))

        def gradient(x):
            y, [vjp] = m.compute_with_vjp(init=dict(x=x), v=dict(_y=1.0))
            return vjp

        def hessian_vector_product(x, v):
            Dv = 0

            replay = forward_operator.precompute(x=x)

            for operator in self.residuals:
                with Builder() as m:
                    xx = m.input('x')
                    fx = replay(xx)
                    r = operator(*fx)
                    m.output(y=r)

                y, [Dv1] = m.compute_with_gnDp(vout='y',
                            init=dict(x=x),
                            v=dict(x_=v))
                Dv = Dv + Dv1
            # H is 2 JtJ, see wikipedia on Gauss Newton.
            return Dv * 2

        BaseProblem.__init__(self,
                        vs = vectorspace,
                        objective=objective,
                        gradient=gradient,
                        hessian_vector_product=hessian_vector_product)

class EpochScheduler(dict):
    """ An object to schedule hyper parameters in multi-epoch
        training.
    """
    def __init__(self):
        pass

    def __call__(self, argname, values):
        self.schedule(argname, values)

    @property
    def min_nepochs(self):
        return max(list(self.keys()) + [0]) + 1

    def schedule(self, argname, values):
        """ schedule a training parameter `argname`
            at the given epochs.

            values can be either a list or a dict, or a scalar.

            list : one value per epoch
            dict : step function per epoch, key is the epoch to change
            scalar : at the 0th epoch.

            The nameed optimizer parameter will be set to the
            given value on that epoch as if all epochs are played
            sequentially.
        """
        if isinstance(values, (list, tuple)):
            iter = enumerate(values)
        elif isinstance(values, (dict,)):
            iter = values.items()
        else:
            # scalar
            iter = enumerate([values])

        for epoch, value in iter:
            args = self.get(epoch, {})
            args[argname] = value
            self[epoch] = args

    def get_args(self, epoch):
        """ move the trainer to the epoch"""
        args = {}
        for epochi in sorted(self.keys()):
            if epochi <= epoch:
                for argname, value in self[epochi].items():
                    args[argname] = value

        return args

class Epoch(object):
    def __init__(self, epoch, nepochs):
        self.epoch = epoch
        self.nepochs = nepochs
    def __repr__(self):
        return "Epoch %d / %d" % (self.epoch, self.nepochs)

class MAPInversion:
    """ Inversion of a ChiSquareProblem with a ForwardOperator;
        given data finding the optimal reconstruction.

        The inversion is done in multiple epochs to accelerate
        convergence and avoid local attractors. Conceptually this
        is similar to stochastic gradient descent.

        Two schedulers are involved. schedule_optimizer controls
        the parameters of the trainer (e.g., maxiter, atol, rtol, etc).
        schedule_problem controls the parameters of the problem
        (subsampling, changing prior, etc).
    """
    def __init__(self, optimizer, ForwardOperator):
        self.optimizer = optimizer
        self.ForwardOperator = ForwardOperator
        self.schedule_optimizer = EpochScheduler()
        self.schedule_problem = EpochScheduler()

    @classmethod
    def create_like(kls, other, epochs=None):
        """ create a new inversion that is a copy of other.
            notice that some of the attributes may be lost;
            not supposed to add new attributes.
        """
        self = object.__new__(kls)
        self.__dict__.update(other.__dict__)

        self.ForwardOperator = other.ForwardOperator
        self.optimizer = other.optimizer
        self.schedule_optimizer = EpochScheduler()
        self.schedule_problem = EpochScheduler()
        if epochs is None:
            self.schedule_optimizer.update(other.schedule_optimizer)
            self.schedule_problem.update(other.schedule_problem)
        else:
            # only take the selected epochs
            for i, epoch in enumerate(epochs):
                self.schedule_optimizer[i] = other.schedule_optimizer.get_args(epoch)
                self.schedule_problem[i] = other.schedule_problem.get_args(epoch)
        return self

    def apply(self, d, s0, epochs=None, monitor_epoch=None, monitor_progress=None):
        """ Apply MAP inversion on synthetic data.

            returns the MAP estimation of the signal.

            epochs : a list of epochs to run; if None, run all of them.

        """
        if epochs is None:
            epochs = range(self.nepochs)

        s1 = s0
        for epoch in epochs:
            if monitor_epoch:
                monitor_epoch(Epoch(epoch, max(epochs) + 1))

            problem = self.get_problem(d, epoch)

            self.optimizer.__dict__.update(self.schedule_optimizer.get_args(epoch))

            state = self.optimizer.minimize(problem, s1, monitor=monitor_progress)
            s1 = state['x']
        return s1

    @property
    def nepochs(self):
        return max(self.schedule_optimizer.min_nepochs,
                     self.schedule_problem.min_nepochs)

    def get_problem(self, d, epoch=0):
        """ Create a problem to be solved for the epoch """
        # if you see exception here, see add a schedule_problem for the arg.
        return self.problem_factory(d=d, **self.schedule_problem.get_args(epoch))

    def problem_factory(self, d, **args):
        """ Create a problem object.

            override this method in subclasses; must
            accept at least the data to be inverted.

            args are set by

            >> mapinversion.schedule_problem(argname, argvalue)
        """
        raise NotImplementedError

