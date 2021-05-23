import logging
from typing import List, Tuple, Callable
from collections import namedtuple

import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt
import seaborn as sns

from pyautonlp.constants import HessianApprox, ConvergenceCriteria, LearningRateStrategy
from pyautonlp.utils import hessian
from pyautonlp.charting import plot_alpha, plot_training_loss, plot_convergence
from pyautonlp.constr.constr_solver import ConstrainedSolver

CacheItem = namedtuple('CacheItem', 'x m loss alpha penalty')


class ConstrainedNewtonSolver(ConstrainedSolver):
    def __init__(
            self,
            loss_fn: Callable,
            eq_constr: List[Callable] = None,
            ineq_constr: List[Callable] = None,
            guess: jnp.ndarray = None,
            hessian_approx: str = HessianApprox.EXACT,
            lr_strategy: str = LearningRateStrategy.CONST,
            lr: float = 0.01,  # relevant if lr strategy is Constant
            beta: float = 0.5,  # relevant if lr strategy is Backtracking
            gamma: float = 0.1,  # relevant if lr strategy is Backtracking + Armijo
            sigma: float = 1.0,  # relevant if lr strategy is Backtracking + Merit Function
            max_iter: int = 100,
            conv_criteria: str = ConvergenceCriteria.KKT_VIOLATION,
            conv_params: Tuple = (10, 1e-4),
            verbose: bool = False,
            visualize: bool = False,
    ):
        # logger
        self._logger = logging.getLogger('newton_solver')
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.info(f'Initializing solver.')

        # loss
        self._loss_fn = loss_fn

        # initial guess
        # TODO fix guess is empty
        # TODO move guess to solve()
        assert guess is not None
        self._x_dims = len(guess)
        self._initial_x = jnp.array(guess, dtype=jnp.float32)

        self._logger.info(f'Dimensions of the state vector: {self._x_dims}.')

        # constraints
        if eq_constr is None or not eq_constr:
            raise ValueError('Equality constraints not found.')
        if ineq_constr is not None:
            raise ValueError('Inequality constraints found for Newton solver.')

        self._eq_constr = eq_constr
        self._constr_fns = eq_constr
        self._multiplier_dims = len(eq_constr)

        self._logger.info(f'Dimensions of the multiplier vector: {self._multiplier_dims}.')

        self._initial_multipliers = jnp.zeros(shape=(self._multiplier_dims,), dtype=jnp.float32)

        # convergence
        conv_n, conv_tol = conv_params
        self._convergence_fn = self._get_convergence_fn(conv_criteria)
        self._max_iter = max_iter
        self._tol = conv_tol

        # learning rate
        self._alpha = lr
        self._beta = beta
        self._gamma = gamma
        self._sigma = sigma
        self._step_size_fn = self._get_step_size_fn(lr_strategy)

        # save intermediate step info
        self._cache = {}

        # plotting params
        self._visualize = visualize

        # grad & hessian functions
        # compile with JAX in advance
        self._grad_loss_x_fn = grad(self._loss_fn)  # N-by-1
        self._grad_constr_x_fns = [grad(f) for f in self._constr_fns]

        self._hessian_approx = hessian_approx
        if hessian_approx == HessianApprox.EXACT:
            self._hess_lagr_xx_fn = hessian(self._lagrangian)  # N-by-N

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:
        curr_x = self._initial_x
        curr_m = self._initial_multipliers

        kkt_n = self._x_dims + self._multiplier_dims
        kkt_matrix = jnp.zeros(shape=(kkt_n, kkt_n), dtype=jnp.float32)

        converged, conv_penalty = self._convergence_fn(curr_x, curr_m)
        n_iter = 0
        while (not converged) and (n_iter < self._max_iter):
            # calculate direction
            # KKT vector - r(x, lambda)
            grad_loss_x = self._grad_loss_x_fn(curr_x)
            c_vals = self._eval_constraints(curr_x)
            kkt_state = jnp.concatenate((grad_loss_x, c_vals))

            # KKT matrix - r'(x, lambda)
            if self._hessian_approx == HessianApprox.EXACT:
                B = self._hess_lagr_xx_fn(curr_x, curr_m)
            elif self._hessian_approx == HessianApprox.GAUSS_NEWTON:
                B = grad_loss_x @ jnp.transpose(grad_loss_x)
            elif self._hessian_approx == HessianApprox.STEEPEST_DESCENT:
                B = jnp.eye(N=self._x_dims)
            elif self._hessian_approx == HessianApprox.BFGS:
                raise NotImplementedError
            else:
                raise NotImplementedError

            # TODO check H for PD + regularization (Powell's trick)
            if (self._hessian_approx == HessianApprox.EXACT or
                    self._hessian_approx == HessianApprox.GAUSS_NEWTON):
                eig_vals, eig_vecs = jnp.linalg.eigh(B)

                B = self._hess_lagr_xx_fn(curr_x, curr_m)
            elif self._hessian_approx == HessianApprox.STEEPEST_DESCENT:
                continue
            else:
                raise NotImplementedError





            kkt_matrix = kkt_matrix.at[:self._x_dims, :self._x_dims].set(B)

            constr_grad_x = self._eval_constraint_gradients(curr_x)
            kkt_matrix = kkt_matrix.at[:self._x_dims, self._x_dims:].set(constr_grad_x)
            kkt_matrix = kkt_matrix.at[self._x_dims:, :self._x_dims].set(jnp.transpose(constr_grad_x))

            # Note: state is multiplied by (-1)
            direction = jnp.linalg.solve(kkt_matrix, -kkt_state)

            # calculate step size (line search)
            # curr_x, grad_loss_x, direction
            step_size = self._step_size_fn(curr_x=curr_x, grad_loss_x=grad_loss_x, direction=direction)

            # save cache + logs
            loss = self._loss_fn(curr_x)
            self._cache[n_iter] = CacheItem(curr_x, curr_m, loss, step_size, conv_penalty)
            self._logger.info(self._get_log_str(n_iter, loss, step_size, conv_penalty))

            # update state
            curr_x += step_size * direction[:self._x_dims]
            curr_m = (1 - step_size) * curr_m + step_size * direction[self._x_dims:]

            # TODO check update sigma
            # self._sigma = jnp.max(curr_m) + .01

            # check convergence
            converged, conv_penalty = self._convergence_fn(curr_x, curr_m)

            # increment counter
            n_iter += 1

        # log and print last results
        loss = self._loss_fn(curr_x)
        self._cache[n_iter] = CacheItem(curr_x, curr_m, loss, .0, conv_penalty)
        self._logger.info(self._get_log_str(n_iter, loss, .0, conv_penalty))

        # fill additional info
        info = (converged, loss, n_iter)

        # visualization
        if self._visualize:
            self.visualize()

        return curr_x, info

    def visualize(self):
        sns.set()

        assert self._cache
        assert len(self._cache) > 1

        # ax = plot_convergence(self._cache, self._loss_fn)
        # ax.set_title('Loss(t) (log scale)')
        # plt.show()

        ax = plot_training_loss(self._cache)
        ax.set_title('Loss(t) (log scale)')
        plt.show()

        ax = plot_alpha(self._cache)
        ax.set_title('Alpha(t)')
        plt.show()
