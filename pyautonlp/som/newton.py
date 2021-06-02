import logging
from typing import List, Tuple, Callable, Union

import jax.numpy as jnp
from jax import grad

from pyautonlp.solver import Solver
from pyautonlp.constants import Direction, ConvergenceCriteria, LineSearch, HessianRegularization


class Newton(Solver):
    def __init__(
            self,
            loss_fn: Callable,
            guess: Union[Tuple, jnp.ndarray] = None,
            direction: str = Direction.EXACT_NEWTON,
            reg: str = HessianRegularization.NONE,
            # line_search: str = LineSearch.CONST,
            alpha: float = 1.0,  # relevant for Constant step line search
            # beta: float = 0.5,  # relevant for Backtracking line search
            # gamma: float = 0.1,  # relevant for Backtracking + Armijo line search
            conv_params: Tuple = (1, 1e-6),
            max_iter: int = 50,
            verbose: bool = False,
    ):
        # logger
        self._logger = logging.getLogger('newton')
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

        # convergence
        # self._convergence_fn = self._get_convergence_fn(conv_criteria)
        self._max_iter = max_iter
        conv_n_steps, conv_tol = conv_params
        self._conv_n_steps = conv_n_steps
        self._tol = conv_tol

        # learning rate
        # self._step_size_fn = self._get_step_size_fn(line_search)
        self._alpha = alpha
        # self._beta = beta
        # self._gamma = gamma

        # save intermediate step info
        self._cache = {}

        # grad & hessian functions
        # compile with JAX in advance
        self._grad_loss_x_fn = grad(self._loss_fn)  # N-by-1

        self._direction = direction
        if direction == Direction.EXACT_NEWTON:
            self._hess_loss_xx_fn = self._hessian_exact(fn=self._loss_fn)  # N-by-N
        elif direction == Direction.BFGS:
            self._hess_loss_xx_fn = self._hessian_bfgs_approx  # N-by-N
        elif direction == Direction.GAUSS_NEWTON:
            self._hess_loss_xx_fn = self._hessian_gn_approx  # N-by-N
        elif direction == Direction.STEEPEST_DESCENT:
            self._hess_loss_xx_fn = self._hessian_sd_approx  # N-by-N
        else:
            raise NotImplementedError

        self._reg = reg

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:
        x_curr = self._initial_x
        self._logger.info(f'Initial guess is: {x_curr}.')

        converged = False
        k = 0
        n_small_diffs_in_a_row = 0
        while (not converged) and (k < self._max_iter):
            g_k = self._grad_loss_x_fn(x_curr)

            # calculate B_k
            if self._direction == Direction.EXACT_NEWTON:
                B_k = self._hess_loss_xx_fn(x_curr)
            elif self._direction == Direction.GAUSS_NEWTON:
                B_k = self._hess_loss_xx_fn(g_k)
            else:
                raise NotImplementedError

            # TODO Hessian Regularization

            # find direction
            d_k = jnp.linalg.solve(B_k, g_k)

            # update params
            x_next = x_curr - self._alpha * d_k

            # check convergence
            diff = jnp.linalg.norm(x_next - x_curr, 2)
            if diff < self._tol:
                n_small_diffs_in_a_row += 1
            else:
                n_small_diffs_in_a_row = 0

            if n_small_diffs_in_a_row == self._conv_n_steps:
                converged = True

            # logging
            loss = self._loss_fn(x_curr)
            self._logger.info(f'Iteration {k}: Loss: {loss:.3f}; ' \
                              f'Alpha: {self._alpha:.4f}; Step diff: {diff:.5f}.')

            x_curr = x_next
            k += 1

        # fill additional info
        loss = self._loss_fn(x_curr)
        self._logger.info(f'Iteration {k}: Loss: {loss:.3f}; ' \
                          f'Alpha: {self._alpha:.4f}; Step diff: {diff:.5f}.')

        info = (converged, loss, k)

        return x_curr, info
