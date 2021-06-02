import logging
from typing import List, Tuple, Callable, Union

import jax.numpy as jnp
from jax import grad

from pyautonlp.constants import LineSearch, ConvergenceCriteria
from pyautonlp.solver import Solver


class GD(Solver):
    def __init__(
            self,
            loss_fn: Callable,
            guess: Union[Tuple, jnp.ndarray] = None,
            alpha: float = 0.01,  # relevant for Constant step line search
            max_iter: int = 500,
            conv_params: Tuple = (3, 1e-6),
            verbose: bool = False,
    ):
        # logger
        self._logger = logging.getLogger('gradient_descent')
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

        # compile with JAX in advance
        self._grad_loss_x_fn = grad(self._loss_fn)  # N-by-1

        # convergence
        # self._convergence_fn = self._get_convergence_fn(conv_criteria)
        self._max_iter = max_iter
        conv_n_steps, conv_tol = conv_params
        self._conv_n_steps = conv_n_steps
        self._tol = conv_tol

        # learning rate
        # self._step_size_fn = self._get_step_size_fn(line_search)
        self._alpha = alpha

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:
        x_curr = self._initial_x
        self._logger.info(f'Initial guess is: {x_curr}.')

        converged = False
        k = 0
        n_small_diffs_in_a_row = 0
        while (not converged) and (k < self._max_iter):
            d_k = self._grad_loss_x_fn(x_curr)
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
