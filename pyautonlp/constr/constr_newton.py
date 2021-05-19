import logging
from typing import List, Tuple, Callable

import jax.numpy as jnp
from jax import grad, jit, vmap

from pyautonlp.constants import HessianApprox, ConvergenceCriteria, LearningRateStrategy
from pyautonlp.utils import hessian
from pyautonlp.constr.constr_solver import ConstrainedSolver


class ConstrainedNewtonSolver(ConstrainedSolver):
    def __init__(
            self,
            loss_fn: Callable,
            eq_constr: List[Callable] = None,
            ineq_constr: List[Callable] = None,
            guess: jnp.ndarray = None,
            hessian_approx: str = HessianApprox.EXACT,
            lr_strategy: str = LearningRateStrategy.CONST,
            lr: float = 0.01,
            gamma: float = 0.1,  # relevant if lr strategy is Backtracking
            beta: float = 0.5,  # relevant if lr strategy is Backtracking
            sigma: float = 1.0,  # merit function parameter
            max_iter: int = 500,
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
        self._x0 = jnp.array(guess, dtype=jnp.float32)

        self._logger.info(f'Dimensions of the state vector: {self._x_dims}.')

        # constraints
        if eq_constr is None or not eq_constr:
            raise ValueError('Equality constraints not found.')
        if ineq_constr is not None:
            raise ValueError('Inequality constraints found for Newton solver.')

        self._eq_constr = eq_constr
        self._constr_fns = eq_constr
        self._lambda_dims = len(eq_constr)

        self._logger.info(f'Dimensions of the multiplier vector: {self._lambda_dims}.')

        self._lambda0 = jnp.zeros(shape=(self._lambda_dims,), dtype=jnp.float32)

        # convergence
        conv_n, conv_tol = conv_params
        self._convergence_fn = self._get_convergence_fn(conv_criteria)
        self._max_iter = max_iter
        self._tol = conv_tol

        # learning rate
        # TODO add diff strategies
        self._alpha = lr

        # grad & hessian functions
        # compile with JAX in advance
        # TODO add diff strategies
        self._grad_fn = grad(self._lagrangian_full)
        self._hess_fn = hessian(self._lagrangian_full)

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:

        converged = False
        n_iter = 0

        curr_state = jnp.concatenate([self._x0, self._lambda0])

        while not converged and n_iter < self._max_iter:
            # calculate direction
            grad_at_point = self._grad_fn(curr_state)
            hess_at_point = self._hess_fn(curr_state)
            direction = jnp.linalg.solve(hess_at_point, grad_at_point)

            # calculate step size (line search)
            alpha_t = self._alpha

            # update state
            curr_state -= alpha_t * direction

            # check convergence
            converged = self._convergence_fn(grad_at_point, self._tol)

            # increment params
            n_iter += 1

        # fill additional info
        info = (converged, n_iter)

        return curr_state, info
