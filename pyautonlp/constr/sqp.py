import logging
from typing import List, Tuple, Callable, Optional, Union

import numpy as np
import jax.numpy as jnp
from jax import grad
from quadprog import solve_qp

from pyautonlp.constants import *
from pyautonlp.constr.constr_solver import ConstrainedSolver


class SQP(ConstrainedSolver):
    def __init__(
            self,
            loss_fn: Callable,
            eq_constr: List[Callable] = None,
            ineq_constr: List[Callable] = None,
            guess: Union[Tuple, jnp.ndarray] = None,
            direction: str = Direction.STEEPEST_DESCENT,
            gn_hessian_fn: Optional[Callable] = None,
            reg: str = HessianRegularization.NONE,
            line_search: str = LineSearch.CONST,
            alpha: float = 0.01,  # relevant for Constant step line search
            beta: float = 0.5,  # relevant for Backtracking line search
            gamma: float = 0.1,  # relevant for Backtracking + Armijo line search
            sigma: float = 1.0,  # relevant for Backtracking + Merit Function line search
            conv_criteria: str = ConvergenceCriteria.KKT_VIOLATION,
            conv_tol: float = 1e-8,
            max_iter: int = 50,
            verbose: bool = False,
            **kwargs
    ):
        # logger
        self._logger = logging.getLogger('sqp_solver')
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
        self._eq_constr = eq_constr
        self._ineq_constr = ineq_constr
        self._constr_fns = []
        if eq_constr is not None and eq_constr:
            self._constr_fns.extend(eq_constr)
        if ineq_constr is not None and ineq_constr:
            self._constr_fns.extend(ineq_constr)
        assert self._constr_fns

        self._eq_mult_dims = 0 if eq_constr is None else len(eq_constr)
        self._logger.info(f'Dimensions of the equality multiplier vector: {self._eq_mult_dims}.')

        self._ineq_mult_dims = 0 if ineq_constr is None else len(ineq_constr)
        self._logger.info(f'Dimensions of the inequality multiplier vector: {self._ineq_mult_dims}.')
        self._multiplier_dims = self._eq_mult_dims + self._ineq_mult_dims

        self._initial_multipliers = jnp.zeros(shape=(self._multiplier_dims,), dtype=jnp.float32)
        self._logger.info(f'Dimensions of the multiplier vector: {self._multiplier_dims}.')

        # convergence
        self._convergence_fn = self._get_convergence_fn(conv_criteria)
        self._max_iter = max_iter
        self._tol = conv_tol

        # learning rate
        self._step_size_fn = self._get_step_size_fn(line_search)
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._sigma = sigma

        # save intermediate step info
        self._cache = {}
        self._step_cache = {}

        # grad & hessian functions
        # compile with JAX in advance
        self._grad_loss_x_fn = grad(self._loss_fn)  # N-by-1
        self._grad_lagr_x_fn = grad(self._lagrangian)
        self._grad_constr_x_fns = [grad(f) for f in self._constr_fns]

        self._direction = direction
        if direction == Direction.EXACT_NEWTON:
            self._hess_lagr_xx_fn = self._hessian_exact(fn=self._lagrangian)  # N-by-N
        elif direction == Direction.BFGS:
            self._hess_lagr_xx_fn = self._hessian_bfgs_approx  # N-by-N
        elif direction == Direction.GAUSS_NEWTON:
            assert gn_hessian_fn is not None
            self._hess_lagr_xx_fn = gn_hessian_fn
        elif direction == Direction.STEEPEST_DESCENT:
            self._hess_lagr_xx_fn = self._hessian_sd_approx  # N-by-N
        else:
            raise NotImplementedError

        self._reg = reg

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:
        x_k = self._initial_x
        self._logger.info(f'Initial guess is: {x_k}.')

        m_k = self._initial_multipliers
        self._logger.info(f'Initial multipliers are: {m_k}.')

        converged, conv_penalty = self._convergence_fn(x_k, m_k)
        k = 0
        B_prev = None
        x_prev = None
        g_prev = None
        g_k = None
        while (not converged) and (k < self._max_iter):
            # calculate B_k
            if self._direction == Direction.EXACT_NEWTON:
                B_k = self._hess_lagr_xx_fn(x_k, m_k)
            elif self._direction == Direction.BFGS:
                g_k = self._grad_lagr_x_fn(x_k, m_k)
                B_k = self._hess_lagr_xx_fn(B_prev, g_k, g_prev, x_k, x_prev)
            elif self._direction == Direction.GAUSS_NEWTON:
                # TODO correct params
                g_k = self._grad_lagr_x_fn(x_k, m_k)
                B_k = self._hess_lagr_xx_fn(g_k)
            else:
                B_k = self._hess_lagr_xx_fn()

            # ensure B_k is pd
            # B_k_is_pd = None
            # eq_constr_grad_x = self._eval_eq_constraint_gradients(x_k)
            # B_k, B_k_is_pd = self._regularize_hessian(B_k, eq_constr_grad_x)
            B_k, B_k_is_pd = self._regularize_full_hessian(B_k)

            # find direction by solving QP
            grad_loss_x = self._grad_loss_x_fn(x_k)
            c_k = self._eval_constraints(x_k)
            constr_grad_x = self._eval_constraint_gradients(x_k)
            d_k = self._solve_qp(B_k, grad_loss_x, c_k, constr_grad_x, self._eq_mult_dims)

            # calculate step size (line search)
            alpha_k = self._step_size_fn(x_k=x_k, grad_loss_x=grad_loss_x, direction=d_k, max_iter=7)

            # update params for BFGS
            if self._direction == Direction.BFGS:
                B_prev = B_k
                x_prev = x_k
                g_prev = g_k

            # save cache + logs
            loss = self._loss_fn(x_k)
            self._step_cache.update({
                'k': k, 'B_k_is_pd': B_k_is_pd,
                'alpha': alpha_k, 'sigma': self._sigma,
                'loss': loss, 'penalty': conv_penalty, 'x': x_k,
            })
            self._cache[k] = dict(self._step_cache)
            self._log_step()

            # update state
            x_k += alpha_k * d_k[:self._x_dims]
            m_k = (1 - alpha_k) * m_k + alpha_k * d_k[self._x_dims:]

            self._logger.info(f'Updated x_k: {x_k}.')

            # TODO check update sigma
            self._sigma = jnp.max(jnp.abs(m_k)) + 0.1

            # check convergence
            converged, conv_penalty = self._convergence_fn(x_k, m_k)

            # increment counter
            k += 1

        # log and print last results
        loss = self._loss_fn(x_k)

        self._step_cache.update({
            'k': k, 'sigma': self._sigma,
            'loss': loss, 'penalty': conv_penalty, 'x': x_k,
        })
        self._cache[k] = dict(self._step_cache)
        self._log_step()

        # fill additional info
        info = (converged, loss, k, self._cache)

        return x_k, info
